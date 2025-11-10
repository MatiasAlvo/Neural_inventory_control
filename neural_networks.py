from shared_imports import *
from quantile_forecaster import FullyConnectedForecaster
import torch.nn.functional as F
import math


class MyNeuralNetwork(nn.Module):

    def __init__(self, args, device='cpu'):
        """"
        Initialize neural network with given parameters

        Parameters
        ----------
        args: dictionary
            Dictionary with the following
            - inner_layer_activations: dictionary with the activation function for each neural net module (master, stores, warehouse, context net)
            - output_layer_activation: dictionary with the activation function for the output layer of each neural net module
            - neurons_per_hidden_layer: dictionary with the number of neurons for each hidden layer of each neural net module
            - output_sizes: dictionary with the output size for each neural net module
            - initial_bias: dictionary with the initial bias for each neural net module
        device: str
            Device where the neural network will be stored
        """

        super().__init__() # initialize super class
        self.device = device

        # Some models are not trainable (e.g. news-vendor policies), so we need to flag it to the trainer
        # so it does not perform greadient steps (otherwise, it will raise an error)
        self.trainable = True
        
        # Get gradient clipping value from config (if specified)
        self.gradient_clipping_norm_value = args.get('gradient_clipping_norm_value', None)
        
        # Define activation functions, which will be called in forward method
        self.activation_functions = {
            'relu': nn.ReLU(), 
            'elu': nn.ELU(), 
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
            }
        
        # If warehouse_upper_bound is not None, then we will use it to multiply the output of the warehouse neural network
        self.warehouse_upper_bound = 0

        self.layers = {}
        # Create nn.ModuleDict to store multiple neural networks
        self.net = self.create_module_dict(args)

        # Initialize bias if given
        if args['initial_bias'] is not None:
            for key, val in args['initial_bias'].items():
                if val is not None:
                    # Position of last linear layer depends on whether there is an output layer activation function
                    pos = -2 if args['output_layer_activation'][key] else -1
                    self.initialize_bias(key, pos, val)
    
    def forward(self, observation):
        raise NotImplementedError
    
    def create_module_dict(self, args):
        """
        Create a dictionary of neural networks, where each key is a neural network module (e.g. master, store, warehouse, context)
        """
        
        return nn.ModuleDict({key: 
                              self.create_sequential_net(
                                  key,
                                  args['inner_layer_activations'][key], 
                                  args['output_layer_activation'][key], 
                                  args['neurons_per_hidden_layer'][key], 
                                  args['output_sizes'][key]
                                  ) 
                                  for key in args['output_sizes']
                                  }
                                  )
    
    def create_sequential_net(self, name, inner_layer_activations, output_layer_activation, neurons_per_hidden_layer, output_size):
        """
        Create a neural network with the given parameters
        """

        # Define layers
        layers = []
        for i, output_neurons in enumerate(neurons_per_hidden_layer):
            layers.append(nn.LazyLinear(output_neurons))
            layers.append(self.activation_functions[inner_layer_activations])

        if len(neurons_per_hidden_layer) == 0:
            layers.append(nn.LazyLinear(output_size))

        # If there is at least one inner layer, then we know the last layer's shape
        # We therefore create a Linear layer in case we want to initialize it to a certain value (not possible with LazyLinear)
        else: 
            layers.append(nn.Linear(neurons_per_hidden_layer[-1], output_size))
        
        # If output_layer_activation is not None, then we add the activation function to the last layer
        if output_layer_activation is not None:
            layers.append(self.activation_functions[output_layer_activation])
        
        self.layers[name] = layers

        # Define network as a sequence of layers
        return nn.Sequential(*layers)

    def initialize_bias(self, key, pos, value):
        self.layers[key][pos].bias.data.fill_(value)
    
    def apply_proportional_allocation(self, desired_allocations, available_inventory, transshipment=False):
        """
        Apply proportional allocation when desired allocations exceed available inventory.
        
        Args:
            desired_allocations: Tensor of shape [batch_size, n_allocations] with desired quantities
            available_inventory: Tensor of shape [batch_size] or [batch_size, 1] with available inventory
            transshipment: If True, the supplying node cannot hold inventory (no clipping at 1)
        
        Returns:
            Allocated quantities scaled proportionally when inventory is insufficient
        """
        # Ensure available_inventory is 1D
        if available_inventory.dim() > 1:
            available_inventory = available_inventory.sum(dim=1)
        
        # Sum of all desired allocations
        sum_desired = desired_allocations.sum(dim=1)
        
        # Calculate scaling factor
        scaling_factor = available_inventory / (sum_desired + 1e-10)
        
        # Apply clipping only if not transshipment
        if not transshipment:
            scaling_factor = torch.clip(scaling_factor, max=1.0)
        
        # Apply scaling to all allocations
        return desired_allocations * scaling_factor[:, None]
    
    def apply_softmax_feasibility_function(self, store_intermediate_outputs, warehouse_inventory, transshipment=False):
        """
        Apply softmax across store intermediate outputs, and multiply by warehouse inventory on-hand
        If transshipment is False, then we add a column of ones to the softmax inputs, to allow for inventory to be held at the warehouse
        """

        total_warehouse_inv = warehouse_inventory[:, :, 0].sum(dim=1)  # warehouse's inventory on-hand
        softmax_inputs = store_intermediate_outputs

        # If warehouse can hold inventory, then concatenate a tensor of ones to the softmax inputs
        if not transshipment:
            softmax_inputs = torch.cat((
                softmax_inputs, 
                torch.ones_like(softmax_inputs[:, 0]).to(self.device)[:, None]
                ), 
                dim=1
                )
        softmax_outputs = self.activation_functions['softmax'](softmax_inputs)

        # If warehouse can hold inventory, then remove last column of softmax outputs
        if not transshipment:
            softmax_outputs = softmax_outputs[:, :-1]

        return torch.multiply(
            softmax_outputs, 
            total_warehouse_inv[:, None]
            )

    def flatten_then_concatenate_tensors(self, tensor_list, dim=1):
        """
        Flatten tensors in tensor_list, and concatenate them along dimension dim
        """

        return torch.cat([
            tensor.flatten(start_dim=dim) for tensor in tensor_list
            ], 
            dim=dim)
    
    def concatenate_signal_to_object_state_tensor(self, object_state, signal):
        """
        Concatenate signal (e.g. context vector) to every location's local state (e.g. store inventories or warehouse inventories).
        Signal is tipically of shape (num_samples, signal_dim) and object_state is of shape (num_samples, n_objects, object_state_dim),
        and results in a tensor of shape (num_samples, n_objects, object_state_dim + signal_dim)
        """

        n_objects = object_state.size(1)
        signal = signal.unsqueeze(1).expand(-1, n_objects, -1)
        return torch.cat((object_state, signal), dim=2)
    
    def unpack_args(self, args, keys):
        """
        Unpacks arguments from a dictionary
        """
        return [args[key] for key in keys] if len(keys) > 1 else args[keys[0]]

class DataDrivenNet(MyNeuralNetwork):
    """
    Fully connected neural network for data-driven ordering.
    Supports both single and multiple warehouse configurations.
    """
    
    def __init__(self, args, scenario=None, device='cpu'):
        super().__init__(args, device)
        self.scenario = scenario
    
    def forward(self, observation):
        """
        Utilize inventory information and demand features to output orders.
        For real data: uses past demands, underage costs, days from Christmas.
        For synthetic data: uses mean and std demand parameters.
        """
        # Build input features list
        input_features = [observation['store_inventories']]

        print(f"length of input_features: {len(observation['store_inventories'][0, 0])}")
        
        # Add demand-related features (data driven is always real data)
        input_features.extend([
            observation['past_demands'],
        ])
        print(f"length of input_features: {len(observation['past_demands'][0, 0])}")
        
        # # Add instock features if available
        # if 'past_instocks' in observation:
        #     input_features.append(observation['past_instocks'])
        
        # # Add time product features if available
        # time_product_feature_keys = [k for k in observation.keys() if k.startswith('past_time_product_feature_')]
        # for key in sorted(time_product_feature_keys):  # Sort to ensure consistent ordering
        #     input_features.append(observation[key])
        
        # Add all time features specified in observation_params
        if hasattr(self, 'scenario') and self.scenario and 'observation_params' in self.scenario.__dict__:
            time_features_list = self.scenario.observation_params.get('time_features', [])
            for feature_name in time_features_list:
                print(f"Feature name: {feature_name}")
                if feature_name in observation:
                    print(f'found')
                    input_features.append(observation[feature_name])
        
        # # Add product features if available - need to expand to match other features
        # if 'product_features' in observation:
        #     # Product features have shape [samples, features], need to expand to [samples, stores, features]
        #     # product_features = observation['product_features']  # [samples, features]
        #     product_features = observation['product_features'].clone()
        #     # n_stores = observation['store_inventories'].size(1)
        #     # Expand to [samples, stores, features] to match other features
        #     # Use repeat() to create a proper copy that maintains gradient flow
        #     # product_features_expanded = product_features.unsqueeze(1).repeat(1, n_stores, 1)
        #     product_features_expanded = product_features
        #     # product_features_expanded = product_features.detach().unsqueeze(1).expand(-1, n_stores, -1).contiguous()


        #     input_features.append(product_features_expanded)
        
        # Flatten and concatenate all features
        input_tensor = self.flatten_then_concatenate_tensors(input_features)
        print(f"input_tensor shape: {input_tensor.shape}")
        print()
        # print("Checking gradients:")
        # for i, feat in enumerate(input_features):
        #     print(f"Feature {i}: requires_grad={feat.requires_grad}")
        #     print(f"Feature {i}: shape={feat.shape}")
        # print(f"input_tensor requires_grad: {input_tensor.requires_grad}")
        
        # Single warehouse or no warehouse
        return {'stores': self.net['master'](input_tensor).unsqueeze(2)}

class DataDrivenNetWithForecasts(MyNeuralNetwork):
    """
    Fully connected neural network for data-driven ordering with LGBM forecast features.
    Extends DataDrivenNet by incorporating forecast features from the data pipeline.
    Supports both single and multiple warehouse configurations.
    """
    
    def __init__(self, args, scenario=None, device='cpu'):
        super().__init__(args, device)
        self.scenario = scenario
    
    def forward(self, observation):
        """
        Utilize inventory information, demand features, and LGBM forecasts to output orders.
        For real data: uses past demands, underage costs, days from Christmas, and forecast features.
        For synthetic data: uses mean and std demand parameters.
        """
        # Build input features list (same as DataDrivenNet)
        input_features = [observation['store_inventories']]
        
        # Add demand-related features (data driven is always real data)
        input_features.extend([
            observation['past_demands'],
        ])
        
        # Add instock features if available
        if 'past_instocks' in observation:
            input_features.append(observation['past_instocks'])
        
        # Add time product features if available
        time_product_feature_keys = [k for k in observation.keys() if k.startswith('past_time_product_feature_')]
        for key in sorted(time_product_feature_keys):  # Sort to ensure consistent ordering
            input_features.append(observation[key])
        
        # Add all time features specified in observation_params
        if hasattr(self, 'scenario') and self.scenario and 'observation_params' in self.scenario.__dict__:
            time_features_list = self.scenario.observation_params.get('time_features', [])
            for feature_name in time_features_list:
                if feature_name in observation:
                    input_features.append(observation[feature_name])
        
        # Add product features if available - need to expand to match other features
        if 'product_features' in observation:
            product_features = observation['product_features'].clone()
            product_features_expanded = product_features
            input_features.append(product_features_expanded)
        
        # Add LGBM forecast features if available (now from observation, not loaded in NN)
        if 'forecast_features' in observation:
            input_features.append(observation['forecast_features'])
        
        # Flatten and concatenate all features
        input_tensor = self.flatten_then_concatenate_tensors(input_features)
        
        # Single warehouse or no warehouse
        return {'stores': self.net['master'](input_tensor).unsqueeze(2)}
        # return {'stores': self.net['master'](input_tensor).unsqueeze(2) + 0.5}

class MeanLastXBaseline(MyNeuralNetwork):
    """
    Weekly mean-of-last-X baseline:
      base = mean(past_demands over last K weeks)
      target = base * coverage_weeks
      order = relu(target - inventory_position)
    """

    # --- tiny no-input network that outputs a positive scalar via softplus ---
    class _ScalarCoverage(nn.Module):
        def __init__(self, init_weeks: float):
            super().__init__()
            # initialize raw so softplus(raw) ~= init_weeks
            self.raw = nn.Parameter(torch.log(torch.exp(torch.tensor(init_weeks)) - 1.0))
        def forward(self):
            return F.softplus(self.raw)  # > 0

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)
        print(f'args: {args}')

        wml = args.get('wml', {})
        self.lookback_weeks    = int(wml.get('lookback_weeks', 8))
        self.round_orders      = bool(wml.get('round_orders', False))
        self.use_instock_mask = bool(wml.get('use_instock_mask', True))
        self.eps               = float(wml.get('eps', 1e-6))

        self.learn_coverage = args['mlx']['learn_coverage']
        init_cov = float(args['mlx']['coverage_weeks'])

        if self.learn_coverage:
            # one-parameter "NN" with no inputs
            self.coverage_net = self._ScalarCoverage(init_cov)
            self.trainable = True
        else:
            self.register_buffer('coverage_weeks_const', torch.tensor(init_cov, dtype=torch.float32))
            self.trainable = False

    def _coverage_weeks(self):
        return self.coverage_net() if self.learn_coverage else self.fixed_coverage_weeks

    def forward(self, observation):
        past_demands = observation['past_demands']            # [B,S,Lw]
        store_inventories = observation['store_inventories']  # [B,S,Dinv]
        instocks = observation.get('past_instocks', None)   # [B,S,Lw] or None

        B, S, Lw = past_demands.shape
        K = min(Lw, self.lookback_weeks)
        if K <= 0:
            return {'stores': torch.zeros(B, S, 1, device=past_demands.device)}

        # NOTE: preserving your masking logic exactly as provided
        if instocks is not None:
            mask = instocks.float()                          # 1 where instock occurred
            masked_demands = past_demands * mask
            sum_demands = masked_demands[:, :, -K:].sum(dim=2)
            count = mask[:, :, -K:].sum(dim=2).clamp(min=self.eps)
            mean_weekly_demand = sum_demands / count
        else:
            mean_weekly_demand = past_demands[:, :, -K:].mean(dim=2)

        target = mean_weekly_demand * self._coverage_weeks()  # scalar * [B,S] -> [B,S]
        inv_pos = store_inventories.sum(dim=2)                # [B,S]
        orders = torch.relu(target - inv_pos)                 # [B,S]
        if self.round_orders:
            orders = torch.round(orders)

        return {'stores': orders.unsqueeze(2)}                # [B,S,1]

class JustInTime(MyNeuralNetwork):
    """"
    Non-admissible policy, that looks into the future and orders so that units arrive just-in-time so satisfy demand
    Can be considered as an "oracle policy"
    """

    def __init__(self, args, scenario=None, device='cpu'):
        super().__init__(args=args, device=device) # Initialize super class
        self.scenario = scenario
        self.trainable = False

    def forward(self, observation):
        """
        Get store allocation by looking into the future and ordering so that units arrive just-in-time to satisfy demand
        Supports both single and multiple warehouse configurations.
        """
        
        current_period = observation['current_period']
        demands, period_shift = self.unpack_args(observation['internal_data'], ["demands", "period_shift"])
        
        num_samples, num_stores, max_lead_time = demands.shape
        
        # Check if we have warehouses
        n_warehouses = observation['warehouse_inventories'].size(1) if 'warehouse_inventories' in observation else 0
        # Handle no warehouse case first
        if n_warehouses == 0:
            # No warehouse case (stores only)
            lead_times = observation['lead_times'][:, :, 0]  # Use first warehouse
            
            future_demands = torch.stack([
                demands[:, j][
                    torch.arange(num_samples), 
                    torch.clip(
                        current_period.to(self.device) + period_shift + lead_times[:, j].long(),
                        max=max_lead_time - 1
                    )
                ] 
                for j in range(num_stores)
            ], dim=1)
            
            return {"stores": torch.clip(future_demands, min=0).unsqueeze(2)}
        
        # One or more warehouses case
        # lead_times is 3D [batch, n_stores, n_warehouses]
        lead_times = observation['lead_times']
        warehouse_lead_times = observation['warehouse_lead_times']  # [batch, n_warehouses]
        
        # Get warehouse-store adjacency matrix
        warehouse_store_adjacency = self.scenario.problem_params['warehouse_store_adjacency']
        adjacency_tensor = torch.tensor(warehouse_store_adjacency, dtype=torch.float32, device=lead_times.device)
        
        # Initialize output tensors
        store_allocation = torch.zeros(num_samples, num_stores, n_warehouses, device=lead_times.device)
        warehouse_orders = torch.zeros(num_samples, n_warehouses, device=lead_times.device)
        
        # For each store, find its connected warehouse and calculate orders
        for store_idx in range(num_stores):
            # Find connected warehouses for this store
            connected_warehouses = adjacency_tensor[:, store_idx].nonzero(as_tuple=True)[0]
            
            if len(connected_warehouses) > 0:
                # Pick the warehouse with shortest lead time for this store
                # Get lead times for all connected warehouses for this store
                store_lead_times_to_warehouses = lead_times[:, store_idx, connected_warehouses]
                # Find the warehouse index with minimum lead time (averaged across batch)
                min_lead_time_idx = torch.argmin(store_lead_times_to_warehouses.mean(dim=0))
                w_idx = connected_warehouses[min_lead_time_idx].item()
                
                
                # Calculate what the store needs to order from this warehouse
                # Orders placed now arrive at current_period + lead_time
                store_demand_time = torch.clip(
                    current_period.to(self.device) + lead_times[:, store_idx, w_idx].long() + period_shift,
                    max=max_lead_time - 1
                )
                
                store_future_demand = demands[:, store_idx][
                    torch.arange(num_samples), 
                    store_demand_time
                ]
                
                # Assign this demand to the warehouse-store allocation
                store_allocation[:, store_idx, w_idx] = store_future_demand
                
                
                # Calculate what the warehouse needs to order from supplier
                # Orders placed now arrive at warehouse at current_period + warehouse_lead_time
                # Then take lead_time more to reach store
                warehouse_demand_time = torch.clip(
                    current_period.to(self.device) + 
                    warehouse_lead_times[:, w_idx].long() + lead_times[:, store_idx, w_idx].long() + period_shift,
                    max=max_lead_time - 1
                )
                
                warehouse_future_demand = demands[:, store_idx][
                    torch.arange(num_samples),
                    warehouse_demand_time
                ]
                
                # Add to warehouse orders (accumulate for all stores connected to this warehouse)
                warehouse_orders[:, w_idx] += warehouse_future_demand
                
        
        return {
            "stores": torch.clip(store_allocation, min=0),
            "warehouses": torch.clip(warehouse_orders, min=0).unsqueeze(2)  # [batch, n_warehouses, 1]
        }

class NeuralNetworkCreator:
    """
    Class to create neural networks
    """

    def set_default_output_size(self, module_name, problem_params):
        
        n_stores = problem_params['n_stores']
        n_warehouses = problem_params['n_warehouses']
        
        # For multiple warehouses, we need outputs for each store-warehouse pair
        if n_warehouses > 1:
            master_size = n_stores * n_warehouses + n_warehouses
        else:
            master_size = n_stores + n_warehouses
            
        default_sizes = {
            'master': master_size, 
            'store': 1, 
            'warehouse': 1, 
            'context': None
            }
        return default_sizes[module_name]

    def get_architecture(self, name):

        architectures = {
            'data_driven': DataDrivenNet,
            'data_driven_with_forecasts': DataDrivenNetWithForecasts,
            'mean_last_x': MeanLastXBaseline,
            'just_in_time': JustInTime,
            }
        return architectures[name]
    
    def get_warehouse_upper_bound(self, warehouse_upper_bound_mult, scenario, device='cpu'):
        """
        Get the warehouse upper bound, which is the sum of all store demands multiplied 
        by warehouse_upper_bound_mult (specified in config file)
        """
        mean = scenario.store_params['demand']['mean']
        if type(mean) == float:
            mean = [mean]
        return torch.tensor([warehouse_upper_bound_mult*sum(mean)]).float().to(device)
    
    def create_neural_network(self, scenario, nn_params, device='cpu'):

        nn_params_copy = copy.deepcopy(nn_params)

        # If not specified in config file, set output size to default value
        for key, val in nn_params_copy['output_sizes'].items():
            if val is None:
                nn_params_copy['output_sizes'][key] = self.set_default_output_size(key, scenario.problem_params)

        # Special handling for architectures that need scenario
        if nn_params_copy['name'] in ['gnn', 'vanilla_warehouse', 'just_in_time', 'data_driven', 'data_driven_with_forecasts']:
            model = self.get_architecture(nn_params_copy['name'])(
                nn_params_copy,
                scenario,
                device=device
            )
        else:
            model = self.get_architecture(nn_params_copy['name'])(
                nn_params_copy, 
                device=device
            )
        
        # Calculate warehouse upper bound if specified in config file
        if 'warehouse_upper_bound_mult' in nn_params.keys():
            model.warehouse_upper_bound = self.get_warehouse_upper_bound(nn_params['warehouse_upper_bound_mult'], scenario, device)
        
        return model.to(device)
