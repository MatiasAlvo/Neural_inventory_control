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
    
    def apply_proportional_allocation(self, store_intermediate_outputs, warehouse_inventories):
        """
        Apply proportional allocation feasibility enforcement function to store intermediate outputs.
        It assigns inventory proportionally to the store order quantities, whenever inventory at the
        warehouse is not sufficient.
        """

        total_limiting_inventory = warehouse_inventories[:, :, 0].sum(dim=1)  # Total inventory at the warehouse
        sum_allocation = store_intermediate_outputs.sum(dim=1)  # Sum of all store order quantities

        # Multiply current allocation by minimum between inventory/orders and 1
        final_allocation = \
            torch.multiply(store_intermediate_outputs,
                           torch.clip(total_limiting_inventory / (sum_allocation + 0.000000000000001), max=1)[:, None])
        return final_allocation
    
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

class VanillaOneStore(MyNeuralNetwork):
    """
    Fully connected neural network for settings with one store and no warehouse
    """
    
    def forward(self, observation):
        """
        Uses store inventories as input and directly outputs store orders
        """
        x = observation['store_inventories']

        # Flatten input, except for the batch dimension
        x = x.flatten(start_dim=1)

        # Pass through network
        # NN architecture has to be such that output is non-negative
        x = self.net['master'](x) + 1
        x = self.activation_functions['softplus'](x)

        return {'stores': x}

class BaseStock(MyNeuralNetwork):
    """
    Base stock policy
    """

    def forward(self, observation):
        """
        Get a base-level, which is the same across all stores and sample
        Calculate allocation as max(base_level - inventory_position, 0)
        """
        x = observation['store_inventories']
        inv_pos = x.sum(dim=2)
        x = self.net['master'](torch.tensor([0.0]).to(self.device))  # Constant base stock level
        return {'stores': torch.clip(x - inv_pos, min=0)} # Clip output to be non-negative

class EchelonStock(MyNeuralNetwork):
    """
    Echelon stock policy
    """

    def forward(self, observation):
        """
        Get a base-level for each location, which is the same across all samples.
        We obtain base-levels via partial sums, which allowed us to avoid getting "stuck" in bad local minima.
        Calculate allocation as max(base_level - inventory_position, 0) truncated above by previous location's inventory/
        Contrary to how we define other policies, we will follow and indexing where locations are ordered from upstream to downstream (last is store).
        """
        store_inventories, warehouse_inventories, echelon_inventories = self.unpack_args(
            observation, ['store_inventories', 'warehouse_inventories', 'echelon_inventories'])
        n_extra_echelons = echelon_inventories.size(1)
        
        x = self.activation_functions['softplus'](self.net['master'](torch.tensor([0.0]).to(self.device)) + 10.0)  # Constant base stock levels
        # The base level for each location wil be calculated as the outputs corresponding to all downstream locations and its own
        base_levels = torch.cumsum(x, dim=0).flip(dims=[0])

        # Inventory position (NOT echelon inventory position) for each location
        stacked_inv_pos = torch.concat((echelon_inventories.sum(dim=2), warehouse_inventories.sum(dim=2), store_inventories.sum(dim=2)), dim=1)
        
        # Tensor with the inventory on hand for the preceding location for each location k
        # For the left-most location, we set it to a large number, so that it does not truncate the allocation
        shifted_inv_on_hand = torch.concat((
            1000000*torch.ones_like(warehouse_inventories[:, :, 0]), 
            echelon_inventories[:, :, 0], 
            warehouse_inventories[:, :, 0]), 
            dim=1
            )

        # print(f'base_levels: {base_levels}')
        # print(f'stacked_inv_pos: {stacked_inv_pos[0]}')
        # print(f'echelon_pos: {torch.stack([(stacked_inv_pos[:, k:].sum(dim=1)) for k in range(2 + n_extra_echelons)], dim=1)[0]}')

        # Allocations before truncating by previous locations inventory on hand.
        # We get them by subtracting the echelon inventory position (i.e., sum of inventory positions from k onwards) from the base levels, 
        # and truncating below by 0.
        tentative_allocations = torch.clip(
            torch.stack([base_levels[k] - (stacked_inv_pos[:, k:].sum(dim=1)) 
                         for k in range(2 + n_extra_echelons)], 
                         dim=1), 
                         min=0)
        
        # print(f'tentative_allocations: {tentative_allocations[0]}')
        # Truncate below by previous locations inventory on hand
        allocations = torch.minimum(tentative_allocations, shifted_inv_on_hand)

        # print(f'shifted_inv_on_hand: {shifted_inv_on_hand[0]}')
        # print(f'allocations: {allocations[0]}')

        # print(f'stacked_inv_on_hand.shape: {shifted_inv_on_hand.shape}')
        # print()

        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : n_extra_echelons],
                } 

class CappedBaseStock(MyNeuralNetwork):
    """"
    Simlar to BaseStock, but with a cap on the order
    """

    def forward(self, observation):
        """
        Get a base-level and cap, which is the same across all stores and sample
        Calculate allocation as min(base_level - inventory_position, cap) and truncated below from 0
        """
        x = observation['store_inventories']
        inv_pos = x.sum(dim=2)
        x = self.net['master'](torch.tensor([0.0]).to(self.device))  # Constant base stock level
        base_level, cap = x[0], x[1]  # We interpret first input as base level, and second output as cap on the order
        
        return {'stores': torch.clip(base_level - inv_pos, min=torch.tensor([0.0]).to(self.device), max=cap)} # Clip output to be non-negative


class VanillaSerial(MyNeuralNetwork):
    """
    Vanilla NN for serial system
    """

    def forward(self, observation):
        """
        We apply a sigmoid to the output of the master neural network, and multiply by the inventory on hand for the preceding location,
        except for the left-most location, where we multiply by an upper bound on orders.
        """
        store_inventories, warehouse_inventories, echelon_inventories = self.unpack_args(
            observation, ['store_inventories', 'warehouse_inventories', 'echelon_inventories'])
        n_extra_echelons = echelon_inventories.size(1)
        
        input_tensor = self.flatten_then_concatenate_tensors([store_inventories, warehouse_inventories, echelon_inventories])
        x = self.net['master'](torch.tensor(input_tensor).to(self.device))  # Constant base stock levels
        # print(f'self.warehouse_upper_bound: {self.warehouse_upper_bound}')
        # assert False

        # Tensor with the inventory on hand for the preceding location for each location k
        # For the left-most location, we set it to an upper bound (same as warehouse upper bound). Currently 4 times mean demand.
        shifted_inv_on_hand = torch.concat((
            self.warehouse_upper_bound.unsqueeze(1).expand(echelon_inventories.shape[0], -1),
            echelon_inventories[:, :, 0], 
            warehouse_inventories[:, :, 0]), 
            dim=1
            )
        # print(f'x: {x.shape}')
        # print(f'shifted_inv_on_hand: {shifted_inv_on_hand.shape}')
        # print(f"self.activation_functions['sigmoid'](x): {self.activation_functions['sigmoid'](x)[0]}")
        allocations = self.activation_functions['sigmoid'](x)*shifted_inv_on_hand
        # print(f'allocations[0]: {allocations[0]}')


        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : n_extra_echelons],
                } 


class VanillaOneWarehouse(MyNeuralNetwork):
    """
    Fully connected neural network for settings with one warehouse (or transshipment center) and many stores
    """

    def forward(self, observation):
        """
        Use store and warehouse inventories and output intermediate outputs for stores and warehouses.
        For stores, apply softmax to intermediate outputs (concatenated with a 1 when inventory can be held at the warehouse)
          and multiply by warehouse inventory on-hand
        For warehouses, apply sigmoid to intermediate outputs and multiply by warehouse upper bound.
        """
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        input_tensor = torch.cat((store_inventories.flatten(start_dim=1), warehouse_inventories.flatten(start_dim=1)), dim=1)
        intermediate_outputs = self.net['master'](input_tensor)
        store_intermediate_outputs, warehouse_intermediate_outputs = intermediate_outputs[:, :n_stores], intermediate_outputs[:, n_stores:]

        # Apply softmax to store intermediate outputs
        # If class name is not VanillaTransshipment, then we will add a column of ones to the softmax inputs (so that inventory can be held at warehouse)
        store_allocation = \
            self.apply_softmax_feasibility_function(
                store_intermediate_outputs, 
                warehouse_inventories,
                transshipment=(self.__class__.__name__ == 'VanillaTransshipment')
                )
        
        # Apply sigmoid to warehouse intermediate outputs and multiply by warehouse upper bound
        warehouse_allocation = self.activation_functions['sigmoid'](warehouse_intermediate_outputs)*(self.warehouse_upper_bound.unsqueeze(1))

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class SymmetryAware(MyNeuralNetwork):
    """
    Symmetry-aware neural network for settings with one warehouse and many stores
    """

    def forward(self, observation):
        """
        Use store and warehouse inventories and output a context vector.
        Then, use the context vector alongside warehouse/store local state to output intermediate outputs for warehouses/store.
        For stores, interpret intermediate outputs as ordered, and apply proportional allocation whenever inventory is scarce.
        For warehouses, apply sigmoid to intermediate outputs and multiply by warehouse upper bound.
        """

        # Get tensor of store parameters
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        store_params = torch.stack([observation[k] for k in ['mean', 'std', 'underage_costs', 'lead_times']], dim=2)
        
        # Get context vector using local state (and without using local parameters)
        input_tensor = self.flatten_then_concatenate_tensors([store_inventories, warehouse_inventories])
        context = self.net['context'](input_tensor)

        # Concatenate context vector to warehouselocal state, and get intermediate outputs
        warehouses_and_context = \
                self.concatenate_signal_to_object_state_tensor(warehouse_inventories, context)
        warehouse_intermediate_outputs = self.net['warehouse'](warehouses_and_context)[:, :, 0]

        # Concatenate context vector to each store's local state and parameters, and get intermediate outputs
        store_inventory_and_params = torch.concat([store_inventories, store_params], dim=2)
        stores_and_context = \
                self.concatenate_signal_to_object_state_tensor(store_inventory_and_params, context)
        store_intermediate_outputs = self.net['store'](stores_and_context)[:, :, 0]  # third dimension has length 1, so we remove it

        # Apply proportional allocation whenever inventory at the warehouse is scarce.
        store_allocation = self.apply_proportional_allocation(
            store_intermediate_outputs, 
            warehouse_inventories
            )

        # Apply sigmoid to warehouse intermediate outputs and multiply by warehouse upper bound        
        # Sigmoid is applied if specified in the config file!
        warehouse_allocation = warehouse_intermediate_outputs*(self.warehouse_upper_bound.unsqueeze(1))

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

    
class VanillaTransshipment(VanillaOneWarehouse):
    """
    Fully connected neural network for setting with one transshipment center (that cannot hold inventory) and many stores
    """

    pass

class DataDrivenNet(MyNeuralNetwork):
    """
    Fully connected neural network
    """
    
    def forward(self, observation):
        """
        Utilize inventory on-hand, past demands, underage costs, and days from Christmas to output store orders directly
        """

        # Input tensor is given by full store inventories, past demands, underage costs for 
        # each sample path, and days from Christmas
        input_tensor = self.flatten_then_concatenate_tensors(
                [observation['store_inventories']] + 
                [observation[key] for key in ['past_demands', 'underage_costs', 'days_from_christmas', 'lead_times']]
            )

        return {'stores': self.net['master'](input_tensor)} # Clip output to be non-negative

class QuantilePolicy(MyNeuralNetwork):
    """
    Base class for quantile policies.
    These policies rely on mappings from features to desired quantiles, and then "invert" the quantiles using a 
    quantile forecaster to get base-stock levels.
    """

    def __init__(self, args, device='cpu'):

        super().__init__(args=args, device=device) # Initialize super class
        self.fixed_nets = {'quantile_forecaster': self.load_forecaster(args, requires_grad=False)}
        self.allow_back_orders = False  # We will set to True only for non-admissible ReturnsNV policy
        
    def load_forecaster(self, nn_params, requires_grad=True):
        """"
        Create quantile forecaster and load weights from file
        """
        quantile_forecaster = FullyConnectedForecaster([128, 128], lead_times=nn_params['forecaster_lead_times'], qs=np.arange(0.05, 1, 0.05))
        quantile_forecaster = quantile_forecaster
        quantile_forecaster.load_state_dict(torch.load(f"{nn_params['forecaster_location']}"))
        
        # Set requires_grad to False for all parameters if we are not training the forecaster
        for p in quantile_forecaster.parameters():
            p.requires_grad_(requires_grad)
        return quantile_forecaster.to(self.device)
    
    def forecast_base_stock_allocation(self, past_demands, days_from_christmas, store_inventories, lead_times, quantiles, allow_back_orders=False):
        """"
        Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
        """

        base_stock_levels = \
            self.fixed_nets['quantile_forecaster'].get_quantile(
                torch.cat([
                    past_demands, 
                    days_from_christmas.unsqueeze(1).expand(past_demands.shape[0], past_demands.shape[1], 1)
                    ], dim=2
                    ), 
                    quantiles, 
                    lead_times
                    )

        # If we allow back orders, then we don't clip at zero from below
        if allow_back_orders:
            store_allocation = base_stock_levels - store_inventories.sum(dim=2)
        else:
            store_allocation = torch.clip(base_stock_levels - store_inventories.sum(dim=2), min=0)
        
        return {"stores": store_allocation}
    
    def compute_desired_quantiles(self, args):

        raise NotImplementedError
    
    def forward(self, observation):
        """
        Get store allocation by mapping features to quantiles for each store.
        Then, with the quantile forecaster, we "invert" the quantiles to get base-stock levels and obtain the store allocation.
        """

        underage_costs, holding_costs, lead_times, past_demands, days_from_christmas, store_inventories = [observation[key] for key in ['underage_costs', 'holding_costs', 'lead_times', 'past_demands', 'days_from_christmas', 'store_inventories']]

        # Get the desired quantiles for each store, which will be used to forecast the base stock levels
        # This function is different for each type of QuantilePolicy
        quantiles = self.compute_desired_quantiles({'underage_costs': underage_costs, 'holding_costs': holding_costs})

        # Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
        return self.forecast_base_stock_allocation(
            past_demands, days_from_christmas, store_inventories, lead_times, quantiles, allow_back_orders=self.allow_back_orders
            )
        
class TransformedNV(QuantilePolicy):

    def compute_desired_quantiles(self, args):
        """"
        Maps the newsvendor quantile (u/[u+h]) to a new quantile
        """

        return self.net['master'](args['underage_costs']/(args['underage_costs'] + args['holding_costs']))

class QuantileNV(QuantilePolicy):

    def __init__(self, args, device='cpu'):

        super().__init__(args=args, device=device) # Initialize super class
        self.trainable = False

    def compute_desired_quantiles(self, args):
        """"
        Returns the newsvendor quantile (u/[u+h])
        """

        return args['underage_costs']/(args['underage_costs'] + args['holding_costs'])

class ReturnsNV(QuantileNV):
    """"
    Same as QuantileNV, but allows back orders (so it is a non-admissible policy)
    """

    def __init__(self, args, device='cpu'):

        super().__init__(args=args, device=device) # Initialize super class
        self.trainable = False
        self.allow_back_orders = True

class FixedQuantile(QuantilePolicy):

    def compute_desired_quantiles(self, args):
        """"
        Returns the same quantile for all stores and periods
        """

        return self.net['master'](torch.tensor([0.0]).to(self.device)).unsqueeze(1).expand(args['underage_costs'].shape[0], args['underage_costs'].shape[1])


class JustInTime(MyNeuralNetwork):
    """"
    Non-admissible policy, that looks into the future and orders so that units arrive just-in-time so satisfy demand
    Can be considered as an "oracle policy"
    """

    def __init__(self, args, device='cpu'):
        super().__init__(args=args, device=device) # Initialize super class
        self.trainable = False

    def forward(self, observation):
        """
        Get store allocation by looking into the future and ordering so that units arrive just-in-time to satisfy demand
        """
        
        current_period, lead_times \
            = self.unpack_args(observation, ["current_period", "lead_times"])
        demands, period_shift = self.unpack_args(observation['internal_data'], ["demands", "period_shift"])
    
        num_samples, num_stores, max_lead_time = demands.shape

        # For every sample and store, get the demand 'lead_times' periods from now.
        # Does not currently work for backlogged demand setting!
        future_demands = torch.stack([
            demands[:, j][
                torch.arange(num_samples), 
                torch.clip(current_period.to(self.device) + period_shift + lead_times[:, j].long(), max=max_lead_time - 1)
                ] 
            for j in range(num_stores)
            ], dim=1
            )

        return {"stores": torch.clip(future_demands, min=0)}

class GNN(MyNeuralNetwork):
    """
    Graph Neural Network for one warehouse many stores case.
    Uses message passing between warehouse and stores.
    """
    
    def __init__(self, args, scenario, device='cpu'):
        super().__init__(args, device)
        
        # Store scenario for creating graph network in forward pass
        self.scenario = scenario
        
    def prepare_graph_and_features(self, observation):
        """
        Prepares graph structure and node features from observation.
        
        Args:
            observation: Current observation from environment
            
        Returns:
            tuple: (graph_network dict, node_features tensor)
        """
        device = observation['store_inventories'].device
        problem_params = self.scenario.problem_params
        
        n_stores = problem_params.get('n_stores', 0)
        n_warehouses = problem_params.get('n_warehouses', 0)
        n_extra_echelons = problem_params.get('n_extra_echelons', 0)
        
        # Initialize adjacency as None, will be created in each case
        adjacency = None
        has_outside_supplier = None
        has_demand = None
        
        # Serial network case (when n_extra_echelons exists)
        if n_extra_echelons > 0:
            # Serial network: echelons -> warehouse -> store
            # Node ordering: [echelon_0, echelon_1, ..., warehouse, store]
            n_nodes = n_extra_echelons + n_warehouses + n_stores
            adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=device)
            has_outside_supplier = torch.zeros(n_nodes, dtype=torch.float32, device=device)
            has_demand = torch.zeros(n_nodes, dtype=torch.float32, device=device)
            
            # Define node indices
            warehouse_idx = n_extra_echelons
            store_idx = n_extra_echelons + n_warehouses  # Only one store
            
            # First echelon connects to outside supplier
            has_outside_supplier[0] = 1
            
            # Connect echelons in series
            for i in range(n_extra_echelons - 1):
                adjacency[i, i + 1] = 1  # Echelon i -> Echelon i+1
            
            # Last echelon connects to warehouse (warehouse always exists in serial case)
            adjacency[n_extra_echelons - 1, warehouse_idx] = 1  # Last echelon -> Warehouse
            
            # Warehouse connects to store (store always exists and is single in serial case)
            adjacency[warehouse_idx, store_idx] = 1
            
            # Store connects to demand
            has_demand[store_idx] = 1
            
        # Warehouse-stores network case (no extra echelons)
        elif n_warehouses > 0 and n_stores > 0:
            # Node ordering: [warehouse_0, warehouse_1, ..., store_0, store_1, ...]
            n_nodes = n_warehouses + n_stores
            adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=device)
            has_outside_supplier = torch.zeros(n_nodes, dtype=torch.float32, device=device)
            has_demand = torch.zeros(n_nodes, dtype=torch.float32, device=device)
            
            # All warehouses connect to outside supplier
            has_outside_supplier[:n_warehouses] = 1
            
            # All stores connect to demand
            has_demand[n_warehouses:] = 1
            
            if n_warehouses == 1:
                # Single warehouse case: all stores connect to the single warehouse
                # Create connections from warehouse to all stores
                for store_idx in range(n_stores):
                    adjacency[0, n_warehouses + store_idx] = 1
            else:
                # Multiple warehouses case: use adjacency matrix from config
                warehouse_store_adjacency = problem_params.get('warehouse_store_adjacency', None)
                
                if warehouse_store_adjacency is None:
                    raise ValueError(
                        f"Multiple warehouses ({n_warehouses}) detected but no 'warehouse_store_adjacency' "
                        f"matrix found in problem_params. Please specify which stores connect to which warehouses."
                    )
                
                # Convert list/array to tensor
                warehouse_store_adjacency = torch.tensor(warehouse_store_adjacency, dtype=torch.float32, device=device)
                
                # Copy the warehouse-store connections into the full adjacency matrix
                # Warehouse nodes are indices [0, n_warehouses)
                # Store nodes are indices [n_warehouses, n_warehouses + n_stores)
                adjacency[:n_warehouses, n_warehouses:] = warehouse_store_adjacency
        
        # Prepare node features - collect in order: [echelons?, warehouse, stores]
        features_list = []
        inv_lengths = []
        
        # Echelon features first (if present)
        if n_extra_echelons > 0:
            echelon_inv_len = observation['echelon_inventories'].size(-1)
            echelon_features = torch.cat([
                observation['echelon_inventories'],
                observation['echelon_holding_costs'].unsqueeze(-1)
            ], dim=-1)
            features_list.append(echelon_features)
            inv_lengths.append(echelon_inv_len)
        
        # Warehouse features
        warehouse_inv_len = observation['warehouse_inventories'].size(-1)
        warehouse_features = torch.cat([
            observation['warehouse_inventories'],
            observation['warehouse_holding_costs'].unsqueeze(-1)
        ], dim=-1)
        features_list.append(warehouse_features)
        inv_lengths.append(warehouse_inv_len)
        
        # Store features last
        store_inv_len = observation['store_inventories'].size(-1)
        store_features = torch.cat([
            observation['store_inventories'],
            observation['holding_costs'].unsqueeze(-1),
            observation['underage_costs'].unsqueeze(-1),
        ], dim=-1)
        features_list.append(store_features)
        inv_lengths.append(store_inv_len)
        
        # Determine max sizes for uniform padding
        max_inv_len = max(inv_lengths)
        max_states_len = max(
            feat.size(-1) - inv_len 
            for feat, inv_len in zip(features_list, inv_lengths)
        )
        
        # Pad all features to same size
        padded_features = [
            self._pad_features(feat, inv_len, max_inv_len, max_states_len)
            for feat, inv_len in zip(features_list, inv_lengths)
        ]
        
        # Single concatenation - already in order: [echelons?, warehouse, stores]
        all_features = torch.cat(padded_features, dim=1)
        
        # Create lead time matrix matching adjacency structure
        lead_time_matrix = self._create_lead_time_matrix(adjacency, has_outside_supplier, observation)
        
        graph_network = {
            'adjacency': adjacency,
            'has_outside_supplier': has_outside_supplier,
            'has_demand': has_demand,
            'lead_time_matrix': lead_time_matrix,  # Matrix of lead times matching adjacency
        }
        
        return graph_network, all_features
    
    def _create_lead_time_matrix(self, adjacency, has_outside_supplier, observation):
        """
        Create a lead time matrix matching the adjacency matrix structure.
        lead_time_matrix[i, j] = lead time for edge from node i to node j (0 if no edge)
        """
        n_nodes = adjacency.size(0)
        device = adjacency.device
        lead_time_matrix = torch.zeros(n_nodes, n_nodes, device=device)
        
        n_extra_echelons = self.scenario.problem_params.get('n_extra_echelons', 0)
        n_warehouses = self.scenario.problem_params.get('n_warehouses', 0)
        n_stores = self.scenario.problem_params.get('n_stores', 0)
        
        # Serial network case
        if n_extra_echelons > 0:
            # Echelon to echelon lead times
            for i in range(n_extra_echelons - 1):
                if adjacency[i, i + 1] > 0:
                    lead_time_matrix[i, i + 1] = observation['echelon_lead_times'][0, i]
            
            # Last echelon to warehouse
            if n_extra_echelons > 0 and n_warehouses > 0:
                echelon_idx = n_extra_echelons - 1
                warehouse_idx = n_extra_echelons
                if adjacency[echelon_idx, warehouse_idx] > 0:
                    lead_time_matrix[echelon_idx, warehouse_idx] = observation['echelon_lead_times'][0, echelon_idx]
            
            # Warehouse to store
            if n_warehouses > 0 and n_stores > 0:
                warehouse_idx = n_extra_echelons
                store_idx = n_extra_echelons + n_warehouses
                if adjacency[warehouse_idx, store_idx] > 0:
                    lead_time_matrix[warehouse_idx, store_idx] = observation['warehouse_lead_times'][0, 0]
        
        # Warehouse-store network case
        else:
            for w_idx in range(n_warehouses):
                for s_idx in range(n_stores):
                    if adjacency[w_idx, n_warehouses + s_idx] > 0:
                        lead_time_matrix[w_idx, n_warehouses + s_idx] = observation['lead_times'][0, s_idx]
        
        # Add a row for virtual supplier lead times (we'll handle this in create_edges)
        # For nodes that connect to outside suppliers, store their lead times
        supplier_lead_times = torch.zeros(n_nodes, device=device)
        if n_extra_echelons > 0 and has_outside_supplier[0] > 0:
            # First echelon gets from supplier
            supplier_lead_times[0] = observation['echelon_lead_times'][0, 0]
        else:
            # Warehouses get from suppliers
            for w_idx in range(n_warehouses):
                if has_outside_supplier[w_idx] > 0:
                    supplier_lead_times[w_idx] = observation['warehouse_lead_times'][0, w_idx]
        
        # Store supplier lead times as an additional attribute
        lead_time_matrix = {
            'matrix': lead_time_matrix,
            'supplier_lead_times': supplier_lead_times
        }
        
        return lead_time_matrix
    
    def _pad_features(self, tensor, inv_len, max_inv_len, max_states_len):
        """Helper to pad inventory and state features to uniform size."""
        inv = tensor[:,:,:inv_len]
        states = tensor[:,:,inv_len:]
        return torch.cat([
            F.pad(inv, (0, max_inv_len - inv_len)),
            F.pad(states, (0, max_states_len - (tensor.size(2) - inv_len)))
        ], dim=2)
    
    
    def create_edge_features(self, graph_network, nodes):
        """
        Create edge features for the supply chain graph based on adjacency matrix.
        
        Args:
            graph_network: Dictionary with adjacency, lead times, and node flags
            nodes: Node embeddings tensor
            
        Returns:
            edge_features: Tensor of edge features [batch_size, num_edges, edge_feature_dim]
        """
        adjacency = graph_network['adjacency']
        has_outside_supplier = graph_network['has_outside_supplier']
        has_demand = graph_network['has_demand']
        lead_time_info = graph_network['lead_time_matrix']
        lead_time_matrix = lead_time_info['matrix']
        supplier_lead_times = lead_time_info['supplier_lead_times']
        
        batch_size = nodes.size(0)
        device = nodes.device
        
        edge_list = []
        
        # Create edges from adjacency matrix
        # Find all edges (i, j) where adjacency[i, j] = 1
        # This represents directed edges: source node i → target node j
        edge_indices = adjacency.nonzero(as_tuple=False)  # [num_edges, 2]
        
        # Extract source and target node indices for each directed edge
        source_indices = edge_indices[:, 0]  # Source nodes (suppliers)
        target_indices = edge_indices[:, 1]  # Target nodes (recipients)
        
        # Get node features for all batches at once
        source_features = nodes[:, source_indices]  # [batch_size, num_edges, node_dim]
        target_features = nodes[:, target_indices]  # [batch_size, num_edges, node_dim]
        
        # Extract lead times from the matrix for each edge
        edge_lead_times = lead_time_matrix[source_indices, target_indices].unsqueeze(-1)  # [num_edges, 1]
        edge_lead_times = edge_lead_times.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_edges, 1]
        
        # Concatenate source, target, and lead time features
        batch_edges = torch.cat([source_features, target_features, edge_lead_times], dim=-1)
        edge_list.append(batch_edges)
        
        # Create edges from outside suppliers (virtual supplier nodes)
        supplier_nodes = has_outside_supplier.nonzero(as_tuple=False).squeeze(-1)
        # Virtual supplier node features (zeros) for all batches
        supplier_features = torch.zeros(batch_size, supplier_nodes.size(0), nodes.size(-1), device=device)
        target_features = nodes[:, supplier_nodes]
        
        # Get supplier lead times from the stored vector
        lead_times = supplier_lead_times[supplier_nodes].unsqueeze(-1)  # [num_supplier_nodes, 1]
        lead_times = lead_times.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_supplier_nodes, 1]
        
        supplier_edges = torch.cat([supplier_features, target_features, lead_times], dim=-1)
        edge_list.append(supplier_edges)
        
        # Create edges to customers (demand nodes)
        demand_nodes = has_demand.nonzero(as_tuple=False).squeeze(-1)
        source_features = nodes[:, demand_nodes]
        # Virtual customer node features (zeros) for all batches
        customer_features = torch.zeros(batch_size, demand_nodes.size(0), nodes.size(-1), device=device)
        customer_lead_times = torch.zeros(batch_size, demand_nodes.size(0), 1, device=device)
        
        customer_edges = torch.cat([source_features, customer_features, customer_lead_times], dim=-1)
        edge_list.append(customer_edges)
        
        # Add self-loops for nodes that supply to others
        # Nodes with outgoing edges (excluding edges to customers)
        supplying_nodes = (adjacency.sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(-1)
        if supplying_nodes.size(0) > 0:
            node_features = nodes[:, supplying_nodes]
            self_loop_lead_times = torch.zeros(batch_size, supplying_nodes.size(0), 1, device=device)
            
            self_loops = torch.cat([node_features, node_features, self_loop_lead_times], dim=-1)
            edge_list.append(self_loops)
        
        # Concatenate all edge features
        edge_features = torch.cat(edge_list, dim=1)
        return edge_features
    
    def aggregate_messages(self, edges):
        """Aggregate messages for nodes based on aggregation method."""
        n_stores = self.scenario.problem_params.get('n_stores', 0)
        
        # Incoming messages aggregation
        # Warehouse gets messages from supplier edge and self-loop
        warehouse_incoming = (edges[:, 0:1] + edges[:, -1:]) / math.sqrt(2)
        
        # Stores get messages from warehouse edges
        store_incoming = edges[:, 1:1+n_stores]
        
        # Outgoing messages aggregation  
        # Warehouse sends to stores (using mean aggregation)
        warehouse_outgoing = edges[:, 1:1+n_stores].mean(dim=1, keepdim=True)
        
        warehouse_outgoing = (warehouse_outgoing + edges[:, -1:]) / math.sqrt(2)
        
        # Stores send to customers
        store_outgoing = edges[:, 1+n_stores:1+2*n_stores]
        
        # Combine aggregated messages
        incoming = torch.cat([warehouse_incoming, store_incoming], dim=1)
        outgoing = torch.cat([warehouse_outgoing, store_outgoing], dim=1)
        
        return incoming, outgoing
    
    def message_passing_step(self, nodes, edges):
        """Perform one step of message passing."""
        n_stores = self.scenario.problem_params.get('n_stores', 0)
        
        # Aggregate messages
        incoming, outgoing = self.aggregate_messages(edges)
        
        # Update nodes
        node_input = torch.cat([nodes, incoming, outgoing], dim=-1)
        node_updates = self.net['node_update'](node_input)
        
        # Apply residual connections
        nodes = nodes + node_updates
        
        # Prepare edge sources and targets for edge update
        supplier_node = torch.zeros_like(nodes[:, :1])
        warehouse_node = nodes[:, :1]
        store_nodes = nodes[:, 1:]
        customer_nodes = torch.zeros_like(store_nodes)
        
        edge_sources = torch.cat([
            supplier_node,  # Supplier -> Warehouse
            warehouse_node.expand(-1, n_stores, -1),  # Warehouse -> Stores
            store_nodes,  # Stores -> Customers
        ], dim=1)
        
        edge_targets = torch.cat([
            warehouse_node,  # Supplier -> Warehouse
            store_nodes,  # Warehouse -> Stores
            customer_nodes,  # Stores -> Customers
        ], dim=1)
        
        # Add self-loop
        edge_sources = torch.cat([edge_sources, warehouse_node], dim=1)
        edge_targets = torch.cat([edge_targets, warehouse_node], dim=1)
        
        # Update edges
        edge_input = torch.cat([edges, edge_sources, edge_targets], dim=-1)
        edge_updates = self.net['edge_update'](edge_input)
        
        # Apply residual connections
        edges = edges + edge_updates
        
        return nodes, edges
    
    def forward(self, observation):
        """
        Forward pass through GNN for one warehouse many stores case.
        """
        # Prepare graph structure and node features
        graph_network, all_features = self.prepare_graph_and_features(observation)
        
        # Initial node embeddings
        nodes = self.net['initial_node'](all_features)
        
        # Create initial edge features using graph network structure
        edge_features = self.create_edge_features(graph_network, nodes)
        
        # Initial edge embeddings
        edges = self.net['initial_edge'](edge_features)
        
        # Message passing iterations (default: 2 steps)
        for _ in range(2):
            nodes, edges = self.message_passing_step(nodes, edges)
        
        # Generate outputs from edges
        n_stores = self.scenario.problem_params.get('n_stores', 0)
        warehouse_order = self.net['output'](edges[:, 0:1]).squeeze(-1)
        store_intermediate = self.net['output'](edges[:, 1:1+n_stores]).squeeze(-1)
        
        # Apply proportional allocation for stores based on warehouse inventory
        # Include self-loop in the allocation
        warehouse_self_loop_output = self.net['output'](edges[:, -1:]).squeeze(-1)
        all_outputs = torch.cat([store_intermediate, warehouse_self_loop_output], dim=1)
        allocations = self.apply_proportional_allocation(
            all_outputs, 
            observation['warehouse_inventories']
        )
        store_orders = allocations[:, :-1]
        self_loop_order = allocations[:, -1:]
        
        return {
            'warehouses': warehouse_order,
            'stores': store_orders,
            'warehouse_self_loop_orders': self_loop_order,
            'graph_network': graph_network
        }


class NeuralNetworkCreator:
    """
    Class to create neural networks
    """

    def set_default_output_size(self, module_name, problem_params):
        
        default_sizes = {
            'master': problem_params['n_stores'] + problem_params['n_warehouses'], 
            'store': 1, 
            'warehouse': 1, 
            'context': None
            }
        return default_sizes[module_name]

    def get_architecture(self, name):

        architectures = {
            'vanilla_one_store': VanillaOneStore, 
            'base_stock': BaseStock,
            'capped_base_stock': CappedBaseStock,
            'echelon_stock': EchelonStock,
            'vanilla_serial': VanillaSerial,
            'vanilla_transshipment': VanillaTransshipment,
            'vanilla_one_warehouse': VanillaOneWarehouse,
            'symmetry_aware': SymmetryAware,
            'data_driven': DataDrivenNet,
            'transformed_nv': TransformedNV,
            'fixed_quantile': FixedQuantile,
            'quantile_nv': QuantileNV,
            'returns_nv': ReturnsNV,
            'just_in_time': JustInTime,
            'gnn': GNN,
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

        # Special handling for GNN architecture - pass scenario directly
        if nn_params_copy['name'] == 'gnn':
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
