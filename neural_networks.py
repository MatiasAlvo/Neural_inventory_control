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

        return {'stores': x.unsqueeze(2)}

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
        return {'stores': torch.clip(x - inv_pos, min=0).unsqueeze(2)}

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

        stores_alloc = allocations[:, -1:].unsqueeze(2)
        warehouses_alloc = allocations[:, -2: -1].unsqueeze(2)
        echelons_alloc = allocations[:, : n_extra_echelons].unsqueeze(2)

        return {
            'stores': stores_alloc,
            'warehouses': warehouses_alloc,
            'echelons': echelons_alloc,
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
        
        return {'stores': torch.clip(base_level - inv_pos, min=torch.tensor([0.0]).to(self.device), max=cap).unsqueeze(2)}


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

        stores_alloc = allocations[:, -1:].unsqueeze(2)
        warehouses_alloc = allocations[:, -2: -1].unsqueeze(2)
        echelons_alloc = allocations[:, : n_extra_echelons].unsqueeze(2)

        return {
            'stores': stores_alloc,
            'warehouses': warehouses_alloc,
            'echelons': echelons_alloc,
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

        # Reshape store_allocation to [batch, n_stores, n_warehouses] for consistency
        # n_warehouses = 1 for VanillaOneWarehouse
        store_allocation = store_allocation.unsqueeze(2)  # [batch, n_stores, 1]
        
        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation.unsqueeze(2)
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
        # Lead times are 3D [batch, n_stores, n_warehouses], use first warehouse
        lead_times_2d = observation['lead_times'][:, :, 0]
        store_params = torch.stack([observation[k] for k in ['mean', 'std', 'underage_costs']] + [lead_times_2d], dim=2)
        
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

        # Reshape store_allocation to [batch, n_stores, n_warehouses] for consistency
        # n_warehouses = 1 for SymmetryAware
        store_allocation = store_allocation.unsqueeze(2)  # [batch, n_stores, 1]
        
        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation.unsqueeze(2)
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
        # Lead times are 3D [batch, n_stores, n_warehouses], use first warehouse
        lead_times_2d = observation['lead_times'][:, :, 0]
        input_tensor = self.flatten_then_concatenate_tensors(
                [observation['store_inventories']] + 
                [observation[key] for key in ['past_demands', 'underage_costs', 'days_from_christmas']] +
                [lead_times_2d]
            )

        return {'stores': self.net['master'](input_tensor).unsqueeze(2)}

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
        
        return {"stores": store_allocation.unsqueeze(2)}
    
    def compute_desired_quantiles(self, args):

        raise NotImplementedError
    
    def forward(self, observation):
        """
        Get store allocation by mapping features to quantiles for each store.
        Then, with the quantile forecaster, we "invert" the quantiles to get base-stock levels and obtain the store allocation.
        """

        # Lead times are 3D [batch, n_stores, n_warehouses], use first warehouse
        lead_times = observation['lead_times'][:, :, 0]
        underage_costs, holding_costs, past_demands, days_from_christmas, store_inventories = [observation[key] for key in ['underage_costs', 'holding_costs', 'past_demands', 'days_from_christmas', 'store_inventories']]

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
        # lead_times is 3D [batch, n_stores, n_warehouses], use first warehouse
        future_demands = torch.stack([
            demands[:, j][
                torch.arange(num_samples), 
                torch.clip(current_period.to(self.device) + period_shift + lead_times[:, j, 0].long(), max=max_lead_time - 1)
                ] 
            for j in range(num_stores)
            ], dim=1
            )

        return {"stores": torch.clip(future_demands, min=0).unsqueeze(2)}

class GNN(MyNeuralNetwork):
    """
    Graph Neural Network for one warehouse many stores case.
    Uses message passing between warehouse and stores.
    """
    
    def __init__(self, args, scenario, device='cpu'):
        super().__init__(args, device)
        
        # Store scenario for creating graph network in forward pass
        self.scenario = scenario
        
        # Check if this is a transshipment case (warehouse cannot hold inventory)
        self.transshipment = args.get('transshipment', False)
        
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
        
        # Determine number of message passing steps based on network topology
        if n_extra_echelons > 0:
            # Serial network: n_echelons + 1 (for warehouse-store connection)
            num_message_passing = n_extra_echelons + 1
        else:
            # Warehouse-store network: 1 step
            num_message_passing = 1
        
        # Create edge index mapping
        edge_index_mapping = self._create_edge_index_mapping(
            adjacency, has_outside_supplier, has_demand,
            n_stores, n_warehouses, n_extra_echelons
        )
        
        # Get on-hand inventory for each node
        node_inventories = self._get_node_inventories(observation, n_stores, n_warehouses, n_extra_echelons)
        
        graph_network = {
            'adjacency': adjacency,
            'has_outside_supplier': has_outside_supplier,
            'has_demand': has_demand,
            'lead_time_matrix': lead_time_matrix,  # Matrix of lead times matching adjacency
            'num_message_passing': num_message_passing,  # Number of message passing iterations
            'edge_index_mapping': edge_index_mapping,  # Dict mapping output types to edge indices
            'node_inventories': node_inventories,  # On-hand inventory for each node
            'n_stores': n_stores,  # Needed for output tensor dimensions
            'n_warehouses': n_warehouses,  # Needed for output tensor dimensions
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
        
        # Serial network case (n_warehouses=1, n_stores=1 by definition)
        if n_extra_echelons > 0:
            # Echelon to echelon lead times
            for i in range(n_extra_echelons - 1):
                if adjacency[i, i + 1] > 0:
                    lead_time_matrix[i, i + 1] = observation['echelon_lead_times'][0, i]
            
            # Last echelon to warehouse
            echelon_idx = n_extra_echelons - 1
            warehouse_idx = n_extra_echelons
            if adjacency[echelon_idx, warehouse_idx] > 0:
                # Use warehouse_lead_times for the connection from echelon to warehouse
                lead_time_matrix[echelon_idx, warehouse_idx] = observation['warehouse_lead_times'][0, 0]
            
            # Warehouse to store (always single warehouse to single store in serial system)
            warehouse_idx = n_extra_echelons
            store_idx = n_extra_echelons + n_warehouses
            if adjacency[warehouse_idx, store_idx] > 0:
                # Lead times are now always 3D [batch, n_stores, n_warehouses]
                # For serial system it's [batch, 1, 1]
                lead_time_matrix[warehouse_idx, store_idx] = observation['lead_times'][0, 0, 0]
        
        # Warehouse-store network case
        else:
            # Lead times are always 3D now [batch, n_stores, n_warehouses]
            warehouse_store_lt_matrix = observation['lead_times'][0]  # [n_stores, n_warehouses]
            for w_idx in range(n_warehouses):
                for s_idx in range(n_stores):
                    if adjacency[w_idx, n_warehouses + s_idx] > 0:
                        # Extract lead time from store s to warehouse w
                        lead_time_matrix[w_idx, n_warehouses + s_idx] = warehouse_store_lt_matrix[s_idx, w_idx]
        
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
    
    def _create_edge_index_mapping(self, adjacency, has_outside_supplier, has_demand,
                                  n_stores, n_warehouses, n_extra_echelons):
        """
        Create mapping from output types to edge indices based on known edge ordering.
        Edge ordering: internal edges, supplier edges, demand edges, self-loops
        Returns a dict with keys: 'stores', 'warehouses', 'echelons'
        All values are 2D arrays for consistency
        """
        mapping = {}
        
        # Count edges to track indices
        edge_indices = adjacency.nonzero(as_tuple=False)
        n_internal_edges = edge_indices.size(0)
        
        if n_extra_echelons > 0:
            # Serial network: echelons → warehouse → store
            # n_stores = 1 and n_warehouses = 1 always in serial networks
            # Internal edges are in order: echelon-to-echelon edges, then last-echelon-to-warehouse, then warehouse-to-store
            
            # The last internal edge is warehouse-to-store
            # Use 2D structure: [n_stores][n_warehouses] for consistency
            mapping['stores'] = [[n_internal_edges - 1]]  # Last internal edge
            
            # The second to last internal edge is last-echelon-to-warehouse
            # Use 2D structure: [n_warehouses][suppliers_per_warehouse] for consistency
            mapping['warehouses'] = [[n_internal_edges - 2]]
            
            # First echelon orders from supplier (first supplier edge after internal edges)
            # Use 2D structure: [n_extra_echelons][suppliers_per_echelon] for consistency
            mapping['echelons'] = [[n_internal_edges]]  # First supplier edge
            
        else:
            # Warehouse-stores network
            # Always use 2D structure for stores [n_stores][n_warehouses] for consistency
            mapping['stores'] = [[] for _ in range(n_stores)]
            
            # Parse internal edges to build store mapping
            for i, (src, tgt) in enumerate(edge_indices):
                if src < n_warehouses and tgt >= n_warehouses:
                    store_idx = (tgt - n_warehouses).item()
                    mapping['stores'][store_idx].append(i)
            
            # Warehouse orders from suppliers (one supplier edge per warehouse)
            # Use 2D structure: [n_warehouses][suppliers_per_warehouse] for consistency
            mapping['warehouses'] = [[n_internal_edges + w] for w in range(n_warehouses)]
        
        return mapping
    
    def _get_node_inventories(self, observation, n_stores, n_warehouses, n_extra_echelons):
        """
        Get on-hand inventory for each node in the graph.
        Returns tensor of shape [batch_size, n_nodes] with inventory for each node.
        """
        batch_size = observation['store_inventories'].size(0)
        device = observation['store_inventories'].device
        n_nodes = n_extra_echelons + n_warehouses + n_stores
        
        node_inventories = torch.zeros(batch_size, n_nodes, device=device)
        node_idx = 0
        
        # Echelon inventories (if present)
        if n_extra_echelons > 0:
            echelon_inv = observation['echelon_inventories'][:, :, 0]  # On-hand inventory
            node_inventories[:, node_idx:node_idx + n_extra_echelons] = echelon_inv
            node_idx += n_extra_echelons
        
        # Warehouse inventories
        if n_warehouses > 0:
            warehouse_inv = observation['warehouse_inventories'][:, :, 0]  # On-hand inventory
            node_inventories[:, node_idx:node_idx + n_warehouses] = warehouse_inv
            node_idx += n_warehouses
        
        # Store inventories
        if n_stores > 0:
            store_inv = observation['store_inventories'][:, :, 0]  # On-hand inventory
            node_inventories[:, node_idx:node_idx + n_stores] = store_inv
        
        return node_inventories
    
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
        
        Edge ordering in the returned tensor:
        1. Internal edges (from adjacency matrix) - connections between network nodes
        2. Supplier edges - from virtual supplier nodes to nodes receiving external supply
        3. Demand edges - from nodes serving demand to virtual customer nodes
        4. Self-loops - for nodes that supply to other nodes
        
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
    
    def aggregate_messages(self, graph_network, edges):
        """
        Aggregate messages for nodes based on graph structure.
        
        Assumes edges are ordered as: internal, supplier, demand, self-loops
        (as created by create_edge_features method)
        
        Args:
            graph_network: Dictionary with graph structure information
            edges: Edge embeddings to aggregate
            
        Returns:
            incoming: Aggregated incoming messages for each node
            outgoing: Aggregated outgoing messages for each node
        """
        adjacency = graph_network['adjacency']
        has_outside_supplier = graph_network['has_outside_supplier']
        has_demand = graph_network['has_demand']
        
        n_nodes = adjacency.size(0)
        batch_size = edges.size(0)
        edge_dim = edges.size(-1)
        device = edges.device
        
        # Initialize message tensors for each node
        incoming = torch.zeros(batch_size, n_nodes, edge_dim, device=device)
        outgoing = torch.zeros(batch_size, n_nodes, edge_dim, device=device)
        
        edge_idx = 0
        
        # Aggregate messages from internal edges (adjacency matrix)
        edge_indices = adjacency.nonzero(as_tuple=False)
        n_internal_edges = edge_indices.size(0)
        internal_edges = edges[:, edge_idx:edge_idx+n_internal_edges]
        
        for i, (src, tgt) in enumerate(edge_indices):
            # Incoming: target node receives from source
            incoming[:, tgt] += internal_edges[:, i]
            # Outgoing: source node sends to target
            outgoing[:, src] += internal_edges[:, i]
        
        edge_idx += n_internal_edges
        
        # Aggregate messages from supplier edges
        supplier_nodes = has_outside_supplier.nonzero(as_tuple=False).squeeze(-1)
        n_supplier_edges = supplier_nodes.size(0)
        supplier_edges = edges[:, edge_idx:edge_idx+n_supplier_edges]
        
        for i, node_idx in enumerate(supplier_nodes):
            # Nodes receive from suppliers
            incoming[:, node_idx] += supplier_edges[:, i]
        
        edge_idx += n_supplier_edges
        
        # Aggregate messages from demand edges
        demand_nodes = has_demand.nonzero(as_tuple=False).squeeze(-1)
        n_demand_edges = demand_nodes.size(0)
        demand_edges = edges[:, edge_idx:edge_idx+n_demand_edges]
        
        for i, node_idx in enumerate(demand_nodes):
            # Nodes send to customers
            outgoing[:, node_idx] += demand_edges[:, i]
        
        edge_idx += n_demand_edges
        
        # Aggregate messages from self-loops
        supplying_nodes = (adjacency.sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(-1)
        if supplying_nodes.size(0) > 0:
            n_self_loops = supplying_nodes.size(0)
            self_loop_edges = edges[:, edge_idx:edge_idx+n_self_loops]
            
            for i, node_idx in enumerate(supplying_nodes):
                # Self-loops contribute to both incoming and outgoing
                incoming[:, node_idx] += self_loop_edges[:, i]
                outgoing[:, node_idx] += self_loop_edges[:, i]
        
        # Normalize aggregated messages by number of connections
        # Count incoming connections for each node
        in_degree = adjacency.sum(dim=0).float()  # How many nodes send to this node
        in_degree += has_outside_supplier.float()  # Add supplier connections
        
        # Add self-loops to in-degree for supplying nodes
        for node_idx in supplying_nodes:
            in_degree[node_idx] += 1
        
        # Count outgoing connections for each node
        out_degree = adjacency.sum(dim=1).float()  # How many nodes this node sends to
        out_degree += has_demand.float()  # Add customer connections
        
        # Add self-loops to out-degree for supplying nodes
        for node_idx in supplying_nodes:
            out_degree[node_idx] += 1
        
        # Normalize (avoid division by zero)
        in_degree = torch.where(in_degree > 0, in_degree, torch.ones_like(in_degree))
        out_degree = torch.where(out_degree > 0, out_degree, torch.ones_like(out_degree))
        
        incoming = incoming / torch.sqrt(in_degree).unsqueeze(0).unsqueeze(-1)
        outgoing = outgoing / torch.sqrt(out_degree).unsqueeze(0).unsqueeze(-1)
        
        return incoming, outgoing
    
    def message_passing_step(self, graph_network, nodes, edges):
        """
        Perform one step of message passing using graph structure.
        
        Args:
            graph_network: Dictionary with graph structure information
            nodes: Current node embeddings
            edges: Current edge embeddings
            
        Returns:
            Updated nodes and edges
        """
        adjacency = graph_network['adjacency']
        has_outside_supplier = graph_network['has_outside_supplier']
        has_demand = graph_network['has_demand']
        
        # Aggregate messages based on graph structure
        incoming, outgoing = self.aggregate_messages(graph_network, edges)
        
        # Update nodes
        node_input = torch.cat([nodes, incoming, outgoing], dim=-1)
        node_updates = self.net['node_update'](node_input)
        
        # Apply residual connections
        nodes = nodes + node_updates
        
        # Reconstruct edge source and target features for edge update
        edge_sources = []
        edge_targets = []
        device = nodes.device
        
        # Internal edges from adjacency matrix
        edge_indices = adjacency.nonzero(as_tuple=False)
        source_indices = edge_indices[:, 0]
        target_indices = edge_indices[:, 1]
        edge_sources.append(nodes[:, source_indices])
        edge_targets.append(nodes[:, target_indices])
        
        # Supplier edges
        supplier_nodes = has_outside_supplier.nonzero(as_tuple=False).squeeze(-1)
        edge_sources.append(torch.zeros(nodes.size(0), supplier_nodes.size(0), nodes.size(-1), device=device))
        edge_targets.append(nodes[:, supplier_nodes])
        
        # Demand edges
        demand_nodes = has_demand.nonzero(as_tuple=False).squeeze(-1)
        edge_sources.append(nodes[:, demand_nodes])
        edge_targets.append(torch.zeros(nodes.size(0), demand_nodes.size(0), nodes.size(-1), device=device))
        
        # Self-loops for nodes with outgoing edges
        supplying_nodes = (adjacency.sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(-1)
        if supplying_nodes.size(0) > 0:
            edge_sources.append(nodes[:, supplying_nodes])
            edge_targets.append(nodes[:, supplying_nodes])
        
        # Concatenate all edge sources and targets
        all_edge_sources = torch.cat(edge_sources, dim=1)
        all_edge_targets = torch.cat(edge_targets, dim=1)
        
        # Update edges
        edge_input = torch.cat([edges, all_edge_sources, all_edge_targets], dim=-1)
        edge_updates = self.net['edge_update'](edge_input)
        
        # Apply residual connections
        edges = edges + edge_updates
        
        return nodes, edges
    
    def forward(self, observation):
        """
        Forward pass through GNN.
        """
        # Prepare graph structure and node features
        graph_network, all_features = self.prepare_graph_and_features(observation)
        
        # Initial node embeddings
        nodes = self.net['initial_node'](all_features)
        
        # Create initial edge features using graph network structure
        edge_features = self.create_edge_features(graph_network, nodes)
        
        # Initial edge embeddings
        edges = self.net['initial_edge'](edge_features)
        
        # Message passing iterations
        num_message_passing = graph_network['num_message_passing']
        for _ in range(num_message_passing):
            nodes, edges = self.message_passing_step(graph_network, nodes, edges)
        
        # Now we need to determine which edges represent which orders
        # Apply output network and allocate based on edge types
        outputs = self._generate_outputs_from_edges(edges, graph_network, observation)
        
        return outputs
    
    def _generate_outputs_from_edges(self, edges, graph_network, observation):
        """
        Generate outputs from edges: apply output network to all edges,
        apply proportional allocation, then use mapping to organize returns.
        """
        mapping = graph_network['edge_index_mapping']
        
        batch_size = edges.size(0)
        device = edges.device
        
        # Apply output network to ALL edges at once
        all_edge_outputs = self.net['output'](edges).squeeze(-1)
        
        # Apply proportional allocation based on graph structure
        # For nodes with limited inventory, scale their outgoing edges
        allocated_outputs = self._apply_proportional_allocation_to_graph(
            all_edge_outputs, graph_network
        )
        
        # Now use mapping to organize the allocated outputs into the return dict
        outputs = {}
        
        # Loop through mapping and extract outputs
        for output_type, edge_indices in mapping.items():
            if not edge_indices:  # Skip empty lists
                continue
            
            # All edge_indices are now 2D (list of lists)
            # Create 2D output tensor
            n_rows = len(edge_indices)
            n_cols = max(len(row) for row in edge_indices) if edge_indices else 0
            result = torch.zeros(batch_size, n_rows, n_cols, device=device)
            for i, row_edges in enumerate(edge_indices):
                for j, edge_idx in enumerate(row_edges):
                    result[:, i, j] = allocated_outputs[:, int(edge_idx)]
            
            # Expand to 3D for stores, warehouses, echelons
            outputs[output_type] = result
        
        return outputs
    
    def _apply_proportional_allocation_to_graph(self, edge_outputs, graph_network):
        """
        Apply proportional allocation to edge outputs based on inventory constraints.
        For each node, scales its outgoing edges proportionally to available inventory.
        """
        adjacency = graph_network['adjacency']
        has_outside_supplier = graph_network['has_outside_supplier']
        has_demand = graph_network['has_demand']
        node_inventories = graph_network['node_inventories']
        
        allocated_outputs = edge_outputs.clone()
        n_nodes = adjacency.size(0)
        
        # Build edge index mapping for quick lookup
        edge_idx = 0
        edge_indices = adjacency.nonzero(as_tuple=False)
        
        # Map from source node to list of edge indices
        node_to_edges = {i: [] for i in range(n_nodes)}
        
        # Internal edges
        for src, _ in edge_indices:
            node_to_edges[src.item()].append(edge_idx)
            edge_idx += 1
        
        # Skip supplier edges (they don't have inventory constraints)
        edge_idx += has_outside_supplier.sum().item()
        
        # Skip demand edges
        edge_idx += has_demand.sum().item()
        
        # Self-loops
        supplying_nodes = (adjacency.sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(-1)
        for node_idx in supplying_nodes:
            node_to_edges[node_idx.item()].append(edge_idx)
            edge_idx += 1
        
        # Apply proportional allocation for each node with outgoing edges
        for node_idx, edge_list in node_to_edges.items():
            if not edge_list:
                continue
            
            # Gather desired allocations for this node's outgoing edges
            # Ensure indices are integers
            desired_allocations = torch.stack([edge_outputs[:, int(idx)] for idx in edge_list], dim=1)
            
            # Get node's available inventory
            available_inv = node_inventories[:, node_idx]
            
            # Apply proportional allocation using the general method
            scaled_allocations = self.apply_proportional_allocation(desired_allocations, available_inv, self.transshipment)
            
            # Write back the scaled allocations
            for i, idx in enumerate(edge_list):
                allocated_outputs[:, int(idx)] = scaled_allocations[:, i]
        
        return allocated_outputs


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
