from shared_imports import *
from data_handling import *
from neural_networks import *
import gymnasium as gym
from gymnasium import spaces

class Simulator(gym.Env):
    """
    Simulator class, defining a differentable simulator
    """

    metadata = {"render_modes": None}

    def __init__(self, device='cpu'):
 
        self.device = device
        self.problem_params, self.observation_params, self.maximize_profit = None, None, None
        self.batch_size, self.n_stores, self.periods, self.observation, self._internal_data = None, None, None, None, None

        # Place holders. Will be overrided in reset method (as observation and action spaces depend on batch size, which might change during execution)
        self.action_space = spaces.Dict({'stores': spaces.Box(low=0.0, high=np.inf, shape=(1, 1), dtype=np.float32)})
        self.observation_space = spaces.Dict({'stores': spaces.Box(low=0.0, high=np.inf, shape=(1, 1), dtype=np.float32)})
    
    def reset(self, periods, problem_params,  data, observation_params):
        """
        Reset the environment, including initializing the observation, and return first observation

        Parameters
        ----------
        periods : int
            Number of periods in the simulation
        problem_params: dict
            Dictionary containing the problem parameters, specifying the number of warehouses, stores, whether demand is lost
            and whether to maximize profit or minimize underage+overage costs
        data : dict
            Dictionary containing the data for the simulation, including initial inventories, demands, holding costs and underage costs
        observation_params: dict
            Dictionary containing the observation parameters, specifying which features to include in the observations
        """

        self.problem_params = problem_params
        self.observation_params = observation_params

        self.batch_size, self.n_stores, self.periods = len(data['initial_inventories']), problem_params['n_stores'], periods
        

        # Data that can only be used by the simulator. E.g., all demands (including future)...
        self._internal_data = {
            'demands': data['demands'],
            'period_shift': observation_params['demand']['period_shift'],
            }
        
        if observation_params['time_features'] is not None:
            self._internal_data.update({k: data[k] for k in observation_params['time_features']})
        if observation_params['sample_features'] is not None:
            self._internal_data.update({k: data[k] for k in observation_params['sample_features']})

        # We create "shift" tensors to calculate in which position of the long vector of the entire batch we have to add the corresponding allocation
        # This is necessary whenever the lead time is different for each sample or/and store
        self._internal_data['allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_inventories'].shape).long().to(self.device)

        if self.problem_params['n_warehouses'] > 0:
            self._internal_data['warehouse_allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_warehouse_inventories'].shape).long().to(self.device)
        
        if self.problem_params['n_extra_echelons'] > 0:
            self._internal_data['echelon_allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_echelon_inventories'].shape).long().to(self.device)

        self._internal_data['zero_allocation_tensor'] = self.initialize_zero_allocation_tensor(data['initial_inventories'].shape[: -1]).to(self.device)

        self.observation = self.initialize_observation(data, observation_params)
        self.action_space = self.initialize_action_space(self.batch_size, problem_params, observation_params)
        self.observation_space = self.initialize_observation_space(self.observation, periods, problem_params,)
        self.maximize_profit = problem_params['maximize_profit']
        
        return self.observation, None
    
    def initialize_shifts_for_allocation_put(self, shape):
        """
        We will add store's allocations into corresponding position by flatenning out the state vector of the
        entire batch. We create allocation_shifts to calculate in which position of that long vector we have
        to add the corresponding allocation
        """

        batch_size, n_stores, lead_time_max = shape

        # Results in a vector of lenght batch_size, where each entry corresponds to the first position of an element of a given sample
        # in the long vector of the entire batch
        n_instance_store_shift = (
            torch.arange(batch_size) * (lead_time_max * n_stores)
            ).to(self.device)

        # Results in a tensor of shape batch_size x stores, where each entry corresponds to the number of positions to move 'to the right'
        # for each store, beginning from the first position within a sample
        store_n_shift = (
            torch.arange(n_stores) * (lead_time_max)
            ).expand(batch_size, n_stores).to(self.device)
        
        # Results in a vector of shape batch_size x stores, where each entry corresponds to the first position of an element of a given (sample, store)
        # in the long vector of the entire batch.
        # We then add the corresponding lead time to obtain the actual position in which to insert the action
        return n_instance_store_shift[:, None] + store_n_shift

    def initialize_zero_allocation_tensor(self, shape):
        """
        Initialize a tensor of zeros with the same shape as the allocation tensor
        """

        return torch.zeros(shape).to(self.device)
    
    def step(self, action):
        """
        Simulate one step in the environment, returning the new observation and the reward (per sample)

        Parameters
        ----------
        action : dict
            Dictionary containing the actions to be taken in the environment for each type of location (stores and warehouses). 
            Each value is a tensor of size batch_size x n_locations, where n_locations is the number of stores or warehouses
        """

        current_demands = self.get_current_demands(
            self._internal_data, 
            current_period=self.observation['current_period'].item()
            )
        
        # Update observation corresponding to past demands.
        # Do this before updating current period (as we consider current period + 1)
        self.update_past_data()

        # Update time related features (e.g., days to christmas)
        self.update_time_features(
            self._internal_data, 
            self.observation, 
            self.observation_params, 
            current_period=self.observation['current_period'].item() + 1
            )

        # Calculate reward and update store inventories
        reward = self.calculate_store_reward_and_update_store_inventories(
            current_demands,
            action,
            self.observation,
            self.maximize_profit
            )
        
        # Calculate reward and update warehouse inventories
        if self.problem_params['n_warehouses'] > 0:

            w_reward = self.calculate_warehouse_reward_and_update_warehouse_inventories(
                action,
                self.observation,
                )
            reward += w_reward
        
        # Calculate reward and update other echelon inventories
        if self.problem_params['n_extra_echelons'] > 0:

            e_reward = self.calculate_echelon_reward_and_update_echelon_inventories(
                action,
                self.observation,
                )
            reward += e_reward
        
        # Update current period
        self.observation['current_period'] += 1

        terminated = self.observation['current_period'] >= self.periods

        return self.observation, reward, terminated, None, None
    
    def get_current_demands(self, data, current_period):
        """
        Get the current demands for the current period.
        period_shift specifies what we should consider as the first period in the data
        """
        
        return data['demands'][:, :, current_period + self._internal_data['period_shift']]
    
    def calculate_store_reward_and_update_store_inventories(self, current_demands, action, observation, maximize_profit=False):
        """
        Calculate reward and observation after demand and action is executed for stores
        """

        store_inventory = self.observation['store_inventories']
        inventory_on_hand = store_inventory[:, :, 0]
        
        post_inventory_on_hand = self.observation['store_inventories'][:, :, 0] - current_demands

        # Reward given by -sales*price + holding_costs
        if maximize_profit:
            reward = (
                -observation['underage_costs'] * torch.minimum(inventory_on_hand, current_demands) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
                )
        
        # Reward given by underage_costs + holding_costs
        else:
            reward = (
                observation['underage_costs'] * torch.clip(-post_inventory_on_hand, min=0) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
                )
        
        # If we are in a lost demand setting, we cannot have negative inventory
        if self.problem_params['lost_demand']:
            post_inventory_on_hand = torch.clip(post_inventory_on_hand, min=0)

        # Update store inventories based on the lead time corresponding to each sample and store.
        # If all lead times are the same, we can simplify this step by just adding the allocation to the last
        # column of the inventory tensor and moving all columns "to the left".
        # We leave this method as it is more general!
        
        store_orders = action['stores']
        
        store_lead_times = observation['lead_times']
        
        observation['store_inventories'] = self.update_inventory_for_heterogeneous_lead_times(
            store_inventory, 
            post_inventory_on_hand, 
            store_orders, 
            store_lead_times, 
            self._internal_data['allocation_shift']
            )
        
        # # Uncomment the following line to simplify the update of store inventories.
        # # Only works when all lead times are the same, and the lead time is larger than 1.
        # # Only tested for one-store setting.
        # observation['store_inventories'] = self.move_left_add_first_col_and_append(
        #     post_inventory_on_hand, 
        #     observation["store_inventories"], 
        #     int(observation["lead_times"][0, 0]), 
        #     action["stores"]
        #     )
        
        return reward.sum(dim=1)
    
    def calculate_warehouse_reward_and_update_warehouse_inventories(self, action, observation):
        """
        Calculate reward and observation after action is executed for warehouses
        """

        warehouse_inventory = self.observation['warehouse_inventories']
        warehouse_inventory_on_hand = warehouse_inventory[:, :, 0]
        
        # Calculate store orders per warehouse
        # action['stores'] is [batch, n_stores, n_warehouses]
        # Sum across stores to get total orders for each warehouse
        warehouse_store_orders = action['stores'].sum(dim=1)  # [batch, n_warehouses]
        
        post_warehouse_inventory_on_hand = warehouse_inventory_on_hand - warehouse_store_orders

        reward = observation['warehouse_holding_costs'] * torch.clip(post_warehouse_inventory_on_hand, min=0)
        # Expand warehouse_lead_times to 3D to match allocation shape
        warehouse_lead_times_3d = observation['warehouse_lead_times'].unsqueeze(2)
        observation['warehouse_inventories'] = self.update_inventory_for_heterogeneous_lead_times(
            warehouse_inventory, 
            post_warehouse_inventory_on_hand, 
            action['warehouses'], 
            warehouse_lead_times_3d, 
            self._internal_data['warehouse_allocation_shift']
            )

        return reward.sum(dim=1)
    
    def calculate_echelon_reward_and_update_echelon_inventories(self, action, observation):
        """
        Calculate reward and observation after action is executed for warehouses
        """

        echelon_inventories = self.observation['echelon_inventories']
        echelon_inventory_on_hand = echelon_inventories[:, :, 0]
        # Sum across source dimension (dim=2) for both echelons and warehouses
        # For echelons, we need orders from echelon i+1 (hence [:, 1:])
        # For warehouses, we sum all warehouse orders for the last echelon
        echelon_orders_from_downstream = action['echelons'][:, 1:, :].sum(dim=2)  # Sum across sources
        warehouse_total_orders = action['warehouses'].sum(dim=(1, 2)).unsqueeze(1)  # Sum across warehouses and sources
        actions_to_subtract = torch.concat([echelon_orders_from_downstream, warehouse_total_orders], dim=1)
        post_echelon_inventory_on_hand = echelon_inventory_on_hand - actions_to_subtract

        reward = observation['echelon_holding_costs'] * torch.clip(post_echelon_inventory_on_hand, min=0)

        # Expand echelon_lead_times to 3D to match allocation shape
        echelon_lead_times_3d = observation['echelon_lead_times'].unsqueeze(2)
        observation['echelon_inventories'] = self.update_inventory_for_heterogeneous_lead_times(
            echelon_inventories, 
            post_echelon_inventory_on_hand, 
            action['echelons'], 
            echelon_lead_times_3d, 
            self._internal_data['echelon_allocation_shift']
            )

        return reward.sum(dim=1)

    def initialize_observation(self, data, observation_params):
        """
        Initialize the observation of the environment
        """

        observation = {
            'store_inventories': data['initial_inventories'],
            'current_period': torch.tensor([0])
            }
        
        if observation_params['include_warehouse_inventory']:
            observation['warehouse_lead_times'] = data['warehouse_lead_times']
            observation['warehouse_holding_costs'] = data['warehouse_holding_costs']
            observation['warehouse_inventories'] = data['initial_warehouse_inventories']
        
        if self.problem_params['n_extra_echelons'] > 0:
            observation['echelon_lead_times'] = data['echelon_lead_times']
            observation['echelon_holding_costs'] = data['echelon_holding_costs']
            observation['echelon_inventories'] = data['initial_echelon_inventories']

        # Include static features in observation (e.g., holding costs, underage costs, lead time and upper bounds)
        for k, v in observation_params['include_static_features'].items():
            if v:
                observation[k] = data[k]


        # Initialize past demands in the observation
        if observation_params['demand']['past_periods'] > 0:
            observation['past_demands'] = self.update_past_demands(data, observation_params, self.batch_size, self.n_stores, current_period=0)

        # Initialize time-related features, such as days to christmas
        if observation_params['time_features']:
            self.update_time_features(data, observation, observation_params, current_period=0)
        
        # Initialize sample-related features, such as the store number to which each sample belongs
        # This is just left as an example, and is not currently used by any policy
        self.create_sample_features(
            data, 
            observation, 
            observation_params
            )

        return observation
    
    def initialize_action_space(self, batch_size, problem_params, observation_params):
        """
        Initialize the action space by creating a dict with spaces.Box with shape batch_size x locations
        """

        d = {'stores': spaces.Box(low=0.0, high=np.inf, shape=(batch_size, problem_params['n_stores']), dtype=np.float32)}

        for k1, k2 in zip(['warehouses', 'extra_echelons'], ['n_warehouses', 'n_extra_echelons']):
            if problem_params[k2] > 0:
                d[k1] = spaces.Box(low=0.0, high=np.inf, shape=(batch_size, problem_params[k2]), dtype=np.float32)

        return spaces.Dict(d)
    
    def initialize_observation_space(self, initial_observation, periods, problem_params):
        
        # Initialize defaultdict to a dictionary with specific keys
        box_values = DefaultDict(lambda: {'low': -np.inf, 'high': np.inf, 'dtype': np.float32})
        box_values.update({
            'holding_costs': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'lead_times': {'low': 0, 'high': 2*10, 'dtype': np.int8},
            'days_to_christmas': {'low': -365, 'high': 365, 'dtype': np.int8},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'store_inventories': {'low': 0 if problem_params['lost_demand'] else -np.inf, 'high': np.inf, 'dtype': np.float32},
            'warehouse_inventories': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'warehouse_lead_times': {'low': 0, 'high': 2*10, 'dtype': np.int8},
            'extra_echelons_inventories': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'underage_costs': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'warehouse_upper_bound': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'current_period': {'low': 0, 'high': periods, 'dtype': np.int8},
        })

        return spaces.Dict(
            {
            k: spaces.Box(
                low=box_values[k]['low'], 
                high=box_values[k]['high'], 
                shape=v.shape,
                dtype=box_values[k]['dtype']
                ) 
                for k, v in initial_observation.items()
                })
    
    def update_inventory_for_heterogeneous_lead_times(self, inventory, inventory_on_hand, allocation, lead_times, allocation_shifter):
        """
        Update the inventory for heterogeneous lead times (something simpler can be done for homogeneous lead times).
        We add the inventory into corresponding position by flatenning out the state vector of the
        entire batch. We created allocation_shifts earlier, which dictates the position shift of that long vector
        for each store and each sample. We then add the corresponding lead time to obtain the actual position in 
        which to insert the action
        
        Now handles 3D allocation tensors for multi-warehouse scenarios.
        """
        
        # Allocation is always 3D: [batch, n_stores, n_warehouses]
        # Create the base inventory tensor with zeros for new orders
        # Each warehouse's allocation will be added at its specific lead time position
        updated_inventory = torch.stack(
            [
                inventory_on_hand + inventory[:, :, 1], 
                *self.move_columns_left(inventory, 1, inventory.shape[2] - 1), 
                torch.zeros(allocation.shape[0], allocation.shape[1], device=allocation.device, dtype=inventory.dtype)
            ], 
            dim=2
        )
        
        # Now add each warehouse's contribution with its specific lead time
        batch_size, n_stores, n_warehouses = allocation.shape
        
        # Expand allocation_shifter to handle warehouse dimension
        # allocation_shifter is [batch, n_stores], need [batch, n_stores, n_warehouses]
        allocation_shifter_3d = allocation_shifter.unsqueeze(2).expand(-1, -1, n_warehouses)
        
        # Flatten everything for the put operation
        indices = (allocation_shifter_3d + lead_times.long() - 1).flatten()
        values = allocation.flatten()
        
        # Filter out zero allocations to avoid unnecessary operations
        non_zero_mask = values != 0
        if non_zero_mask.any():
            indices = indices[non_zero_mask]
            values = values[non_zero_mask]
            # Ensure values have the same dtype as updated_inventory
            values = values.to(updated_inventory.dtype)
            updated_inventory = updated_inventory.put(indices, values, accumulate=True)
        
        return updated_inventory

    def update_past_demands(self, data, observation_params, batch_size, stores, current_period):
        """
        Update the past demands in the observation
        """
        
        past_periods = observation_params['demand']['past_periods']
        current_period_shifted = current_period + self._internal_data['period_shift']
        
        if current_period_shifted == 0:
            past_demands = torch.zeros(batch_size, stores, past_periods).to(self.device)
        # If current_period_shifted < past_periods, we fill with zeros at the left
        else:
            past_demands = data['demands'][:, :, max(0, current_period_shifted - past_periods): current_period_shifted]

            fill_with_zeros = past_periods - (current_period_shifted - max(0, current_period_shifted - past_periods))
            if fill_with_zeros > 0:
                past_demands = torch.cat([
                    torch.zeros(batch_size, stores, fill_with_zeros).to(self.device), 
                    past_demands
                    ], 
                    dim=2)
        
        return past_demands
    
    def update_time_features(self, data, observation, observation_params, current_period):
        """
        Update all data that depends on time in the observation (e.g., days to christmas)
        """
        if observation_params['time_features'] is not None:
            for k in observation_params['time_features']:
                if data[k].shape[2] + 2 < current_period:
                    raise ValueError('Current period is greater than the number of periods in the data')
                observation[k] = data[k][:, :, min(current_period + observation_params['demand']['period_shift'], data[k].shape[2] - 1)]
    
    def create_sample_features(self, data, observation, observation_params):
        """
        Create features that only depend on the sample index (and not on the period) in the observation
        """

        if observation_params['sample_features'] is not None:
            for k in observation_params['sample_features']:
                observation[k] = data[k]
    
    def update_days_to_christmas(self, data, observation_params, current_period):
        """
        Update the days to christmas in the observation
        """
        days_to_christmas = data['days_to_christmas'][current_period + observation_params['demand']['period_shift']]
        
        return days_to_christmas

    def update_past_data(self):
        """
        Update the past data observations (e.g. last demands) in the observation
        """

        if self._internal_data['demands'].shape[2] + 2 < self.observation['current_period'].item():
            raise ValueError('Current period is greater than the number of periods in the data')
        if self.observation_params['demand']['past_periods'] > 0:
            self.observation['past_demands'] = self.update_past_demands(
                self._internal_data,
                self.observation_params,
                self.batch_size,
                self.n_stores,
                current_period=min(self.observation['current_period'].item() + 1, self._internal_data['demands'].shape[2])  # do this before updating current period!
                )

    def move_columns_left(self, tensor_to_displace, start_index, end_index):
        """
        Move all columns in given array to the left, and return as list
        """

        return [tensor_to_displace[:, :, i + 1] for i in range(start_index, end_index)]
    
    def move_left_and_append(self, tensor_to_displace, tensor_to_append, start_index=0, end_index=None, dim=2):
        """
        Move all columns in given array to the left, and append a new tensor at the end
        """
        
        if end_index is None:
            end_index = tensor_to_displace.shape[dim] - 1
        
        return torch.stack([*self.move_columns_left(tensor_to_displace, start_index, end_index), tensor_to_append],
                           dim=dim)

    def move_left_add_first_col_and_append(self, post_inventory_on_hand, inventory, lead_time, action):
        """
        Move columns of inventory (deleting first column, as post_inventory_on_hand accounts for inventory_on_hand after demand arrives)
          to the left, add inventory_on_hand to first column, and append action at the end
        """

        return torch.stack([
            post_inventory_on_hand + inventory[:, :, 1], 
            *self.move_columns_left(inventory, 1, lead_time - 1), 
            action
            ], dim=2)