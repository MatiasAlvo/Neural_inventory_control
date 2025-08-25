from shared_imports import *

class Scenario():
    '''
    Class to generate an instance. 
    First samples parameters (e.g, mean demand and std for each store, costs, lead times, etc...) if there are parameters to be sampled.
    Then, creates demand traces, and the initial values (e.g., of inventory) to be used.
    '''
    def __init__(self, periods, problem_params, store_params, warehouse_params, echelon_params, num_samples, observation_params, seeds=None):

        self.problem_params = problem_params
        self.store_params = store_params
        self.warehouse_params = warehouse_params
        self.echelon_params = echelon_params
        self.num_samples = num_samples
        self.periods = periods
        self.observation_params = observation_params
        self.seeds = seeds
        self.demands = self.generate_demand_samples(problem_params, store_params, store_params['demand'], seeds)
        self.underage_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['underage_cost'], seeds['underage_cost'], discrete=False)
        self.holding_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['holding_cost'], seeds['holding_cost'], discrete=False)
        self.lead_times = self.generate_lead_times(problem_params, store_params['lead_time'], seeds['lead_time'])
            
        self.means, self.stds = self.generate_means_and_stds(observation_params, store_params)
        self.initial_inventories = self.generate_initial_inventories(problem_params, store_params, self.demands, self.lead_times, seeds['initial_inventory'])
        
        self.initial_warehouse_inventories = self.generate_initial_warehouse_inventory(warehouse_params)
        self.warehouse_lead_times = self.generate_warehouse_data(warehouse_params, 'lead_time')
        self.warehouse_holding_costs = self.generate_warehouse_data(warehouse_params, 'holding_cost')
        self.warehouse_edge_costs = self.generate_warehouse_data(warehouse_params, 'edge_cost') if warehouse_params and 'edge_cost' in warehouse_params else None

        self.initial_echelon_inventories = self.generate_initial_echelon_inventory(echelon_params)
        self.echelon_lead_times = self.generate_echelon_data(echelon_params, 'lead_time')
        self.echelon_holding_costs = self.generate_echelon_data(echelon_params, 'holding_cost')

        time_and_sample_features = {'time_features': {}, 'sample_features': {}}

        for feature_type, feature_file in zip(['time_features', 'sample_features'], ['time_features_file', 'sample_features_file']):
            if observation_params.get(feature_type) and observation_params.get(feature_file):
                features = pd.read_csv(observation_params[feature_file])
                for k in observation_params[feature_type]:
                    tensor_to_append = torch.tensor(features[k].values)
                    if feature_type == 'time_features':
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.problem_params['n_stores'], -1)
                    elif feature_type == 'sample_features':  # Currently only supports the one store case
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(1).expand(-1, self.problem_params['n_stores'])
            
        self.time_features = time_and_sample_features['time_features']
        self.sample_features = time_and_sample_features['sample_features']

        # Creates a dictionary specifying which data has to be split by sample index and which by period (when dividing into train, dev, test sets)
        self.split_by = self.define_how_to_split_data()

    def get_data(self):
        """
        Return the generated data. Will be part of a Dataset
        """

        data =  {'demands': self.demands,
                'underage_costs': self.underage_costs,
                'holding_costs': self.holding_costs,
                'lead_times': self.lead_times,
                'mean': self.means,
                'std': self.stds,
                'initial_inventories': self.initial_inventories,
                'initial_warehouse_inventories': self.initial_warehouse_inventories,
                'warehouse_lead_times': self.warehouse_lead_times,
                'warehouse_holding_costs': self.warehouse_holding_costs,
                'warehouse_edge_costs': self.warehouse_edge_costs,
                'initial_echelon_inventories': self.initial_echelon_inventories,
                'echelon_holding_costs': self.echelon_holding_costs,
                'echelon_lead_times': self.echelon_lead_times,
                }
        
        for k, v in self.time_features.items():
            data[k] = v
        
        for k, v in self.sample_features.items():
            data[k] = v
        
        return {k: v.float() for k, v in data.items() if v is not None}
    
    def define_how_to_split_data(self):
        """
        Define how to split the data into different samples
        If demand comes from real data, the training and dev sets correspond to different periods.
        However, if it is generated, the split is according to sample indexes
        """

        split_by = {'sample_index': ['underage_costs', 'holding_costs', 'lead_times', 'initial_inventories'], 
                    'period': []}
        
        # Only include warehouse inventories if there are warehouses
        if self.problem_params['n_warehouses'] > 0:
            split_by['sample_index'].append('initial_warehouse_inventories')
            split_by['sample_index'].append('warehouse_lead_times')
            split_by['sample_index'].append('warehouse_holding_costs')
            split_by['sample_index'].append('warehouse_edge_costs')

        # Include echelon data if there are extra echelons
        if self.problem_params['n_extra_echelons'] > 0:
            split_by['sample_index'].append('initial_echelon_inventories')
            split_by['sample_index'].append('echelon_holding_costs')
            split_by['sample_index'].append('echelon_lead_times')
        
        if self.store_params['demand']['distribution'] == 'real':
            split_by['period'].append('demands')
        else:
            split_by['sample_index'].append('demands')
        
        # Add mean and std if they're going to be generated (not None)
        if 'mean' in self.observation_params['include_static_features'] and self.observation_params['include_static_features']['mean']:
            split_by['sample_index'].append('mean')
        if 'std' in self.observation_params['include_static_features'] and self.observation_params['include_static_features']['std']:
            split_by['sample_index'].append('std')
        
        for k in self.time_features.keys():
            split_by['period'].append(k)

        for k in self.sample_features.keys():
            split_by['sample_index'].append(k)
        
        return split_by
    
    def generate_demand_samples(self, problem_params, store_params, demand_params, seeds):
        """
        Generate demand data
        """
                
        # Sample parameters to generate demand if necessary (otherwise, does nothing)
        self.generate_demand_parameters(problem_params, demand_params, seeds)

        demand_generator_functions = {
            "normal": self.generate_normal_demand, 
            'poisson': self.generate_poisson_demand,
            'real': self.read_real_demand_data,
            }

        # Changing demand seed for consistency with results prensented in the manuscript
        self.adjust_seeds_for_consistency(problem_params, store_params, seeds)

        # Sample demand traces
        demand = demand_generator_functions[demand_params['distribution']](problem_params, demand_params, seeds['demand'])

        if demand_params['clip']:  # Truncate at 0 from below if specified
            demand = np.clip(demand, 0, None)
        
        return torch.tensor(demand)

    def adjust_seeds_for_consistency(self, problem_params, store_params, seeds):
        """
        Adjust seeds for consistency with results prensented in the manuscript
        """

        if problem_params['n_warehouses'] == 0 and problem_params['n_stores'] == 1 and store_params['demand']['distribution'] != 'real':
            try:
                # Changing demand seed for consistency with results prensented in the manuscript
                seeds['demand'] = seeds['demand'] + int(store_params['lead_time']['value'] + 10*store_params['underage_cost']['value'])
            except Exception as e:
                print(f'Error: {e}')
    
    def read_real_demand_data(self, problem_params, demand_params, seed):
        """
        Read real demand data
        """

        demand = torch.load(demand_params['file_location'])[: self.num_samples]
        return demand

    def generate_demand_parameters(self, problem_params, demand_params, seeds):
        """
        Sample parameters of demand distribution, if necessary
        """
        
        if demand_params['sample_across_stores']:  # only supported for normal demand
            demand_params.update(self.sample_normal_mean_and_std(problem_params, demand_params, seeds))
    
    def generate_normal_demand(self, problem_params, demand_params, seed):
        """
        Generate normal demand data
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        if problem_params['n_stores'] == 1:
            demand = np.random.normal(demand_params['mean'], 
                                      demand_params['std'], 
                                      size=(self.num_samples, 1, self.periods)
                                      )
        else:
            # Calculate covariance matrix and sample from multivariate normal
            correlation = demand_params['correlation']
            cov_matrix = [[correlation*v1*v2 if i!= j else v1*v2 
                           for i, v1 in enumerate(demand_params['std'])
                           ] 
                           for j, v2 in enumerate(demand_params['std'])
                           ]
            demand = np.random.multivariate_normal(demand_params['mean'], cov=cov_matrix, size=(self.num_samples, self.periods))
            demand = np.transpose(demand, (0, 2, 1))

        return demand

    def generate_poisson_demand(self, problem_params, demand_params, seed):

        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.poisson(demand_params['mean'], size=(self.num_samples, problem_params['n_stores'], self.periods))

    def generate_data(self, demand_params, **kwargs):
        """
        Generate demand data
        """
        demand_generator_functions = {"normal": self.generate_normal_demand_for_one_store}
        demand = demand_generator_functions[demand_params['distribution']](demand_params, **kwargs)
        
        if demand_params['clip']:
            demand = np.clip(demand, 0, None)

        return torch.tensor(demand)
        
    def sample_normal_mean_and_std(self, problem_params, demand_params, seeds):
        """
        Sample mean and std for normal demand
        """
            
        # Set seed
        np.random.seed(seeds['mean'])

        means = np.random.uniform(demand_params['mean_range'][0], demand_params['mean_range'][1], problem_params['n_stores']).round(3)
        np.random.seed(seeds['coef_of_var'])
        coef_of_var = np.random.uniform(demand_params['coef_of_var_range'][0], demand_params['coef_of_var_range'][1], problem_params['n_stores'])
        stds = (means * coef_of_var).round(3)
        return {'mean': means, 'std': stds}
    
    def generate_data_for_samples_and_stores(self, problem_params, cost_params, seed, discrete=False):
        """
        Generate cost or lead time data, for each sample and store
        """
        
        np.random.seed(seed)

        # We first create a default dict from the params dictionary, so that we return False by default
        # whenever we query a key that was not set by the user
        params_copy = DefaultDict(lambda: False, copy.deepcopy(cost_params))

        sample_functions = {False: np.random.uniform, True: np.random.randint}
        this_sample_function = sample_functions[discrete]
        

        if params_copy['file_location']:
            params_copy['value'] = torch.load(params_copy['file_location'])[: self.num_samples]
        if params_copy['sample_across_stores']:
            return torch.tensor(this_sample_function(*params_copy['range'], problem_params['n_stores'])).expand(self.num_samples, -1)
        elif params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples)).unsqueeze(1).expand(-1, problem_params['n_stores'])
        elif params_copy['expand']:
            # Check if value is a 2D matrix (for warehouse-to-store lead times)
            value_tensor = torch.tensor(params_copy['value'])
            if value_tensor.dim() == 2:
                # This is a [n_stores, n_warehouses] matrix for warehouse-to-store lead times
                # Expand to [num_samples, n_stores, n_warehouses]
                return value_tensor.unsqueeze(0).expand(self.num_samples, -1, -1)
            else:
                # Original behavior for 1D values
                return value_tensor.expand(self.num_samples, problem_params['n_stores'])
        else:
            return torch.tensor(params_copy['value'])
    
    def generate_lead_times(self, problem_params, lead_time_params, seed):
        """
        Generate lead times ensuring proper 3D dimensionality [num_samples, n_stores, n_warehouses]
        """
        lead_times_raw = self.generate_data_for_samples_and_stores(problem_params, lead_time_params, seed, discrete=True)
        
        if lead_times_raw.dim() == 2:
            n_warehouses = problem_params.get('n_warehouses', 0)
            if n_warehouses > 0:
                lead_times = lead_times_raw.unsqueeze(2).expand(-1, -1, n_warehouses)
            else:
                lead_times = lead_times_raw.unsqueeze(2)
        else:
            lead_times = lead_times_raw
        
        return lead_times.to(torch.int64)
    
    def generate_initial_inventories(self, problem_params, store_params, demands, lead_times, seed):
        """
        Generate initial inventory data
        """
        # Set seed
        np.random.seed(seed)

        if store_params['initial_inventory']['sample']:
            demand_mean = demands.float().mean(dim=2).mean(dim=0)
            demand_mults = np.random.uniform(*store_params['initial_inventory']['range_mult'], 
                                             size=(self.num_samples, 
                                                   problem_params['n_stores'], 
                                                   max(store_params['initial_inventory']['inventory_periods'], lead_times.max().item()) 
                                                   )
                                            )
            return demand_mean[None, :, None] * demand_mults

        else:
            return torch.zeros(self.num_samples, 
                               problem_params['n_stores'], 
                               store_params['initial_inventory']['inventory_periods'])
    
    def generate_initial_warehouse_inventory(self, warehouse_params):
        """
        Generate initial warehouse inventory data for multiple warehouses
        """
        if warehouse_params is None:
            return None
        
        n_warehouses = self.problem_params['n_warehouses']
        
        # Handle lead_time as either scalar or list
        if isinstance(warehouse_params['lead_time'], list):
            max_lead_time = max(warehouse_params['lead_time'])
        else:
            max_lead_time = warehouse_params['lead_time']
        
        return torch.zeros(self.num_samples, 
                           n_warehouses, 
                           max_lead_time)
    
    def generate_initial_echelon_inventory(self, echelon_params):
        """
        Generate initial echelon inventory data
        """
        if echelon_params is None:
            return None
        
        return torch.zeros(self.num_samples, 
                           len(echelon_params['lead_time']), 
                           max(echelon_params['lead_time'])
                           )
    
    def generate_warehouse_data(self, warehouse_params, key):
        """
        Generate warehouse data for multiple warehouses
        Supports both scalar values (same for all warehouses) and lists (different per warehouse)
        """
        if warehouse_params is None:
            return None
        
        n_warehouses = self.problem_params['n_warehouses']
        value = warehouse_params[key]
        
        if isinstance(value, list):
            # Different values for each warehouse
            if len(value) != n_warehouses:
                raise ValueError(f"warehouse_params['{key}'] list length {len(value)} doesn't match n_warehouses {n_warehouses}")
            return torch.tensor(value).unsqueeze(0).expand(self.num_samples, -1)
        else:
            # Same value for all warehouses
            return torch.tensor([value]).expand(self.num_samples, n_warehouses)
    
    def generate_echelon_data(self, echelon_params, key):
        """
        Generate echelon_params data
        For now, it is simply fixed across all samples
        """
        if echelon_params is None:
            return None
        
        return torch.tensor(echelon_params[key]).unsqueeze(0).expand(self.num_samples, -1)
    
    def generate_means_and_stds(self, observation_params, store_params):
        """
        Create tensors with store demand's means and stds.
        Will be used as inputs for the symmetry-aware NN.
        """

        to_return = {'mean': None, 'std': None}
        for k in ['mean', 'std']:
            if k in observation_params['include_static_features'] and observation_params['include_static_features'][k]:
                to_return[k] = torch.tensor(store_params['demand'][k]).unsqueeze(0).expand(self.num_samples, -1)
        return to_return['mean'], to_return['std']

class MyDataset(Dataset):

    def __init__(self, num_samples, data):
        self.data = data
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    

class DatasetCreator():

    def __init__(self):

        pass

    def create_datasets(self, scenario, split=True, by_period=False, by_sample_indexes=False, periods_for_split=None, sample_index_for_split=None):

        if split:
            if by_period:
                return [self.create_single_dataset(data) for data in self.split_by_period(scenario, periods_for_split)]
            elif by_sample_indexes:
                train_data, dev_data = self.split_by_sample_index(scenario, sample_index_for_split)
            else:
                raise NotImplementedError
            return self.create_single_dataset(train_data), self.create_single_dataset(dev_data)
        else:
            return self.create_single_dataset(scenario.get_data())
    
    def split_by_sample_index(self, scenario, sample_index_for_split):
        """
        Split dataset into dev and train sets by sample index
        We consider the first entries to correspomd to the dev set (so that size of train set does not impact it)
        This should be used when demand is synthetic (otherwise, if demand is real, there would be data leakage)
        """

        data = scenario.get_data()

        dev_data = {k: v[:sample_index_for_split] for k, v in data.items()}
        train_data = {k: v[sample_index_for_split:] for k, v in data.items()}

        return train_data, dev_data
    
    def split_by_period(self, scenario, periods_for_split):

        data = scenario.get_data()
        # Only include keys that actually exist in the data
        common_data = {k: data[k] for k in scenario.split_by['sample_index'] if k in data}
        out_datasets = []

        for period_range in periods_for_split:
            this_data = copy.deepcopy(common_data)
            # Change period_range to slice object (it is currently of type string)
            period_range = slice(*map(int, period_range.strip('() ').split(',')))

            for k in scenario.split_by['period']:
                if k in data:  # Only add if key exists
                    this_data[k] = data[k][:, :, period_range]
            out_datasets.append(this_data)
        
        return out_datasets

    
    def create_single_dataset(self, data):
        """
        Create a single dataset
        """

        num_samples = len(data['initial_inventories'])

        return MyDataset(num_samples, data)
    