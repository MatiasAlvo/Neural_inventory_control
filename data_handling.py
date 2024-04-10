from shared_imports import *

class Scenario():
    '''
    Class to generate a setting when there are parameters to be sampled.
    For example, we might sample the mean demand and std for each store, which is later going to be used to sample actual demand samples
    '''
    def __init__(self, periods, problem_params, store_params, warehouse_params, num_samples, observation_params, seeds=None):

        self.problem_params = problem_params
        self.store_params = store_params
        self.warehouse_params = warehouse_params
        self.num_samples = num_samples
        self.periods = periods
        self.observation_params = observation_params
        self.seeds = seeds
        self.demands = self.generate_demand_samples(problem_params, store_params, store_params['demand'], seeds)
        self.underage_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['underage_cost'], seeds['underage_cost'], discrete=False)
        self.holding_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['holding_cost'], seeds['holding_cost'], discrete=False)
        self.lead_times = self.generate_data_for_samples_and_stores(problem_params, store_params['lead_time'], seeds['lead_time'], discrete=True).to(torch.int64)
        self.means, self.stds = self.generate_means_and_stds(observation_params, store_params)
        self.initial_inventories = self.generate_initial_inventories(problem_params, store_params, self.demands, self.lead_times, seeds['initial_inventory'])

        # print(f'self.underage_costs: {self.underage_costs}')
        # print(f'self.holding_costs: {self.holding_costs}')
        # print(f'self.lead_times: {self.lead_times}')
        # print(f'self.means: {self.means}')
        # print(f'self.stds: {self.stds}')
        # print(f'self.initial_inventories: {self.initial_inventories}')
        
        self.initial_warehouse_inventories = self.generate_initial_warehouse_inventory(warehouse_params)
        self.warehouse_lead_times = self.generate_warehouse_data(warehouse_params, 'lead_time')
        self.warehouse_holding_costs = self.generate_warehouse_data(warehouse_params, 'holding_cost')

        time_and_sample_features = {'time_features': {}, 'sample_features': {}}
        # self.time_features = {}

        for feature_type, feature_file in zip(['time_features', 'sample_features'], ['time_features_file', 'sample_features_file']):
            if observation_params[feature_type] and observation_params[feature_file]:
            # if observation_params['time_features'] and observation_params['time_features_file']:
                features = pd.read_csv(observation_params[feature_file])
                for k in observation_params[feature_type]:
                    tensor_to_append = torch.tensor(features[k].values)
                    if feature_type == 'time_features':
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.problem_params['n_stores'], -1)
                    elif feature_type == 'sample_features':  # currently only supports the one store case
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(1).expand(-1, self.problem_params['n_stores'])
                    # self.time_features[k] = torch.tensor(time_features[k].values).unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.problem_params['n_stores'], -1)
            
        self.time_features = time_and_sample_features['time_features']
        self.sample_features = time_and_sample_features['sample_features']

        # creates a dictionary specifying which data has to be split by sample index and which by period (when dividing into train, dev, test sets)
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
                }
        
        for k, v in self.time_features.items():
            data[k] = v
        
        for k, v in self.sample_features.items():
            data[k] = v
        
        return {k: v.float() for k, v in data.items() if v is not None}
    
    def get_fixed_across_samples_data(self):

        params_fixed_across_samples = {
            'days_from_christmas': self.days_from_christmas
            }

        return {k: v.float() for k, v in params_fixed_across_samples.items() if v is not None}
    
    def define_how_to_split_data(self):
        """
        Define how to split the data into different samples
        If demand comes from real data, the training and dev sets correspond to different periods.
        However, if it is generated, the split is according to sample indexes
        """

        split_by = {'sample_index': ['underage_costs', 'holding_costs', 'lead_times', 'initial_inventories', 'initial_warehouse_inventories'], 
                    'period': []}

        if self.store_params['demand']['distribution'] == 'real':
            split_by['period'].append('demands')
        else:
            split_by['sample_index'].append('demands')
        
        for k in self.time_features.keys():
            split_by['period'].append(k)

        for k in self.sample_features.keys():
            split_by['sample_index'].append(k)
            # split_by['period'].append('days_from_christmas')

        # print(f'self.get_data(): {self.get_data().keys()}')
        
        return split_by
    
    def generate_demand_samples(self, problem_params, store_params, demand_params, seeds):
        """
        Generate demand data
        """
                
        # sample parameters to generate demand if necessary (otherwise, does nothing)
        self.generate_demand_parameters(problem_params, demand_params, seeds)

        demand_generator_functions = {
            "normal": self.generate_normal_demand, 
            'poisson': self.generate_poisson_demand,
            'real': self.read_real_demand_data,
            }
        # print(f'demand_params: {demand_params}')

        # changing demand seed for consistency with results prensented in the manuscript
        self.adjust_seeds_for_consistency(problem_params, store_params, seeds)

        # sample demand
        demand = demand_generator_functions[demand_params['distribution']](problem_params, demand_params, seeds['demand'])
        # print(f'demand: {demand[0]}')
        # print(f'demand: {demand.shape}')

        if demand_params['clip']:  # truncate at 0 from below if specified
            demand = np.clip(demand, 0, None)
        
        return torch.tensor(demand)

    def adjust_seeds_for_consistency(self, problem_params, store_params, seeds):

        if problem_params['n_warehouses'] == 0 and problem_params['n_stores'] == 1 and store_params['demand']['distribution'] != 'real':
            try:
                # changing demand seed for consistency with results prensented in the manuscript
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

        # set seed
        if seed is not None:
            np.random.seed(seed)
        
        if problem_params['n_stores'] == 1:
            demand = np.random.normal(demand_params['mean'], 
                                      demand_params['std'], 
                                      size=(self.num_samples, 1, self.periods)
                                      )
        else:
            # calculate covariance matrix and sample from multivariate normal
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

        # set seed
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

        # set seed
        np.random.seed(seeds['mean'])

        means = np.random.uniform(demand_params['mean_range'][0], demand_params['mean_range'][1], problem_params['n_stores']).round(3)
        # del demand_params['mean_range']
        np.random.seed(seeds['coef_of_var'])
        coef_of_var = np.random.uniform(demand_params['coef_of_var_range'][0], demand_params['coef_of_var_range'][1], problem_params['n_stores'])
        # del demand_params['coef_of_var_range']
        stds = (means * coef_of_var).round(3)
        return {'mean': means, 'std': stds}
    
    def generate_data_for_samples_and_stores(self, problem_params, cost_params, seed, discrete=False):
        """
        Generate cost or lead time data, for each sample and store
        """
        
        np.random.seed(seed)

        # we first create a default dict from the params dictionary, so that we return False by default
        # whenever we query a key that was not set by the user
        params_copy = DefaultDict(lambda: False, copy.deepcopy(cost_params))

        sample_functions = {False: np.random.uniform, True: np.random.randint}
        # sample uniformly from the range (discrete)
        this_sample_function = sample_functions[discrete]
        

        if params_copy['file_location']:
            params_copy['value'] = torch.load(params_copy['file_location'])[: self.num_samples]
        if params_copy['sample_across_stores']:
            return torch.tensor(this_sample_function(*params_copy['range'], problem_params['n_stores'])).expand(self.num_samples, -1)
        elif params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples)).unsqueeze(1).expand(-1, problem_params['n_stores'])
        elif params_copy['expand']:
            # this_n_stores = 3
            # expanded = torch.tensor(params_copy['value']).expand(self.num_samples, this_n_stores)
            # print(f"expand: {expanded[3]}")
            # assert False
            # print(f'value: {params_copy["value"]}')
            return torch.tensor(params_copy['value']).expand(self.num_samples, problem_params['n_stores'])
        else:
            # print(f'value: {params_copy["value"]}')
            # assert False
            return torch.tensor(params_copy['value'])
    
    def generate_initial_inventories(self, problem_params, store_params, demands, lead_times, seed):
        """
        Generate initial inventory data
        """
        # set seed
        np.random.seed(seed)

        if store_params['initial_inventory']['sample']:
            # change type of demands to float

            # demand_mean = demands.float().mean(dim=0)
            demand_mean = demands.float().mean(dim=2).mean(dim=0)
            demand_mults = np.random.uniform(*store_params['initial_inventory']['range_mult'], 
                                             size=(self.num_samples, 
                                                   problem_params['n_stores'], 
                                                   max(store_params['initial_inventory']['inventory_periods'], lead_times.max()) 
                                                   )
                                            )
            return demand_mean[None, :, None] * demand_mults

        else:
            return torch.zeros(self.num_samples, 
                               problem_params['n_stores'], 
                               store_params['initial_inventory']['inventory_periods'])
    
    def generate_initial_warehouse_inventory(self, warehouse_params):
        """
        Generate initial warehouse inventory data
        """
        if warehouse_params is None:
            return None
        
        return torch.zeros(self.num_samples, 
                           1, 
                           warehouse_params['lead_time']
                           )
    
    def generate_warehouse_data(self, warehouse_params, key):
        """
        Generate warehouse lead time data
        For now, it is simply fixed across all samples
        """
        if warehouse_params is None:
            return None
        
        return torch.tensor([warehouse_params[key]]).expand(self.num_samples, self.problem_params['n_warehouses'])

    def generate_days_from_christmas(self, store_params):

        if store_params['demand']['distribution'] == 'real' and False:
            raise NotImplementedError
        else:
            return None
    
    def generate_means_and_stds(self, observation_params, store_params):

        to_return = {'mean': None, 'std': None}
        for k in ['mean', 'std']:
            if k in observation_params['include_static_features'] and observation_params['include_static_features'][k]:
                to_return[k] = torch.tensor(store_params['demand'][k]).unsqueeze(0).expand(self.num_samples, -1)
                # to_return[k] = store_params['demand'][k]
        return to_return['mean'], to_return['std']
        # return torch.tensor(to_return['mean']).unsqueeze(0).expand(self.num_samples, -1), torch.tensor(to_return['std']).unsqueeze(0).expand(self.num_samples, -1)

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
                # train_data, dev_data = self.split_by_period(scenario, period_for_split)
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
        common_data = {k: data[k] for k in scenario.split_by['sample_index']}
        out_datasets = []

        for period_range in periods_for_split:
            this_data = copy.deepcopy(common_data)
            # change period_range to slice object (it is a string)
            period_range = slice(*map(int, period_range.strip('() ').split(',')))

            for k in scenario.split_by['period']:
                this_data[k] = data[k][:, :, period_range]
            out_datasets.append(this_data)
        
        return out_datasets

    
    def create_single_dataset(self, data):
        """
        Create a single dataset
        """

        num_samples = len(data['initial_inventories'])

        return MyDataset(num_samples, data)
    