import yaml
import os
import pandas as pd
from trainer import *

# Load the configuration files
# config_setting_file = 'Code_to_submit/config_files/settings/one_store_backlogged.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/vanilla_one_store.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/base_stock.yml'

config_setting_file = 'Code_to_submit/config_files/settings/one_store_lost.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/capped_base_stock.yml'
config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/vanilla_one_store.yml'

# config_setting_file = 'Code_to_submit/config_files/settings/transshipment_backlogged.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/vanilla_transshipment.yml'

# config_setting_file = 'Code_to_submit/config_files/settings/one_warehouse_lost_demand.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/symmetry_aware.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/vanilla_transshipment.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/vanilla_one_warehouse.yml'

# config_setting_file = 'Code_to_submit/config_files/settings/one_store_real_data_lost_demand.yml'
# config_setting_file = 'Code_to_submit/config_files/settings/one_store_real_data_read_file_example.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/just_in_time.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/returns_nv.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/quantile_nv.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/fixed_quantile.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/transformed_nv.yml'
# config_hyperparams_file = 'Code_to_submit/config_files/policies_and_hyperparams/data_driven_net.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)

with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'sample_data_params'
hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
seeds, test_seeds, problem_params, params_by_dataset, observation_params, store_params, warehouse_params, sample_data_params = [config_setting[key] for key in setting_keys]
trainer_params, optimizer_params, nn_params = [config_hyperparams[key] for key in hyperparams_keys]
observation_params = DefaultDict(lambda: None, observation_params)

# create a tensor of random numbers, of size samples x n_stores
# tensor = torch.randint(0, 10, (params_by_dataset['train']['n_samples'] + params_by_dataset['dev']['n_samples'], 3)).float()
# torch.save(tensor, 'Code_to_submit/data_files/underage_cost_3_stores.pt')
# tensor = torch.load('Code_to_submit/data_files/underage_cost_3_stores.pt')
# print(tensor)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset_creator = DatasetCreator()

if sample_data_params['split_by_period']:
    
    scenario = Scenario(
        periods=None,  # period info for each dataset is given in sample_data_params
        problem_params=problem_params, 
        store_params=store_params, 
        warehouse_params=warehouse_params, 
        num_samples=params_by_dataset['train']['n_samples'],  # in this case, num_samples=number of products, which has to be the same across all datasets
        observation_params=observation_params, 
        seeds=seeds
        )
    
    train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
        scenario, 
        split=True, 
        by_period=True, 
        periods_for_split=[sample_data_params[k] for  k in ['train_periods', 'dev_periods', 'test_periods']],)

else:
    scenario = Scenario(
        periods=params_by_dataset['train']['periods'], 
        problem_params=problem_params, 
        store_params=store_params, 
        warehouse_params=warehouse_params, 
        num_samples=params_by_dataset['train']['n_samples'] + params_by_dataset['dev']['n_samples'], 
        observation_params=observation_params, 
        seeds=seeds
        )

    train_dataset, dev_dataset = dataset_creator.create_datasets(scenario, split=True, by_sample_indexes=True, sample_index_for_split=params_by_dataset['dev']['n_samples'])

    scenario = Scenario(
        params_by_dataset['test']['periods'], 
        problem_params, 
        store_params, 
        warehouse_params, 
        params_by_dataset['test']['n_samples'], 
        observation_params, 
        test_seeds
        )

    test_dataset = dataset_creator.create_datasets(scenario, split=False)

train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False)

# print(f'scenario.demand: {scenario.demands[0]}')
# print(f"train_dataset[0]['days_from_christmas']: {train_dataset[0]['days_from_christmas']}")
# print(f"dev_dataset[0]['days_from_christmas']: {dev_dataset[0]['days_from_christmas']}")
# print(f"test_dataset[0]['days_from_christmas']: {test_dataset[0]['days_from_christmas']}")

data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

neural_net_creator = NeuralNetworkCreator
model = neural_net_creator().create_neural_network(scenario, nn_params, device=device)

loss_function = PolicyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])

simulator = Simulator(device=device)
trainer = Trainer(device=device)

# we will create a folder for each day of the year, and a subfolder for each model
# when executing with different problem primitives (i.e. instance), it might be useful to create an additional subfolder for each instance
trainer_params['base_dir'] = 'Code_to_submit/saved_models'
trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['name']]
# we will simply name the model with the current time stamp
trainer_params['save_model_filename'] = trainer.get_time_stamp()

# Load previous model if load_model is set to True in the config file
if trainer_params['load_previous_model']:
    print(f'Loading model from {trainer_params["load_model_path"]}')
    model, optimizer = trainer.load_model(model, optimizer, trainer_params['load_model_path'])

trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)

# deploy on test set, and enforce the discrete allocation if the demand is poisson
average_test_loss, average_test_loss_to_report = trainer.test(
    loss_function, 
    simulator, 
    model, 
    data_loaders, 
    optimizer, 
    problem_params, 
    observation_params, 
    params_by_dataset, 
    discrete_allocation=store_params['demand']['distribution'] == 'poisson'
    )

print(f'test loss: {average_test_loss_to_report}')