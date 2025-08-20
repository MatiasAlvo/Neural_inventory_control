import yaml
import pandas as pd
from trainer import *
import sys

# Check if command-line arguments for setting and hyperparameter filenames are provided (which corresponds to third and fourth parameters)
if len(sys.argv) == 4:
    setting_name = sys.argv[2]
    hyperparams_name = sys.argv[3]

elif len(sys.argv) == 2:
    setting_name = 'one_store_lost'  # One store under lost demand setting
    hyperparams_name = 'vanilla_one_store'  # Vanilla neural network for one store setting
else:
    print(f'Number of parameters provided including script name: {len(sys.argv)}')
    print(f'Number of parameters should be either 4 or 2 (so that last 2 parameters defined in main_run.py)')
    assert False

train_or_test = sys.argv[1]

print(f'Setting file name: {setting_name}')
print(f'Hyperparams file name: {hyperparams_name}\n')

config_setting_file = f'config_files/settings/{setting_name}.yml'
config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)

with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params'
hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
seeds, test_seeds, problem_params, params_by_dataset, observation_params, store_params, warehouse_params, echelon_params, sample_data_params = [
    config_setting[key] for key in setting_keys
    ]

trainer_params, optimizer_params, nn_params = [config_hyperparams[key] for key in hyperparams_keys]
observation_params = DefaultDict(lambda: None, observation_params)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset_creator = DatasetCreator()

# For realistic data, train, dev and test sets correspond to the same products, but over disjoint periods.
# We will therefore create one scenario, and then split the data into train, dev and test sets by 
# "copying" all non-period related information, and then splitting the period related information
if sample_data_params['split_by_period']:
    
    scenario = Scenario(
        periods=None,  # period info for each dataset is given in sample_data_params
        problem_params=problem_params, 
        store_params=store_params, 
        warehouse_params=warehouse_params, 
        echelon_params=echelon_params, 
        num_samples=params_by_dataset['train']['n_samples'],  # in this case, num_samples=number of products, which has to be the same across all datasets
        observation_params=observation_params, 
        seeds=seeds
        )
    
    train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
        scenario, 
        split=True, 
        by_period=True, 
        periods_for_split=[sample_data_params[k] for  k in ['train_periods', 'dev_periods', 'test_periods']],)

# For synthetic data, we will first create a scenario that we will divide into train and dev sets by sample index.
# Then, we will create a separate scenario for the test set, which will be exaclty the same as the previous scenario, 
# but with different seeds to generate demand traces, and with a longer time horizon.
# One can use this method of generating scenarios to train a model using some specific problem primitives, 
# and then test it on a different set of problem primitives, by simply creating a new scenario with the desired primitives.
else:
    scenario = Scenario(
        periods=params_by_dataset['train']['periods'], 
        problem_params=problem_params, 
        store_params=store_params, 
        warehouse_params=warehouse_params, 
        echelon_params=echelon_params, 
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
        echelon_params, 
        params_by_dataset['test']['n_samples'], 
        observation_params, 
        test_seeds
        )

    test_dataset = dataset_creator.create_datasets(scenario, split=False)

train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False)
data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

neural_net_creator = NeuralNetworkCreator
model = neural_net_creator().create_neural_network(scenario, nn_params, problem_params, device=device)

loss_function = PolicyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])

simulator = Simulator(device=device)
trainer = Trainer(device=device)

# We will create a folder for each day of the year, and a subfolder for each model
# When executing with different problem primitives (i.e. instance), it might be useful to create an additional subfolder for each instance
trainer_params['base_dir'] = 'saved_models'
trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['name']]

# We will simply name the model with the current time stamp
trainer_params['save_model_filename'] = trainer.get_time_stamp()

# Load previous model if load_model is set to True in the config file
if trainer_params['load_previous_model']:
    print(f'Loading model from {trainer_params["load_model_path"]}')
    model, optimizer = trainer.load_model(model, optimizer, trainer_params['load_model_path'])

if train_or_test == 'train':
    trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)

elif train_or_test in ['test', 'train']:
    # Deploy on test set, and enforce discrete allocation if the demand is poisson
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

    print(f'Average per-period test loss: {average_test_loss_to_report}')

else:
    print(f'Invalid argument: {train_or_test}')
    assert False