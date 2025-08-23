# Neural Inventory Control
Implementation of Hindsight Differentiable Policy Optimization, as described in the paper [Neural Inventory Control in Networks via Hindsight Differentiable Policy Optimization](https://arxiv.org/abs/2306.11246).

## Introduction

Inventory management offers unique opportunities for reliably evaluating and applying deep reinforcement learning (DRL). We introduce Hindsight Differentiable Policy Optimization (HDPO), facilitating direct optimization of a DRL policy's hindsight performance using stochastic gradient descent. HDPO leverages two key elements: (i) an ability to backtest any policy's performance on a sample of historical scenarios, and (ii) the differentiability of the total cost incurred in a given scenario. We assess this approach in four problem classes where we can benchmark performance against the true optimum. HDPO algorithms consistently achieve near-optimal performance across all these classes, even when dealing with up to 60-dimensional raw state vectors. Moreover, we propose a natural neural network architecture to address problems with weak (or aggregate) coupling constraints between locations in an inventory network. This architecture utilizes weight duplication for "sibling" locations and state summarization. We demonstrate empirically that this design significantly enhances sample efficiency and provide justification for it through an asymptotic performance guarantee. Lastly, we assess HDPO in a setting that incorporates real sales data from a retailer, demonstrating its superiority over generalized newsvendor strategies.

## Citation

You can cite our work using the following bibtex entry:

```
@article{alvo2023neural,
  title={Neural inventory control in networks via hindsight differentiable policy optimization},
  author={Alvo, Matias and Russo, Daniel and Kanoria, Yash},
  journal={arXiv preprint arXiv:2306.11246},
  year={2023}
}
```


## Installation

To set up the environment for this project, follow these steps:

1. Clone this repository to your local machine:

```
git clone git@github.com:MatiasAlvo/Neural_inventory_control.git
```

2. Navigate to the project directory:
```
cd Neural_inventory_control
```

3. Create a conda environment using the provided environment.yml file
```
conda env create -f environment.yml
```

4. Activate the conda environment:
```
conda activate neural_inventory_control
```

5. Install torch with pip:
```
pip install torch
```

## Usage

The main functionalities for the code are in the following scripts:

1. `data_handling.py`: Defines how to create a Scenarios class (which is actually a collection of scenarios). This includes sampling parameters (such as demand or costs when they are obtained by sampling), and sampling the demand traces. It also defines the Dataset class.

2. `environment.py`: Defines the functionalities of the environment (simulator).

3. `main_run.py`: Runs HDPO. We also provide a Jupyter notebook for the same purpose (`main_run.ipynb`).

4. `neural_networks.py`: Defines the neural network functionalities.

5. `trainer.py`: Defines the Trainer class, which is in charge of the training itself, including handling the interactions with the simulator and updating the neural network's weights.

Parameters for settings/instances and policies are to be defined in a config file under the **config_files/settings** and **config_files/policies_and_hyperparams**, respectively. Instructions on how to do this are detailed in a later section. 

The code can be executed from the terminal by running the following:
```
python3 main_run.py [mode] [config_file_path] [hyperparameters_file_path]
```
Here, `[mode]` can be either `train` or `test`. If `train` is specified, the model is executed on the test set after training. The last two parameters define the filenames for the configuration files for the environment and hyperparameters, respectively. For example, if you want to train (and then test) a model for the setting defined in `one_store_lost`, considering the hyperparameters (including neural network architecture) defined in the file `vanilla_one_store`, you should run:
```
python3 main_run.py train one_store_lost vanilla_one_store
```

We allow for providing the filenames for the setting and hyperparameter config files in `main_run.py`, in which case the last 2 parameters have to be omitted in the terminal. For example, you can run:
```
python3 main_run.py train
```
and specify the filenames within the main script.

## Settings and policy classes

We provide config files for all the settings and all the policies described in our paper. For detailed descriptions of the settings and neural network architectures that we consider, as well as which policy classes/architectures are considered for each setting, please refer to our [paper](https://arxiv.org/abs/2306.11246).

The settings considered are the following:

- `one_store_backlogged`: One store under a backlogged demand assumption.
- `one_store_lost`: One store under a lost demand assumption.
- `one_store_real_data_lost_demand`: One store under a lost demand assumption, but considering demand traces built using sales data from a real retailer.
- `one_store_real_data_read_file_example`: Same as the previous one, but in which all problem primitives (lead times, underage costs and holding costs) are read from files. We provide this example for instructional purposes.
- `one_warehouse_lost_demand`: One warehouse and many stores, under a lost demand assumption.
- `serial_system`: Serial system, in which inventory flows linearly from upstream towards downstream locations. Considers a backlogged demand assumption
- `transshipment_backlogged`: One transshipment center (i.e., a warehouse that cannot hold inventory) and many stores, under a backlogged demand assumption.

The policy classes we consider are the following:
- `base_stock`: Base stock policy (optimal for the setting of one store under a backlogged demand assumption).
- `capped_base_stock`: Capped base stock policy (well-performing for the setting of one store under a lost demand assumption).
- `data_driven_net`: Defines a direct mapping from time-series data (previous demands) and current inventory on-hand to an order amount. In our paper, we refer to it as HDPO (end-to-end) in the section "HDPO with real time series data". This policy class can be modified to utilize a different set of features.
- `echelon_stock`: Echelon stock policy (optimal for a serial system under a backlogged demand assumption).
- `fixed_quantile`: Generalized newsvendor policy, that utilizes the same quantile for each scenario.
- `gnn`: Graph Neural Network for one warehouse many stores settings, using message passing between nodes.
- `just_in_time`: Non-admissible oracle policy, that looks into the future and orders to precisely meet future demand.
- `quantile_nv`: Generalized newsvendor policy, that utilizes the newsvendor quantile (p/[p+h]).
- `returns_nv`: Generalized newsvendor policy, that utilizes the newsvendor quantile (p/[p+h]), but allows for negative orders. It defines a non-admissible policy.
- `symmetry_aware`: Symmetry-aware neural network for settings with one warehouse and many stores.
- `transformed_nv`: Generalized newsvendor policy, that considers a flexible mapping from newsvendor quantile (p/[p+h]) to a new quantile. This quantile is therefore different across scenarios, but fixed across time for each scenario.
- `vanilla_one_store`: Vanilla neural network for settings with one store and no warehouse.
- `vanilla_one_warehouse`: Vanilla neural network for settings with one warehouse and many stores.
- `vanilla_serial`: Vanilla neural network for the serial system setting.
- `vanilla_transshipment`: Vanilla neural network for the setting with one transshipment center and many stores.

To create a new policy class, follow these steps:

1. Open the `neural_networks.py` file.
2. Inside `neural_networks.py`, create a new class that inherits from `MyNeuralNetwork`. Define the `forward` method within this class to specify the forward pass of your neural network architecture.
3. In the `get_architecture` method of the `NeuralNetworkCreator` class (located in neural_networks.py), add your newly created neural network to the dictionary of architectures.
4. Finally, create a config file for your policy class under `config/files/policies_and_hyperparams`. This config should define the necessary parameters to instantiate your policy, as well as other parameters (to be defined below).

## Populating a config file

### Setting config file

#### `seeds` and `test_seeds`:
- `underage_cost`: Seed to sample underage costs, in case they are random.
- `holding_cost`: Seed to sample holding costs, in case they are random.
- `mean`: Seed to sample mean parameters, in case they are random.
- `coef_of_var`: Seed to sample the coefficient of variations (which defines the standard deviation), in case they are random.
- `lead_time`: Seed to sample lead times, in case they are random.
- `demand`: Seed to sample demand traces.
- `initial_inventory`: Seed to sample the value of initial inventories at the stores.

#### `sample_data_params`:
- `split_by_period`: Whether to split the dataset by period or by sample. If True (currently only used for setting with real data), train, dev and test sets
are generated by copying all values (costs and lead times), and splitting demand traces into disjoint sets of periods.

#### `problem_params`: 
- `n_stores`: The number of stores in the inventory network.
- `n_warehouses`: The number of warehouses in the inventory network.
- `n_extra_echelons`: The number of extra echelons in the inventory network.
- `lost_demand`: Whether to consider unmet demand to be lost or backlogged.
- `maximize_profit`: True if objective is to maximize profit. False if objective is to minimize cost.

#### `params_by_dataset`: one dictionary for each of train, dev and test, containing
- `n_samples`: The number of samples in the dataset.
- `batch_size`: The batch size for training.
- `periods`: The number of periods to simulate.
- `ignore_periods`: The number of initial periods to ignore for purposes of reporting costs.

#### `observation_params`: defines which information is containted in the observation
- `include_warehouse_inventory`: Whether to include warehouse inventory in observations.
- `include_static_features`: 
  - `holding_costs`: Whether to include holding costs in observations.
  - `underage_costs`: Whether to include underage costs in observations.
  - `lead_times`: Whether to include lead times in observations.
- `demand`: 
  - `past_periods`: The number of past periods to include in demand observations.
  - `period_shift`: The shift in demand periods. It shifts the beginning of the planning horizon from 0 to this value.
- `time_features_file`: Path to the file containing time-related features.
- `time_features`: List of time-related features to include (e.g., 'days_from_christmas').
- `sample_features_file`: Path to the file containing scenario-related features (e.g., product type).
- `sample_features`: List of scenario-related features to include (e.g., 'store_nbr').

#### `store_params`:
- `demand`: 
  - `sample_across_stores`: Whether to sample the parameters for each store. Only works for normal distributions. If True, sample in the ranges specified in `mean_range` and `coef_of_var_range`.
  - `vary_across_samples`: Whether to sample the parameters for each different scenario. Only works for normal distributions. If True, sample in the ranges specified in `mean_range` and `coef_of_var_range`.
  - `expand`: Whether the copy demand parameters across all stores and scenarios by "expanding". If True, we copy the values of `mean` and `std`.
  - `mean_range`: The range from which to sample the mean of demands.
  - `coef_of_var_range`: The range of the coefficient of variation of the demand distribution from which to sample.
  - `mean`: Mean of the distribution, which is fixed across all stores and samples.
  - `std`: Standard deviation of the distribution, which is fixed across all stores and samples.
  - `distribution`: The distribution of demand. Can be normal, poisson or real.
  - `correlation`: The correlation coefficient for pairwise demands.
  - `clip`: Whether to clip demand values below from 0.
  - `decimals`: The number of decimals for demand values.
- `lead_time/holding_cost/underage_cost`: 
  - `sample_across_stores`: Whether to sample the parameters for each store.
  - `vary_across_samples`: Whether to sample the parameters for each different scenario. If True, sample in the range specified in `range`.
  - `expand`: Whether the copy demand parameters across all stores and scenarios by "expanding". If True, we copy the value `value`.
  - `range`: The range from which to sample the parameter values.
  - `value`: Value to be copied across all stores and scenarios
  - `file_location`: Location for directly reading the values of the parameters, instead of defining them manually.
- `initial_inventory`: 
  - `sample`: True if considering a random initial inventory, given as random_value*mean_demand_per_store.
  - `range_mult`: The range from which to sample random_value.
  - `inventory_periods`: How many periods to consider for the inventories state (which might be larger that lead time, if specified).

#### `warehouse_params`:
- `holding_cost`: The holding cost for the warehouse.
- `lead_time`: The lead time for the warehouse.

#### `echelon_params`:
  - `holding_cost`: A list specifying the holding cost for each echelon.
  - `lead_time`: A list specifying the lead time for each echelon.

### Config file for Hyperparameters and Neural Network Architectures

#### `trainer_params`:
- `epochs`: The number of epochs to train the neural network.
- `do_dev_every_n_epochs`: Perform evaluation on the dev dataset every `n` epochs.
- `early_stopping_patience_epochs`: Stop training if the chosen metric (dev_loss or train_loss) doesn't improve for this many epochs. Set to `null` to disable early stopping.
- `print_results_every_n_epochs`: Print training results every `n` epochs.
- `save_model`: Whether to save the trained model.
- `epochs_between_save`: Number of epochs between model saving (and only if performance improved).
- `choose_best_model_on`: Performance on which to select the best model. Can be `dev_loss` or `train_loss`
- `load_previous_model`: Whether to load a previously trained model.
- `load_model_path`: Path to the previously trained model.

#### `optimizer_params`:
- `learning_rate`: The learning rate used by the optimizer.

#### `nn_params`:
- `name`: The name of the neural network architecture.
- `inner_layer_activations`: Activation functions for inner layers of each component/module of the neural network.
- `output_layer_activation`: Activation function for the output layer of each component/module of the neural network.
- `neurons_per_hidden_layer`: Number of neurons for each hidden layer of each component/module of the neural network.
- `output_sizes`: Size of the output layer for each component of the neural network.
- `initial_bias`: Initial bias for the last layer of each component/module of the neural network.
- `gradient_clipping_norm_value`: Maximum norm for gradient clipping. If specified, gradients will be clipped to this value during training to prevent exploding gradients. Set to `null` to disable gradient clipping.
- `forecaster_location`: Location for loading quantile forecaster, if used within the policy.
- `warehouse_upper_bound_mult`: Multiplier to calculate the upper bound for warehouse outputs.

## License

MIT License

Copyright 2024 Matias Alvo, Daniel Russo, Yashodhan Kanoria

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.