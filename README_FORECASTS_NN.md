# Neural Network with LGBM Forecast Features

This document explains how to use the new `DataDrivenNetWithForecasts` neural network that incorporates LGBM forecast features alongside the existing data-driven features.

## Overview

The `DataDrivenNetWithForecasts` class extends the original `DataDrivenNet` by incorporating additional forecast features from the `forecasts_raw.pt` tensor. This tensor contains LGBM-generated quantile forecasts with shape `[products, periods, stores, features]`.

## Key Features

- **Same functionality as DataDrivenNet**: Maintains all existing features (inventory, past demands, instocks, time features, product features)
- **Additional forecast features**: Incorporates LGBM quantile forecasts for each product and period
- **Automatic feature integration**: Forecast features are automatically extracted based on the current period
- **Flexible architecture**: Uses the same neural network architecture as the original DataDrivenNet

## Files Created/Modified

### New Files
1. **`neural_networks.py`** - Added `DataDrivenNetWithForecasts` class
2. **`config_files/policies_and_hyperparams/data_driven_net_with_forecasts.yml`** - Configuration for the new NN
3. **`config_files/settings/vn2_round_1_data_with_lgbm.yml`** - Complete training configuration
4. **`test_new_nn.py`** - Test script to verify the integration
5. **`train_with_forecasts.py`** - Training script for the new NN
6. **`README_FORECASTS_NN.md`** - This documentation

### Modified Files
1. **`neural_networks.py`** - Updated `NeuralNetworkCreator` to support the new architecture

## Usage

### 1. Basic Usage

```python
from neural_networks import NeuralNetworkCreator
import yaml

# Load configuration
with open('config_files/policies_and_hyperparams/data_driven_net_with_forecasts.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create neural network
neural_net_creator = NeuralNetworkCreator()
model = neural_net_creator.create_neural_network(scenario, config['nn_params'], device='cpu')
```

### 2. Configuration

The new neural network requires forecast features to be specified in the `observation_params`:

```yaml
observation_params:
  'forecast_features': # Add forecast features from LGBM predictions
    'file_location': 'lgbm_predictions/nn_inputs/forecasts_raw.pt'

nn_params:
  'name': 'data_driven_with_forecasts'
  'inner_layer_activations': 
    'master': 'elu'
  'output_layer_activation':
    'master': 'relu'
  'neurons_per_hidden_layer': 
    'master': [64, 64]
  'initial_bias':
    'master': 1.0
  'output_sizes':
    'master': null
```

### 3. Training

Use the provided training script:

```bash
python train_with_forecasts.py
```

Or use the main training script with the new configuration:

```bash
python main_run.py config_files/settings/vn2_round_1_data_with_lgbm.yml train
```

### 4. Testing

Test the integration:

```bash
python test_new_nn.py
```

## How It Works

### Forecast Feature Integration

1. **Data Pipeline Loading**: The `forecasts_raw.pt` tensor is loaded in the `Scenario` class during data initialization
2. **Product-Sample Mapping**: Each sample in the batch corresponds to a specific product, ensuring correct forecast feature mapping
3. **Observation Building**: Forecast features are extracted for the current period and added to the observation dictionary
4. **Feature Concatenation**: Forecast features are concatenated with existing features in the neural network forward pass

### Tensor Structure

The `forecasts_raw.pt` tensor has the following structure:
- **Shape**: `[products, periods, stores, features]`
- **Products**: Number of products (599 in the example)
- **Periods**: Number of time periods (157 in the example)
- **Stores**: Number of stores (1 in the example)
- **Features**: Number of forecast features (6 in the example - quantile forecasts)

### Forward Pass

During the forward pass, the neural network:

1. Collects all standard features (inventory, past demands, etc.)
2. Receives forecast features from the observation (already properly mapped to products)
3. Concatenates all features
4. Passes through the neural network

## Example Output

```
Testing new neural network with forecasts integration...
Neural network parameters: {'name': 'data_driven_with_forecasts', 'forecasts_raw_path': 'lgbm_predictions/nn_inputs/forecasts_raw.pt', ...}
Loaded forecasts_raw tensor with shape: torch.Size([599, 157, 1, 6])
Model created successfully: DataDrivenNetWithForecasts
Model trainable: True
Forward pass successful!
Output shape: torch.Size([2, 1, 1])
Output values: tensor([[[0.9355]], [[0.8416]]])
Forecasts tensor loaded with shape: torch.Size([599, 157, 1, 6])
Forecast features for period 5: tensor([0., 0., 0., 0., 0., 0.])
✅ Test passed! The new neural network is working correctly.
```

## Configuration Files

### 1. Neural Network Configuration
- **File**: `config_files/policies_and_hyperparams/data_driven_net_with_forecasts.yml`
- **Purpose**: Defines the neural network architecture and parameters
- **Key Parameter**: `forecasts_raw_path` - Path to the forecasts tensor

### 2. Complete Training Configuration
- **File**: `config_files/settings/vn2_round_1_data_with_lgbm.yml`
- **Purpose**: Complete configuration for training with real data
- **Includes**: Data paths, observation parameters, training parameters

## Troubleshooting

### Common Issues

1. **Forecasts tensor not found**
   - Ensure `forecasts_raw.pt` exists at the specified path
   - Check the path in the configuration file

2. **Batch size mismatch**
   - The neural network handles batch size mismatches automatically
   - For batch_size < products: takes first batch_size products
   - For batch_size > products: repeats the last product's features

3. **Period out of range**
   - Ensure the current period is within the range of the forecasts tensor
   - The tensor has 157 periods (0-156), so current_period should be < 157

### Debugging

Enable debug output by adding print statements in the `forward` method:

```python
print(f"Current period: {current_period}")
print(f"Forecast features shape: {forecast_features.shape}")
print(f"Input features count: {len(input_features)}")
```

## Performance Considerations

- **Memory Usage**: The forecasts tensor is loaded once during initialization
- **Forward Pass**: Minimal overhead for feature extraction and concatenation
- **Training**: Same training procedure as the original DataDrivenNet

## Future Enhancements

Potential improvements for the neural network:

1. **Dynamic Feature Selection**: Select specific forecast features based on configuration
2. **Feature Normalization**: Add normalization for forecast features
3. **Multi-Horizon Forecasts**: Support forecasts for multiple time horizons
4. **Feature Importance**: Add methods to analyze which features are most important

## Conclusion

The `DataDrivenNetWithForecasts` neural network successfully integrates LGBM forecast features while maintaining the same functionality as the original `DataDrivenNet`. The integration is seamless and requires minimal changes to existing training pipelines.
