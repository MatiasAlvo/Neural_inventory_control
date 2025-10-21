#!/usr/bin/env python3
"""
Training script for the new DataDrivenNetWithForecasts neural network.
This script demonstrates how to train the neural network with LGBM forecast features.
"""

import yaml
import torch
from pathlib import Path
from main_run import main

def train_with_forecasts():
    """Train the neural network with forecast features."""
    
    # Configuration file for the new neural network
    config_file = 'config_files/settings/vn2_round_1_data_with_lgbm.yml'
    
    print(f"Training neural network with forecast features using config: {config_file}")
    print("="*60)
    
    # Check if the config file exists
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Please create the configuration file first.")
        return False
    
    # Load and display the configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully!")
    print(f"Neural network type: {config['nn_params']['name']}")
    print(f"Forecasts path: {config['nn_params'].get('forecasts_raw_path', 'Not specified')}")
    print(f"Training epochs: {config['trainer_params']['epochs']}")
    print(f"Learning rate: {config['optimizer_params']['learning_rate']}")
    print("="*60)
    
    # Run the training
    try:
        main(config_file, 'train')
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False

if __name__ == "__main__":
    success = train_with_forecasts()
    if success:
        print("\n🎉 Neural network training with forecast features completed successfully!")
    else:
        print("\n💥 Training failed. Check the error messages above.")
