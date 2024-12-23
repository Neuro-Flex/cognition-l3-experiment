"""Initialization module for AI consciousness implementation."""
import os
import torch
from typing import Tuple

from core.config import ModelConfig, TrainingConfig, HardwareConfig
from core.hardware import HardwareManager
from models.base_model import BaseModel, create_train_state

def initialize_system() -> Tuple[BaseModel, HardwareConfig, ModelConfig, TrainingConfig]:
    """Initialize the AI consciousness system with CPU optimization."""
    # Initialize hardware manager
    hw_manager = HardwareManager()
    hw_manager.setup_environment()

    # Get hardware configuration
    hw_config = HardwareConfig.from_environment()

    # Initialize model configuration with CPU optimizations
    model_config = ModelConfig()
    model_config.optimize_for_cpu()

    # Initialize training configuration
    train_config = TrainingConfig()
    train_config.optimize_for_cpu()

    # Create base model
    model = BaseModel(config=model_config)

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Test basic model functionality
    batch_size = model_config.batch_size
    seq_length = model_config.max_sequence_length
    test_input = torch.ones((batch_size, seq_length), dtype=torch.long)

    try:
        # Test forward pass
        output, pooled = model(test_input)
        print("Model initialization successful")
        print(f"Output shape: {output.shape}")
        print(f"Pooled output shape: {pooled.shape}")

        # Log hardware configuration
        hw_info = hw_manager.get_hardware_info()
        print("\nHardware Configuration:")
        for key, value in hw_info.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

    return model, hw_config, model_config, train_config

if __name__ == "__main__":
    initialize_system()
