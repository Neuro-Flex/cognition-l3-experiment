"""Configuration module for AI consciousness implementation.

This module handles hardware-specific configurations and model parameters.
"""
from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class HardwareConfig:
    """Hardware configuration settings."""
    device_type: str = "cpu"  # Current device type (cpu/cuda)
    num_devices: int = 1      # Number of available devices
    memory_limit: Optional[int] = None  # Memory limit in bytes

    @classmethod
    def from_environment(cls) -> "HardwareConfig":
        """Initialize configuration from current environment."""
        # Detect available CUDA devices
        if torch.cuda.is_available():
            device_type = "cuda"
            num_devices = torch.cuda.device_count()
            # Get memory info for first GPU
            if num_devices > 0:
                memory_limit = torch.cuda.get_device_properties(0).total_memory
        else:
            device_type = "cpu"
            num_devices = 1
            memory_limit = None

        return cls(
            device_type=device_type,
            num_devices=num_devices,
            memory_limit=memory_limit
        )

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_sequence_length: int = 512
    vocab_size: int = 50000
    dropout_rate: float = 0.1

    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    use_amp: bool = False  # Automatic Mixed Precision flag

    def optimize_for_device(self, device_type: str):
        """Adjust parameters based on device type."""
        if device_type == "cpu":
            self.batch_size = min(self.batch_size, 32)
            self.use_amp = False
            self.gradient_accumulation_steps = max(self.gradient_accumulation_steps, 4)
        elif device_type == "cuda":
            self.batch_size = 64
            self.use_amp = True
            self.gradient_accumulation_steps = 1

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # PyTorch specific settings
    optimizer: str = "AdamW"
    scheduler: str = "linear"
    gradient_clip_val: float = 1.0

    def optimize_for_device(self, device_type: str):
        """Adjust training parameters based on device type."""
        if device_type == "cpu":
            self.batch_size = 16
            self.gradient_accumulation_steps = 4
            self.eval_steps = 1000
        elif device_type == "cuda":
            self.batch_size = 32
            self.gradient_accumulation_steps = 1
            self.eval_steps = 500