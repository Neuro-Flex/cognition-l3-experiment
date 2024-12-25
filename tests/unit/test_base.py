"""
Base test configuration and utilities for consciousness model tests.
"""
import torch
import pytest
import random

class ConsciousnessTestBase:
    """Base class for consciousness model tests."""

    @pytest.fixture
    def seed(self):
        """Fixture to provide a random seed."""
        return random.randint(0, 2**32 - 1)

    @pytest.fixture
    def batch_size(self):
        """Default batch size for tests."""
        return 2

    @pytest.fixture
    def seq_length(self):
        """Default sequence length for tests."""
        return 8

    @pytest.fixture
    def hidden_dim(self):
        """Default hidden dimension."""
        return 128

    @pytest.fixture
    def num_heads(self):
        """Default number of attention heads."""
        return 4

    @pytest.fixture
    def deterministic(self):
        """Default deterministic mode for testing."""
        return True

    def create_inputs(self, seed: int, batch_size: int, seq_length: int, hidden_dim: int) -> torch.Tensor:
        """
        Create input tensors with a specific seed for reproducibility.

        Args:
            seed (int): Seed value for random number generator.
            batch_size (int): Number of samples in a batch.
            seq_length (int): Sequence length.
            hidden_dim (int): Dimension of hidden layers.

        Returns:
            torch.Tensor: Generated input tensor.
        """
        torch.manual_seed(seed)
        return torch.randn(batch_size, seq_length, hidden_dim)

    def assert_output_shape(self, output, expected_shape):
        """Assert output has expected shape."""
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    def assert_valid_attention(self, attention_weights):
        """Assert attention weights are valid probabilities."""
        # Check shape and values
        assert torch.all(attention_weights >= 0), "Attention weights must be non-negative"
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1))), "Attention weights must sum to 1"
