"""
Base test configuration and utilities for consciousness model tests.
"""
import torch
import pytest

class ConsciousnessTestBase:
    """Base class for consciousness model tests."""

    @pytest.fixture
    def seed(self):
        """Provide reproducible random seed."""
        return 42

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
        return 64

    @pytest.fixture
    def num_heads(self):
        """Default number of attention heads."""
        return 4

    @pytest.fixture
    def deterministic(self):
        """Default deterministic mode for testing."""
        return True

    def create_inputs(self, seed, batch_size, seq_length, hidden_dim):
        """Create random input tensors."""
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
