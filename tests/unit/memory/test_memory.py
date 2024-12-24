"""
Tests for memory components of consciousness model.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.unit.test_base import ConsciousnessTestBase
from models.memory import WorkingMemory, InformationIntegration, GRUCell

class TestMemoryComponents(ConsciousnessTestBase):
    """Test suite for memory components."""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_length(self):
        return 8

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def working_memory(self, hidden_dim):
        """Create working memory module for testing."""
        return WorkingMemory(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout_rate=0.1
        ).to(self.device)

    @pytest.fixture
    def info_integration(self, hidden_dim):
        """Create information integration module for testing."""
        return InformationIntegration(
            hidden_dim=hidden_dim,
            num_modules=4,
            dropout_rate=0.1
        ).to(self.device)

    @pytest.fixture
    def gru_cell(self, hidden_dim):
        """Create GRU cell for testing."""
        return GRUCell(input_dim=hidden_dim, hidden_dim=hidden_dim).to(self.device)

    def test_gru_state_updates(self, gru_cell, device, batch_size, hidden_dim):
        """Test GRU cell state updates."""
        x = torch.randn(batch_size, hidden_dim, device=device)
        h = torch.randn(batch_size, hidden_dim, device=device)

        # Initialize and run forward pass
        new_h = gru_cell(x, h)

        # Verify shapes
        self.assert_output_shape(new_h, (batch_size, hidden_dim))

        # State should be updated (different from initial state)
        assert not torch.allclose(new_h, h, rtol=1e-5)

    def test_memory_sequence_processing(self, working_memory, device, batch_size, seq_length, hidden_dim):
        """Test working memory sequence processing."""
        # Test with different sequence lengths
        for test_length in [4, 8, 16]:
            inputs = self.create_inputs(self.seed, batch_size, test_length, hidden_dim)
            initial_state = torch.zeros(batch_size, hidden_dim, device=device)

            output, final_state = working_memory(inputs, initial_state, deterministic=True)

            # Verify shapes adapt to sequence length
            self.assert_output_shape(output, (batch_size, test_length, hidden_dim))

    def test_context_aware_gating(self, working_memory, device, batch_size, seq_length, hidden_dim):
        """Test context-aware gating mechanisms."""
        # Create two different input sequences with controlled differences
        base_inputs = self.create_inputs(self.seed, batch_size, seq_length, hidden_dim)

        # Create similar and different inputs
        similar_inputs = base_inputs + torch.randn_like(base_inputs) * 0.1
        different_inputs = torch.randn(batch_size, seq_length, hidden_dim, device=device)

        initial_state = torch.zeros(batch_size, hidden_dim, device=device)

        # Process sequences
        _, state_base = working_memory(base_inputs, initial_state, deterministic=True)
        _, state_similar = working_memory(similar_inputs, initial_state, deterministic=True)
        _, state_different = working_memory(different_inputs, initial_state, deterministic=True)

        # Similar inputs should produce more similar states than different inputs
        base_similar_diff = torch.mean(torch.abs(state_base - state_similar))
        base_different_diff = torch.mean(torch.abs(state_base - state_different))
        assert base_similar_diff < base_different_diff

    def test_information_integration(self, info_integration, device, batch_size, seq_length, hidden_dim):
        """Test information integration computation."""
        # Create inputs with proper shape for information integration
        inputs = torch.stack([
            self.create_inputs(self.seed, batch_size, seq_length, hidden_dim)
            for _ in range(info_integration.num_modules)
        ], dim=1)  # Shape: [batch, num_modules, seq_length, hidden_dim]

        # Initialize and run forward pass
        output, phi = info_integration(inputs, deterministic=True)

        # Verify shapes
        expected_output_shape = (batch_size, info_integration.num_modules, seq_length, hidden_dim)
        self.assert_output_shape(output, expected_output_shape)

        # Phi should be a scalar per batch element
        assert phi.shape == (batch_size,)
        # Phi should be non-negative and finite
        assert torch.all(phi >= 0) and torch.all(torch.isfinite(phi))

    def test_memory_retention(self, working_memory, device, batch_size, seq_length, hidden_dim):
        """Test memory retention over sequences."""
        # Create a sequence with a distinctive pattern
        pattern = torch.ones(batch_size, 1, hidden_dim, device=device)
        inputs = torch.cat([
            pattern,
            self.create_inputs(self.seed, batch_size, seq_length-2, hidden_dim),
            pattern
        ], dim=1)

        initial_state = torch.zeros(batch_size, hidden_dim, device=device)

        output, final_state = working_memory(inputs, initial_state, deterministic=True)

        # Final state should capture pattern information
        assert torch.any(torch.abs(final_state) > 0.1)  # Non-zero activations
