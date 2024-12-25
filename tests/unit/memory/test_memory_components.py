"""
Unit tests for working memory and GRU components.
"""
import torch
import torch.nn as nn
import pytest

from models.memory import GRUCell, WorkingMemory, InformationIntegration

class TestGRUCell:
    @pytest.fixture
    def gru_cell(self):
        return GRUCell(input_dim=32, hidden_dim=64)  # Changed input_dim to match test cases

    def test_gru_state_updates(self, gru_cell):
        # Test dimensions
        batch_size = 2
        input_dim = 32
        hidden_dim = 64

        # Create sample inputs
        x = torch.randn(batch_size, input_dim)
        h = torch.randn(batch_size, hidden_dim)

        # Initialize parameters
        gru_cell.reset_parameters()

        # Apply GRU cell
        new_h = gru_cell(x, h)

        # Test output shape
        assert new_h.shape == (batch_size, hidden_dim)

        # Test state update properties
        # Values should be bounded by tanh activation
        assert torch.all(torch.abs(new_h) <= 1.0)

        # Test multiple updates maintain reasonable values
        for _ in range(10):
            h = new_h
            new_h = gru_cell(x, h)
            assert torch.all(torch.isfinite(new_h))
            assert torch.all(torch.abs(new_h) <= 1.0)

    def test_gru_reset_gate(self, gru_cell):
        batch_size = 2
        input_dim = 32
        hidden_dim = 64

        x = torch.randn(batch_size, input_dim)
        h = torch.randn(batch_size, hidden_dim)

        gru_cell.reset_parameters()

        # Test with zero input
        x_zero = torch.zeros_like(x)
        h_zero = gru_cell(x_zero, h)

        # With zero input, new state should be influenced by reset gate
        # and should be different from previous state
        assert not torch.allclose(h_zero, h)

        # Test with zero state
        h_zero = torch.zeros_like(h)
        new_h = gru_cell(x, h_zero)

        # With zero state, output should be primarily determined by input
        assert not torch.allclose(new_h, h_zero)

class TestWorkingMemory:
    @pytest.fixture
    def memory_module(self):
        return WorkingMemory(input_dim=32, hidden_dim=64, dropout_rate=0.1)  # Changed input_dim

    def test_sequence_processing(self, memory_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32
        hidden_dim = 64

        # Create sample sequence
        inputs = torch.randn(batch_size, seq_length, input_dim)
        
        # Process sequence (removed reset_parameters call)
        outputs, final_state = memory_module(inputs, deterministic=True)

        # Test output shapes
        assert outputs.shape == (batch_size, seq_length, hidden_dim)
        assert final_state.shape == (batch_size, hidden_dim)

        # Test temporal consistency
        # Later timesteps should be influenced by earlier ones
        first_half = outputs[:, :seq_length//2, :]
        second_half = outputs[:, seq_length//2:, :]

        # Calculate temporal correlation
        combined = torch.cat([
            first_half.reshape(-1, hidden_dim),
            second_half.reshape(-1, hidden_dim)
        ], dim=0)

        correlation_matrix = torch.corrcoef(combined)
        
        # Extract correlations between first_half and second_half
        num_first = first_half.reshape(-1, hidden_dim).shape[0]
        correlations = correlation_matrix[num_first:, :num_first]

        correlation = torch.mean(torch.abs(correlations))

        # Assert that the average correlation is above the threshold
        assert correlation > 0.1

    def test_memory_retention(self, memory_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        inputs = torch.randn(batch_size, seq_length, input_dim)

        # Test with different initial states
        initial_state = torch.randn(batch_size, 64)

        outputs1, final_state1 = memory_module(inputs, prev_state=initial_state)  # Changed initial_state to prev_state

        outputs2, final_state2 = memory_module(inputs, prev_state=torch.zeros_like(initial_state))

        # Different initial states should lead to different outputs
        assert not torch.allclose(outputs1, outputs2)
        assert not torch.allclose(final_state1, final_state2)
