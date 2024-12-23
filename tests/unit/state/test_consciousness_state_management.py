"""
Tests for consciousness state management.
"""
import pytest
import torch
import numpy as np
from tests.unit.test_base import ConsciousnessTestBase
from models.consciousness_state import ConsciousnessStateManager

@pytest.fixture
def state_manager():
    """Create a ConsciousnessStateManager instance for testing."""
    return ConsciousnessStateManager(hidden_dim=64, num_states=4)

class TestStateManagement(ConsciousnessTestBase):
    """Test suite for consciousness state management."""

    def test_state_updates(self, state_manager, batch_size, hidden_dim):
        """Test consciousness state updates."""
        consciousness_state = torch.randn(batch_size, hidden_dim)
        integrated_output = torch.randn(batch_size, hidden_dim)

        # Initialize and run forward pass
        new_state, metrics = state_manager(
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Verify shapes
        assert new_state.shape == (batch_size, hidden_dim)
        assert 'state_value' in metrics
        assert 'energy_cost' in metrics

    def test_rl_optimization(self, state_manager, batch_size, hidden_dim):
        """Test reinforcement learning optimization."""
        consciousness_state = torch.randn(batch_size, hidden_dim)
        integrated_output = torch.randn(batch_size, hidden_dim)

        # Run multiple updates
        states = []
        values = []
        for _ in range(3):
            new_state, metrics = state_manager(
                consciousness_state,
                integrated_output,
                deterministic=True
            )
            states.append(new_state)
            values.append(metrics['state_value'])
            consciousness_state = new_state

        # Check state evolution
        states = torch.stack(states)
        values = torch.stack(values)

        # States should change over time
        assert not torch.allclose(states[0], states[-1], rtol=1e-5)

    def test_energy_efficiency(self, state_manager, batch_size, hidden_dim):
        """Test energy efficiency metrics."""
        # Test with different complexity inputs
        simple_state = torch.zeros(batch_size, hidden_dim)
        complex_state = torch.randn(batch_size, hidden_dim)
        integrated_output = torch.randn(batch_size, hidden_dim)

        # Compare energy costs
        _, metrics_simple = state_manager(
            simple_state,
            integrated_output,
            deterministic=True
        )
        _, metrics_complex = state_manager(
            complex_state,
            integrated_output,
            deterministic=True
        )

        # Complex states should require more energy
        assert metrics_complex['energy_cost'] > metrics_simple['energy_cost']

    def test_state_value_estimation(self, state_manager, batch_size, hidden_dim):
        """Test state value estimation."""
        consciousness_state = torch.randn(batch_size, hidden_dim)
        integrated_output = torch.randn(batch_size, hidden_dim)

        # Test value estimation consistency
        _, metrics1 = state_manager(
            consciousness_state,
            integrated_output,
            deterministic=True
        )
        _, metrics2 = state_manager(
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Same input should give same value estimate
        assert torch.allclose(metrics1['state_value'], metrics2['state_value'], rtol=1e-5)

    def test_adaptive_gating(self, state_manager, batch_size, hidden_dim):
        """Test adaptive gating mechanisms."""
        consciousness_state = torch.randn(batch_size, hidden_dim)

        # Test with different integrated outputs
        integrated_outputs = [torch.randn(batch_size, hidden_dim) for _ in range(3)]

        # Track gating behavior
        new_states = []
        for integrated_output in integrated_outputs:
            new_state, _ = state_manager(
                consciousness_state,
                integrated_output,
                deterministic=True
            )
            new_states.append(new_state)

        # Different inputs should lead to different states
        states = torch.stack(new_states)
        for i in range(len(states)-1):
            assert not torch.allclose(states[i], states[i+1], rtol=1e-5)

def test_state_manager_forward(state_manager):
    """Test forward pass of the state manager."""
    batch_size = 2
    hidden_dim = 64
    state = torch.randn(batch_size, hidden_dim)
    inputs = torch.randn(batch_size, hidden_dim)
    new_state, metrics = state_manager(state, inputs)
    assert new_state.shape == (batch_size, hidden_dim)
    assert 'memory_gate' in metrics
    assert 'energy_cost' in metrics
    assert 'state_value' in metrics

if __name__ == '__main__':
    pytest.main([__file__])
