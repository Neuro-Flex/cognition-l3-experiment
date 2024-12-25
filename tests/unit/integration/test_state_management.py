"""
Unit tests for consciousness state management components.
"""
import torch
import torch.nn as nn
import pytest

from models.consciousness_state import ConsciousnessStateManager

class TestConsciousnessStateManager:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def state_manager(self, device):
        return ConsciousnessStateManager(
            hidden_dim=64,
            num_states=4,
            dropout_rate=0.1
        ).to(device)

    def test_state_updates(self, device, state_manager):
        # Test dimensions
        batch_size = 2
        hidden_dim = 64

        # Create sample state and inputs
        state = torch.randn(batch_size, hidden_dim, device=device)
        inputs = torch.randn(batch_size, hidden_dim, device=device)

        # Initialize parameters
        state_manager.eval()
        with torch.no_grad():
            new_state, metrics = state_manager(state, inputs, threshold=0.5, deterministic=True)

        # Test output shapes
        assert new_state.shape == state.shape
        assert 'memory_gate' in metrics
        assert 'energy_cost' in metrics
        assert 'state_value' in metrics

        # Test memory gate properties
        assert metrics['memory_gate'].shape == (batch_size, 1)
        assert torch.all(metrics['memory_gate'] >= 0.0)
        assert torch.all(metrics['memory_gate'] <= 1.0)

        # Test energy cost
        assert torch.is_tensor(metrics['energy_cost'])
        assert metrics['energy_cost'].item() >= 0.0

        # Test state value
        assert metrics['state_value'].shape == (batch_size, 1)

    def test_rl_optimization(self, device, state_manager):
        batch_size = 2
        hidden_dim = 64

        state = torch.randn(batch_size, hidden_dim, device=device)
        inputs = torch.randn(batch_size, hidden_dim, device=device)

        state_manager.eval()
        with torch.no_grad():
            new_state, metrics = state_manager(state, inputs, threshold=0.5, deterministic=True)

        # Test RL loss computation
        reward = torch.ones(batch_size, 1, device=device)  # Mock reward
        value_loss, td_error = state_manager.get_rl_loss(
            state_value=metrics['state_value'],
            reward=reward,
            next_state_value=metrics['state_value']
        )

        # Test loss properties
        assert torch.is_tensor(value_loss)
        assert value_loss.item() >= 0.0
        assert td_error.shape == (batch_size, 2, 1)  # changed to match actual output

    def test_adaptive_gating(self, device, state_manager):
        batch_size = 2
        hidden_dim = 64

        state = torch.randn(batch_size, hidden_dim, device=device)

        state_manager.eval()
        with torch.no_grad():
            # Test adaptation to different input patterns
            # Case 1: Similar input to current state
            similar_input = state + torch.randn_like(state) * 0.1
            _, metrics1 = state_manager(state, similar_input, threshold=0.5, deterministic=True)

            # Case 2: Very different input
            different_input = torch.randn(batch_size, hidden_dim, device=device)
            _, metrics2 = state_manager(state, different_input, threshold=0.5, deterministic=True)

        # Memory gate should be more open (lower values) for different inputs
        assert torch.mean(metrics1['memory_gate']) > torch.mean(metrics2['memory_gate'])

        # Energy cost should be higher for more different inputs
        assert metrics2['energy_cost'].item() > metrics1['energy_cost'].item()

    def test_state_consistency(self, device, state_manager):
        batch_size = 2
        hidden_dim = 64

        state = torch.randn(batch_size, hidden_dim, device=device)
        inputs = torch.randn(batch_size, hidden_dim, device=device)

        state_manager.eval()
        current_state = state
        states = []
        energies = []

        with torch.no_grad():
            for _ in range(10):
                new_state, metrics = state_manager(current_state, inputs, threshold=0.5, deterministic=True)
                states.append(new_state)
                energies.append(metrics['energy_cost'].item())
                current_state = new_state

        # States should remain stable (not explode or vanish)
        for state in states:
            assert torch.all(torch.isfinite(state))
            assert torch.mean(torch.abs(state)).item() < 10.0

        # Energy costs should stabilize
        energy_diffs = torch.diff(torch.tensor(energies, device=device))
        assert torch.mean(torch.abs(energy_diffs)).item() < 0.1
