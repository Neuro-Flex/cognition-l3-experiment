"""
Unit tests for cognitive process integration components.
"""
import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

from models.consciousness_state import CognitiveProcessIntegration

class TestCognitiveProcessIntegration:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def integration_module(self, device):
        return CognitiveProcessIntegration(
            hidden_dim=64,
            num_heads=4
        ).to(device)

    def test_cross_modal_attention(self, device, integration_module):
        # Test dimensions
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        # Create multi-modal inputs
        inputs = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'textual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'numerical': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        # Initialize parameters
        input_shape = (64,)
        integration_module.eval()
        with torch.no_grad():
            consciousness_state, attention_maps = integration_module(inputs, deterministic=True)

        # Test output shapes
        assert consciousness_state.shape == (batch_size, seq_length, 64)

        # Test attention maps
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    map_key = f"{target}-{source}"
                    assert map_key in attention_maps, f"Missing attention map for {map_key}"
                    attention_map = attention_maps[map_key]
                    # Check attention map properties
                    assert attention_map.shape[-2:] == (seq_length, seq_length)
                    # Verify attention weights sum to 1
                    assert torch.allclose(
                        attention_map.sum(dim=-1),
                        torch.ones((batch_size, seq_length), device=device)
                    )

    def test_modality_specific_processing(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        # Test with single modality
        single_input = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device)
        }
        input_shape = (64,)
        integration_module.eval()
        with torch.no_grad():
            consciousness_state1, _ = integration_module(single_input, deterministic=True)

        # Test with multiple modalities
        multi_input = {
            'visual': single_input['visual'],
            'textual': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        with torch.no_grad():
            consciousness_state2, _ = integration_module(multi_input, deterministic=True)

        # Multi-modal processing should produce different results
        assert not torch.allclose(consciousness_state1, consciousness_state2), "Consciousness states should differ for single vs multiple modalities"

    def test_integration_stability(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        inputs = {
            'modality1': torch.randn(batch_size, seq_length, input_dim, device=device),
            'modality2': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        integration_module.eval()
        states = []
        with torch.no_grad():
            for _ in range(5):
                state, _ = integration_module(inputs, deterministic=True)
                states.append(state)

        # All forward passes should produce identical results
        for i in range(1, len(states)):
            assert torch.allclose(states[0], states[i])

        # Test with dropout
        integration_module.train()
        states_dropout = []
        for i in range(5):
            state, _ = integration_module(inputs, deterministic=False)
            states_dropout.append(state)

        # Dropout should produce different results
        assert not all(
            torch.allclose(states_dropout[0], state)
            for state in states_dropout[1:]
        )

    def test_cognitive_integration(self, device, integration_module):
        # Test dimensions
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        # Create multi-modal inputs
        inputs = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'textual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'numerical': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        # Initialize parameters
        input_shape = (64,)
        integration_module.eval()
        with torch.no_grad():
            consciousness_state, attention_maps = integration_module(inputs, deterministic=True)

        # Adjust assertions as needed
        assert consciousness_state.shape == (batch_size, seq_length, 64)
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    map_key = f"{target}-{source}"
                    assert map_key in attention_maps, f"Missing attention map for {map_key}"
                    attention_map = attention_maps[map_key]
                    assert attention_map.shape[-2:] == (seq_length, seq_length)
                    assert torch.allclose(
                        attention_map.sum(dim=-1),
                        torch.ones((batch_size, seq_length), device=device)
                    )

    def test_edge_cases(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        # Test with empty input
        empty_input = {}
        with pytest.raises(ValueError):
            integration_module(empty_input, deterministic=True)

        # Test with mismatched input dimensions
        mismatched_input = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'textual': torch.randn(batch_size, seq_length, input_dim // 2, device=device)  # Different input dimension
        }
        with pytest.raises(ValueError):
            integration_module(mismatched_input, deterministic=True)

    def test_dropout_behavior(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 64  # Updated input_dim to match the expected input shape

        inputs = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device),
            'textual': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        # Test with dropout enabled
        integration_module.train()
        state1, _ = integration_module(inputs, deterministic=False)
        state2, _ = integration_module(inputs, deterministic=False)

        # Outputs should be different due to dropout
        assert not torch.allclose(state1, state2)

        # Test with dropout disabled
        integration_module.eval()
        with torch.no_grad():
            state3, _ = integration_module(inputs, deterministic=True)
            state4, _ = integration_module(inputs, deterministic=True)

        # Outputs should be identical with dropout disabled
        assert torch.allclose(state3, state4)
