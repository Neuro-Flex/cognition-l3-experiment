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
        input_dim = 32

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
            consciousness_state, attention_maps = integration_module(inputs)

        # Test output shapes
        assert consciousness_state.shape == (batch_size, seq_length, 64)

        # Test attention maps
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    map_key = f"{target}-{source}"
                    assert map_key in attention_maps
                    attention_map = attention_maps[map_key]
                    # Check attention map properties
                    assert attention_map.shape[-2:] == (seq_length, seq_length)
                    # Verify attention weights sum to 1
                    assert torch.allclose(
                        attention_map.sum(dim=-1),
                        torch.ones((batch_size, 4, seq_length), device=device)
                    )

    def test_modality_specific_processing(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        # Test with single modality
        single_input = {
            'visual': torch.randn(batch_size, seq_length, input_dim, device=device)
        }
        input_shape = (64,)
        integration_module.eval()
        with torch.no_grad():
            consciousness_state1, _ = integration_module(single_input)

        # Test with multiple modalities
        multi_input = {
            'visual': single_input['visual'],
            'textual': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        with torch.no_grad():
            consciousness_state2, _ = integration_module(multi_input)

        # Multi-modal processing should produce different results
        assert not torch.allclose(consciousness_state1, consciousness_state2)

    def test_integration_stability(self, device, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        inputs = {
            'modality1': torch.randn(batch_size, seq_length, input_dim, device=device),
            'modality2': torch.randn(batch_size, seq_length, input_dim, device=device)
        }

        integration_module.eval()
        states = []
        with torch.no_grad():
            for _ in range(5):
                state, _ = integration_module(inputs)
                states.append(state)

        # All forward passes should produce identical results
        for i in range(1, len(states)):
            assert torch.allclose(states[0], states[i])

        # Test with dropout
        integration_module.train()
        states_dropout = []
        for i in range(5):
            state, _ = integration_module(inputs)
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
        input_dim = 32

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
            consciousness_state, attention_maps = integration_module(inputs)

        # Adjust assertions as needed
        assert consciousness_state.shape == (batch_size, seq_length, 64)
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    map_key = f"{target}-{source}"
                    assert map_key in attention_maps
                    attention_map = attention_maps[map_key]
                    assert attention_map.shape[-2:] == (seq_length, seq_length)
                    assert torch.allclose(
                        attention_map.sum(dim=-1),
                        torch.ones((batch_size, 4, seq_length), device=device)
                    )
