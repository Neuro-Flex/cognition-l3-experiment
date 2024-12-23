"""
Unit tests for consciousness attention mechanisms.
"""
import torch
import torch.nn as nn
import pytest

from models.attention import ConsciousnessAttention, GlobalWorkspace

torch.manual_seed(0)

class TestConsciousnessAttention:
    @pytest.fixture
    def attention_module(self):
        return ConsciousnessAttention(
            num_heads=4,
            head_dim=32,
            dropout_rate=0.1
        )

    def test_scaled_dot_product_attention(self, attention_module):
        # Test input shapes
        batch_size = 2
        seq_length = 8
        input_dim = 128

        # Create sample inputs
        inputs_q = torch.randn(batch_size, seq_length, input_dim)
        inputs_kv = torch.randn(batch_size, seq_length, input_dim)

        # Initialize parameters
        attention_module.eval()  # Set to evaluation mode

        # Apply attention
        with torch.no_grad():
            output, attention_weights = attention_module(inputs_q, inputs_kv)

        # Test output shapes
        assert output.shape == (batch_size, seq_length, input_dim)
        assert attention_weights.shape == (batch_size, 4, seq_length, seq_length)

        # Test attention weight properties
        # Weights should sum to 1 along the key dimension
        weight_sums = torch.sum(attention_weights, dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums))

        # Test masking
        mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        mask[:, -1] = False  # Mask out last position

        with torch.no_grad():
            output_masked, attention_weights_masked = attention_module(inputs_q, inputs_kv, mask=mask)

        # Verify masked positions have zero attention
        assert torch.allclose(attention_weights_masked[..., -1], torch.zeros_like(attention_weights_masked[..., -1]))

    def test_attention_dropout(self, attention_module):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs_q = torch.randn(batch_size, seq_length, input_dim)
        inputs_kv = torch.randn(batch_size, seq_length, input_dim)

        attention_module.train()  # Set to training mode

        # Test with dropout enabled (training mode)
        output1, _ = attention_module(inputs_q, inputs_kv)

        output2, _ = attention_module(inputs_q, inputs_kv)

        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)

        attention_module.eval()  # Set to evaluation mode

        # Test with dropout disabled (inference mode)
        with torch.no_grad():
            output3, _ = attention_module(inputs_q, inputs_kv)

            output4, _ = attention_module(inputs_q, inputs_kv)

        # Outputs should be identical with dropout disabled
        assert torch.allclose(output3, output4)

    def test_attention_output_shape(self, attention_module):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs_q = torch.randn(batch_size, seq_length, input_dim)
        inputs_kv = torch.randn(batch_size, seq_length, input_dim)

        attention_module.eval()  # Set to evaluation mode

        with torch.no_grad():
            output, _ = attention_module(inputs_q, inputs_kv)

        assert output.shape == inputs_q.shape  # Adjusted expected shape

class TestGlobalWorkspace:
    @pytest.fixture
    def workspace_module(self):
        return GlobalWorkspace(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            dropout_rate=0.1
        )

    def test_global_workspace_broadcasting(self, workspace_module):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs = torch.randn(batch_size, seq_length, input_dim)
        workspace_module.eval()  # Set to evaluation mode

        with torch.no_grad():
            output, attention_weights = workspace_module(inputs)

        # Test output shapes
        assert output.shape == inputs.shape
        assert attention_weights.shape == (batch_size, 4, seq_length, seq_length)

        # Test residual connection
        # Output should not be too different from input due to residual
        assert torch.mean(torch.abs(output - inputs)) < 1.2  # Adjust threshold
