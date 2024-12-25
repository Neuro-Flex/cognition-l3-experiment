"""
Comprehensive tests for attention mechanisms in consciousness model.
"""
import pytest
import torch
import torch.nn.functional as F
from tests.unit.test_base import ConsciousnessTestBase
from models.attention import ConsciousnessAttention, GlobalWorkspace

class TestAttentionMechanisms(ConsciousnessTestBase):
    """Test suite for attention mechanisms."""

    @pytest.fixture
    def hidden_dim(self):
        return 128

    @pytest.fixture
    def num_heads(self):
        return 4

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_length(self):
        return 8

    @pytest.fixture
    def attention_module(self, hidden_dim, num_heads):
        """Create attention module for testing."""
        return ConsciousnessAttention(
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            dropout_rate=0.1
        )

    def test_scaled_dot_product(self, attention_module, batch_size, seq_length, hidden_dim):
        """Test scaled dot-product attention computation."""
        # Create inputs
        inputs_q = torch.randn(batch_size, seq_length, hidden_dim)
        inputs_kv = torch.randn(batch_size, seq_length, hidden_dim)

        # Initialize and run forward pass
        attention_module.eval()
        with torch.no_grad():
            output, attention_weights = attention_module(
                inputs_q, 
                inputs_kv
            )

        # Verify output shape
        self.assert_output_shape(output, (batch_size, seq_length, hidden_dim))

        # Verify attention weights
        self.assert_valid_attention(attention_weights)

    def test_attention_mask(self, attention_module, batch_size, seq_length, hidden_dim):
        """Test attention mask handling."""
        # Create inputs and mask
        inputs_q = torch.randn(batch_size, seq_length, hidden_dim)
        inputs_kv = torch.randn(batch_size, seq_length, hidden_dim)
        mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        mask[:, seq_length//2:] = False  # Mask out second half

        # Initialize and run forward pass
        attention_module.eval()
        with torch.no_grad():
            output, attention_weights = attention_module(
                inputs_q, 
                inputs_kv,
                mask=mask
            )

        # Verify masked attention weights are zero
        assert torch.allclose(attention_weights[..., seq_length//2:], torch.zeros_like(attention_weights[..., seq_length//2:]))

    def test_consciousness_broadcasting(self, attention_module, batch_size, seq_length, hidden_dim):
        """Test consciousness-aware broadcasting."""
        inputs_q = torch.randn(batch_size, seq_length, hidden_dim)
        inputs_kv = torch.randn(batch_size, seq_length, hidden_dim)

        # Test with and without dropout
        attention_module.eval()
        with torch.no_grad():
            output1, _ = attention_module(inputs_q, inputs_kv)
            output2, _ = attention_module(inputs_q, inputs_kv)

        # Outputs should be identical when deterministic
        assert torch.allclose(output1, output2, rtol=1e-5)

    def test_global_workspace_integration(self, batch_size, seq_length, hidden_dim, num_heads):
        """Test global workspace integration."""
        workspace = GlobalWorkspace(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            dropout_rate=0.1
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Initialize and run forward pass
        workspace.eval()
        with torch.no_grad():
            output, attention_weights = workspace(inputs, deterministic=True)

        # Verify shapes
        self.assert_output_shape(output, (batch_size, seq_length, hidden_dim))

        # Test residual connection
        # Output should be different from input due to processing
        assert not torch.allclose(output, inputs, rtol=1e-5)

    def test_attention_dropout(self, attention_module, batch_size, seq_length, hidden_dim):
        """Test attention dropout behavior."""
        inputs_q = torch.randn(batch_size, seq_length, hidden_dim)
        inputs_kv = torch.randn(batch_size, seq_length, hidden_dim)

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

    def test_attention_output_shape(self, attention_module, batch_size, seq_length, hidden_dim):
        """Test attention output shape."""
        inputs_q = torch.randn(batch_size, seq_length, hidden_dim)
        inputs_kv = torch.randn(batch_size, seq_length, hidden_dim)

        attention_module.eval()  # Set to evaluation mode

        with torch.no_grad():
            output, _ = attention_module(inputs_q, inputs_kv)

        assert output.shape == inputs_q.shape  # Adjusted expected shape
