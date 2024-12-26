import pytest
import torch
import torch.nn as nn
from models.dynamic_attention import DynamicAttention
from models.consciousness import ConsciousnessModel

class TestDynamicAttention:
    @pytest.fixture
    def attention(self):
        return DynamicAttention(hidden_dim=128, num_heads=4, dropout_rate=0.1)

    @pytest.fixture
    def sample_input(self):
        batch_size = 2
        seq_len = 5
        hidden_dim = 128
        return {
            'query': torch.randn(batch_size, seq_len, hidden_dim),
            'key': torch.randn(batch_size, seq_len, hidden_dim),
            'value': torch.randn(batch_size, seq_len, hidden_dim),
            'goals': torch.randn(batch_size, hidden_dim),
            'context': torch.randn(batch_size, hidden_dim)
        }

    def test_initialization(self, attention):
        """Test proper initialization of DynamicAttention module"""
        assert isinstance(attention, nn.Module)
        assert attention.hidden_dim == 128
        assert attention.num_heads == 4
        assert isinstance(attention.attention_threshold, torch.Tensor)
        assert float(attention.attention_threshold) == pytest.approx(0.1, rel=1e-5)

    def test_forward_pass(self, attention, sample_input):
        """Test complete forward pass"""
        attended_value, metrics = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value'],
            sample_input['goals'],
            sample_input['context']
        )
        
        assert attended_value.shape == sample_input['query'].shape
        assert 'priority_weights' in metrics
        assert 'attention_weights' in metrics
        assert 'attention_threshold' in metrics
        
        # Check priority weights shape
        assert metrics['priority_weights'].shape == (sample_input['query'].size(0), attention.num_heads)

    def test_priority_computation(self, attention, sample_input):
        """Test priority weight computation"""
        priority_weights = attention.compute_priority_weights(
            sample_input['query'].mean(dim=1),
            sample_input['goals']
        )
        
        assert priority_weights.shape == (sample_input['query'].size(0), attention.num_heads)
        assert torch.allclose(priority_weights.sum(dim=-1), torch.ones(sample_input['query'].size(0)))

    def test_threshold_adaptation(self, attention, sample_input):
        """Test attention threshold adaptation"""
        initial_threshold = attention.attention_threshold.clone()
        attention.update_threshold(sample_input['context'])
        
        assert attention.attention_threshold != initial_threshold
        assert 0 <= float(attention.attention_threshold) <= 1

    def test_context_integration(self, attention, sample_input):
        """Test context integration in attention"""
        # Test with context
        output_with_context, metrics_with_context = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value'],
            context=sample_input['context']
        )
        
        # Test without context
        output_no_context, metrics_no_context = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value']
        )
        
        # Outputs should be different with and without context
        assert not torch.allclose(output_with_context, output_no_context)

    def test_attention_masking(self, attention, sample_input):
        """Test attention masking based on threshold"""
        _, metrics = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value']
        )
        
        attention_weights = metrics['attention_weights']
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)

    def test_goal_directed_attention(self, attention, sample_input):
        """Test goal-directed attention behavior"""
        # Test with different goals
        goal1 = torch.ones_like(sample_input['goals'])
        goal2 = -torch.ones_like(sample_input['goals'])
        
        _, metrics1 = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value'],
            goals=goal1
        )
        
        _, metrics2 = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value'],
            goals=goal2
        )
        
        # Different goals should produce different attention patterns
        assert not torch.allclose(
            metrics1['attention_weights'],
            metrics2['attention_weights']
        )

    def test_integration_with_consciousness(self, attention, sample_input):
        """Test integration with ConsciousnessModel"""
        model = ConsciousnessModel(
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_states=3,
            dropout_rate=0.1
        )

        inputs = {
            'sensory': torch.randn(2, 5, 128)  # [batch_size, seq_len, hidden_dim]
        }
        
        # Run forward pass
        output, _ = model(inputs)
        
        # Basic validation checks
        assert 'broadcasted' in output, "Output should contain broadcasted data"

    @pytest.mark.parametrize('batch_size', [1, 4, 8])
    def test_batch_processing(self, attention, batch_size):
        """Test processing different batch sizes"""
        inputs = {
            'query': torch.randn(batch_size, 5, 128),
            'key': torch.randn(batch_size, 5, 128),
            'value': torch.randn(batch_size, 5, 128)
        }
        
        output, metrics = attention(
            inputs['query'],
            inputs['key'],
            inputs['value']
        )
        
        assert output.shape == (batch_size, 5, 128)
        assert metrics['priority_weights'].shape == (batch_size, attention.num_heads)

    def test_attention_stability(self, attention, sample_input):
        """Test stability of attention patterns"""
        outputs = []
        for _ in range(5):
            output, _ = attention(
                sample_input['query'],
                sample_input['key'],
                sample_input['value']
            )
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        variance = torch.var(outputs, dim=0).mean()
        assert variance < 1.0, "Attention patterns should be relatively stable"

    @pytest.mark.parametrize('noise_level', [0.0, 0.5, 1.0])
    def test_noise_resilience(self, attention, sample_input, noise_level):
        """Test resilience to input noise"""
        noisy_query = sample_input['query'] + torch.randn_like(sample_input['query']) * noise_level
        
        clean_output, clean_metrics = attention(
            sample_input['query'],
            sample_input['key'],
            sample_input['value']
        )
        
        noisy_output, noisy_metrics = attention(
            noisy_query,
            sample_input['key'],
            sample_input['value']
        )
        
        # Check that attention maintains some stability under noise
        similarity = torch.cosine_similarity(
            clean_output.mean(dim=1),
            noisy_output.mean(dim=1),
            dim=-1
        ).mean()
        
        assert similarity > 0.5, f"Attention too sensitive to noise: similarity={similarity}"

if __name__ == '__main__':
    pytest.main([__file__])
