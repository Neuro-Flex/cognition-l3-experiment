import pytest
import torch
from models.consciousness import ConsciousnessModel
from models.simulated_emotions import SimulatedEmotions

@pytest.fixture
def model():
    return ConsciousnessModel(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_states=3,
        dropout_rate=0.1
    )

@pytest.fixture
def sample_input():
    batch_size = 2
    seq_len = 5
    hidden_dim = 128
    return {
        'attention': torch.randn(batch_size, seq_len, hidden_dim)
    }

class TestConsciousnessModel:
    def test_model_initialization(self, model):
        assert isinstance(model, ConsciousnessModel)
        assert hasattr(model, 'emotional_processor')
        assert isinstance(model.emotional_processor, SimulatedEmotions)
    
    def test_emotional_integration(self, model, sample_input):
        """Test if emotional processing is properly integrated."""
        output, metrics = model(sample_input)
        
        assert 'emotional_state' in metrics
        assert 'emotion_intensities' in metrics
        assert 'emotional_influence' in metrics
        
        # Check emotional influence on output
        assert metrics['emotional_influence'].shape == output.shape
        assert torch.any(metrics['emotional_influence'] != 0)
        
    def test_emotion_cognition_progress(self, model):
        """Test if emotional states affect cognition progress."""
        metrics = {
            'phi': 0.8,
            'coherence': 0.7,
            'emotion_intensities': torch.tensor([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        }
        
        progress = model.calculate_cognition_progress(metrics)
        assert 'emotional_coherence' in metrics
        assert 0 <= progress <= 100

    def test_forward_pass(self, model, sample_input):
        """Test basic forward pass with emotional processing"""
        output, metrics = model(sample_input)
        
        # Check output shape
        assert output.shape == (sample_input['attention'].size(0), model.hidden_dim)
        
        # Verify emotional metrics
        assert all(k in metrics for k in ['emotional_state', 'emotion_intensities'])
