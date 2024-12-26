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
    seq_len = 1
    hidden_dim = 128
    return {
        'attention': torch.randn(batch_size, seq_len, hidden_dim),
        'perception': torch.randn(batch_size, seq_len, hidden_dim),
        'memory': torch.randn(batch_size, seq_len, hidden_dim)
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
        assert metrics['emotional_influence'].shape == output['broadcasted'].shape
    
    def test_memory_retrieval_shape(self, model, sample_input):
        """Test if memory retrieval produces correct shapes"""
        output, metrics = model(sample_input)
        
        # Check if retrieved_memory exists and has correct shape
        assert 'retrieved_memory' in metrics
        retrieved_memory = metrics['retrieved_memory']
        assert retrieved_memory.shape == (sample_input['attention'].size(0), model.hidden_dim)

    def test_goal_state_updates(self, model, sample_input):
        """Test if goal state updates properly"""
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
        assert output['broadcasted'].shape == (sample_input['attention'].size(0), model.hidden_dim)
        
        # Verify emotional metrics
        assert all(k in metrics for k in ['emotional_state', 'emotion_intensities'])

    def test_global_workspace_integration(self, model, sample_input):
        """Test if global workspace properly integrates information"""
        output, metrics = model(sample_input)
        
        # Check workspace metrics
        assert 'workspace_attention' in metrics
        assert 'competition_weights' in metrics
        
        # Verify shapes
        assert metrics['workspace_attention'].shape == (
            sample_input['attention'].size(0),
            3,  # num_modalities
            3   # seq_len (since each modality has seq_len=1, concatenated seq_len=3)
        )
        
        # Test competition mechanism
        competition_weights = metrics['competition_weights']
        assert torch.all(competition_weights >= 0)
        assert torch.allclose(competition_weights.sum(dim=-1), 
                            torch.ones_like(competition_weights.sum(dim=-1)))

    def test_information_broadcast(self, model, sample_input):
        """Test if information is properly broadcasted"""
        output, metrics = model(sample_input)
        
        # Output should be influenced by all modalities
        assert output['broadcasted'].shape == (sample_input['attention'].size(0), model.hidden_dim)
        
        # Test if output contains integrated information
        prev_output, _ = model(sample_input)
        assert not torch.allclose(output['broadcasted'], prev_output['broadcasted'], atol=1e-6)
