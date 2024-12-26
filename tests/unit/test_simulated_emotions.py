import pytest
import torch
from models.simulated_emotions import SimulatedEmotions

class TestSimulatedEmotions:
    @pytest.fixture
    def emotion_model(self):
        return SimulatedEmotions(hidden_dim=128)
        
    @pytest.fixture
    def sample_state(self):
        return torch.randn(2, 128)  # batch_size=2, hidden_dim=128
        
    def test_emotion_generation(self, emotion_model, sample_state):
        emotions = emotion_model.generate_emotions(sample_state)
        assert emotions.shape == (2, 6)  # batch_size=2, num_emotions=6
        assert torch.allclose(emotions.sum(dim=-1), torch.ones(2))
        
    def test_emotion_regulation(self, emotion_model, sample_state):
        emotions = emotion_model.generate_emotions(sample_state)
        regulated = emotion_model.regulate_emotions(sample_state, emotions)
        assert regulated.shape == emotions.shape
        assert torch.all(regulated >= 0) and torch.all(regulated <= 1)
        
    def test_emotional_influence(self, emotion_model, sample_state):
        modified_state, metrics = emotion_model(sample_state)
        assert modified_state.shape == sample_state.shape
        assert 'emotions' in metrics
        assert 'emotion_intensities' in metrics
        assert 'emotional_influence' in metrics
        
    def test_emotion_decay(self, emotion_model, sample_state):
        # Test multiple steps of emotion decay
        initial_state, _ = emotion_model(sample_state)
        for _ in range(5):
            new_state, metrics = emotion_model(sample_state)
            assert torch.any(metrics['emotion_intensities'] != 0)  # Emotions should persist
            assert torch.all(metrics['emotion_intensities'] <= 1)  # Should not exceed 1
