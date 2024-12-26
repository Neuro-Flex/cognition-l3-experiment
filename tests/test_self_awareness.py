import pytest
import torch
import torch.nn as nn
from models.self_awareness import SelfAwareness
from models.consciousness import ConsciousnessModel

class TestSelfAwareness:
    @pytest.fixture
    def self_awareness(self):
        return SelfAwareness(hidden_dim=128, num_heads=4, dropout_rate=0.1)

    @pytest.fixture
    def sample_state(self):
        return torch.randn(2, 128)  # batch_size=2, hidden_dim=128

    def test_initialization(self, self_awareness):
        """Test proper initialization of SelfAwareness module"""
        assert isinstance(self_awareness, nn.Module)
        assert self_awareness.hidden_dim == 128
        assert len(self_awareness.state_history) == 0

    def test_state_history_update(self, self_awareness, sample_state):
        """Test state history maintenance"""
        self_awareness.update_state_history(sample_state)
        assert len(self_awareness.state_history) == 1
        
        # Test history size limit
        for _ in range(1000):
            self_awareness.update_state_history(sample_state)
        assert len(self_awareness.state_history) == self_awareness.history_size

    def test_self_representation(self, self_awareness, sample_state):
        """Test self-representation computation"""
        self_rep = self_awareness.compute_self_representation(sample_state)
        assert self_rep.shape == sample_state.shape
        assert torch.is_tensor(self_rep)

    def test_state_monitoring(self, self_awareness, sample_state):
        """Test state monitoring capabilities"""
        monitoring_results = self_awareness.monitor_state(sample_state)
        
        assert 'attended_state' in monitoring_results
        assert 'state_change' in monitoring_results
        assert 'anomaly_score' in monitoring_results
        
        assert monitoring_results['attended_state'].shape == sample_state.shape
        assert monitoring_results['anomaly_score'].shape == (sample_state.shape[0], 1)

    def test_metacognition(self, self_awareness, sample_state):
        """Test metacognitive assessment"""
        metacog_results = self_awareness.assess_metacognition(sample_state)
        
        assert 'confidence' in metacog_results
        assert 'error_prediction' in metacog_results
        assert 'adaptation_rate' in metacog_results
        
        assert metacog_results['confidence'].shape == (sample_state.shape[0], 1)
        assert 0 <= metacog_results['confidence'].min() <= 1
        assert 0 <= metacog_results['confidence'].max() <= 1

    def test_forward_pass(self, self_awareness, sample_state):
        """Test complete forward pass"""
        updated_state, metrics = self_awareness(sample_state)
        
        assert updated_state.shape == sample_state.shape
        assert 'self_representation' in metrics
        assert 'attended_state' in metrics
        assert 'confidence' in metrics

    def test_anomaly_detection(self, self_awareness):
        """Test anomaly detection with normal and anomalous inputs"""
        normal_state = torch.randn(2, 128)
        anomalous_state = torch.ones(2, 128) * 10  # Extreme values
        
        normal_results = self_awareness.monitor_state(normal_state)
        anomalous_results = self_awareness.monitor_state(anomalous_state)
        
        assert normal_results['anomaly_score'].mean() < anomalous_results['anomaly_score'].mean() + 0.01

    def test_confidence_calibration(self, self_awareness):
        """Test confidence calibration with different input qualities"""
        clear_state = torch.randn(2, 128)
        noisy_state = clear_state + torch.randn(2, 128) * 2
        
        clear_metacog = self_awareness.assess_metacognition(clear_state)
        noisy_metacog = self_awareness.assess_metacognition(noisy_state)
        
        assert clear_metacog['confidence'].mean() > noisy_metacog['confidence'].mean()

    def test_integration_with_consciousness(self):
        """Test integration with ConsciousnessModel"""
        consciousness = ConsciousnessModel(
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_states=3,
            dropout_rate=0.1
        )
        
        # Single input modality with proper shape
        inputs = {
            'attention': torch.randn(2, 5, 128)  # [batch_size, seq_len, hidden_dim]
        }
        
        # Run forward pass
        output, metrics = consciousness(inputs)
        print(f"metrics keys: {metrics.keys()}")
        print(f"retrieved_memory shape: {metrics.get('retrieved_memory', 'Not Found')}")
        
        # Check if 'retrieved_memory' is in metrics
        assert 'retrieved_memory' in metrics, "retrieved_memory not found in metrics"
        
        # Verify the shape of retrieved_memory
        retrieved_memory = metrics['retrieved_memory']
        assert retrieved_memory.shape == (2, 128), (
            f"retrieved_memory has shape {retrieved_memory.shape}, expected (2, 128)"
        )

    @pytest.mark.parametrize('batch_size', [1, 4, 8])
    def test_batch_processing(self, self_awareness, batch_size):
        """Test processing different batch sizes"""
        state = torch.randn(batch_size, 128)
        updated_state, metrics = self_awareness(state)
        
        assert updated_state.shape == (batch_size, 128)
        assert metrics['self_representation'].shape == (batch_size, 128)
        assert metrics['confidence'].shape == (batch_size, 1)

    def test_state_persistence(self, self_awareness):
        """Test persistence of internal state representations"""
        initial_state = torch.randn(2, 128)
        
        # Process same state multiple times
        states = []
        for _ in range(5):
            state, _ = self_awareness(initial_state)
            states.append(state)
        
        # Check for stability in representations
        states = torch.stack(states)
        variance = torch.var(states, dim=0).mean()
        assert variance < 1.0, "State representations should be relatively stable"

    def test_adaptation_over_time(self, self_awareness):
        """Test adaptation of internal representations over time"""
        sequence = [torch.randn(2, 128) for _ in range(10)]
        confidences = []
        
        for state in sequence:
            _, metrics = self_awareness(state)
            confidences.append(metrics['confidence'].mean().item())
            
        # Check for adaptation pattern
        confidence_changes = [b - a for a, b in zip(confidences[:-1], confidences[1:])]
        assert any(change != 0 for change in confidence_changes), "Model should show adaptation"

    @pytest.mark.parametrize('noise_level', [0.0, 0.5, 1.0])
    def test_noise_resilience(self, self_awareness, noise_level):
        """Test resilience to different noise levels"""
        base_state = torch.randn(2, 128)
        noisy_state = base_state + torch.randn_like(base_state) * noise_level
        
        _, base_metrics = self_awareness(base_state)
        _, noisy_metrics = self_awareness(noisy_state)
        
        confidence_ratio = noisy_metrics['confidence'].mean() / base_metrics['confidence'].mean()
        assert confidence_ratio <= 1.01, "Confidence should not increase with noise"

if __name__ == '__main__':
    pytest.main([__file__])
