import pytest
import torch
from models.reasoning import BasicReasoning

@pytest.fixture
def model():
    return BasicReasoning(hidden_dim=128)

@pytest.fixture
def sample_input():
    return torch.randn(2, 5, 128)  # [batch_size, seq_len, hidden_dim]

class TestBasicReasoning:
    def test_reasoning_scores(self, model, sample_input):
        output = model(sample_input)
        
        # Check all components exist
        assert 'logical_score' in output['metrics']
        assert 'pattern_score' in output['metrics']
        assert 'causal_score' in output['metrics']
        
        # Verify score ranges
        for key in ['logical_score', 'pattern_score', 'causal_score']:
            assert 0 <= output['metrics'][key] <= 1
        
        # Calculate overall score
        score = model.calculate_reasoning_score(output['metrics'])
        assert 0 <= score <= 100

    def test_confidence_estimation(self, model, sample_input):
        output = model(sample_input)
        assert 'confidence' in output
        assert output['confidence'].shape == (2, 5, 1)  # [batch_size, seq_len, 1]
        assert torch.all(output['confidence'] >= 0) and torch.all(output['confidence'] <= 1)
