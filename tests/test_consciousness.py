"""Test suite for consciousness module implementation."""
import pytest
import torch
import torch.nn as nn
from models.consciousness_model import ConsciousnessModel
from tests.unit.test_base import ConsciousnessTestBase

class TestConsciousnessModel(ConsciousnessTestBase):
    """Test cases for the consciousness model."""

    @pytest.fixture
    def model(self, hidden_dim, num_heads):
        """Create a consciousness model for testing."""
        return ConsciousnessModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=4,
            num_states=4,
            dropout_rate=0.1,
            input_dim=hidden_dim  # Added input_dim argument
        )

    def create_inputs(self, seed_fixture, batch_size, seq_length, hidden_dim):
        """Create sample input tensors with proper seed handling."""
        seed = seed_fixture() if hasattr(seed_fixture, '__call__') else seed_fixture
        return super().create_inputs(seed, batch_size, seq_length, hidden_dim)
        
    @pytest.fixture
    def sample_input(self, seed, batch_size, seq_length, hidden_dim):
        """Create sample input data for testing."""
        seed_val = seed  # Don't call fixture directly
        inputs = {
            'attention': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim),
            'memory': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim),
            'reasoning': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim),
            'emotion': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim)
        }
        return inputs

    @pytest.fixture
    def deterministic(self):
        return True

    def test_model_initialization(self, model):
        """Test that consciousness model initializes correctly."""
        assert isinstance(model, ConsciousnessModel)
        assert model.hidden_dim == 128
        assert model.num_heads == 4
        assert model.num_layers == 4
        assert model.num_states == 4
        assert model.input_dim == 128  # Check for input_dim

    def test_model_forward_pass(self, model, sample_input, deterministic):
        """Test forward pass through consciousness model."""
        # Initialize model
        input_shape = (model.hidden_dim,)
        model.eval() if deterministic else model.train()

        # Run forward pass
        with torch.no_grad() if deterministic else torch.enable_grad():
            state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
            new_state, metrics = model(sample_input, initial_state=state, deterministic=deterministic)

        # Check output structure and shapes
        batch_size = next(iter(sample_input.values())).shape[0]
        assert new_state.shape == (batch_size, model.hidden_dim)

        # Verify metrics
        assert all(k in metrics for k in ['memory_state', 'attention_weights', 'phi', 'attention_maps'])

        # Validate attention weights
        self.assert_valid_attention(metrics['attention_weights'])

    def test_model_config(self, model):
        """Test model configuration methods."""
        config = model.get_config()
        assert config['hidden_dim'] == 128
        assert config['num_heads'] == 4
        assert config['num_layers'] == 4
        assert config['num_states'] == 4
        assert config['dropout_rate'] == 0.1

        default_config = ConsciousnessModel.create_default_config()
        assert isinstance(default_config, dict)
        assert all(k in default_config for k in [
            'hidden_dim', 'num_heads', 'num_layers', 'num_states', 'dropout_rate'
        ])

    def test_model_state_initialization(self, model, sample_input, deterministic):
        """Test initialization of the model state."""
        input_shape = (model.hidden_dim,)
        model.eval() if deterministic else model.train()
        with torch.no_grad() if deterministic else torch.enable_grad():
            state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
        assert state.shape == (sample_input['attention'].shape[0], model.hidden_dim)

    def test_model_state_update(self, model, sample_input, deterministic):
        """Test updating the model state."""
        input_shape = (model.hidden_dim,)
        model.eval() if deterministic else model.train()
        with torch.no_grad() if deterministic else torch.enable_grad():
            state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
            new_state, metrics = model(sample_input, initial_state=state, deterministic=deterministic)
        assert new_state is not None
        assert 'memory_state' in metrics

    def test_model_attention_weights(self, model, sample_input, deterministic):
        """Test attention weights in the model."""
        input_shape = (model.hidden_dim,)
        model.eval() if deterministic else model.train()
        with torch.no_grad() if deterministic else torch.enable_grad():
            state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
            _, metrics = model(sample_input, initial_state=state, deterministic=deterministic)
        attention_weights = metrics['attention_weights']
        assert attention_weights.ndim == 4  # (batch, heads, seq, seq)
        assert torch.all(attention_weights >= 0)
        assert torch.allclose(torch.sum(attention_weights, dim=-1), torch.tensor(1.0))

if __name__ == '__main__':
    pytest.main([__file__])
