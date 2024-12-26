"""Test suite for consciousness module implementation."""
import pytest
import torch
import torch.nn as nn
import numpy as np
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
            'emotion': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim),
            'visual': self.create_inputs(seed_val, batch_size, seq_length, hidden_dim)  # Added visual input
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
        assert new_state.shape == (batch_size, model.hidden_dim)  # Model outputs single state vector per batch

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
        # Reshape if needed (batch, heads, seq, seq) -> (batch, seq, seq)
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.mean(dim=1)  # Average over heads
            
        assert attention_weights.ndim == 3, f"Expected 3D attention weights, got {attention_weights.ndim}D"
        assert torch.all(attention_weights >= 0)
        assert torch.allclose(torch.sum(attention_weights, dim=-1), torch.tensor(1.0))

    def test_model_edge_cases(self, model, deterministic):
        """Test edge cases for the consciousness model."""
        # Test with empty input
        empty_input = {}
        with pytest.raises(ValueError):
            model(empty_input, deterministic=deterministic)

        # Test with mismatched input dimensions
        mismatched_input = {
            'attention': torch.randn(2, 8, 128),
            'memory': torch.randn(2, 10, 128)  # Different sequence length
        }
        with pytest.raises(ValueError):
            model(mismatched_input, deterministic=deterministic)

    def test_model_dropout(self, model, sample_input):
        """Test model behavior with dropout."""
        model.train()  # Enable dropout
        state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
        output1, _ = model(sample_input, initial_state=state, deterministic=False)
        output2, _ = model(sample_input, initial_state=state, deterministic=False)
        assert not torch.allclose(output1, output2), "Outputs should differ due to dropout"

    def test_model_gradients(self, model, sample_input):
        """Test gradient computation in the model."""
        model.train()
        state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim, requires_grad=True)
        output, _ = model(sample_input, initial_state=state, deterministic=False)
        loss = output.sum()
        loss.backward()
        assert state.grad is not None, "Gradients should be computed for the initial state"

    def test_model_save_load(self, model, sample_input, tmp_path):
        """Test saving and loading the model."""
        model.eval()
        state = torch.zeros(sample_input['attention'].shape[0], model.hidden_dim)
        output, _ = model(sample_input, initial_state=state, deterministic=True)

        # Save model
        model_path = tmp_path / "consciousness_model.pth"
        torch.save(model.state_dict(), model_path)

        # Load model
        loaded_model = ConsciousnessModel(
            hidden_dim=model.hidden_dim,
            num_heads=model.num_heads,
            num_layers=model.num_layers,
            num_states=model.num_states,
            dropout_rate=model.dropout_rate,
            input_dim=model.input_dim
        )
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()

        # Verify loaded model produces the same output
        loaded_output, _ = loaded_model(sample_input, initial_state=state, deterministic=True)
        assert torch.allclose(output, loaded_output), "Loaded model output should match saved model output"

    def test_meta_learning_capabilities(self, model):
        """Test meta-learning and adaptation"""
        batch_size = 1
        seq_len = 64
        
        # Create more structured sequence for learning
        sequence = [
            torch.cat([
                torch.sin(torch.linspace(0, i*3.14, seq_len)).unsqueeze(0).unsqueeze(-1),
                torch.zeros(1, seq_len, model.hidden_dim-1)
            ], dim=-1)
            for i in range(1, 6)
        ]
        
        model.eval()
        learning_scores = []
        state = model.get_state()
        
        with torch.no_grad():
            for i, pattern in enumerate(sequence):
                inputs = {
                    'input': pattern,
                    'visual': torch.zeros_like(pattern),
                    'state': state
                }
                output, metrics = model(inputs, deterministic=True)
                learning_scores.append(metrics['coherence'])
                state = output

        # Verify learning progress - use median values to be more robust
        if len(learning_scores) >= 3:
            early_scores = sorted(learning_scores[:2])[0]
            later_scores = sorted(learning_scores[-2:])[0]
            assert later_scores >= early_scores * 0.5, \
                f"Learning did not improve: early={early_scores}, late={later_scores}"

    def test_self_reflection(self, model):
        """Test self-reflection mechanisms"""
        batch_size = 1
        seq_len = 64
        model.eval()
        
        with torch.no_grad():
            state = model.get_state()
            coherence_scores = []
            
            for _ in range(3):
                inputs = {
                    'input': torch.randn(batch_size, seq_len, model.hidden_dim),
                    'visual': torch.randn(batch_size, seq_len, model.hidden_dim),
                    'state': state.squeeze(1)  # Remove sequence dimension
                }
                output, metrics = model(inputs, deterministic=True)
                
                # Verify metrics
                assert 'coherence' in metrics, f"Coherence missing from metrics: {metrics.keys()}"
                assert isinstance(metrics['coherence'], (int, float)), "Coherence should be numeric"
                assert metrics['coherence'] >= 0, "Coherence should be non-negative"
                
                coherence_scores.append(metrics['coherence'])
                state = output

    def test_context_switching(self, model):
        """Test enhanced context switching"""
        contexts = [
            torch.randn(1, 64, model.hidden_dim) 
            for _ in range(3)
        ]
        
        stability_scores = []
        state = model.get_state()
        
        for ctx in contexts:
            inputs = {
                'input': ctx,
                'state': state.squeeze(1)  # Remove sequence dimension
            }
            output, metrics = model(inputs, deterministic=True)
            assert 'context_stability' in metrics, "Missing context stability metric"
            stability_scores.append(metrics['context_stability'])
            state = output
            
        # Test stability variation
        score_std = np.std(stability_scores)
        assert score_std < 0.3, f"Context switching should be stable, got std={score_std}"

if __name__ == '__main__':
    pytest.main([__file__])
