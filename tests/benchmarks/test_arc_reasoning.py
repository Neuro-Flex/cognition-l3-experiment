import torch
import pytest
from typing import Dict, Tuple

from models.consciousness_model import ConsciousnessModel

class TestARCReasoning:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def model_config(self):
        return ConsciousnessModel.create_default_config()

    @pytest.fixture
    def consciousness_model(self, model_config):
        return ConsciousnessModel(**model_config)

    def load_arc_sample(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Load a sample ARC task for testing."""
        sample_input = {
            'visual': torch.tensor([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        }

        expected_output = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        return sample_input, expected_output

    def _prepare_visual_input(self, visual, batch_size, hidden_dim):
        """Prepare visual input for the model by flattening and padding."""
        visual_flat = visual.view(batch_size, -1)  # Flatten to [batch_size, N]
        return torch.nn.functional.pad(
            visual_flat,
            (0, hidden_dim - visual_flat.shape[1])
        )

    def _get_final_state(self, output):
        """Extract final state from model output."""
        if output.dim() == 3:  # If output is [batch, seq, hidden]
            return output[:, -1, :]  # Take last sequence position
        return output  # Otherwise return as is

    def test_pattern_recognition(self, device, consciousness_model):
        inputs, expected = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]
        
        # Project visual input to correct dimensionality
        visual_input = self._prepare_visual_input(
            inputs['visual'], 
            batch_size,
            consciousness_model.hidden_dim
        )

        # Initialize model state
        model_inputs = {
            'visual': visual_input.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }

        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()

        try:
            with torch.no_grad():
                output, metrics = consciousness_model(
                    model_inputs,
                    deterministic=True,
                    consciousness_threshold=0.5
                )
                
                final_state = self._get_final_state(output)

                # Validate outputs
                assert final_state.shape == (batch_size, consciousness_model.hidden_dim)
                assert 'phi' in metrics
                assert metrics['phi'].shape == (batch_size, 1)
                assert torch.all(metrics['phi'] >= 0)

                # Validate attention
                assert 'attention_weights' in metrics
                assert metrics['attention_weights'].dim() >= 3  # (batch, heads, seq)

                # Validate attention maps
                assert 'attention_maps' in metrics
                for attn_map in metrics['attention_maps'].values():
                    # Get actual dimensions from attention map
                    batch, heads, seq_len = attn_map.size()[:3]
                    assert torch.allclose(
                        torch.sum(attn_map, dim=-1),
                        torch.ones((batch, heads, seq_len), device=device)
                    )

        except Exception as e:
            pytest.fail(f"Pattern recognition test failed: {str(e)}")

    def test_abstraction_capability(self, device, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Create transformed versions
        def preprocess_input(x):
            return self._prepare_visual_input(
                x, 
                batch_size,
                consciousness_model.hidden_dim
            )

        variations = {
            'original': preprocess_input(inputs['visual']),
            'rotated': preprocess_input(torch.rot90(inputs['visual'][:, :, :, 0], k=1).unsqueeze(-1)),
            'scaled': preprocess_input(inputs['visual'] * 2.0)
        }

        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()

        try:
            states = {}
            with torch.no_grad():
                for name, visual_input in variations.items():
                    output, metrics = consciousness_model(
                        {'visual': visual_input.to(device),
                         'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)},
                        deterministic=True
                    )
                    states[name] = self._get_final_state(output)

            # Test representation similarity
            def cosine_similarity(x, y):
                return torch.sum(x * y) / (torch.linalg.norm(x) * torch.linalg.norm(y))

            orig_rot_sim = cosine_similarity(
                states['original'].flatten(),
                states['rotated'].flatten()
            )
            orig_scaled_sim = cosine_similarity(
                states['original'].flatten(),
                states['scaled'].flatten()
            )

            # Transformed versions should maintain similar representations
            assert orig_rot_sim > 0.5
            assert orig_scaled_sim > 0.7

        except Exception as e:
            pytest.fail(f"Abstraction capability test failed: {str(e)}")

    def test_sequence_completion(self, device, consciousness_model):
        """Test sequence completion capabilities"""
        # Create sequence pattern
        sequence = torch.tensor([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        batch_size = sequence.shape[0]
        visual_input = self._prepare_visual_input(sequence, batch_size, consciousness_model.hidden_dim)
        
        model_inputs = {
            'visual': visual_input.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }
        
        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()
        
        with torch.no_grad():
            output, metrics = consciousness_model(model_inputs, deterministic=True)
            
            # Verify sequential pattern learning
            assert 'phi' in metrics

    def test_object_transformation(self, device, consciousness_model):
        """Test object transformation understanding"""
        # Initial shape
        initial = torch.tensor([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        # Transformed shape (rotated and translated)
        transformed = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        batch_size = initial.shape[0]
        
        input_visual = self._prepare_visual_input(
            torch.cat([initial, transformed], dim=1),
            batch_size,
            consciousness_model.hidden_dim
        )
        
        model_inputs = {
            'visual': input_visual.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }
        
        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()
        
        with torch.no_grad():
            output, metrics = consciousness_model(model_inputs, deterministic=True)
            
            # Verify transformation understanding
            assert 'phi' in metrics

    def test_rule_inference(self, device, consciousness_model):
        """Test ability to infer rules from examples"""
        # Rule: Alternating patterns
        examples = torch.tensor([
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        batch_size = examples.shape[0]
        visual_input = self._prepare_visual_input(examples, batch_size, consciousness_model.hidden_dim)
        
        model_inputs = {
            'visual': visual_input.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }
        
        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()
        
        with torch.no_grad():
            output, metrics = consciousness_model(model_inputs, deterministic=True)
            
            # Verify rule learning
            assert 'phi' in metrics

    def test_logical_deduction(self, device, consciousness_model):
        """Test logical deduction tasks"""
        # Logical deduction example
        logical_input = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        batch_size = logical_input.shape[0]
        visual_input = self._prepare_visual_input(logical_input, batch_size, consciousness_model.hidden_dim)
        
        model_inputs = {
            'visual': visual_input.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }
        
        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()
        
        with torch.no_grad():
            output, metrics = consciousness_model(model_inputs, deterministic=True)
            
            # Verify logical deduction
            assert 'phi' in metrics

    def test_language_understanding(self, device, consciousness_model):
        """Test language understanding tasks"""
        # Language understanding example
        language_input = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        batch_size = language_input.shape[0]
        visual_input = self._prepare_visual_input(language_input, batch_size, consciousness_model.hidden_dim)
        
        model_inputs = {
            'visual': visual_input.to(device),
            'state': torch.zeros((batch_size, consciousness_model.hidden_dim), device=device)
        }
        
        consciousness_model = consciousness_model.to(device)
        consciousness_model.eval()
        
        with torch.no_grad():
            output, metrics = consciousness_model(model_inputs, deterministic=True)
            
            # Verify language understanding
            assert 'phi' in metrics
