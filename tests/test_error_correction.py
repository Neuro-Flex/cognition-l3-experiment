import unittest
import torch
import torch.nn as nn
from models.consciousness_model import ConsciousnessModel
from models.error_handling import ErrorHandler, ErrorCorrection

class TestErrorCorrection(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64
        self.error_correction = ErrorCorrection(hidden_dim=self.hidden_dim)
        self.error_handler = ErrorHandler(logger=None)
        self.model = ConsciousnessModel(
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_layers=2,
            num_states=3
        )

    def test_error_correction_shape(self):
        """Test if error correction maintains correct tensor shape"""
        batch_size = 8
        input_state = torch.randn(batch_size, self.hidden_dim)
        corrected_state, error_prob = self.error_correction(input_state)
        
        self.assertEqual(corrected_state.shape, input_state.shape)
        self.assertTrue(isinstance(error_prob, float))
        self.assertTrue(0 <= error_prob <= 1)

    def test_error_detection(self):
        """Test if error detection works for invalid states"""
        # Test with valid state
        valid_state = torch.randn(4, self.hidden_dim)
        valid_state = torch.nn.functional.normalize(valid_state, dim=-1)
        _, valid_error = self.error_correction(valid_state)
        
        # Test with invalid state (NaN values)
        invalid_state = torch.full((4, self.hidden_dim), float('nan'))
        _, invalid_error = self.error_correction(invalid_state)
        
        self.assertLess(valid_error, invalid_error)

    def test_error_correction_recovery(self):
        """Test if error correction can recover from corrupted states"""
        # Create original state
        original_state = torch.randn(4, self.hidden_dim)
        original_state = torch.nn.functional.normalize(original_state, dim=-1)
        
        # Create corrupted state with some NaN values
        corrupted_state = original_state.clone()
        corrupted_state[:, :10] = float('nan')
        
        # Apply error correction
        corrected_state, error_prob = self.error_correction(corrupted_state)
        
        # Check if NaN values were fixed
        self.assertFalse(torch.isnan(corrected_state).any())
        self.assertTrue(error_prob > 0.5)  # Should detect high error probability

    def test_error_handling_integration(self):
        """Test integration of error correction with error handling"""
        batch_size = 4
        seq_len = 3
        
        # Create input with some invalid values
        inputs = {
            'visual': torch.randn(batch_size, seq_len, self.hidden_dim),
            'textual': torch.randn(batch_size, seq_len, self.hidden_dim)
        }
        inputs['visual'][0, 0] = float('nan')  # Introduce error
        
        # Process through model
        try:
            state, metrics = self.model(inputs)
            self.assertTrue('error_prob' in metrics)
            self.assertFalse(torch.isnan(state).any())
        except Exception as e:
            self.fail(f"Error correction should handle NaN values: {str(e)}")

    def test_error_correction_consistency(self):
        """Test if error correction is consistent across multiple runs"""
        input_state = torch.randn(4, self.hidden_dim)
        
        # Run multiple corrections
        results = []
        for _ in range(5):
            corrected, prob = self.error_correction(input_state)
            results.append((corrected.clone(), prob))
        
        # Check consistency
        for i in range(1, len(results)):
            torch.testing.assert_close(results[0][0], results[i][0])
            self.assertAlmostEqual(results[0][1], results[i][1])

    def test_error_correction_gradients(self):
        """Test if error correction maintains gradient flow"""
        input_state = torch.randn(4, self.hidden_dim, requires_grad=True)
        corrected_state, _ = self.error_correction(input_state)
        
        # Check if gradients can flow
        loss = corrected_state.sum()
        loss.backward()
        
        self.assertIsNotNone(input_state.grad)
        self.assertFalse(torch.isnan(input_state.grad).any())

    def test_error_correction_bounds(self):
        """Test if error correction maintains value bounds"""
        # Test with extreme values
        extreme_state = torch.randn(4, self.hidden_dim) * 1000
        corrected_state, _ = self.error_correction(extreme_state)
        
        # Check if values are normalized
        self.assertTrue(torch.all(corrected_state <= 1))
        self.assertTrue(torch.all(corrected_state >= -1))

    def test_error_logging(self):
        """Test if errors are properly logged"""
        # Create invalid state
        invalid_state = torch.full((4, self.hidden_dim), float('nan'))
        
        # Process with error handler
        self.error_handler.log_error(
            "state_error",
            "Invalid state detected",
            {"state": invalid_state}
        )
        
        # Check error history
        self.assertTrue(len(self.error_handler.error_history) > 0)
        latest_error = self.error_handler.error_history[-1]
        self.assertEqual(latest_error['type'], "state_error")

    def test_error_correction_with_noise(self):
        """Test error correction with different noise levels"""
        base_state = torch.randn(4, self.hidden_dim)
        noise_levels = [0.1, 0.5, 1.0]
        
        for noise in noise_levels:
            noisy_state = base_state + torch.randn_like(base_state) * noise
            corrected_state, error_prob = self.error_correction(noisy_state)
            
            # Higher noise should lead to higher error probability
            self.assertTrue(
                error_prob >= noise * 0.1,
                f"Error probability too low for noise level {noise}"
            )

if __name__ == '__main__':
    unittest.main()
