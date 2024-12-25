"""
Unit tests for Information Integration Theory (IIT) components.
"""
import torch
import torch.nn as nn
import pytest

from models.memory import InformationIntegration

class TestInformationIntegration:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def integration_module(self, device):
        return InformationIntegration(
            hidden_dim=64,
            num_modules=4,
            input_dim=32,  # Add input_dim parameter
            dropout_rate=0.1
        ).to(device)

    def test_phi_metric_computation(self, device, integration_module):
        # Test dimensions
        batch_size = 2
        num_modules = 4
        input_dim = 32  # Updated to match expected shapes

        # Create sample inputs
        inputs = torch.randn(batch_size, num_modules, input_dim, device=device)

        # Initialize parameters
        integration_module.eval()
        with torch.no_grad():
            output, phi = integration_module(inputs)

        # Test output shapes
        assert output.shape == inputs.shape
        assert phi.shape == (batch_size,)  # Phi should be a scalar per batch element

        # Test phi properties
        assert torch.all(torch.isfinite(phi))  # Phi should be finite
        assert torch.all(phi >= 0.0)  # Phi should be non-negative

        # Test with different input patterns
        # More structured input should lead to higher phi
        structured_input = torch.randn(batch_size, 1, input_dim, device=device).repeat(1, num_modules, 1)
        with torch.no_grad():
            _, phi_structured = integration_module(structured_input)

        random_input = torch.randn(batch_size, num_modules, input_dim, device=device)
        with torch.no_grad():
            _, phi_random = integration_module(random_input)

        # Structured input should have higher integration
        assert torch.all(phi_structured > phi_random)

    def test_information_flow(self, device, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32  # Updated to match expected shapes

        inputs = torch.zeros(batch_size, num_modules, input_dim, device=device)  # ensure shape matches the model

        # Test with and without dropout
        integration_module.train()
        output1, _ = integration_module(inputs, deterministic=False)

        integration_module.eval()
        with torch.no_grad():
            output2, _ = integration_module(inputs, deterministic=True)

        # Test residual connection properties
        # Output should maintain some similarity with input
        input_output_correlation = torch.mean(torch.abs(
            torch.corrcoef(inputs.view(-1, input_dim).T, output2.view(-1, input_dim).T)
        ))
        assert input_output_correlation > 0.1

        # Test module interactions
        # Compute cross-module correlations
        module_correlations = torch.corrcoef(output2.view(batch_size * num_modules, input_dim).T)

        # There should be some correlation between modules
        avg_cross_correlation = torch.mean(torch.abs(module_correlations))
        assert avg_cross_correlation > 0.1

    def test_entropy_calculations(self, device, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32  # Updated to match expected shapes

        # Test with different input distributions
        # Uniform distribution
        uniform_input = torch.ones(batch_size, num_modules, input_dim, device=device)
        integration_module.eval()
        with torch.no_grad():
            _, phi_uniform = integration_module(uniform_input)

        # Concentrated distribution
        concentrated_input = torch.zeros(batch_size, num_modules, input_dim, device=device)
        concentrated_input[:, :, 0] = 1.0
        with torch.no_grad():
            _, phi_concentrated = integration_module(concentrated_input)

        # Uniform distribution should have higher entropy
        assert torch.all(phi_uniform > phi_concentrated)

    def test_memory_integration(self, device, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32  # Updated to match expected shapes

        inputs = torch.randn(batch_size, num_modules, input_dim, device=device)

        # Process through integration
        integration_module.eval()
        with torch.no_grad():
            output, phi = integration_module(inputs)

        # Test output shapes
        assert output.shape == inputs.shape
        assert phi.shape == (batch_size,)  # Phi should be a scalar per batch element

        # Test phi properties
        assert torch.all(torch.isfinite(phi))  # Phi should be finite
        assert torch.all(phi >= 0.0)  # Phi should be non-negative

        # Test with different input patterns
        # More structured input should lead to higher phi
        structured_input = torch.randn(batch_size, 1, input_dim, device=device).repeat(1, num_modules, 1)
        with torch.no_grad():
            _, phi_structured = integration_module(structured_input)

        random_input = torch.randn(batch_size, num_modules, input_dim, device=device)
        with torch.no_grad():
            _, phi_random = integration_module(random_input)

        # Structured input should have higher integration
        assert torch.all(phi_structured > phi_random)
