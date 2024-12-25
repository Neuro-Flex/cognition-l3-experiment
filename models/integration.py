class InformationIntegration(nn.Module):
    def forward(self, inputs, deterministic=True):
        """Process inputs with enhanced validation."""
        # Input tensor validation
        if isinstance(inputs, torch.Tensor):
            if inputs.size(0) == 0 or inputs.size(1) == 0:
                raise ValueError("Empty input dimensions")
            if torch.isnan(inputs).any():
                raise ValueError("Input contains NaN values")
            if inputs.size(-1) != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {inputs.size(-1)}")

        # Process input after validation
        processed = self.input_projection(inputs)
        normed = self.layer_norm(processed)
        
        if not deterministic:
            normed = self.dropout(normed)

        # Calculate integration metric (phi)
        phi = torch.mean(torch.abs(normed), dim=(-2, -1))
        
        return normed, phi
