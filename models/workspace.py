def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    # Check for empty input
    if inputs.size(0) == 0 or inputs.size(1) == 0 or inputs.size(2) == 0:
        raise ValueError("Input tensor has zero-sized dimension")

    if torch.isnan(inputs).any():
        raise ValueError("Input tensor contains NaN values")

    if inputs.dim() != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {inputs.shape}")

    # Rest of the workspace implementation
    # ...
