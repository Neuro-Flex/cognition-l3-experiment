class ConsciousnessAttention(nn.Module):
    def forward(self, x, mask=None):
        # Input validation
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("Empty input tensor")
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
            
        # ...existing code...

class GlobalWorkspace(nn.Module):
    def forward(self, inputs):
        # Input validation
        if inputs.size(0) == 0 or inputs.size(1) == 0:
            raise ValueError("Empty input tensor")
        if torch.isnan(inputs).any():
            raise ValueError("Input contains NaN values")
            
        # ...existing code...
