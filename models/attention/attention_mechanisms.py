class ConsciousnessAttention(nn.Module):
    def forward(self, query, key=None, value=None, mask=None):
        # Validate inputs
        if query.size(0) == 0 or query.size(1) == 0:
            raise ValueError("Empty input tensor")
        if torch.isnan(query).any():
            raise ValueError("Input contains NaN values")
            
        # ...existing code...

class GlobalWorkspace(nn.Module):
    def forward(self, x):
        # Validate input
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("Empty input tensor")
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
            
        # ...existing code...
