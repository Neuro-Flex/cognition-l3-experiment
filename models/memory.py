import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Tuple

class GRUCell(nn.Module):
    """
    Gated Recurrent Unit cell with consciousness-aware gating mechanisms.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Combined gates
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        """
        Apply GRU cell with consciousness-aware updates.

        Args:
            x: Input tensor [batch_size, input_dim]
            h: Hidden state [batch_size, hidden_dim]
        """
        # Ensure inputs are float tensors
        x = torch.as_tensor(x, dtype=torch.float32)
        h = torch.as_tensor(h, dtype=torch.float32)

        # Concatenate inputs for efficiency
        inputs = torch.cat([x, h], dim=-1)

        # Create update gate
        z = torch.sigmoid(self.update(inputs))

        # Create reset gate
        r = torch.sigmoid(self.reset(inputs))

        # Create candidate activation
        h_reset = r * h
        h_concat = torch.cat([x, h_reset], dim=-1)
        h_tilde = torch.tanh(self.candidate(h_concat))

        # Final update
        h_new = torch.clamp((1.0 - z) * h + z * h_tilde, -1.0, 1.0)

        return h_new

class WorkingMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, prev_state=None):
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.layer_norm(x)
        
        if prev_state is None:
            prev_state = torch.zeros((1, x.size(0), self.hidden_dim), device=x.device)
            
        output, new_state = self.gru(x.unsqueeze(0), prev_state)
        return output.squeeze(0), new_state

class InformationIntegration(nn.Module):
    """
    Implementation of Information Integration Theory (IIT) components.
    """
    def __init__(self, hidden_dim: int, num_modules: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        self.dropout_rate = dropout_rate
        
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, deterministic=True):
        # Project inputs if needed
        if inputs.size(-1) != self.hidden_dim:
            inputs = self.input_projection(inputs)
            
        # Apply layer normalization
        x = self.layer_norm(inputs)
        
        # Apply self-attention
        y, _ = self.multihead_attn(x, x, x)
        
        if not deterministic:
            y = self.dropout(y)
            
        # Add residual connection
        output = x + y
        
        # Calculate integration metric (phi)
        phi = torch.mean(torch.abs(output), dim=(-2, -1))
        
        return output, phi
