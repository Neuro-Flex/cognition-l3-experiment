import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Tuple
import math
class GRUCell(nn.Module):
    """
    Gated Recurrent Unit cell with consciousness-aware gating mechanisms.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Combined gates - update input dimensions
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
        # Ensure inputs are float tensors and properly shaped
        x = torch.as_tensor(x, dtype=torch.float32)
        h = torch.as_tensor(h, dtype=torch.float32)
        
        # Ensure proper dimensions
        if x.dim() == 2:
            if x.size(1) != self.input_dim:
                raise ValueError(f"Expected input dim {self.input_dim}, got {x.size(1)}")

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

    def reset_parameters(self):
        """
        Initialize parameters of the GRU cell.
        """
        nn.init.kaiming_uniform_(self.update.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.reset.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.candidate.weight, a=math.sqrt(5))
        if self.update.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.update.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.update.bias, -bound, bound)
        if self.reset.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.reset.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.reset.bias, -bound, bound)
        if self.candidate.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.candidate.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.candidate.bias, -bound, bound)

class WorkingMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, prev_state=None):
        # Project input to hidden dimension
        x = self.dropout(self.input_projection(x))
        x = self.layer_norm(x)
        
        if prev_state is None:
            prev_state = torch.zeros((1, x.size(0), self.hidden_dim), device=x.device)
            
        output, new_state = self.gru(x.unsqueeze(0), prev_state)
        return output.squeeze(0), new_state

class InformationIntegration(nn.Module):
    """
    Implementation of Information Integration Theory (IIT) components.
    """
    def __init__(self, hidden_dim: int, num_modules: int, input_dim: int = None, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        
        # Update input projection
        self.input_projection = nn.Linear(self.input_dim, self.input_dim)  # Changed to maintain input dim
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,  # Changed to use input_dim
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, deterministic=True):
        # Project inputs if needed
        x = self.input_projection(inputs)
            
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply self-attention
        y, _ = self.multihead_attn(x, x, x)
        
        if not deterministic:
            y = self.dropout(y)
            
        # Add residual connection
        output = x + y

        # Prevent potential NaNs by clamping
        output = torch.clamp(output, min=-1e6, max=1e6)

        # Calculate integration metric (phi)
        phi = torch.mean(torch.abs(output), dim=(-2, -1))
        
        return output, phi
