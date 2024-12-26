import torch
import torch.nn as nn
from typing import Optional, Tuple

class WorkingMemory(nn.Module):
    """Working memory component for maintaining and updating information"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Memory cells
        self.memory_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Process through LSTM
        output, (h_n, c_n) = self.memory_rnn(inputs, prev_state)
        
        # Apply update gate
        if prev_state is not None:
            prev_h = prev_state[0]
            gate = self.update_gate(torch.cat([output, prev_h[-1:]], dim=-1))
            output = gate * output + (1 - gate) * prev_h[-1:]
            
        # Project output
        output = self.output_projection(output)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output, (h_n, c_n)
