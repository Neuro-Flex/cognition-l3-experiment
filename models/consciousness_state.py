import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class CognitiveProcessIntegration(nn.Module):
    """
    Extended transformer architecture for multi-modal task management.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Updated input projection to handle correct dimensions
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)  # Changed input_dim to match hidden_dim
        
        # Add multihead attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Add layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.cross_modal_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = True):
        # Process input dictionary to create a single tensor
        x = next(iter(inputs.values()))  # Take first input as base
        
        # Apply input projection to each sequence in batch
        x = x.view(-1, x.size(-1))  # Reshape to (batch * seq, hidden_dim)
        x = self.input_projection(x)
        x = x.view(*inputs[next(iter(inputs))].shape[:-1], -1)  # Restore original shape
        
        # Rest of processing
        attention_maps = {}
        if not deterministic:
            x = self.dropout(x)
        x = self.layer_norm(x)
        
        return x, attention_maps

class ConsciousnessStateManager(nn.Module):
    """
    Manages consciousness state transitions with adaptive memory gates and RL optimization.
    """
    def __init__(self, hidden_dim: int, input_dim: Optional[int] = None, 
                 num_states: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        input_dim = input_dim or hidden_dim  # Ensure input_dim is set
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_states = num_states
        self.dropout_rate = dropout_rate

        # Core components
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.gate_network = nn.Linear(hidden_dim, hidden_dim)
        
        # State management
        self.state_embedding = nn.Parameter(torch.randn(num_states, hidden_dim))
        self.state_transition = nn.Linear(hidden_dim * 2, num_states)
        
        # Optional input projection
        self.input_projection = None
        if self.input_dim != hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, hidden_dim)

        # For RL "value"
        self.value_network = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, inputs: torch.Tensor,
                threshold: float = 0.5, deterministic: bool = True):
        # Layer norm
        state = self.layer_norm(state)
        inputs = self.layer_norm(inputs)
        if not deterministic:
            drop = nn.Dropout(self.dropout_rate)
            state = drop(state)
            inputs = drop(inputs)

        # Compute memory gate
        raw_gate = self.gate_network(state * inputs)
        memory_gate = torch.sigmoid(raw_gate)         # shape [batch_size, hidden_dim]

        # Apply threshold masking
        mask = (memory_gate >= threshold).float()
        memory_gate = memory_gate * mask

        # Update state
        new_state = state + memory_gate * inputs

        # Calculate metrics
        energy_cost = 1.0 - memory_gate.mean()  # single scalar
        state_value = self.value_network(new_state)   # shape [batch_size, 1]
        metrics = {
            'memory_gate': memory_gate,
            'energy_cost': energy_cost,
            'state_value': state_value
        }
        return new_state, metrics

    def get_rl_loss(self, state_value, reward, next_state_value, gamma=0.99):
        """
        Compute RL loss for optimizing state transitions.

        Args:
            state_value: Estimated value of current state
            reward: Immediate reward (e.g., task performance)
            next_state_value: Estimated value of next state
            gamma: Discount factor
        """
        # Ensure reward has the same shape as state_value
        td_target = reward + gamma * next_state_value
        td_error = td_target - state_value

        # Value loss (MSE)
        value_loss = torch.mean(td_error ** 2)
        return value_loss, td_error
