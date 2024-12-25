import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch as F
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
        
        # Add modality combination layer
        self.modality_combination = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = True):
        """Process multiple modalities and generate cross-modal attention maps."""
        batch_size = next(iter(inputs.values())).size(0)
        seq_length = next(iter(inputs.values())).size(1)
        attention_maps = {}
        processed_states = {}
        
        # First pass: Project all inputs
        for modality, tensor in inputs.items():
            processed = self.input_projection(tensor)
            processed_states[modality] = processed

        # Initialize combined state with zeros
        combined_state = torch.zeros(
            batch_size, seq_length, self.hidden_dim,
            device=next(iter(inputs.values())).device
        )

        # Generate attention maps between all modality pairs
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    query = processed_states[target]
                    key = processed_states[source]
                    value = processed_states[source]
                    
                    attn_output, attn_weights = self.attention(
                        query=query,
                        key=key,
                        value=value
                    )
                    
                    # Store attention map
                    map_key = f"{target}-{source}"
                    attention_maps[map_key] = attn_weights
                    
                    # Add to combined state
                    combined_state = combined_state + attn_output

        # Final processing
        combined_state = self.modality_combination(combined_state)
        if not deterministic:
            combined_state = self.dropout(combined_state)
        combined_state = self.layer_norm(combined_state)
        
        return combined_state, attention_maps

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
        
        # Add similarity computation layer
        self.similarity_projection = nn.Linear(hidden_dim, hidden_dim)

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
        
        # Compute input similarity
        similarity = F.cosine_similarity(
            self.similarity_projection(state),
            self.similarity_projection(inputs),
            dim=-1
        ).unsqueeze(-1)
        
        # Modulate gate based on similarity
        memory_gate = torch.sigmoid(raw_gate) * similarity

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
