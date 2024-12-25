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
        
        # Input projection layer
        self.input_projection = nn.Linear(32, hidden_dim)  # Assuming input_dim=32
        
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
        processed_modalities = {}
        for modality, x in inputs.items():
            # Project input to hidden dimension first
            x = self.input_projection(x)
            x = self.layer_norm(x)  # Now x has shape [batch, seq_len, hidden_dim]
            if not deterministic:
                x = self.dropout(x)
            processed_modalities[modality] = x

        # Cross-modal attention integration
        integrated_features = []
        attention_maps = {}

        for target_modality, target_features in processed_modalities.items():
            cross_modal_contexts = []
            for source_modality, source_features in processed_modalities.items():
                if source_modality != target_modality:
                    # Reuse self.attention and pass 3D tensors directly
                    attended, weights = self.attention(
                        query=target_features,
                        key=source_features,
                        value=source_features,
                        need_weights=True,
                        average_attn_weights=False
                    )
                    cross_modal_contexts.append(attended)
                    attention_maps[f"{target_modality}-{source_modality}"] = weights

            # Ensure tensor shapes match before combining
            if cross_modal_contexts:
                combined = torch.mean(torch.stack(cross_modal_contexts), dim=0)
                combined = self.cross_modal_projection(combined)
                integrated = target_features + combined
            else:
                integrated = target_features

            integrated_features.append(integrated)
        # Final integration across all modalities
        final_state = torch.mean(torch.stack(integrated_features), dim=0)
        return final_state, attention_maps

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
