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
            for source_modality, _ in processed_modalities.items():
                if source_modality != target_modality:
                    attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout_rate)
                    mask = None  # Define 'mask' variable
                    attended, _ = attention(processed_modalities[source_modality].unsqueeze(0), target_features.unsqueeze(0), target_features.unsqueeze(0), attn_mask=mask)
                    cross_modal_contexts.append(attended.squeeze(0))
                    attention_maps[f"{target_modality}-{source_modality}"] = attended

            # Ensure tensor shapes match before combining
            if cross_modal_contexts:
                combined = torch.mean(torch.stack(cross_modal_contexts), dim=0)
                combined = nn.Linear(combined.size(-1), target_features.size(-1))(combined)
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
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim or hidden_dim
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
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = True):
        # Project inputs if needed
        if self.input_projection is not None:
            processed_inputs = {k: self.input_projection(v) for k, v in inputs.items()}
        else:
            processed_inputs = inputs

        # Process modalities
        processed_modalities = {}
        for modality, x in processed_inputs.items():
            x = self.layer_norm(x)
            if not deterministic:
                x = nn.Dropout(p=self.dropout_rate)(x)
            processed_modalities[modality] = x

        # Cross-modal attention integration
        integrated_features = []
        attention_maps = {}

        for target_modality, target_features in processed_modalities.items():
            cross_modal_contexts = []
            for source_modality, _ in processed_modalities.items():
                if source_modality != target_modality:
                    attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout_rate)
                    mask = None  # Define 'mask' variable
                    attended, _ = attention(processed_modalities[source_modality].unsqueeze(0), target_features.unsqueeze(0), target_features.unsqueeze(0), attn_mask=mask)
                    cross_modal_contexts.append(attended.squeeze(0))
                    attention_maps[f"{target_modality}-{source_modality}"] = attended

            # Ensure tensor shapes match before combining
            if cross_modal_contexts:
                combined = torch.mean(torch.stack(cross_modal_contexts), dim=0)
                combined = nn.Linear(combined.size(-1), target_features.size(-1))(combined)
                integrated = target_features + combined
            else:
                integrated = target_features

            integrated_features.append(integrated)
        # Final integration across all modalities
        final_state = torch.mean(torch.stack(integrated_features), dim=0)
        return final_state, attention_maps

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
        reward = reward.unsqueeze(-1)
        td_target = reward + gamma * next_state_value
        td_error = td_target - state_value

        # Value loss (MSE)
        value_loss = torch.mean(td_error ** 2)
        return value_loss, td_error
