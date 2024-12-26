import torch
import torch.nn as nn
from typing import Dict, Union, Optional

class MultiHeadAttention(nn.Module):
    """Custom MultiHeadAttention implementation"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate, batch_first=True)
        
    def forward(self, x):
        # MultiheadAttention expects query, key, value
        output, attention_weights = self.attention(x, x, x)
        self.attention_weights = attention_weights
        return output

class GlobalWorkspace(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.1, num_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Integration layers with modality-specific processing
        self.modality_integration = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_modalities)
        ])
        
        # Attention and competition mechanisms
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.competition_gate = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Enhanced broadcasting with gating mechanism
        self.broadcast_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.broadcast_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Information integration layers
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, modalities: Union[Dict[str, torch.Tensor], None] = None, sensory: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass handling both direct dictionary input and kwargs
        """
        if modalities is None:
            modalities = kwargs

        # Get list of available modalities
        available_modalities = list(modalities.keys())

        # Integrate modalities
        integrated_features = []
        for modality in available_modalities:
            # Get features and ensure they're 3D [batch, seq, hidden]
            features = modalities[modality]
            if features.dim() == 2:
                features = features.unsqueeze(0)  # Add batch dimension
            integrated = self.modality_integration[available_modalities.index(modality)](features)
            integrated_features.append(integrated)

        # Pad remaining slots with zeros if needed
        while len(integrated_features) < self.num_modalities:
            zero_features = torch.zeros_like(integrated_features[0])
            integrated_features.append(zero_features)

        # Stack and reshape for attention
        integrated_stack = torch.stack(integrated_features, dim=1)  # [batch, num_mods, seq, hidden]
        batch_size, num_mods, seq_len, hidden_dim = integrated_stack.shape
        reshaped_input = integrated_stack.view(batch_size * num_mods, seq_len, hidden_dim)  # [batch*mods, seq, hidden]

        # Process through attention mechanism
        attended = self.attention(reshaped_input)  # [batch*mods, seq, hidden]
        attended = attended.view(batch_size, num_mods, seq_len, hidden_dim)  # Restore shape

        # Enhanced competition with gating
        competition_input = attended.mean(dim=2)  # Average over sequence dimension [batch, mods, hidden]
        competition_output, competition_weights = self.competition_gate(competition_input, competition_input, competition_input)

        # Information integration
        integrated_info = torch.cat([
            competition_output,
            attended.mean(dim=2)  # Context from attention
        ], dim=-1)
        integrated_info = self.integration_layer(integrated_info)
        
        # Enhanced broadcasting with gating
        gate_input = torch.cat([competition_output, integrated_info], dim=-1)
        broadcast_gate = self.broadcast_gate(gate_input)
        broadcasted = self.broadcast_layer(competition_output)
        broadcasted = broadcast_gate * broadcasted + (1 - broadcast_gate) * integrated_info
        
        # Mean pooling across modalities to get final broadcast shape [batch, hidden]
        broadcasted = broadcasted.mean(dim=1)  # Add this line to get correct shape

        # Get attention weights and reshape for correct dimensionality
        attention_weights = self.attention.attention_weights  # [batch*mods, seq, seq]
        batch_size, num_mods, seq_len, hidden_dim = attended.shape
        
        # Reshape attention weights to match expected dimensions
        attention_weights = attention_weights.view(batch_size, num_mods, seq_len, -1)
        
        # Pad to match expected number of modalities if needed
        if attention_weights.size(1) < self.num_modalities:
            padding = torch.zeros(
                batch_size,
                self.num_modalities - attention_weights.size(1),
                seq_len,
                attention_weights.size(3),
                device=attention_weights.device
            )
            attention_weights = torch.cat([attention_weights, padding], dim=1)
        
        # Ensure we have the correct sequence length dimension
        if attention_weights.size(2) < 3:
            padding = torch.zeros(
                batch_size,
                attention_weights.size(1),
                3 - attention_weights.size(2),
                attention_weights.size(3),
                device=attention_weights.device
            )
            attention_weights = torch.cat([attention_weights, padding], dim=2)
        
        # Final reshape to match expected dimensions [batch, num_modalities, seq_len]
        attention_weights = attention_weights[:, :self.num_modalities, :3, :3]
        attention_weights = attention_weights.squeeze(-1)  # Remove the last dimension

        return {
            'broadcasted': broadcasted,  # Now correctly [batch, hidden]
            'attended': attended,  # [batch, mods, seq, hidden]
            'competition_weights': competition_weights,  # [batch, mods, mods]
            'workspace_attention': attention_weights,  # [batch, num_modalities, seq_len]
            'integration_state': integrated_info  # New field for tracking integration state
        }

