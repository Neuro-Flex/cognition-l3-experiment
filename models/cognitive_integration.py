import torch
import torch.nn as nn
from typing import Dict, Any

class CognitiveProcessIntegration(nn.Module):
    """
    Extended transformer architecture for multi-modal task management.
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Layer definitions
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = True):
        # Process each modality separately first
        processed_modalities = {}
        for modality, x in inputs.items():
            x = self.layer_norm(x)
            x = self.dense(x)
            x = self.gelu(x)
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
                    attention = nn.MultiheadAttention(
                        embed_dim=self.hidden_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout_rate,
                        batch_first=True
                    )
                    attended, _ = attention(
                        target_features, 
                        source_features, 
                        source_features,
                        need_weights=False
                    )
                    cross_modal_contexts.append(attended)
                    attention_maps[f"{target_modality}-{source_modality}"] = attended

            if cross_modal_contexts:
                combined = torch.mean(torch.stack(cross_modal_contexts), dim=0)
                combined = nn.Linear(combined.shape[-1], target_features.shape[-1])(combined)
                integrated = target_features + combined
            else:
                integrated = target_features

            integrated_features.append(integrated)

        consciousness_state = torch.mean(torch.stack(integrated_features), dim=0)
        return consciousness_state, attention_maps