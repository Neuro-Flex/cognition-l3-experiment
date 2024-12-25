import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConsciousnessAttention(nn.Module):
    """
    Multi-head attention mechanism for consciousness modeling based on Global Workspace Theory.
    Implements scaled dot-product attention with consciousness-aware broadcasting.
    """
    def __init__(self, num_heads: int, head_dim: int, dropout_rate: float = 0.1, attention_dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.depth = num_heads * head_dim

        # Linear projections
        self.query = nn.Linear(self.depth, self.depth)
        self.key = nn.Linear(self.depth, self.depth)
        self.value = nn.Linear(self.depth, self.depth)
        self.output_projection = nn.Linear(self.depth, self.depth)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs_q.size(0)
        
        # Linear projections
        query = self.query(inputs_q)
        key = self.key(inputs_kv)
        value = self.value(inputs_kv)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        depth_scaling = float(self.head_dim) ** -0.5
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) * depth_scaling

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_logits = attention_logits.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(attention_logits, dim=-1)
        
        if training:
            attention_weights = self.attn_dropout(attention_weights)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.depth)
        output = self.output_projection(attention_output)

        if training:
            output = self.output_dropout(output)

        # Residual connection
        output = output + inputs_q

        return output, attention_weights

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory for consciousness modeling.
    Integrates information from multiple cognitive processes through attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Layer normalizations
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Attention mechanism
        self.attention = ConsciousnessAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, inputs: torch.Tensor, 
                memory_state: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional deterministic mode."""
        # Layer normalization and attention
        x = self.layer_norm1(inputs)
        attended_output, attention_weights = self.attention(
            x, x, mask=None, 
            training=not deterministic  # Convert deterministic to training mode
        )
        
        # First residual connection
        x = inputs + attended_output
        
        # Feed-forward network with residual connection
        y = self.layer_norm2(x)
        y = self.ff_network(y) if not deterministic else self.ff_network.eval()(y)
        output = x + y

        return output, attention_weights
