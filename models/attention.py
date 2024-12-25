import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConsciousnessAttention(nn.Module):
    """
    Multi-head attention mechanism for consciousness modeling based on Global Workspace Theory.
    Implements scaled dot-product attention with consciousness-aware broadcasting.
    """
    def __init__(self, num_heads: int, head_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Linear projections
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key_value, mask=None, training=None):
        """Forward pass of consciousness attention mechanism."""
        # Input validation
        if query.size(0) == 0 or query.size(1) == 0 or query.size(2) == 0:
            raise ValueError("Query tensor cannot be empty")
        if key_value.size(0) == 0 or key_value.size(1) == 0 or key_value.size(2) == 0:
            raise ValueError("Key/Value tensor cannot be empty")
            
        # Validate input dimensions
        if query.size(-1) != self.hidden_dim or key_value.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected input dimension {self.hidden_dim}, got query: {query.size(-1)}, key/value: {key_value.size(-1)}")
            
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expand mask for multiple heads
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~expanded_mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.output_dropout(output)

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
        # Input validation
        if inputs.size(0) == 0 or inputs.size(1) == 0 or inputs.size(2) == 0:
            raise ValueError("Input tensor cannot be empty")
            
        if inputs.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected input dimension {self.hidden_dim}, got {inputs.size(-1)}")

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
