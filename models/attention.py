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
            raise ValueError("Empty input tensor")
        if key_value.size(0) == 0 or key_value.size(1) == 0 or key_value.size(2) == 0:
            raise ValueError("Empty input tensor")
        if query.size(0) != key_value.size(0):
            raise ValueError("Batch size mismatch between query and key_value")
        if query.size(1) != key_value.size(1):
            raise ValueError("Sequence length mismatch between query and key_value")
        if query.nelement() == 0 or key_value.nelement() == 0:
            raise ValueError("Empty input tensor")
            
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

    def _process_attention(self, inputs: torch.Tensor,
                         memory_state: Optional[torch.Tensor] = None,
                         deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process inputs through attention mechanism with residual connection."""
        # Use attention mechanism
        attention_output, attention_weights = self.attention(
            query=inputs,
            key_value=memory_state if memory_state is not None else inputs
        )
        
        # First residual connection and layer norm
        attention_output = inputs + attention_output
        normalized = self.layer_norm2(attention_output)
        
        # Feed-forward network with residual
        ff_output = self.ff_network(normalized)
        output = attention_output + ff_output
        
        return output, attention_weights

    def forward(self, inputs: torch.Tensor,
                memory_state: Optional[torch.Tensor] = None, 
                deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional deterministic mode."""
        # Input validation
        if inputs.numel() == 0:
            raise ValueError("Empty input tensor")
            
        # Get input dimensions
        batch_size, *dims = inputs.size()
        
        # Reshape input if needed to match expected 3D shape [batch, seq, features]
        if len(dims) == 1:
            inputs = inputs.unsqueeze(1)  # Add sequence dimension
        elif len(dims) > 2:
            # Flatten all dimensions after batch into sequence dimension
            inputs = inputs.view(batch_size, -1, dims[-1])
            
        # Apply layer normalization
        normalized = self.layer_norm1(inputs)
        
        # Process through attention layers
        output, attention_weights = self._process_attention(normalized, memory_state, deterministic)
        
        return output, attention_weights
