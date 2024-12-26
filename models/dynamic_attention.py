import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class DynamicAttention(nn.Module):
    """
    Dynamic attention mechanism that adapts based on goals and context.
    Implements goal-directed attention with priority management.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Core attention components
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Goal-directed components
        self.goal_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Priority calculation
        self.priority_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=-1)
        )
        
        # Context integration
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Adaptive threshold
        self.register_buffer('attention_threshold', torch.tensor(0.1))
        self.threshold_adaptor = nn.Linear(hidden_dim, 1)
        
    def update_threshold(self, context: torch.Tensor):
        """Dynamically adjust attention threshold based on context"""
        threshold_delta = torch.sigmoid(self.threshold_adaptor(context)).mean()
        self.attention_threshold = self.attention_threshold * 0.9 + threshold_delta * 0.1
        
    def compute_priority_weights(self, query: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Calculate attention priority weights based on current goals"""
        combined = torch.cat([query, goals], dim=-1)
        return self.priority_network(combined)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                goals: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with dynamic attention allocation.
        
        Args:
            query: Input queries [batch_size, seq_len, hidden_dim]
            key: Input keys
            value: Input values
            goals: Current goals/objectives [batch_size, hidden_dim]
            context: Current context state [batch_size, hidden_dim]
        """
        batch_size = query.size(0)
        
        # Process goals if provided, otherwise use learned default
        if goals is None:
            goals = torch.zeros(batch_size, self.hidden_dim, device=query.device)
        processed_goals = self.goal_processor(goals)
        
        # Calculate priority weights
        priority_weights = self.compute_priority_weights(query.mean(dim=1), processed_goals)
        
        # Apply attention with priority weighting
        attended_value, attention_weights = self.attention(
            query + processed_goals.unsqueeze(1),
            key,
            value
        )
        
        # Integrate context if provided
        if context is not None:
            self.update_threshold(context)
            context_gate = self.context_gate(
                torch.cat([attended_value.mean(dim=1), context], dim=-1)
            )
            attended_value = attended_value * context_gate.unsqueeze(1)
        
        # Apply threshold
        attention_mask = (attention_weights > self.attention_threshold).float()
        filtered_attention = attention_weights * attention_mask
        
        metrics = {
            'priority_weights': priority_weights,
            'attention_weights': filtered_attention,
            'attention_threshold': self.attention_threshold
        }
        
        return attended_value, metrics
