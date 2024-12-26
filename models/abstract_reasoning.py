import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class AbstractReasoning(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pattern_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.causal_analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.symbolic_reasoner = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2
        )
        
        # Add pattern learning components
        self.pattern_memory = nn.Parameter(torch.zeros(100, hidden_dim))
        self.pattern_importance = nn.Parameter(torch.zeros(100))
        
        # Add learning rate controller
        self.learning_rate_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def learn_pattern(self, pattern: torch.Tensor) -> None:
        """Store and learn from new patterns"""
        # Calculate pattern novelty
        similarities = F.cosine_similarity(
            pattern.unsqueeze(0),
            self.pattern_memory,
            dim=1
        )
        
        # Find least important pattern to replace
        if torch.min(similarities) > 0.8:
            idx = torch.argmin(self.pattern_importance)
            self.pattern_memory[idx] = pattern
            self.pattern_importance[idx] = self.learning_rate_controller(pattern).item()

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        patterns = self.pattern_extractor(state)
        causal = self.causal_analyzer(torch.cat([state, context], dim=-1))
        symbolic = self.symbolic_reasoner(state.unsqueeze(0)).squeeze(0)
        
        # Learn from current patterns
        self.learn_pattern(patterns.detach().mean(dim=0))
        
        return {
            'patterns': patterns,
            'causal_relations': causal,
            'symbolic_output': symbolic,
            'pattern_memory': self.pattern_memory
        }
