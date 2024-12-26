import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_dim: int, memory_size: int = 1000, similarity_threshold: float = 0.5):
        super().__init__()
        self.memory_size = memory_size
        self.memory_key = nn.Linear(hidden_dim, hidden_dim)
        self.memory_value = nn.Linear(hidden_dim, hidden_dim)
        self.memory_query = nn.Linear(hidden_dim, hidden_dim)
        self.similarity_threshold = similarity_threshold
        self.importance_scorer = nn.Linear(hidden_dim, 1)
        
        # Initialize memory banks
        self.register_buffer('memory_keys', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('memory_index', torch.zeros(memory_size, dtype=torch.bool))
        self.register_buffer('memory_importance', torch.zeros(memory_size))
        
        # Add adaptive memory management
        self.register_buffer('memory_access_count', torch.zeros(memory_size))
        self.register_buffer('memory_utility', torch.zeros(memory_size))
        self.adaptive_threshold = similarity_threshold

    def consolidate_memory(self) -> None:
        """Consolidate and cleanup memory periodically"""
        # Remove duplicate memories
        unique_keys = {}
        for idx in range(self.memory_size):
            if not self.memory_index[idx]:
                continue
            key = tuple(self.memory_keys[idx].tolist())
            if key in unique_keys:
                # Keep the more important memory
                if self.memory_importance[idx] > self.memory_importance[unique_keys[key]]:
                    self.memory_index[unique_keys[key]] = False
                    unique_keys[key] = idx
                else:
                    self.memory_index[idx] = False
            else:
                unique_keys[key] = idx

    def update_memory_utility(self, idx: int, utility_score: float):
        """Update utility scores for memory management"""
        self.memory_utility[idx] = (self.memory_utility[idx] * self.memory_access_count[idx] + utility_score) / (self.memory_access_count[idx] + 1)
        self.memory_access_count[idx] += 1

    def adaptive_consolidation(self) -> None:
        """Dynamically adjust memory management parameters"""
        if torch.sum(self.memory_index) > 0:
            # Adjust similarity threshold based on memory usage
            usage_ratio = torch.sum(self.memory_index).float() / self.memory_size
            self.adaptive_threshold = self.similarity_threshold * (1.0 + usage_ratio)
            
            # Remove low-utility memories when space is needed
            if usage_ratio > 0.9:
                utility_threshold = torch.quantile(self.memory_utility[self.memory_index], 0.2)
                low_utility = self.memory_utility < utility_threshold
                self.memory_index[low_utility] = False

    def store(self, state: torch.Tensor, force_store: bool = False) -> bool:
        """Store memory with importance scoring and duplicate detection"""
        key = self.memory_key(state)
        value = self.memory_value(state)
        importance = self.importance_scorer(state).item()

        # Check similarity with existing memories
        if not force_store:
            similarities = F.cosine_similarity(key.unsqueeze(0), self.memory_keys[self.memory_index], dim=1)
            if torch.any(similarities > self.adaptive_threshold):
                return False

        # Find slot for new memory
        if torch.any(~self.memory_index):
            # Use empty slot
            idx = torch.where(~self.memory_index)[0][0]
        else:
            # Replace least important memory
            idx = torch.argmin(self.memory_importance)

        self.memory_keys[idx] = key
        self.memory_values[idx] = value
        self.memory_age[idx] = 0
        self.memory_importance[idx] = importance
        self.memory_index[idx] = True
        self.memory_age[self.memory_index] += 1

        # Consolidate periodically
        if torch.sum(self.memory_index) % 100 == 0:
            self.consolidate_memory()
        
        # Add adaptive consolidation
        self.adaptive_consolidation()

        return True

    def recall(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced recall with top-k and importance weighting"""
        q = self.memory_query(query)
        
        # Only compare against stored memories
        active_keys = self.memory_keys[self.memory_index]
        active_values = self.memory_values[self.memory_index]
        active_importance = self.memory_importance[self.memory_index]

        if len(active_keys) == 0:
            return torch.zeros_like(query), torch.zeros(0)

        attention = torch.matmul(q, active_keys.T)
        
        # Weight by importance and recency
        attention = attention * active_importance
        attention = attention / (1 + self.memory_age[self.memory_index])

        # Get top-k memories
        weights = F.softmax(attention, dim=-1)
        values, indices = torch.topk(weights, min(top_k, len(weights)))
        
        recalled = torch.matmul(weights[indices], active_values[indices])
        return recalled, values

    def forget_old_memories(self, age_threshold: int) -> None:
        """Remove memories older than threshold"""
        old_memories = self.memory_age > age_threshold
        self.memory_index[old_memories] = False

    def get_memory_stats(self) -> dict:
        """Return memory usage statistics"""
        return {
            'total_memories': torch.sum(self.memory_index).item(),
            'average_age': torch.mean(self.memory_age[self.memory_index]).item(),
            'average_importance': torch.mean(self.memory_importance[self.memory_index]).item()
        }
