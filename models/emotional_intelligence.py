import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class EmotionalIntelligence(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.emotion_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.emotion_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # 8 basic emotions
        )
        self.emotional_memory = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, state: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        emotion_embedding = self.emotion_encoder(state)
        emotion_logits = self.emotion_detector(emotion_embedding)
        emotion_state = self.emotional_memory(emotion_embedding, context)
        
        return {
            'emotion_logits': emotion_logits,
            'emotion_state': emotion_state
        }
