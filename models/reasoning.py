import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from models.base_model import BaseModel
from core.config import ModelConfig

class ReasoningModule(nn.Module):
    """Module for implementing reasoning capabilities."""
    def __init__(self, config: ModelConfig, base_model: BaseModel):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.context_processor = nn.Linear(config.hidden_size, config.hidden_size)
        self.reasoning_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projector = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base model representations
        hidden_states, pooled = self.base_model(
            input_ids,
            attention_mask=attention_mask
        )

        # Process context if provided
        if context is not None:
            context_features = self.context_processor(context)
            # Ensure context_features matches the shape of hidden_states
            context_features = context_features.unsqueeze(1)
            hidden_states = hidden_states + context_features

        # Apply reasoning transformations
        reasoning_output = self.reasoning_head(hidden_states)

        # Generate output logits
        logits = self.output_projector(reasoning_output)

        return logits, reasoning_output

class CommonSenseReasoning(ReasoningModule):
    """Specialized module for common-sense reasoning."""
    def __init__(self, config: ModelConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.concept_embeddings = nn.Embedding(
            num_embeddings=10000,  # Placeholder size for concept vocabulary
            embedding_dim=config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        concept_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process concepts if provided
        concept_context = None
        if concept_ids is not None:
            concept_context = self.concept_embeddings(concept_ids)

        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            context=concept_context
        )

class MathematicalReasoning(ReasoningModule):
    """Specialized module for mathematical reasoning."""
    def __init__(self, config: ModelConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.symbolic_processor = nn.Linear(config.hidden_size, config.hidden_size)
        self.equation_encoder = nn.Linear(config.hidden_size, config.hidden_size)

    def process_symbolic(
        self,
        symbolic_input: torch.Tensor
    ) -> torch.Tensor:
        """Process symbolic mathematical expressions."""
        return self.symbolic_processor(symbolic_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        symbolic_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process symbolic input if provided
        context = None
        if symbolic_input is not None:
            processed_symbolic = self.process_symbolic(symbolic_input)
            context = processed_symbolic

        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            context=context
        )

class BasicReasoning(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Logical reasoning components
        self.logical_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Pattern recognition
        self.pattern_recognition = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        
        # Causal inference
        self.causal_inference = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Reasoning confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Logical reasoning
        logical_out = self.logical_layer(x)
        
        # Pattern recognition through attention
        pattern_out, pattern_weights = self.pattern_recognition(x, x, x)
        
        # Causal inference by combining logical and pattern outputs
        causal_input = torch.cat([logical_out, pattern_out], dim=-1)
        causal_out = self.causal_inference(causal_input)
        
        # Calculate reasoning confidence
        confidence_input = torch.cat([logical_out, pattern_out, causal_out], dim=-1)
        confidence = self.confidence_estimator(confidence_input)
        
        # Calculate normalized reasoning scores
        with torch.no_grad():
            # Normalize using softmax for pattern weights
            pattern_weights_norm = torch.softmax(pattern_weights, dim=-1)
            pattern_score = torch.mean(pattern_weights_norm)
            
            # Normalize cosine similarities to [0,1] range
            logical_sim = torch.cosine_similarity(logical_out, x, dim=-1)
            logical_score = torch.clamp((logical_sim + 1) / 2, 0, 1).mean()
            
            causal_sim = torch.cosine_similarity(causal_out, x, dim=-1)
            causal_score = torch.clamp((causal_sim + 1) / 2, 0, 1).mean()
        
        return {
            'output': causal_out,
            'confidence': confidence,
            'metrics': {
                'logical_score': logical_score.item(),
                'pattern_score': pattern_score.item(),
                'causal_score': causal_score.item(),
                'reasoning_weights': pattern_weights
            }
        }

    def calculate_reasoning_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall reasoning capability score"""
        weights = {
            'logical': 0.4,
            'pattern': 0.3,
            'causal': 0.3
        }
        
        score = (
            weights['logical'] * metrics['logical_score'] +
            weights['pattern'] * metrics['pattern_score'] +
            weights['causal'] * metrics['causal_score']
        )
        
        return score * 100  # Convert to percentage
