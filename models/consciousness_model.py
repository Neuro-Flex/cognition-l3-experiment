"""
Main consciousness model integrating all components.
"""
import torch
import torch.nn as nn
from typing import Any, Dict

from .attention import GlobalWorkspace
from .memory import WorkingMemory, InformationIntegration
from .consciousness_state import CognitiveProcessIntegration, ConsciousnessStateManager

class ConsciousnessModel(nn.Module):
    """
    Complete consciousness model integrating GWT, IIT, working memory,
    and cognitive process management.
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, num_states: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_states = num_states
        self.dropout_rate = dropout_rate

        # Global Workspace for conscious awareness
        self.global_workspace = GlobalWorkspace(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            dropout_rate=dropout_rate
        )

        # Working memory with GRU cells
        self.working_memory = WorkingMemory(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # Information integration component
        self.information_integration = InformationIntegration(
            hidden_dim=hidden_dim,
            num_modules=num_layers,
            dropout_rate=dropout_rate
        )

        # Cognitive process integration
        self.cognitive_integration = CognitiveProcessIntegration(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        # Consciousness state management
        self.state_manager = ConsciousnessStateManager(
            hidden_dim=hidden_dim,
            num_states=num_states,
            dropout_rate=dropout_rate
        )

        # GRU cell
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Shape alignment layer
        self.align_layer = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

    def forward(self, inputs, state=None, deterministic=True, consciousness_threshold=0.5):
        """
        Process inputs through consciousness architecture.
        """
        # Initialize attention maps dictionary
        attention_maps = {}

        # Validate and process inputs
        batch_size = next(iter(inputs.values())).shape[0]
        inputs = {k: torch.tensor(v, dtype=torch.float32) for k, v in inputs.items()}

        # Initialize consciousness state if none provided
        if state is None:
            state = torch.zeros(batch_size, self.hidden_dim, device=next(iter(inputs.values())).device)
        else:
            state = torch.tensor(state, dtype=torch.float32)

        metrics = {}

        # Global workspace processing
        workspace_input = next(iter(inputs.values()))
        workspace_output, attention_weights = self.global_workspace(
            workspace_input,
            deterministic=deterministic
        )
        metrics['attention_weights'] = attention_weights

        # Working memory update
        memory_output, memory_state = self.working_memory(
            workspace_output,
            deterministic=deterministic,
            initial_state=state
        )
        metrics['memory_state'] = memory_state

        # Information integration
        integrated_output, phi = self.information_integration(
            memory_output,
            deterministic=deterministic
        )
        metrics['phi'] = phi

        # Cognitive process integration
        consciousness_state, attention_maps = self.cognitive_integration(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in inputs.items()},
            deterministic=deterministic
        )
        metrics['attention_maps'] = attention_maps

        # Update consciousness state
        new_state, state_metrics = self.state_manager(
            consciousness_state,
            integrated_output,
            threshold=consciousness_threshold,
            deterministic=deterministic
        )
        metrics.update(state_metrics)

        # Apply multi-head attention
        attn_output, attention_weights = self.attention(
            memory_output,
            memory_output,
            memory_output,
            need_weights=True
        )
        
        # Store attention map
        attention_maps['self_attention'] = attention_weights

        return new_state, {
            'attention_weights': attention_weights,
            'attention_maps': attention_maps,
            'memory_state': memory_state,
            'phi': phi,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_states': self.num_states,
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def create_default_config(cls) -> Dict[str, Any]:
        """Create default model configuration."""
        return {
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'num_states': 4,
            'dropout_rate': 0.1
        }
