import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .working_memory import WorkingMemory
from .information_integration import InformationIntegration
from .self_awareness import SelfAwareness  # Add this import

class MultiHeadAttention(nn.Module):
    """Custom MultiHeadAttention implementation"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate)
        
    def forward(self, x, deterministic=True):
        # Store attention weights for later use
        output, self.attention_weights = self.attention(x, x, x)
        return output

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory for consciousness simulation.
    Manages attention, working memory, and information integration.
    """
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention mechanism for information broadcasting
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)

        # Working memory component
        self.memory_gate = nn.Linear(hidden_dim, hidden_dim)
        self.memory_update = nn.Linear(hidden_dim, hidden_dim)

        # Information integration layers
        self.integration_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, memory_state: Optional[torch.Tensor] = None,
              deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process inputs through attention mechanism
        attended = self.attention(inputs, deterministic=deterministic)
        
        # Ensure memory_state has correct shape
        if memory_state is None:
            memory_state = torch.zeros_like(attended)
        else:
            # Expand memory state if needed
            memory_state = memory_state.unsqueeze(1).expand(-1, attended.size(1), -1)

        # Update working memory with broadcasting
        gate = torch.sigmoid(self.memory_gate(attended))
        update = self.memory_update(attended)
        memory_state = gate * memory_state + (1 - gate) * update

        # Pool across sequence dimension if needed
        if len(memory_state.shape) == 3:
            memory_state = memory_state.mean(dim=1)

        # Integrate information
        integrated = torch.relu(self.integration_layer(
            torch.cat([attended.mean(dim=1), memory_state], dim=-1)
        ))

        # Generate conscious output
        output = self.output_layer(integrated)
        output = self.layer_norm(output)

        return output, memory_state

class ConsciousnessModel(nn.Module):
    """
    Main consciousness module implementing integration of various cognitive processes.
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, num_states: int, dropout_rate: float = 0.1, input_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.num_states = num_states
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim if input_dim is not None else hidden_dim

        # Global Workspace for conscious awareness
        self.global_workspace = GlobalWorkspace(
            hidden_dim=hidden_dim,
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )

        # Working memory
        self.working_memory = WorkingMemory(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # Information integration
        self.information_integration = InformationIntegration(
            hidden_dim=hidden_dim,
            num_modules=num_layers,
            dropout_rate=dropout_rate
        )

        # Add attention for multi-head processing
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Add self-awareness module
        self.self_awareness = SelfAwareness(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # State tracking
        self.previous_state = None

    def get_config(self):
        return {
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_states': self.num_states,
            'dropout_rate': self.dropout_rate
        }

    @staticmethod
    def create_default_config():
        return {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'num_states': 4,
            'dropout_rate': 0.1
        }

    def calculate_phi(self, conscious_output):
        """Calculate information integration metric (phi)"""
        return torch.abs(self.phi_calculator(conscious_output)).mean()

    def calculate_energy_cost(self, cognitive_outputs):
        """Calculate energy cost of processing"""
        return torch.abs(self.energy_tracker(torch.mean(cognitive_outputs, dim=0))).mean()

    def forward(self, inputs: Dict[str, torch.Tensor],
                state: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> Tuple[torch.Tensor, Dict]:
        
        # Get device from inputs
        device = next(iter(inputs.values())).device
        
        # Initialize state if None
        if state is None:
            state = torch.zeros(inputs['attention'].shape[0], self.hidden_dim, device=device)
            
        # Get input tensor
        x = inputs['attention']  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention - x is already in the correct shape
        attn_out, attention_weights = self.attention(x, x, x)
        
        # Process through global workspace with reshaped state
        conscious_out, memory_state = self.global_workspace(attn_out, state, deterministic)
        
        # Process through self-awareness
        aware_state, awareness_metrics = self.self_awareness(
            conscious_out,
            previous_state=self.previous_state
        )
        
        # Update previous state
        self.previous_state = aware_state.detach()
        
        # Calculate integration metrics
        integrated_out, phi = self.information_integration(conscious_out, deterministic)
        
        # Update metrics
        metrics = {
            'attention_weights': attention_weights,
            'memory_state': memory_state,
            'phi': phi,
            'attention_maps': attention_weights,
            **awareness_metrics
        }
        
        return aware_state, metrics

def create_consciousness_module(hidden_dim: int = 512,
                             num_cognitive_processes: int = 4) -> ConsciousnessModel:
    """Creates and initializes the consciousness module."""
    return ConsciousnessModel(
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=4,
        num_states=num_cognitive_processes,
        dropout_rate=0.1
    )
