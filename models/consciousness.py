import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

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

        # Update working memory
        if memory_state is None:
            memory_state = torch.zeros_like(attended)

        gate = torch.sigmoid(self.memory_gate(attended))
        update = self.memory_update(attended)
        memory_state = gate * memory_state + (1 - gate) * update

        # Integrate information
        integrated = torch.relu(self.integration_layer(
            torch.cat([attended, memory_state], dim=-1)
        ))

        # Generate conscious output
        output = self.output_layer(integrated)
        output = self.layer_norm(output)

        return output, memory_state

class ConsciousnessModule(nn.Module):
    """
    Main consciousness module implementing integration of various cognitive processes.
    """
    def __init__(self, hidden_dim: int = 512, num_cognitive_processes: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cognitive_processes = num_cognitive_processes

        # Global workspace for consciousness
        self.global_workspace = GlobalWorkspace(hidden_dim=hidden_dim)

        # Cognitive processes (attention, memory, reasoning, emotion)
        self.cognitive_processes = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_cognitive_processes)
        ])

        # Integration layer
        self.integration = nn.Linear(hidden_dim, hidden_dim)
        
        # Add energy tracking
        self.energy_tracker = nn.Linear(hidden_dim, 1)
        
        # Add phi calculation layer
        self.phi_calculator = nn.Linear(hidden_dim, 1)

    def calculate_phi(self, conscious_output):
        """Calculate information integration metric (phi)"""
        return torch.abs(self.phi_calculator(conscious_output)).mean()

    def calculate_energy_cost(self, cognitive_outputs):
        """Calculate energy cost of processing"""
        return torch.abs(self.energy_tracker(torch.mean(cognitive_outputs, dim=0))).mean()

    def forward(self, inputs: Dict[str, torch.Tensor],
              memory_state: Optional[torch.Tensor] = None,
              deterministic: bool = True) -> Dict[str, torch.Tensor]:
        # Ensure the number of inputs matches the number of cognitive processes
        if len(inputs) != self.num_cognitive_processes:
            raise ValueError("Number of input modalities must match num_cognitive_processes.")

        # Process different cognitive aspects
        cognitive_outputs = []
        for process, (key, value) in zip(self.cognitive_processes, inputs.items()):
            processed = process(value)
            cognitive_outputs.append(processed)

        # Combine cognitive processes by stacking
        combined = torch.stack(cognitive_outputs, dim=1)

        # Process through global workspace
        conscious_output, new_memory_state = self.global_workspace(
            combined, memory_state, deterministic
        )

        # Calculate metrics
        phi = self.calculate_phi(conscious_output)
        energy_cost = self.calculate_energy_cost(torch.stack(cognitive_outputs))
        attention_maps = self.global_workspace.attention.attention_weights

        # Final integration
        integrated = self.integration(conscious_output)
        integrated = torch.relu(integrated)

        return {
            'output': integrated,
            'memory_state': new_memory_state,
            'consciousness_state': conscious_output,
            'phi': phi,
            'energy_cost': energy_cost,
            'attention_maps': attention_maps
        }

def create_consciousness_module(hidden_dim: int = 512,
                             num_cognitive_processes: int = 4) -> ConsciousnessModule:
    """Creates and initializes the consciousness module."""
    return ConsciousnessModule(
        hidden_dim=hidden_dim,
        num_cognitive_processes=num_cognitive_processes
    )