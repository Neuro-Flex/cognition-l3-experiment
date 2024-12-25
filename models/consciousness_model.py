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
            head_dim=hidden_dim // num_heads,
            dropout_rate=dropout_rate
        )

        # Working memory with GRU cells
        self.working_memory = WorkingMemory(
            input_dim=self.input_dim,
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

    def forward(self, inputs, state=None, initial_state=None, deterministic=True, consciousness_threshold=0.5):
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
            workspace_input
        )
        metrics['attention_weights'] = attention_weights

        # Working memory update
        memory_output, memory_state = self.working_memory(
            workspace_output,
            deterministic=deterministic,
            initial_state=initial_state
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

class WorkingMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim

    def forward(self, inputs, deterministic=False, initial_state=None):
        # Project input
        projected = self.input_projection(inputs)
        
        # Get batch size from input
        batch_size = projected.size(0)
        
        # Prepare initial state for GRU
        if initial_state is not None:
            # Reshape to match GRU's expected hidden state shape (num_layers, batch_size, hidden_dim)
            initial_state = initial_state.view(batch_size, self.hidden_dim)
            initial_state = initial_state.unsqueeze(0)
            # Expand to match the expected batch size
            initial_state = initial_state.expand(1, batch_size, self.hidden_dim)
        else:
            # Initialize with zeros if no state provided
            initial_state = torch.zeros(1, batch_size, self.hidden_dim, device=inputs.device)
            
        # Pass through GRU
        memory_output, memory_state = self.gru(projected, initial_state)
        
        # Apply layer normalization and dropout
        if not deterministic:
            memory_output = self.dropout(memory_output)
        memory_output = self.layer_norm(memory_output)
        
        return memory_output, memory_state

class CognitiveProcessIntegration(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)  # Added input_projection
        # ...initialize other necessary layers...

    def forward(self, inputs: Dict[str, torch.Tensor], deterministic: bool = True):
        """Process multiple modalities and generate cross-modal attention maps."""
        batch_size = next(iter(inputs.values())).size(0)
        seq_length = next(iter(inputs.values())).size(1)
        attention_maps = {}
        processed_states = {}

        # First pass: Project all inputs
        for modality, tensor in inputs.items():
            processed = self.input_projection(tensor)  # Use input_projection
            processed_states[modality] = processed

        # Initialize combined state with zeros matching the maximum sequence length
        max_seq_length = max(tensor.size(1) for tensor in processed_states.values())
        combined_state = torch.zeros(
            batch_size, max_seq_length, self.hidden_dim,
            device=next(iter(inputs.values())).device
        )

        # Generate attention maps between all modality pairs
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    query = processed_states[target]
                    key = processed_states[source]
                    value = processed_states[source]

                    attn_output, attn_weights = self.attention(
                        query=query,
                        key=key,
                        value=value
                    )

                    # Store attention map
                    map_key = f"{target}-{source}"
                    attention_maps[map_key] = attn_weights

                    # Pad attn_output if necessary to match combined_state's sequence length
                    if attn_output.size(1) < max_seq_length:
                        pad_size = max_seq_length - attn_output.size(1)
                        attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, pad_size))
                    elif attn_output.size(1) > max_seq_length:
                        attn_output = attn_output[:, :max_seq_length, :]

                    combined_state = combined_state + attn_output

        # ...existing code...
        return combined_state, attention_maps

class InformationIntegration(nn.Module):
    def __init__(self, hidden_dim: int, num_modules: int, dropout_rate: float):
        super().__init__()
        # Store modules in a ModuleList
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_modules)
        ])
        self.phi_layer = nn.Linear(hidden_dim, 1)

    def forward(self, memory_output: torch.Tensor, deterministic: bool = True):
        integrated_output = memory_output
        # Iterate through module_list instead of calling modules()
        for module in self.module_list:
            integrated_output = module(integrated_output)
        
        # Compute phi with non-linearity to introduce variability
        phi = torch.sigmoid(self.phi_layer(integrated_output)).squeeze(-1)
        
        return integrated_output, phi
