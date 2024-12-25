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

        # Add components for ARC reasoning
        self.sequence_predictor = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.transformation_net = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.rule_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, inputs, state=None, initial_state=None, deterministic=True, consciousness_threshold=0.5):
        """
        Process inputs through consciousness architecture.
        """
        # Initialize attention maps dictionary 
        attention_maps = {}

        # Validate and process inputs
        if not inputs:
            raise ValueError("Inputs cannot be empty.")

        # Allow for more flexible input combinations
        required_modalities = {'visual', 'textual'}  # Required modalities
        missing_modalities = required_modalities - inputs.keys()
        if missing_modalities:
            # Auto-populate missing modalities with zero tensors
            batch_size = next(iter(inputs.values())).size(0)
            seq_len = next(iter(inputs.values())).size(1)
            for modality in missing_modalities:
                inputs[modality] = torch.zeros(batch_size, seq_len, self.hidden_dim, device=inputs[next(iter(inputs.keys()))].device)

        # Check input dimensions
        expected_dims = {
            'attention': (None, 8, self.hidden_dim),
            'memory': (None, 10, self.hidden_dim),
            'visual': (None, None, self.hidden_dim),
            'textual': (None, None, self.hidden_dim)
        }

        # Project inputs to correct dimension if needed
        for modality, tensor in inputs.items():
            if modality in expected_dims:
                # Project if dimensions don't match
                if tensor.size(-1) != self.hidden_dim:
                    inputs[modality] = self.input_projection(tensor)

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

        # Add sequence prediction
        sequence_pred = self.sequence_predictor(new_state)
        
        # Add transformation understanding
        if inputs['visual'].shape[1] > 1:  # If we have a sequence
            trans_input = torch.cat([new_state[:,0], new_state[:,1]], dim=1)
            trans_vec = self.transformation_net(trans_input)
        else:
            trans_vec = torch.zeros_like(new_state[:,0])
            
        # Add rule learning
        rule_embed = self.rule_encoder(new_state.mean(dim=1))
        
        metrics.update({
            'sequence_predictions': sequence_pred,
            'transformation_vectors': trans_vec,
            'rule_embeddings': rule_embed,
            'rule_confidence': torch.sigmoid(rule_embed.norm(dim=-1, keepdim=True))
        })

        # Ensure new_state has shape (batch_size, hidden_dim)
        new_state = new_state.mean(dim=1)

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
        if not inputs:
            raise ValueError("Empty input dictionary")

        # Get dimensions from largest input tensor
        seq_lengths = {name: tensor.size(1) if tensor.dim() > 1 else 1 
                      for name, tensor in inputs.items()}
        max_seq_len = max(seq_lengths.values())
        
        # Pad all inputs to match max sequence length
        processed_inputs = {}
        for name, tensor in inputs.items():
            if tensor.dim() == 2:  # [batch, features]
                tensor = tensor.unsqueeze(1)  # Add sequence dimension
            if tensor.size(1) < max_seq_len:
                # Pad sequence dimension to match max length
                pad_size = max_seq_len - tensor.size(1)
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))
            processed_inputs[name] = tensor

        # Continue with regular processing using padded inputs
        # Get dimensions from first input tensor
        first_tensor = next(iter(inputs.values()))
        batch_size = first_tensor.size(0)
        hidden_dim = first_tensor.size(-1)
        
        # Validate all inputs have same sequence length
        seq_length = next(iter(inputs.values())).size(1)
        for name, tensor in inputs.items():
            if tensor.size(1) != seq_length:
                raise ValueError(f"Sequence length mismatch for {name}: expected {seq_length}, got {tensor.size(1)}")
        
        # Initialize combined state with correct dimensions
        combined_state = torch.zeros(
            batch_size, seq_length, hidden_dim,
            device=first_tensor.device
        )

        attention_maps = {}
        processed_states = {}

        # Input validation
        if not inputs:
            raise ValueError("Empty input dictionary")

        # Ensure all inputs have same dimensions
        first_tensor = next(iter(inputs.values()))
        expected_shape = first_tensor.shape[-1]
        for name, tensor in inputs.items():
            if tensor.shape[-1] != expected_shape:
                raise ValueError(f"Mismatched dimensions for {name}: expected {expected_shape}, got {tensor.shape[-1]}")

        # Project and reshape inputs
        for modality, tensor in inputs.items():
            # Ensure 3D shape for attention
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            processed = self.input_projection(tensor)
            processed_states[modality] = processed

        # Generate attention maps between all pairs
        combined_state = torch.zeros(
            batch_size, seq_length, self.hidden_dim,
            device=next(iter(inputs.values())).device
        )

        for source in processed_states.keys():
            for target in processed_states.keys():
                if source != target:
                    query = processed_states[target] 
                    key = processed_states[source]
                    value = processed_states[source]

                    # Ensure 3D shape for attention
                    if query.dim() == 2:
                        query = query.unsqueeze(1)
                    if key.dim() == 2:
                        key = key.unsqueeze(1)
                    if value.dim() == 2:
                        value = value.unsqueeze(1)

                    attn_output, attn_weights = self.attention(
                        query=query,
                        key=key,
                        value=value
                    )
                    
                    attention_maps[f"{target}-{source}"] = attn_weights
                    combined_state = combined_state + attn_output

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
