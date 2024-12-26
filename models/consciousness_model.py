"""
Main consciousness model integrating all components.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple
import torch.nn.functional as F

from .attention import GlobalWorkspace
from .memory import WorkingMemory, InformationIntegration
from .consciousness_state import CognitiveProcessIntegration, ConsciousnessStateManager

class ConsciousnessModel(nn.Module):
    """
    Complete consciousness model integrating GWT, IIT, working memory,
    and cognitive process management.
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, num_states: int, dropout_rate: float = 0.1, input_dim: int = None, advanced_reflection: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_states = num_states
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.advanced_reflection = advanced_reflection

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

        # Add context-switching challenges
        self.context_switching_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_switching_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Add creative problem-solving scenarios
        self.creative_problem_solving_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.creative_problem_solving_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Add meta-learning layer
        self.add_meta_learning_layer()

        # Initialize state history and context history
        self.state_history = []
        self.context_history = []

        # Add context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State tracking
        self.register_buffer('current_state', torch.zeros(1, hidden_dim))

        # Add performance monitoring
        self.register_buffer('performance_history', torch.zeros(1000))
        self.register_buffer('adaptation_rate', torch.tensor(0.1))
        
        # Add adaptive learning components
        self.adaptive_learner = nn.ModuleDict({
            'pattern_detector': nn.Linear(hidden_dim, hidden_dim),
            'error_predictor': nn.Linear(hidden_dim, 1),
            'adaptation_gate': nn.Linear(hidden_dim * 2, hidden_dim)
        })

    def add_meta_learning_layer(self):
        """Add meta-learning capabilities"""
        self.meta_learner = nn.ModuleDict({
            'pattern_recognition': nn.Linear(self.hidden_dim, self.hidden_dim),
            'adaptation_layer': nn.GRU(self.hidden_dim, self.hidden_dim)
        })
        # Register meta_memory as a parameter instead of trying to add it to ModuleDict
        self.register_parameter(
            'meta_memory',
            nn.Parameter(torch.zeros(1, self.hidden_dim))
        )
        
    def self_reflection_mechanism(self, state: torch.Tensor, 
                                previous_states: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """Improved self-reflection mechanism"""
        coherence_score = 0.0
        current_state = state.detach()  # Ensure we don't track unnecessary gradients
        
        # Enhanced coherence calculation
        if previous_states:
            similarities = []
            for prev_state in previous_states[-5:]:
                try:
                    # Handle batch size mismatch
                    if current_state.size(0) != prev_state.size(0):
                        if current_state.size(0) > prev_state.size(0):
                            prev_state = prev_state.expand(current_state.size(0), -1)
                        else:
                            current_state = current_state.mean(0, keepdim=True).expand(prev_state.size(0), -1)
                    # Calculate similarity
                    sim = F.cosine_similarity(current_state, prev_state, dim=-1)
                    similarities.append(sim.mean().item())
                except RuntimeError as e:
                    print(f"Warning: Similarity calculation failed: {str(e)}")
                    similarities.append(0.0)
            
            coherence_score = sum(similarities) / len(similarities) if similarities else 0.0
            coherence_score = min(max(coherence_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        # Generate reflection output
        reflection_output = self.meta_learner['pattern_recognition'](current_state)
        return reflection_output, coherence_score

    def enhanced_context_switching(self, 
                                 inputs: Dict[str, torch.Tensor],
                                 context_history: List[torch.Tensor]) -> torch.Tensor:
        """Improved context switching mechanism"""
        # Use state if available, otherwise use input
        state_tensor = inputs.get('state', next(iter(inputs.values())))
        context_embedding = self.context_encoder(state_tensor)
        
        if not context_history:
            return torch.ones(context_embedding.size(0), device=context_embedding.device)
            
        context_similarity = torch.stack([
            F.cosine_similarity(context_embedding, prev_ctx, dim=-1)
            for prev_ctx in context_history[-3:]  # Look at last 3 contexts
        ]).mean(dim=0)
        
        return F.softmax(context_similarity, dim=-1)

    def adaptive_learning_step(self, state: torch.Tensor, error: float) -> torch.Tensor:
        """Adapt model parameters based on performance"""
        with torch.no_grad():
            pattern = self.adaptive_learner['pattern_detector'](state)
            predicted_error = self.adaptive_learner['error_predictor'](pattern).mean()
            
            # Update adaptation rate based on prediction error
            error_diff = abs(predicted_error.item() - error)
            self.adaptation_rate *= 0.95 if error_diff > 0.5 else 1.05
            self.adaptation_rate.clamp_(0.01, 1.0)
            
            # Generate adaptation signal
            adaptation = self.adaptive_learner['adaptation_gate'](
                torch.cat([state, pattern], dim=-1)
            )
            
            return adaptation * self.adaptation_rate.item()

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
        inputs = {k: v.clone().detach().to(dtype=torch.float32) if isinstance(v, torch.Tensor) 
                 else torch.tensor(v, dtype=torch.float32) 
                 for k, v in inputs.items()}

        # Initialize consciousness state if none provided
        if state is None:
            state = torch.zeros(batch_size, self.hidden_dim, device=next(iter(inputs.values())).device)
        else:
            state = torch.tensor(state, dtype=torch.float32)

        metrics = {}

        # Global workspace processing
        workspace_input = next(iter(inputs.values()))
        workspace_output, workspace_attention = self.global_workspace(workspace_input)
        
        # Ensure attention weights have correct shape (batch, seq, seq)
        attention_weights = workspace_attention.squeeze(1)  # Remove head dimension
        metrics['attention_weights'] = attention_weights
        
        # Working memory update
        memory_output, memory_state = self.working_memory(
            workspace_output,
            deterministic=deterministic,
            initial_state=initial_state
        )

        # Information integration
        integrated_output, phi = self.information_integration(memory_output, deterministic=deterministic)
        
        # Update required metrics
        metrics.update({
            'memory_state': memory_state,
            'attention_weights': attention_weights,
            'phi': phi,
            'attention_maps': attention_maps
        })

        # Fix state shape handling - ensure it matches sequence length
        if 'state' in inputs:
            # Handle 4D state tensor case
            state_tensor = inputs['state']
            if state_tensor.dim() == 4:
                # Remove extra dimensions (batch, extra_dim, seq, hidden)
                state_tensor = state_tensor.squeeze(1)
            elif state_tensor.dim() == 3:
                # Already correct shape (batch, seq, hidden)
                pass
            else:
                # Add sequence dimension if needed
                state_tensor = state_tensor.unsqueeze(1)
            
            # Now expand to match sequence length
            target_seq_len = next(iter(inputs.values())).size(1)
            if state_tensor.size(1) != target_seq_len:
                state_tensor = state_tensor.expand(-1, target_seq_len, -1)
            inputs['state'] = state_tensor

        # Cognitive process integration with fixed state shape
        consciousness_state, attention_maps = self.cognitive_integration(inputs, deterministic=deterministic)

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

        # Add context-switching challenges
        context_switching_output = self.context_switching_net(new_state)
        context_switching_gate = torch.sigmoid(self.context_switching_gate(new_state))
        context_switching_state = context_switching_gate * context_switching_output + (1 - context_switching_gate) * new_state

        # Add creative problem-solving scenarios
        creative_problem_solving_output = self.creative_problem_solving_net(new_state)
        creative_problem_solving_gate = torch.sigmoid(self.creative_problem_solving_gate(new_state))
        creative_problem_solving_state = creative_problem_solving_gate * creative_problem_solving_output + (1 - creative_problem_solving_gate) * new_state

        # Add self-reflection and meta-learning
        reflection_output, coherence = self.self_reflection_mechanism(
            state=new_state,
            previous_states=self.state_history[-5:]
        )
        
        # Enhanced context handling
        context_attention = self.enhanced_context_switching(
            inputs=inputs,
            context_history=self.context_history
        )
        
        metrics['coherence'] = coherence
        metrics['context_stability'] = context_attention.mean().item()

        metrics.update({
            'context_switching_state': context_switching_state,
            'creative_problem_solving_state': creative_problem_solving_state
        })

        # Update meta-learning components more robustly
        if not hasattr(self, 'state_history'):
            self.state_history = []
        if not hasattr(self, 'context_history'):
            self.context_history = []
            
        # Store current state in history (limit size)
        self.state_history = self.state_history[-10:] + [new_state.detach()]
        
        # Add self-reflection with proper error handling
        try:
            reflection_output, coherence = self.self_reflection_mechanism(
                state=new_state,
                previous_states=self.state_history[:-1]  # Exclude current state
            )
            metrics['coherence'] = coherence
        except Exception as e:
            print(f"Warning: Self-reflection failed: {str(e)}")
            metrics['coherence'] = 0.0
            reflection_output = new_state

        # Compute coherence score
        coherence_score = 0.0
        if len(self.state_history) > 0:
            current_state = new_state.detach()
            similarities = []
            for prev_state in self.state_history[-5:]:
                try:
                    # Handle batch size mismatch by broadcasting
                    if current_state.size(0) != prev_state.size(0):
                        if current_state.size(0) > prev_state.size(0):
                            prev_state = prev_state.expand(current_state.size(0), -1)
                        else:
                            current_state = current_state.mean(0, keepdim=True).expand(prev_state.size(0), -1)
                    sim = F.cosine_similarity(current_state, prev_state, dim=-1)
                    similarities.append(sim.mean().item())
                except Exception as e:
                    print(f"Warning: Similarity calculation failed: {str(e)}")
                    similarities.append(0.0)
            coherence_score = sum(similarities) / len(similarities) if similarities else 0.0

        # Update metrics with coherence
        metrics.update({
            'coherence': coherence_score,
            'context_stability': context_attention.mean().item() if isinstance(context_attention, torch.Tensor) else 0.0
        })

        # Update state history with proper shape
        if len(self.state_history) >= 10:
            self.state_history.pop(0)
        self.state_history.append(new_state.detach().mean(dim=0) if new_state.dim() > 2 else new_state.detach())

        # Update current state
        self.current_state = new_state.detach().mean(dim=0, keepdim=True)

        # Ensure all required metrics are present before returning
        required_metrics = ['memory_state', 'attention_weights', 'phi', 'attention_maps']
        for metric in required_metrics:
            if metric not in metrics:
                metrics[metric] = torch.tensor(0.0) if metric != 'attention_maps' else {}

        try:
            # Add performance monitoring
            performance_metric = metrics['coherence']
            self.performance_history = torch.cat([
                self.performance_history[1:],
                torch.tensor([performance_metric])
            ])
            
            # Apply adaptive learning
            adaptation = self.adaptive_learning_step(
                new_state,
                1.0 - performance_metric
            )
            new_state = new_state + adaptation
            
            # Add performance stats to metrics
            metrics.update({
                'adaptation_rate': self.adaptation_rate.item(),
                'average_performance': self.performance_history[-100:].mean().item(),
                'performance_trend': (self.performance_history[-100:] - 
                                    self.performance_history[-200:-100]).mean().item()
            })
            
        except Exception as e:
            print(f"Warning: Adaptation step failed: {str(e)}")
            # Provide default metrics even if adaptation fails
            metrics.update({
                'adaptation_rate': 0.1,
                'average_performance': 0.5,
                'performance_trend': 0.0
            })

        # Update metrics with proper bounds and required fields
        metrics.update({
            'patterns': self.meta_learner['pattern_recognition'](new_state).detach(),
            'pattern_confidence': torch.sigmoid(rule_embed.norm(dim=-1)).mean().item(),
            'coherence': max(min(coherence_score, 1.0), 0.0),  # Ensure coherence is bounded
            'context_stability': context_attention.mean().item() if isinstance(context_attention, torch.Tensor) else 0.0
        })

        try:
            # Performance monitoring with bounded metrics
            performance_metric = min(max(metrics['coherence'], 0.0), 1.0)
            self.performance_history = torch.cat([
                self.performance_history[1:],
                torch.tensor([performance_metric], device=self.performance_history.device)
            ])
            
            # Apply adaptive learning
            adaptation = self.adaptive_learning_step(
                new_state.detach(),
                1.0 - performance_metric
            )
            new_state = new_state + adaptation
            
            # Add performance stats to metrics
            metrics.update({
                'adaptation_rate': float(self.adaptation_rate.mean().item()),
                'average_performance': float(self.performance_history[-100:].mean().item()),
                'performance_trend': float((self.performance_history[-100:] - 
                                         self.performance_history[-200:-100]).mean().item())
            })
            
        except Exception as e:
            print(f"Warning: Adaptation step failed: {str(e)}")
            # Provide default metrics even if adaptation fails
            metrics.update({
                'adaptation_rate': 0.1,
                'average_performance': 0.5,
                'performance_trend': 0.0
            })

        # Ensure output requires gradients
        if not deterministic:
            state = state.detach().requires_grad_(True)

        # Keep original shape for state history
        batch_state = new_state.clone()
        # Store mean across batch dimension for history
        self.state_history = self.state_history[-10:] + [new_state.mean(dim=0, keepdim=True).detach()]

        return new_state, metrics  # Detach output state

    def get_state(self):
        """Get current model state with proper shape"""
        if not hasattr(self, 'current_state'):
            self.current_state = torch.zeros(1, self.hidden_dim)
        return self.current_state.clone()  # Return a clone to prevent modifications

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
