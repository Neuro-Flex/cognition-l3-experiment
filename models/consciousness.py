import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .working_memory import WorkingMemory
from .information_integration import InformationIntegration
from .self_awareness import SelfAwareness  # Add this import
from .dynamic_attention import DynamicAttention
from .long_term_memory import LongTermMemory
from .simulated_emotions import SimulatedEmotions

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

        # Add assertion to ensure output has correct hidden_dim
        assert output.shape[-1] == self.hidden_dim, (
            f"GlobalWorkspace output has hidden_dim {output.shape[-1]}, expected {self.hidden_dim}"
        )
        
        # Add logging for debugging
        print(f"GlobalWorkspace output shape: {output.shape}")
        print(f"memory_state shape: {memory_state.shape}")

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

        # Replace standard attention with dynamic attention
        self.attention = DynamicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Add self-awareness module
        self.self_awareness = SelfAwareness(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Long-term memory
        self.long_term_memory = LongTermMemory(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            memory_size=1000,
            dropout_rate=dropout_rate
        )
        
        # State tracking
        self.previous_state = None

        # Add goal tracking
        self.goal_state = nn.Parameter(torch.randn(1, hidden_dim))
        self.goal_updater = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Context tracking
        self.context_state = None
        self.context_integrator = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Add emotional processing component
        self.emotional_processor = SimulatedEmotions(hidden_dim=hidden_dim)
        
        # Add emotion integration layer
        self.emotion_integration = nn.Linear(hidden_dim * 2, hidden_dim)

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
        config = {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'num_states': 4,
            'dropout_rate': 0.1,
            'emotion_decay': 0.95,
            'emotion_threshold': 0.3
        }
        return config

    def calculate_phi(self, conscious_output):
        """Calculate information integration metric (phi)"""
        return torch.abs(self.phi_calculator(conscious_output)).mean()

    def calculate_energy_cost(self, cognitive_outputs):
        """Calculate energy cost of processing"""
        return torch.abs(self.energy_tracker(torch.mean(cognitive_outputs, dim=0))).mean()

    def update_goals(self, current_state: torch.Tensor):
        """Update goal state based on current conscious state"""
        batch_size = current_state.size(0)
        expanded_goals = self.goal_state.expand(batch_size, -1)
        self.goal_state = nn.Parameter(
            self.goal_updater(current_state, expanded_goals)
        )

    def forward(self, inputs: Dict[str, torch.Tensor],
                state: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> Tuple[torch.Tensor, Dict]:
        # Initialize metrics dictionary at the start
        metrics = {}
        
        # Get device from inputs
        device = next(iter(inputs.values())).device
        
        # Initialize state if None
        if state is None:
            state = torch.zeros(inputs['attention'].shape[0], self.hidden_dim, device=device)
            
        # Get input tensor
        x = inputs['attention']  # [batch_size, seq_len, hidden_dim]
        
        # Apply dynamic attention with goals and context
        attn_out, attention_metrics = self.attention(
            x, x, x,
            goals=self.goal_state.expand(x.size(0), -1),
            context=self.context_state
        )
        
        # Update context state
        if self.context_state is None:
            self.context_state = attn_out.mean(dim=1)
        else:
            self.context_state = self.context_integrator(
                torch.cat([self.context_state, attn_out.mean(dim=1)], dim=-1)
            )
        
        # Process through global workspace with reshaped state
        conscious_out, memory_state = self.global_workspace(attn_out, state, deterministic)
        
        # Add assertion to ensure conscious_out has correct hidden_dim
        assert conscious_out.shape[-1] == self.hidden_dim, (
            f"conscious_out has hidden_dim {conscious_out.shape[-1]}, expected {self.hidden_dim}"
        )
        
        # Add logging to verify conscious_out dimensions
        print(f"conscious_out shape: {conscious_out.shape}")
        
        # Process through self-awareness
        aware_state, awareness_metrics = self.self_awareness(
            conscious_out,
            previous_state=self.previous_state
        )
        
        # Add assertion to ensure aware_state has correct hidden_dim
        assert aware_state.shape[-1] == self.hidden_dim, (
            f"aware_state has hidden_dim {aware_state.shape[-1]}, expected {self.hidden_dim}"
        )
        
        # Update previous state
        self.previous_state = aware_state.detach()
        
        # Calculate integration metrics
        integrated_out, phi = self.information_integration(conscious_out, deterministic)
        
        # Update goals based on conscious output
        self.update_goals(conscious_out)
        
        # Store memory with correct dimensions
        memory_to_store = conscious_out.detach()  # Remove mean reduction
        
        # Use long_term_memory instead of memory
        try:
            # Ensure memory_to_store has correct shape [batch_size, hidden_dim]
            memory_to_store = conscious_out.mean(dim=1) if len(conscious_out.shape) == 3 else conscious_out
            
            # Store memory
            self.long_term_memory.store_memory(memory_to_store)
            
            # Retrieve memory using current state as query
            retrieved_memory = self.long_term_memory.retrieve_memory(memory_to_store)
            
            # Ensure retrieved memory has correct shape
            if retrieved_memory.shape != (memory_to_store.shape[0], self.hidden_dim):
                retrieved_memory = retrieved_memory.view(memory_to_store.shape[0], self.hidden_dim)
                
            metrics['retrieved_memory'] = retrieved_memory
            
        except Exception as e:
            print(f"Memory operation error: {e}")
            # Create zero tensor with correct shape
            metrics['retrieved_memory'] = torch.zeros(
                inputs['attention'].shape[0], 
                self.hidden_dim, 
                device=inputs['attention'].device
            )
        
        # Average over sequence length to get [batch_size, hidden_dim] 
        query = conscious_out.mean(dim=1) if len(conscious_out.shape) > 2 else conscious_out
        print(f"query shape: {query.shape}")
        
        # Ensure query has correct shape before memory retrieval
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Retrieve memory and ensure it's in metrics
        try:
            retrieved_memory = self.long_term_memory.retrieve_memory(query)
            print(f"retrieved_memory shape: {retrieved_memory.shape}")
            metrics['retrieved_memory'] = retrieved_memory
        except Exception as e:
            print(f"Memory retrieval error: {e}")
            metrics['retrieved_memory'] = torch.zeros(
                query.size(0), 
                self.hidden_dim, 
                device=query.device
            )
        
        # Process through emotional system
        emotional_state, emotion_metrics = self.emotional_processor(conscious_out)
        
        # Integrate emotional influence
        combined = torch.cat([conscious_out, emotional_state], dim=-1)
        integrated_state = self.emotion_integration(combined)
        
        # Update metrics
        metrics.update({
            'emotional_state': emotional_state,
            'emotion_intensities': emotion_metrics['emotion_intensities'],
            'emotional_influence': emotion_metrics['emotional_influence']
        })
        
        # Update remaining metrics
        metrics.update(attention_metrics)
        metrics['goal_state'] = self.goal_state
        metrics['context_state'] = self.context_state
        metrics['phi'] = phi
        
        return aware_state, metrics

    def calculate_cognition_progress(self, metrics):
        """
        Calculate cognitive progress based on metrics.
        Returns a value between 0 and 100.
        """
        # Calculate emotional coherence
        emotional_coherence = torch.mean(metrics['emotion_intensities']).item()
        metrics['emotional_coherence'] = emotional_coherence
        
        # Calculate overall progress using phi, coherence and emotional_coherence
        progress = (
            0.4 * metrics['phi'] +
            0.3 * metrics['coherence'] +
            0.3 * emotional_coherence
        ) * 100
        
        return max(0, min(100, progress))  # Ensure result is between 0 and 100

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
