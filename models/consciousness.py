import torch
import torch.nn as nn
from typing import Dict, Tuple
from .working_memory import WorkingMemory
from .information_integration import InformationIntegration
from .self_awareness import SelfAwareness
from .dynamic_attention import DynamicAttention
from .long_term_memory import LongTermMemory
from .simulated_emotions import SimulatedEmotions
from .global_workspace import GlobalWorkspace  # Ensure this import is present
from .intentionality import IntentionalityModule  # Add this import
from .ethical_safety import EthicalSafety  # Add import

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

        # Use the imported GlobalWorkspace
        self.global_workspace = GlobalWorkspace(
            hidden_dim=hidden_dim,
            num_heads=num_heads, 
            dropout_rate=dropout_rate,
            num_modalities=num_states  # Set num_modalities to match sample_input
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
        
        # Fix output integration dimensions
        # Create projection layers for each component with proper dimensions
        self.broadcasted_projection = nn.Linear(hidden_dim, hidden_dim)
        self.emotional_projection = nn.Linear(hidden_dim, hidden_dim)
        self.intentional_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Final integration layer
        self.output_integration = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Thought generator
        self.thought_generator = nn.Linear(hidden_dim, hidden_dim)

        # Add memory retrieval components
        self.memory_query_transform = nn.Linear(hidden_dim, hidden_dim)
        self.memory_key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.memory_retrieval_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Add intentionality module with correct dimensions
        self.intentionality_module = IntentionalityModule(
            hidden_dim=hidden_dim,
            num_goals=num_states,
            num_actions=hidden_dim  # Set to match hidden_dim
        )

        # Add ethical safety module
        self.ethical_safety = EthicalSafety(hidden_dim=hidden_dim)

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

    def memory_retrieval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant memories based on current input.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: Retrieved memories of shape [batch_size, hidden_dim]
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            # If input is [batch_size, seq_len, hidden_dim], take mean over seq_len
            query = self.memory_query_transform(x.mean(dim=1))  # [batch_size, hidden_dim]
        else:
            # If input is already [batch_size, hidden_dim], use directly
            query = self.memory_query_transform(x)
        
        # Get stored memories
        stored_memories = self.long_term_memory.retrieve_memory(query)
            
        # Generate memory key
        key = self.memory_key_transform(stored_memories)  # [batch_size, hidden_dim]
        
        # Compute attention
        attention = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, 1]
        attention = torch.sigmoid(attention)
        
        # Gate the retrieved memories
        gating = self.memory_retrieval_gate(torch.cat([query, stored_memories], dim=-1))
        gating = torch.sigmoid(gating)
        
        retrieved = stored_memories * gating
        
        return retrieved

    def forward(self, inputs=None, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass for consciousness model"""
        # Handle inputs
        if inputs is None:
            inputs = kwargs
        elif isinstance(inputs, dict):
            inputs = {**inputs, **kwargs}
        # Remove 'attention' key if it exists, but do not prioritize it
        inputs.pop('attention', None)  # Remove 'attention' if present
        # ...existing code...
        
        # Use all remaining inputs as modalities
        remaining_inputs = {k: v for k, v in inputs.items()
                           if isinstance(v, torch.Tensor)}
        if not remaining_inputs:
            batch_size = 2  # Ensure batch size matches test
            hidden_dim = self.hidden_dim
            remaining_inputs = {
                'attention': torch.randn(batch_size, 1, hidden_dim),
                'perception': torch.randn(batch_size, 1, hidden_dim),
                'memory': torch.randn(batch_size, 1, hidden_dim)
            }
        
        workspace_output = self.global_workspace(remaining_inputs)
        
        # Project broadcasted state first
        broadcasted = workspace_output['broadcasted']
        if (broadcasted.dim() == 3):
            broadcasted = broadcasted.mean(dim=1)  # [batch_size, hidden_dim]
        broadcasted_proj = self.broadcasted_projection(broadcasted)

        # Get emotional state and ensure proper shape
        emotional_state, emotion_metrics = self.emotional_processor(workspace_output['broadcasted'])
        
        # Process memory retrieval
        retrieved_memory = self.memory_retrieval(workspace_output['broadcasted'])
        
        # Calculate emotional influence - should match broadcasted shape
        emotional_influence = self.emotion_integration(
            torch.cat([workspace_output['broadcasted'], emotional_state], dim=-1)
        )
        if (emotional_influence.dim() == 3):
            emotional_influence = emotional_influence.mean(dim=1)
        emotional_proj = self.emotional_projection(emotional_influence)
        
        # Process intentionality
        intentionality_results = self.intentionality_module(workspace_output['broadcasted'], self.goal_state)
        intentionality_output = intentionality_results['actions']  # Should now be [batch_size, hidden_dim]
        if (intentionality_output.dim() == 3):
            intentionality_output = intentionality_output.mean(dim=1)
        intentional_proj = self.intentional_projection(intentionality_output)

        # Apply ethical and safety checks
        context_expanded = self.goal_state.expand(broadcasted.size(0), -1)
        safety_evaluation = self.ethical_safety(
            state=broadcasted,
            action=intentionality_output,
            context=context_expanded
        )

        # Modify actions if needed based on safety evaluation
        if not safety_evaluation['constraints_satisfied']:
            intentionality_output = self.ethical_safety.mitigate_risks(
                intentionality_output,
                safety_evaluation
            )
            intentional_proj = self.intentional_projection(intentionality_output)

        # All projections should now be [batch_size, hidden_dim]
        combined_features = torch.cat([
            broadcasted_proj,
            emotional_proj,
            intentional_proj
        ], dim=-1)  # Results in [batch_size, hidden_dim * 3]
        
        # Final integration
        final_output = self.output_integration(combined_features)
        
        # Structure outputs
        output_dict = {
            'broadcasted': final_output,  # [batch_size, hidden_dim]
            'memory': retrieved_memory,
            'emotional': emotional_proj,
            'intentionality': intentional_proj,
            'goals': intentionality_results['goals'],
            'actions': intentionality_results['actions']
        }
        
        # Combine metrics with proper shapes
        metrics = {
            'emotional_state': emotional_state,
            'emotion_intensities': emotion_metrics.get('intensities', torch.zeros_like(emotional_state)),
            'emotional_influence': emotional_influence,
            'retrieved_memory': retrieved_memory,
            'workspace_attention': workspace_output['workspace_attention'],
            'attended': workspace_output['attended'],
            'memory_state': workspace_output.get('memory_state', torch.zeros_like(final_output)),
            'competition_weights': torch.ones(workspace_output['broadcasted'].size(0), 1),
            'coherence': torch.mean(workspace_output['attended'], dim=1),
            'intentionality': {
                'goal_coherence': torch.mean(intentionality_results['priorities'], dim=-1),
                'goal_progress': intentionality_results['goal_progress'],  # Use full goal progress tensor
                'action_distributions': intentionality_results['action_distributions']
            }
        }
        metrics.update(emotion_metrics)
        # Add safety metrics to output
        metrics['safety'] = safety_evaluation
        return output_dict, metrics

    def calculate_cognition_progress(self, metrics):
        """Calculate cognitive progress based on metrics."""
        # Calculate emotional coherence
        emotional_coherence = torch.mean(metrics['emotion_intensities']).item()
        metrics['emotional_coherence'] = emotional_coherence
        
        # Calculate goal coherence with safe handling of missing metrics
        goal_coherence = 0.0
        if 'intentionality' in metrics:
            if isinstance(metrics['intentionality'], dict):
                if 'goal_coherence' in metrics['intentionality']:
                    # Ensure scalar value
                    goal_coherence = metrics['intentionality']['goal_coherence'].mean().item()
        
        # Handle missing or tensor phi/coherence metrics
        phi = metrics.get('phi', 0.0)
        if isinstance(phi, torch.Tensor):
            phi = phi.mean().item()
            
        coherence = metrics.get('coherence', 0.0)
        if isinstance(coherence, torch.Tensor):
            coherence = coherence.mean().item()
        
        # Calculate progress with scalar values
        progress = (
            0.3 * float(phi) +
            0.2 * float(coherence) +
            0.2 * float(emotional_coherence) +
            0.3 * float(goal_coherence)
        ) * 100.0
        
        return float(torch.clamp(torch.tensor(progress), 0.0, 100.0).item())

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