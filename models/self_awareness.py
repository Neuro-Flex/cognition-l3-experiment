import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class SelfAwareness(nn.Module):
    """Module for implementing self-awareness, monitoring, and representation."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-representation components
        self.self_embed = nn.Linear(hidden_dim, hidden_dim)
        self.state_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Self-monitoring components
        self.monitor = nn.ModuleDict({
            'attention': nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate),
            'state_tracker': nn.Linear(hidden_dim * 2, hidden_dim),
            'anomaly_detector': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # Metacognitive components
        self.metacognition = nn.ModuleDict({
            'confidence': nn.Linear(hidden_dim, 1),
            'error_prediction': nn.Linear(hidden_dim, hidden_dim),
            'adaptation_net': nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # Store adaptation rate as buffer instead of parameter
        self.register_buffer('adaptation_rate', torch.tensor(0.1))
        
        self.history_size = 1000
        self.state_history = []
        
    def update_state_history(self, state: torch.Tensor):
        """Maintain a history of internal states."""
        self.state_history.append(state.detach())
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
            
    def compute_self_representation(self, current_state: torch.Tensor) -> torch.Tensor:
        """Generate self-representation from current state."""
        self_rep = self.self_embed(current_state)
        historical_context = None
        
        if self.state_history:
            historical_tensor = torch.stack(self.state_history[-10:], dim=1)
            historical_context, _ = self.state_encoder(historical_tensor)
            historical_context = historical_context[:, -1, :]  # Take last state
            
        if historical_context is not None:
            self_rep = self_rep + 0.1 * historical_context
            
        return self_rep
        
    def monitor_state(self, current_state: torch.Tensor, 
                     previous_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Monitor internal state and detect anomalies."""
        # Compare current state with previous if available
        if previous_state is None:
            previous_state = torch.zeros_like(current_state)
            
        # Attend to important aspects of state
        attended_state, _ = self.monitor['attention'](
            current_state.unsqueeze(0), 
            current_state.unsqueeze(0), 
            current_state.unsqueeze(0)
        )
        attended_state = attended_state.squeeze(0)
        
        # Track state changes
        state_diff = self.monitor['state_tracker'](
            torch.cat([current_state, previous_state], dim=-1)
        )
        
        # Calculate state magnitude for anomaly detection
        state_magnitude = torch.norm(current_state, dim=-1, keepdim=True)
        normalized_state = current_state / (state_magnitude + 1e-6)
        
        # Detect anomalies based on normalized state
        anomaly_score = self.monitor['anomaly_detector'](normalized_state)
        
        return {
            'attended_state': attended_state,
            'state_change': state_diff,
            'anomaly_score': anomaly_score
        }
        
    def assess_metacognition(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Assess metacognitive aspects like confidence and error prediction."""
        # Normalize state for more stable confidence estimation
        state_magnitude = torch.norm(state, dim=-1, keepdim=True)
        normalized_state = state / (state_magnitude + 1e-6)
        
        # Calculate confidence based on normalized state
        confidence = torch.sigmoid(self.metacognition['confidence'](normalized_state))
        confidence = confidence * torch.exp(-state_magnitude / 100)  # Reduce confidence for extreme values
        
        error_pred = self.metacognition['error_prediction'](state)
        
        return {
            'confidence': confidence,
            'error_prediction': error_pred,
            'adaptation_rate': self.adaptation_rate
        }
        
    def forward(self, current_state: torch.Tensor, 
                previous_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Process current state through self-awareness mechanisms."""
        # Update state history
        self.update_state_history(current_state)
        
        # Generate self representation
        self_rep = self.compute_self_representation(current_state)
        
        # Monitor state
        monitoring_results = self.monitor_state(current_state, previous_state)
        
        # Assess metacognition
        metacog_results = self.assess_metacognition(self_rep)
        
        # Combine all metrics
        metrics = {
            'self_representation': self_rep,
            **monitoring_results,
            **metacog_results
        }
        
        # Update based on monitoring and metacognition
        updated_state = current_state + \
                       (monitoring_results['attended_state'] * metacog_results['adaptation_rate'])
        
        return updated_state, metrics
