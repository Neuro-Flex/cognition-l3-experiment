import torch
import torch.nn as nn
from typing import Tuple

class InformationIntegration(nn.Module):
    """Module for integrating information across different cognitive processes."""
    
    def __init__(self, hidden_dim: int, num_modules: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        
        # Integration layers
        self.integration_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Phi calculation network
        self.phi_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for information integration.
        Returns integrated information and phi value.
        """
        # Set dropout behavior
        self.train(not deterministic)
        
        # Integrate information
        integrated = self.integration_network(x)
        
        # Calculate phi (information integration measure)
        phi = self.phi_network(integrated)
        
        return integrated, phi
