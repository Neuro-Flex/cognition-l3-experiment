import torch
import pytest
from models.ethical_safety import EthicalSafety

def test_safety_check():
    ethical_safety = EthicalSafety(hidden_dim=64)
    state = torch.randn(2, 64)
    
    is_safe, metrics = ethical_safety.check_safety(state)
    
    assert isinstance(is_safe, torch.Tensor)
    assert 'safety_score' in metrics
    assert metrics['safety_score'].shape == (2, 1)

def test_ethical_evaluation():
    ethical_safety = EthicalSafety(hidden_dim=64)
    action = torch.randn(2, 64)
    context = torch.randn(2, 64)
    
    is_ethical, metrics = ethical_safety.evaluate_ethics(action, context)
    
    assert isinstance(is_ethical, torch.Tensor)
    assert 'ethics_score' in metrics
    assert metrics['ethics_score'].shape == (2, 1)

def test_risk_mitigation():
    ethical_safety = EthicalSafety(hidden_dim=64)
    action = torch.ones(2, 64)
    
    safety_metrics = {
        'is_safe': False,
        'safety_score': torch.tensor([[0.3], [0.6]])
    }
    
    mitigated_action = ethical_safety.mitigate_risks(action, safety_metrics)
    assert torch.all(mitigated_action < action)
