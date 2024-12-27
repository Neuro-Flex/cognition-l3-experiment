import torch
import pytest
from models.consciousness import ConsciousnessModel

def test_intentionality_integration():
    """Test integration of intentionality module with consciousness model"""
    model = ConsciousnessModel(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        num_states=3
    )
    
    # Create sample inputs
    batch_size = 2
    hidden_dim = 64
    inputs = {
        'visual': torch.randn(batch_size, 1, hidden_dim),
        'textual': torch.randn(batch_size, 1, hidden_dim),
        'memory': torch.randn(batch_size, 1, hidden_dim)
    }
    
    # Process through model
    outputs, metrics = model(inputs)
    
    # Check if intentionality outputs are present
    assert 'intentionality' in outputs
    assert 'intentionality' in metrics
    
    # Verify intentionality metrics structure
    assert 'goal_coherence' in metrics['intentionality']
    assert 'goal_progress' in metrics['intentionality']
    
    # Check shapes
    assert outputs['intentionality'].shape == (batch_size, hidden_dim)
    assert metrics['intentionality']['goal_coherence'].shape == (batch_size,)
    
    # Check shapes after integration
    assert outputs['broadcasted'].shape[-1] == hidden_dim, "Wrong output dimension"
    if outputs['intentionality'].dim() == 3:
        assert outputs['intentionality'].shape[-1] == hidden_dim, "Wrong intentionality dimension"
    else:
        assert outputs['intentionality'].shape[-1] == hidden_dim, "Wrong intentionality dimension"
    
    # Check metrics structure with safe access
    assert 'intentionality' in metrics, "Missing intentionality metrics"
    assert isinstance(metrics['intentionality'], dict), "Intentionality metrics should be a dict"
    assert 'goal_coherence' in metrics['intentionality'], "Missing goal coherence"
    assert 'goal_progress' in metrics['intentionality'], "Missing goal progress"

def test_goal_directed_behavior():
    """Test goal-directed behavior in consciousness model"""
    model = ConsciousnessModel(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        num_states=3
    )
    
    batch_size = 2
    hidden_dim = 64
    
    # Create initial state
    state = torch.randn(batch_size, hidden_dim)
    
    # Set initial goal state
    model.goal_state.data = torch.randn(1, hidden_dim)
    
    # Process multiple steps
    for _ in range(3):
        inputs = {
            'visual': torch.randn(batch_size, 1, hidden_dim),
            'memory': torch.randn(batch_size, 1, hidden_dim)
        }
        outputs, metrics = model(inputs)
        
        # Check if goal state is being updated
        assert not torch.allclose(outputs['intentionality'], state)
        state = outputs['intentionality']
        
        # Verify goal progress is being tracked
        assert 'goal_progress' in metrics['intentionality']
        assert metrics['intentionality']['goal_progress'].shape == (batch_size,)
        assert (metrics['intentionality']['goal_progress'] >= 0).all()
        assert (metrics['intentionality']['goal_progress'] <= 1).all()

def test_cognition_progress_with_intentionality():
    """Test cognition progress calculation including intentionality metrics"""
    model = ConsciousnessModel(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        num_states=3
    )
    
    batch_size = 2
    hidden_dim = 64
    inputs = {
        'visual': torch.randn(batch_size, 1, hidden_dim),
        'memory': torch.randn(batch_size, 1, hidden_dim)
    }
    
    outputs, metrics = model(inputs)
    
    # Calculate cognition progress
    progress = model.calculate_cognition_progress(metrics)
    
    # Verify progress includes intentionality components
    assert 0 <= progress <= 100
    assert 'goal_coherence' in metrics['intentionality']
    assert 'goal_progress' in metrics['intentionality']
    
    # Test progress calculation with various intentionality metrics
    metrics['intentionality']['goal_coherence'] = torch.ones(batch_size)
    metrics['intentionality']['goal_progress'] = torch.ones(batch_size)
    
    improved_progress = model.calculate_cognition_progress(metrics)
    assert improved_progress > progress, "Progress should improve with better goal metrics"
