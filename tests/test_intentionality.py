import torch
import pytest
from models.intentionality import IntentionalityModule, GoalGenerator

def test_goal_generator():
    batch_size = 4
    hidden_dim = 64
    num_goals = 5
    
    generator = GoalGenerator(hidden_dim, num_goals)
    state = torch.randn(batch_size, hidden_dim)
    
    goals, priorities = generator(state)
    
    assert goals.shape == (batch_size, num_goals, hidden_dim)
    assert priorities.shape == (batch_size, num_goals)
    assert torch.allclose(priorities.sum(dim=-1), torch.ones(batch_size))

def test_intentionality_module():
    batch_size = 4
    hidden_dim = 64
    num_goals = 5
    num_actions = 10
    
    module = IntentionalityModule(hidden_dim, num_goals, num_actions)
    state = torch.randn(batch_size, hidden_dim)
    context = torch.randn(batch_size, hidden_dim)
    
    output = module(state, context)
    
    assert output['goals'].shape == (batch_size, num_goals, hidden_dim)
    assert output['priorities'].shape == (batch_size, num_goals)
    assert output['actions'].shape == (batch_size, num_actions)
    assert output['progress'].shape == (batch_size, num_goals)
    
def test_goal_update():
    batch_size = 4
    hidden_dim = 64
    num_goals = 5
    num_actions = 10
    
    module = IntentionalityModule(hidden_dim, num_goals, num_actions)
    feedback = torch.randn(batch_size, 1)
    goals = torch.randn(batch_size, num_goals, hidden_dim)
    priorities = torch.softmax(torch.randn(batch_size, num_goals), dim=-1)
    
    new_goals, new_priorities = module.update_goals(feedback, goals, priorities)
    
    assert new_goals.shape == goals.shape
    assert new_priorities.shape == priorities.shape
    assert torch.allclose(new_priorities.sum(dim=-1), torch.ones(batch_size))
