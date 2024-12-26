import pytest
import torch
import torch.nn as nn
from models.long_term_memory import LongTermMemory

class TestLongTermMemory:
    @pytest.fixture
    def long_term_memory(self):
        return LongTermMemory(input_dim=128, hidden_dim=128, memory_size=1000, dropout_rate=0.1)

    @pytest.fixture
    def sample_input(self):
        batch_size = 2
        seq_len = 5
        hidden_dim = 128
        return torch.randn(batch_size, seq_len, hidden_dim)

    def test_initialization(self, long_term_memory):
        """Test proper initialization of LongTermMemory module"""
        assert isinstance(long_term_memory, nn.Module)
        assert long_term_memory.input_dim == 128
        assert long_term_memory.hidden_dim == 128
        assert long_term_memory.memory_size == 1000
        assert long_term_memory.memory_storage.shape == (1000, 128)

    def test_forward_pass(self, long_term_memory, sample_input):
        """Test complete forward pass"""
        output, (h_n, c_n) = long_term_memory(sample_input)
        
        assert output.shape == sample_input.shape
        assert h_n.shape == (2, sample_input.size(0), 128)
        assert c_n.shape == (2, sample_input.size(0), 128)

    def test_memory_storage(self, long_term_memory, sample_input):
        """Test memory storage functionality"""
        long_term_memory.store_memory(sample_input.mean(dim=1))
        assert long_term_memory.memory_index == sample_input.size(0)
        assert torch.allclose(long_term_memory.memory_storage[:sample_input.size(0)], sample_input.mean(dim=1))

    def test_memory_retrieval(self, long_term_memory, sample_input):
        """Test memory retrieval functionality"""
        long_term_memory.store_memory(sample_input.mean(dim=1))
        query = sample_input.mean(dim=1)
        retrieved_memory = long_term_memory.retrieve_memory(query)
        
        assert retrieved_memory.shape == query.shape
        assert torch.allclose(retrieved_memory, sample_input.mean(dim=1), atol=1e-1)

if __name__ == '__main__':
    pytest.main([__file__])
