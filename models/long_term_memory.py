import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LongTermMemory(nn.Module):
    """Long-term memory component for maintaining and recalling episodic information."""
    
    def __init__(self, input_dim: int, hidden_dim: int, memory_size: int = 1000, dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Memory cells
        self.memory_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Initialize memory storage with correct device and shape
        self.register_buffer('memory_storage', torch.zeros(memory_size, hidden_dim))
        self.memory_index = 0
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def store_memory(self, memory: torch.Tensor):
        """Store memory in the long-term memory storage."""
        # Add assertion to ensure memory has shape [batch_size, hidden_dim]
        assert memory.dim() == 2 and memory.size(1) == self.hidden_dim, (
            f"Memory has shape {memory.shape}, expected [batch_size, {self.hidden_dim}]"
        )
        
        batch_size = memory.size(0)
        
        # Store exact memory values without normalization
        if self.memory_index + batch_size > self.memory_size:
            overflow = (self.memory_index + batch_size) - self.memory_size
            self.memory_storage[self.memory_index:] = memory[:batch_size - overflow].detach()
            self.memory_storage[:overflow] = memory[batch_size - overflow:].detach()
            self.memory_index = overflow
        else:
            self.memory_storage[self.memory_index:self.memory_index + batch_size] = memory.detach()
            self.memory_index += batch_size

    def retrieve_memory(self, query):
        """Retrieve relevant memories based on query."""
        try:
            # Ensure query has correct shape [batch_size, hidden_dim]
            if query.dim() == 1:
                query = query.unsqueeze(0)
            elif query.dim() > 2:
                query = query.view(-1, self.hidden_dim)
            
            batch_size = query.size(0)
            
            # Handle empty memory case
            if self.memory_index == 0:
                return query  # Return query itself if no memories stored
            
            # Get valid memories
            if self.memory_index < self.memory_size:
                valid_memories = self.memory_storage[:self.memory_index]
            else:
                valid_memories = self.memory_storage
            
            # Ensure we have at least one memory
            if valid_memories.size(0) == 0:
                return query
            
            # Normalize for similarity computation only
            query_norm = F.normalize(query, p=2, dim=1)
            memories_norm = F.normalize(valid_memories, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.matmul(query_norm, memories_norm.t())
            
            # Get attention weights through softmax
            attention = F.softmax(similarity / 0.1, dim=1)  # Temperature scaling
            
            # Use original memories for weighted sum
            retrieved = torch.matmul(attention, valid_memories)
            
            return retrieved

        except Exception as e:
            print(f"Memory retrieval error: {e}")
            return query.clone()  # Return query itself in case of error
        
    def forward(self, x):
        # Run LSTM
        output, (h_n, c_n) = self.memory_rnn(x)
        
        # Project output
        output = self.output_projection(output)
        output = self.layer_norm(output)
        
        # Retrieve memory ensuring batch size consistency
        retrieved_memory = self.retrieve_memory(x)
        # Ensure retrieved_memory has the same batch size as input
        retrieved_memory = retrieved_memory.view(x.size(0), -1)
        metrics = {'retrieved_memory': retrieved_memory}
        
        return output, (h_n, c_n)
