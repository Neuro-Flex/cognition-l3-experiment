import torch
import torch.nn as nn
from dataclasses import dataclass

class BaseAttention(nn.Module):
    """Multi-head attention mechanism implemented in PyTorch."""
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, training=True):
        batch_size = x.size(0)
        head_dim = self.hidden_size // self.num_heads

        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.num_heads, head_dim)
        k = self.key(x).view(batch_size, -1, self.num_heads, head_dim)
        v = self.value(x).view(batch_size, -1, self.num_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        
        if training:
            weights = self.dropout(weights)

        # Compute weighted sum
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.hidden_size)
        
        return self.output(output)

class BaseModel(nn.Module):
    """Base transformer model implemented in PyTorch."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        self.encoder_layers = nn.ModuleList([
            BaseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_rate=config.dropout_rate
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask=None, training=True):
        assert input_ids.dim() == 2, "input_ids must be of shape (batch_size, seq_len)"
        
        x = self.embeddings(input_ids)

        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape, "attention_mask shape must match input_ids"

        for layer in self.encoder_layers:
            x = layer(x, mask=attention_mask, training=training)

        pooled = self.pooler(x[:, 0])
        return x, pooled

def create_train_state(model: BaseModel, config):
    """Creates optimizer with training settings."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.max_steps,
        T_mult=1,
        eta_min=0
    )

    return optimizer, scheduler
