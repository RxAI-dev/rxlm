import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Basic Feed-forward layer with activation function and optional dropout"""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float = 0.0, *args, **kwargs):
        super(FeedForward, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)


class GatedLinearUnit(nn.Module):
    """Gated linear unit layer with configurable activation (SwiGLU, ReGLU, etc.)"""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: nn.Module, *args, **kwargs):
        super(GatedLinearUnit, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(embed_dim, hidden_dim * 2)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l, g = self.linear(x).chunk(2, dim=-1)
        return l * self.activation(g)


class GatedFeedForward(nn.Module):
    """Gated feed-forward layer with activation function and optional dropout"""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float = 0.0, *args, **kwargs):
        super(GatedFeedForward, self).__init__(*args, **kwargs)
        self.fc1 = GatedLinearUnit(embed_dim, hidden_dim, activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)


def get_activation_layer(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu' or activation == 'swish':
        return nn.SiLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f'Activation {activation} not supported')
