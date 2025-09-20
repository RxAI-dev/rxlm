import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformers.moe import MoeRouter

class MoeFeedForwardVectorized(nn.Module):
    """
    Vectorized MoE - current implementation is incorrect - it calculates all the experts, then selects the correct ones.

    Commented out implementation is fixing this problem, but is causing memory overflows, because of experts weights
    indexing - it's using ~15x more memory, than dense model of similar size, so it's currently not viable.

    It's recommended to use standard MoE from rxnn.transformers.moe instead.
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_experts: int,
            activation: nn.Module,
            top_k: int = 1,
            dropout: float = 0.0,
            *args,
            **kwargs
    ):
        super(MoeFeedForwardVectorized, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoeRouter(embed_dim, num_experts, top_k)

        # Batch all expert parameters together
        self.w1 = nn.Parameter(torch.empty(num_experts, embed_dim, self._w1_dim_factor(hidden_dim)))
        self.b1 = nn.Parameter(torch.zeros(num_experts, self._w1_dim_factor(hidden_dim)))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, embed_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, embed_dim))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_linear_parameters()
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def _init_linear_parameters(self):
        nn.init.kaiming_normal_(self.w1, nonlinearity='relu')
        nn.init.kaiming_normal_(self.w2, nonlinearity='relu')

    def _w1_dim_factor(self, hidden_dim: int) -> int:
        return hidden_dim

    def _activate(self, h: torch.Tensor):
        return self.activation(h)

    def router_loss(self):
        return self.router.aux_loss

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        x = x.view(-1, self.embed_dim)  # [batch*seq_len, embed_dim]

        # Get routing weights and indices
        weights, indices = self.router(x)  # [batch*seq_len, top_k]

        # Create expert masks and combine it with masks
        mask = F.one_hot(indices, self.num_experts).float()  # [batch*seq_len, top_k, num_experts]
        weights = (weights.unsqueeze(-1) * mask).sum(dim=1)  # [batch*seq_len, num_experts]

        # Expert computation
        x = x.unsqueeze(1).expand(-1, self.num_experts, -1)  # [batch*seq_len, num_experts, embed_dim]

        # First linear layer
        h = torch.einsum('bie,ieh->bih', x, self.w1) + self.b1  # [batch*seq_len, num_experts, hidden_dim]
        h = self._activate(h)
        h = self.dropout(h)

        # Second linear layer (projection back to embed_dim)
        out = torch.einsum('bih,ihe->bie', h, self.w2) + self.b2  # [batch*seq_len, num_experts, embed_dim]

        # Weighted sum of expert outputs
        out = (out * weights.unsqueeze(-1)).sum(dim=1)  # [batch*seq_len, embed_dim]

        return out.view(*orig_shape)
        # orig_shape = x.shape
        # x = x.view(-1, self.embed_dim)  # [batch*seq_len, embed_dim]
        #
        # # Get routing weights and indices
        # weights, indices = self.router(x)  # [B*T, top_k], [B*T, top_k]
        #
        # # Flatten indices and weights
        # batch_size = x.shape[0]
        # top_k = indices.shape[1]
        # indices_flat = indices.view(-1)  # [B*T * top_k]
        #
        # # Compute contributions for selected experts without materializing large tensors
        # # First Layer:
        # # Compute all expert contributions first (but this may still be memory-heavy)
        # # Alternative: Compute contributions for selected experts directly
        # # ... (see detailed steps below)
        #
        # # Alternative approach using gather and batched operations
        # x_expanded = x.unsqueeze(1).repeat(1, top_k, 1).view(-1, self.embed_dim)  # [B*T*top_k, D]
        #
        # # Compute first layer contributions using gather
        # # indices_flat has shape [B*T*top_k]
        # # selected_w1 is self.w1[indices_flat], but we compute the product inline
        # h = torch.einsum(
        #     'be, eih -> bh',
        #     x_expanded,
        #     self.w1[indices_flat]
        # ) + self.b1[indices_flat]
        # h = self._activate(h)
        # h = self.dropout(h)
        #
        # # Second layer:
        # out = torch.einsum(
        #     'bh, eho -> beo',
        #     h,
        #     self.w2[indices_flat]
        # ).squeeze(-1) + self.b2[indices_flat]
        #
        # # Reshape and apply weights
        # out = out.view(batch_size, top_k, -1)
        # weights = weights.view(batch_size, top_k, 1)
        # out = (out * weights).sum(dim=1)
        #
        # return out.view(*orig_shape)


class GatedMoeFeedForwardVectorized(MoeFeedForwardVectorized):
    """Gated Mixture-of-Experts Feed-Forward layer - enable GLU-based activations for MoE"""

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_experts: int,
            activation: nn.Module = nn.SiLU(),
            top_k: int = 1,
            dropout: float = 0.1,
            *args,
            **kwargs
    ):
        super(GatedMoeFeedForwardVectorized, self).__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=activation,
            top_k=top_k,
            dropout=dropout,
            *args,
            **kwargs
        )

    def _init_linear_parameters(self):
        nn.init.kaiming_normal_(self.w1, nonlinearity='relu')
        nn.init.kaiming_normal_(self.w2, nonlinearity='linear')

    def _w1_dim_factor(self, hidden_dim: int) -> int:
        return 2 * hidden_dim

    def _activate(self, h: torch.Tensor):
        a, b = h.chunk(2, dim=-1)
        return a * self.activation(b)
