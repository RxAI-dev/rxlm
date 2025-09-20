import torch
import torch.nn as nn
import torch.nn.functional as F

from .ff import FeedForward, GatedFeedForward

class MoeRouter(nn.Module):
    """Mixture-of-Experts Router layer - computes routing weights for each expert."""

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1, *args, **kwargs):
        super(MoeRouter, self).__init__(*args, **kwargs)
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # For expert load balancing
        self.register_buffer('aux_loss', torch.tensor(0.0), persistent=False)

    # def calculate_aux_loss(self, top_k_indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    #     expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
    #     expert_usage = expert_mask.sum(dim=0).mean(dim=0)
    #     mean_probs = probs.mean(dim=0)
    #     return (expert_usage * mean_probs).sum() * self.num_experts

    def calculate_aux_loss(self, top_k_indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # Get shapes
        T, K = top_k_indices.size()  # Batch, Sequence length, Top-K

        # 1. Compute expert selection mask (one-hot encoded)
        expert_mask = F.one_hot(top_k_indices, self.num_experts).to(dtype=probs.dtype)  # (B, S, K, E)

        # 2. Total number of times each expert is selected
        expert_usage = expert_mask.sum(dim=(0, 1))  # (E,)

        # 3. Fraction of tokens assigned to each expert
        total_selections = T * K
        fraction_expert = expert_usage / (total_selections + 1e-6)  # (E,)

        # 4. Sum of probabilities for each expert's selected tokens
        probs_expanded = probs.unsqueeze(1).expand(-1, K, -1)  # (B_K, K, E)
        sum_probs = (probs_expanded * expert_mask).sum(dim=(0, 1))

        # 5. Average probability per expert (avoid division by zero)
        avg_probs = sum_probs / expert_usage.clamp(min=1e-6)  # (E,)

        # 6. Compute load balancing loss
        loss = (fraction_expert * avg_probs).sum() * self.num_experts

        return loss

    def forward(self, x: torch.Tensor):
        # Input shape: [batch*seq_len, embed_dim]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        # Get top-k experts for each token
        top_k_weights, top_k_indices = probs.topk(self.top_k, dim=-1)

        # Normalize weights (sum to 1 for each token)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        # Load Balance Loss
        if self.training:
            self.aux_loss = self.calculate_aux_loss(top_k_indices, probs)

        return top_k_weights, top_k_indices


class MoeFeedForward(nn.Module):
    """Mixture-of-Experts Feed-Forward layer - combines multiple experts into a single model."""

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
        super(MoeFeedForward, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = 1.5

        self.router = MoeRouter(embed_dim, num_experts, top_k)

        # Batch all expert parameters together
        self._init_experts(num_experts, embed_dim, hidden_dim, activation, dropout)

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        self.experts = nn.ModuleList([
            FeedForward(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])

    def router_loss(self):
        return self.router.aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original shape for final step
        orig_shape = x.size()
        # Flatten input sequence
        x = x.view(-1, self.embed_dim)
        num_tokens, embed_dim = x.size()

        # === STEP 1: ROUTING ===
        # Get routing weights and selected experts from router
        # routing_weights: [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]
        routing_weights, selected_experts = self.router(x)

        # === STEP 2: CREATE DISPOSE MAP ===
        # Flatten experts weights and indices.
        flat_selected_experts = selected_experts.view(-1)
        flat_routing_weights = routing_weights.view(-1)

        # Create original token indices tensor
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)

        # === STEP 3: PERMUTE ===
        # Create permute map by sorting flattened selected experts.
        sorted_expert_indices, sorted_order = flat_selected_experts.sort(0)

        # Permute and reorganize tokens
        permuted_token_indices = token_indices[sorted_order]
        permuted_routing_weights = flat_routing_weights[sorted_order]
        # Reorganize flattened input
        dispatched_x = x[permuted_token_indices]

        # === STEP 4: BATCH EXPERT PROCESSING ===
        # Calculate number of tokens per expert
        tokens_per_expert = F.one_hot(sorted_expert_indices, num_classes=self.num_experts).sum(dim=0)

        # Create expert outputs list and start token idx (from 0)
        expert_outputs = []
        start_idx = 0
        # Efficient expert loop
        for i in range(self.num_experts):
            num_tokens_for_expert = tokens_per_expert[i].item()
            if num_tokens_for_expert == 0:
                continue # Skip empty experts

            # Get input tokens for expert
            end_idx = start_idx + num_tokens_for_expert
            expert_input = dispatched_x[start_idx:end_idx]
            # Process input with expert feed forward
            expert_output = self.experts[i](expert_input)
            expert_outputs.append(expert_output)
            start_idx = end_idx

        # Concatenate expert results
        concatenated_outputs = torch.cat(expert_outputs, dim=0)

        # === STEP 5: REVERSE PERMUTATION AND COMBINE RESULTS ===
        # Apply routing weights to expert outputs
        weighted_outputs = concatenated_outputs * permuted_routing_weights.unsqueeze(1)

        # Create empty output tensor
        final_output = torch.zeros_like(x)

        # Create reverse output map
        inverse_sorted_order = sorted_order.argsort(0)

        # Reversed permutation for weighted outputs
        unpermuted_outputs = weighted_outputs[inverse_sorted_order]

        # Create final indices for scatter add operation
        scatter_indices = token_indices.unsqueeze(1).expand(-1, embed_dim)

        # Allocate results in final tensor with scatter add
        final_output.scatter_add_(0, scatter_indices, unpermuted_outputs)

        # Get final output to initial shape
        return final_output.view(orig_shape)


class GatedMoeFeedForward(MoeFeedForward):
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
        super(GatedMoeFeedForward, self).__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=activation,
            top_k=top_k,
            dropout=dropout,
            *args,
            **kwargs
        )

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        self.experts = nn.ModuleList([
            GatedFeedForward(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])
