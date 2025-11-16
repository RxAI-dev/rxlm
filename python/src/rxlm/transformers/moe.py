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

    @torch._dynamo.disable
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
            num_shared_experts: int = 0,  # CHANGED: Added shared experts parameter
            router_amp: bool = False,
            router_dtype: torch.dtype = torch.float32,
            *args,
            **kwargs
    ):
        super(MoeFeedForward, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts  # CHANGED: Store number of shared experts
        self.router_amp = router_amp
        self.router_dtype = router_dtype

        self.router = MoeRouter(embed_dim, num_experts, top_k)

        self.experts = self._init_experts(num_experts, embed_dim, hidden_dim, activation, dropout)

        # CHANGED: Initialize shared experts that are always activated
        if num_shared_experts > 0:
            self.shared_experts = self._init_experts(num_shared_experts, embed_dim, hidden_dim, activation, dropout)

            # CHANGED: For multiple shared experts, use learned weighting via small gating network
            # This prevents numeric overflow and allows the model to balance shared expert contributions
            if num_shared_experts > 1:
                self.shared_expert_gate = nn.Linear(embed_dim, num_shared_experts, bias=False)

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        return nn.ModuleList([
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
        if self.router_amp:
            with torch.amp.autocast(device_type=x.device.type, dtype=self.router_dtype):
                routing_weights, selected_experts = self.router(x)
        else:
            routing_weights, selected_experts = self.router(x)

        # CHANGED: Fast path for single-token processing (autoregressive generation)
        # When processing one token at a time, avoid complex permutation overhead
        if num_tokens == 1:
            # Simple loop over top-k experts for single token
            final_output = torch.zeros_like(x)
            for k in range(self.top_k):
                expert_idx = selected_experts[0, k]
                weight = routing_weights[0, k]
                expert_output = self.experts[expert_idx](x)
                final_output += weight * expert_output
        else:
            # CHANGED: Original batched processing path for multi-token sequences (prompt phase)
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
                num_tokens_for_expert = tokens_per_expert[i]
                if num_tokens_for_expert == 0:
                    continue  # Skip empty experts

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
            final_output.scatter_add_(0, scatter_indices, unpermuted_outputs.to(dtype=final_output.dtype))

        # CHANGED: Add shared expert outputs (if any)
        # Shared experts are applied to all tokens without routing
        if self.num_shared_experts > 0:
            if self.num_shared_experts == 1:
                # Single shared expert: directly add to output
                shared_output = self.shared_experts[0](x)
                final_output = final_output + shared_output
            else:
                # Multiple shared experts: use learned weighted mean
                # Compute gating weights for shared experts
                shared_gate_logits = self.shared_expert_gate(x)  # [num_tokens, num_shared_experts]
                shared_weights = F.softmax(shared_gate_logits, dim=-1)  # [num_tokens, num_shared_experts]

                # Compute all shared expert outputs
                shared_outputs = torch.stack([
                    expert(x) for expert in self.shared_experts
                ], dim=1)  # [num_tokens, num_shared_experts, embed_dim]

                # Apply weighted mean
                shared_combined = (shared_outputs * shared_weights.unsqueeze(-1)).sum(dim=1)  # [num_tokens, embed_dim]
                final_output = final_output + shared_combined

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
            num_shared_experts: int = 0,  # CHANGED: Added shared experts parameter
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
            num_shared_experts=num_shared_experts,  # CHANGED: Pass through shared experts
            *args,
            **kwargs
        )

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        # CHANGED: Use GatedFeedForward for routed experts
        return nn.ModuleList([
            GatedFeedForward(embed_dim, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])


class HeterogeneousMoeFeedForward(MoeFeedForward):
    """Asymmetrical Mixture-of-Experts Feed-Forward layer - use experts with different hidden dimensions"""

    def __init__(
            self,
            embed_dim: int,
            experts_config: dict[int, int],
            shared_experts_config: dict[int, int] = None,
            activation: nn.Module = nn.SiLU(),
            top_k: int = 1,
            dropout: float = 0.1,
            use_gated_ff: bool = True,
            *args,
            **kwargs
    ):
        num_experts = sum(experts_config.keys())
        num_shared_experts = 0 if shared_experts_config is None else sum(shared_experts_config.keys())
        self.experts_config = experts_config
        self.shared_experts_config = shared_experts_config
        self.use_gated_ff = use_gated_ff
        super(HeterogeneousMoeFeedForward, self).__init__(
            embed_dim=embed_dim,
            hidden_dim=0,
            num_experts=num_experts,
            activation=activation,
            top_k=top_k,
            dropout=dropout,
            num_shared_experts=num_shared_experts,  # CHANGED: Pass through shared experts
            *args,
            **kwargs
        )

    def _init_experts(self, num_experts: int, embed_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        config = self.shared_experts_config if num_experts == self.num_shared_experts else self.experts_config

        all_experts = []
        for n, dim in config.items():
            all_experts.extend([
                GatedFeedForward(embed_dim, dim, activation, dropout) if self.use_gated_ff else FeedForward(embed_dim, dim, activation, dropout)
                for _ in range(n)
            ])
        return all_experts

