"""Mixture-of-Experts (MoE) Feed-Forward Layers.

This module provides both standard and vectorized MoE implementations:

Standard (Legacy) Classes:
- MoeFeedForward: Basic MoE with FeedForward experts
- GatedMoeFeedForward: MoE with GatedFeedForward (GLU) experts
- HeterogeneousMoeFeedForward: MoE with variable hidden dimensions per expert

Vectorized (Optimized) Classes:
- VectorizedMoeFeedForward: ~10-50x faster via batched matrix multiplication
- VectorizedGatedMoeFeedForward: Gated version with same speedup

Key Optimization:
The vectorized classes replace the Python loop over experts with a single batched
matrix multiplication (BMM). This is achieved by:
1. Sorting tokens by assigned expert (already done in original)
2. Computing position of each token within its expert group (fully vectorized)
3. Creating padded tensor [num_experts, max_tokens_per_expert, embed_dim]
4. Single BMM operation processes all experts simultaneously
5. Gathering results back to original token order

Memory Considerations:
The padded approach uses O(num_experts * max_tokens_per_expert * embed_dim) memory.
With good load balancing, this is acceptable. Example:
- 80 experts, 8 active per token, 32k total tokens
- max_tokens_per_expert ≈ 3.3k (with balancing)
- Padded memory: [80, 3.3k, 512] ≈ 270MB in FP16

Legacy Compatibility:
Use `from_legacy=True` and `load_weights_from_legacy()` to convert existing models:
    model = VectorizedMoeFeedForward(..., from_legacy=True)
    model.load_state_dict(legacy_state_dict)
    model.load_weights_from_legacy()  # Converts to vectorized format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ff import FeedForward, GatedFeedForward
from ..training.utils import RxTModelProfiler

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

    def forward(self, x: torch.Tensor, profiler: RxTModelProfiler = None) -> torch.Tensor:
        # Store original shape for final step
        orig_shape = x.size()
        # Flatten input sequence
        x = x.view(-1, self.embed_dim)
        num_tokens, embed_dim = x.size()

        # === STEP 1: ROUTING ===
        # Get routing weights and selected experts from router
        # routing_weights: [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]
        if profiler is not None:
            profiler.profile_start('moe_router')


        if self.router_amp:
            with torch.amp.autocast(device_type=x.device.type, dtype=self.router_dtype):
                routing_weights, selected_experts = self.router(x)
        else:
            routing_weights, selected_experts = self.router(x)

        if profiler is not None:
            profiler.profile_end('moe_router')

        if profiler is not None:
            profiler.profile_start('moe_permute')

        # CHANGED: Fast path for single-token processing (autoregressive generation)
        # When processing one token at a time, avoid complex permutation overhead
        if num_tokens == 1:
            if profiler is not None:
                profiler.profile_end('moe_permute')

            if profiler is not None:
                profiler.profile_start('moe_loop')

            # Simple loop over top-k experts for single token
            final_output = torch.zeros_like(x)
            for k in range(self.top_k):
                expert_idx = selected_experts[0, k]
                weight = routing_weights[0, k]
                expert_output = self.experts[expert_idx](x)
                final_output += weight * expert_output

            if profiler is not None:
                profiler.profile_end('moe_loop')

            if profiler is not None:
                profiler.profile_start('moe_reverse')
                profiler.profile_end('moe_reverse')
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

            if profiler is not None:
                profiler.profile_end('moe_permute')

            if profiler is not None:
                profiler.profile_start('moe_loop')

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

            if profiler is not None:
                profiler.profile_end('moe_loop')

            if profiler is not None:
                profiler.profile_start('moe_reverse')

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

            if profiler is not None:
                profiler.profile_end('moe_reverse')

        if profiler is not None:
            profiler.profile_start('moe_shared')

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

        if profiler is not None:
            profiler.profile_end('moe_shared')

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


class VectorizedMoeFeedForward(nn.Module):
    """Vectorized Mixture-of-Experts Feed-Forward layer - uses batched matmul instead of Python loops for ~10-50x speedup."""

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_experts: int,
            activation: nn.Module,
            top_k: int = 1,
            dropout: float = 0.0,
            num_shared_experts: int = 0,
            router_amp: bool = False,
            router_dtype: torch.dtype = torch.float32,
            from_legacy: bool = False,
            *args,
            **kwargs
    ):
        super(VectorizedMoeFeedForward, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.router_amp = router_amp
        self.router_dtype = router_dtype
        self.activation = activation
        self.dropout_p = dropout

        self.router = MoeRouter(embed_dim, num_experts, top_k)

        # Initialize stacked expert weights
        self._init_weights()

        # Dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Legacy compatibility: initialize old-style experts if from_legacy=True
        self._init_legacy_experts(from_legacy)

        # Initialize shared experts (keep as ModuleList - only routed experts are vectorized)
        self._init_shared_experts()

    def _init_weights(self):
        """Initialize stacked expert weights for basic feedforward."""
        # fc1: embed_dim -> hidden_dim, so weight is [hidden_dim, embed_dim]
        # For BMM we need [num_experts, embed_dim, hidden_dim] (transposed)
        self.w1 = nn.Parameter(torch.randn(self.num_experts, self.embed_dim, self.hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

        # fc2: hidden_dim -> embed_dim, so weight is [embed_dim, hidden_dim]
        # For BMM we need [num_experts, hidden_dim, embed_dim] (transposed)
        self.w2 = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, self.embed_dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(self.num_experts, self.embed_dim))

    def _init_legacy_experts(self, from_legacy: bool):
        """Initialize legacy ModuleList experts if from_legacy=True."""
        if from_legacy:
            self.experts = nn.ModuleList([
                FeedForward(self.embed_dim, self.hidden_dim, self.activation, self.dropout_p)
                for _ in range(self.num_experts)
            ])
        else:
            self.experts = None

    def _init_shared_experts(self):
        """Initialize shared experts (always as ModuleList)."""
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(self.embed_dim, self.hidden_dim, self.activation, self.dropout_p)
                for _ in range(self.num_shared_experts)
            ])

            if self.num_shared_experts > 1:
                self.shared_expert_gate = nn.Linear(self.embed_dim, self.num_shared_experts, bias=False)

    def load_weights_from_legacy(self):
        """Transfer weights from legacy ModuleList experts to stacked tensors, then free legacy memory."""
        if self.experts is None:
            raise ValueError("No legacy experts to load from. Was from_legacy=True during init?")

        with torch.no_grad():
            # Stack weights from legacy experts into new parameters
            # fc1.weight is [hidden_dim, embed_dim], we transpose to [embed_dim, hidden_dim]
            self.w1.copy_(torch.stack([e.fc1.weight.T for e in self.experts], dim=0))
            self.b1.copy_(torch.stack([e.fc1.bias for e in self.experts], dim=0))
            # fc2.weight is [embed_dim, hidden_dim], we transpose to [hidden_dim, embed_dim]
            self.w2.copy_(torch.stack([e.fc2.weight.T for e in self.experts], dim=0))
            self.b2.copy_(torch.stack([e.fc2.bias for e in self.experts], dim=0))

        # Free legacy memory
        del self.experts
        self.experts = None
        torch.cuda.empty_cache()

    def router_loss(self):
        return self.router.aux_loss

    def _apply_expert_weights(self, padded_input: torch.Tensor) -> torch.Tensor:
        """
        Apply expert weights to padded input using batched matmul.

        Args:
            padded_input: [num_experts, max_tokens, embed_dim]

        Returns:
            output: [num_experts, max_tokens, embed_dim]
        """
        # First linear layer: [E, T, D] @ [E, D, H] -> [E, T, H]
        hidden = torch.bmm(padded_input, self.w1)  # [num_experts, max_tokens, hidden_dim]
        hidden = hidden + self.b1.unsqueeze(1)  # Add bias

        # Activation
        hidden = self.activation(hidden)

        # Dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Second linear layer: [E, T, H] @ [E, H, D] -> [E, T, D]
        output = torch.bmm(hidden, self.w2)  # [num_experts, max_tokens, embed_dim]
        output = output + self.b2.unsqueeze(1)  # Add bias

        return output

    def _vectorized_expert_forward(
        self,
        dispatched_x: torch.Tensor,
        sorted_expert_indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        starts: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized expert computation using padded batched matmul.

        Args:
            dispatched_x: [total_tokens, embed_dim] - sorted tokens
            sorted_expert_indices: [total_tokens] - expert ID for each token
            tokens_per_expert: [num_experts] - count of tokens per expert
            starts: [num_experts] - start index for each expert in dispatched_x

        Returns:
            [total_tokens, embed_dim] - expert outputs in same order as dispatched_x
        """
        total_tokens = dispatched_x.size(0)

        # Find maximum tokens assigned to any expert (for padding)
        max_tokens = tokens_per_expert.max().item()

        # Early exit if no tokens (shouldn't happen but be safe)
        if max_tokens == 0:
            return torch.zeros_like(dispatched_x)

        # === STEP 1: Compute position of each token within its expert ===
        token_positions = torch.arange(total_tokens, device=dispatched_x.device)
        expert_token_positions = token_positions - starts[sorted_expert_indices]

        # === STEP 2: Create padded tensor [num_experts, max_tokens, embed_dim] ===
        padded_input = torch.zeros(
            self.num_experts, max_tokens, self.embed_dim,
            dtype=dispatched_x.dtype, device=dispatched_x.device
        )

        # === STEP 3: Scatter tokens to proper positions ===
        # Use advanced indexing to place each token at [expert_id, position_in_expert, :]
        padded_input[sorted_expert_indices, expert_token_positions] = dispatched_x

        # === STEP 4: Batched matmul for all experts ===
        output = self._apply_expert_weights(padded_input)

        # === STEP 5: Gather results back ===
        # Extract results at positions corresponding to actual tokens
        expert_output = output[sorted_expert_indices, expert_token_positions]  # [total_tokens, embed_dim]

        return expert_output

    def _single_token_expert_forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward pass for a single token through top-k experts.

        Args:
            x: [1, embed_dim] - single token input
            selected_experts: [top_k] - indices of selected experts
            routing_weights: [top_k] - routing weights for each expert

        Returns:
            output: [1, embed_dim] - weighted sum of expert outputs
        """
        # Select weights and biases for active experts using advanced indexing
        # [num_experts, embed_dim, hidden_dim] -> [top_k, embed_dim, hidden_dim]
        w1_selected = self.w1[selected_experts]  # [top_k, embed_dim, hidden_dim]
        b1_selected = self.b1[selected_experts]  # [top_k, hidden_dim]
        w2_selected = self.w2[selected_experts]  # [top_k, hidden_dim, embed_dim]
        b2_selected = self.b2[selected_experts]  # [top_k, embed_dim]

        # Expand single token for batched computation: [1, embed_dim] -> [top_k, 1, embed_dim]
        x_expanded = x.unsqueeze(0).expand(self.top_k, -1, -1)  # [top_k, 1, embed_dim]

        # First linear layer: [top_k, 1, D] @ [top_k, D, H] -> [top_k, 1, H]
        hidden = torch.bmm(x_expanded, w1_selected)  # [top_k, 1, hidden_dim]
        hidden = hidden + b1_selected.unsqueeze(1)  # Add bias

        # Activation
        hidden = self.activation(hidden)

        # Dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Second linear layer: [top_k, 1, H] @ [top_k, H, D] -> [top_k, 1, D]
        expert_outputs = torch.bmm(hidden, w2_selected)  # [top_k, 1, embed_dim]
        expert_outputs = expert_outputs + b2_selected.unsqueeze(1)  # Add bias

        # Apply routing weights: [top_k, 1, embed_dim] * [top_k, 1, 1] -> [top_k, 1, embed_dim]
        weighted_outputs = expert_outputs * routing_weights.view(-1, 1, 1)

        # Sum across experts: [top_k, 1, embed_dim] -> [1, embed_dim]
        final_output = weighted_outputs.sum(dim=0)

        return final_output

    def forward(self, x: torch.Tensor, profiler: RxTModelProfiler = None) -> torch.Tensor:
        # Store original shape for final step
        orig_shape = x.size()
        # Flatten input sequence
        x = x.view(-1, self.embed_dim)
        num_tokens, embed_dim = x.size()

        # === STEP 1: ROUTING ===
        # Get routing weights and selected experts from router
        # routing_weights: [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]
        if profiler is not None:
            profiler.profile_start('moe_router')

        if self.router_amp:
            with torch.amp.autocast(device_type=x.device.type, dtype=self.router_dtype):
                routing_weights, selected_experts = self.router(x)
        else:
            routing_weights, selected_experts = self.router(x)


        if profiler is not None:
            profiler.profile_end('moe_router')

        if profiler is not None:
            profiler.profile_start('moe_permute')

        # Fast path for single-token processing (autoregressive generation)
        if num_tokens == 1:
            if profiler is not None:
                profiler.profile_end('moe_permute')

            if profiler is not None:
                profiler.profile_start('moe_loop')

            # Vectorized computation for all top-k experts - no Python loop!
            final_output = self._single_token_expert_forward(
                x, selected_experts[0], routing_weights[0]
            )


            if profiler is not None:
                profiler.profile_end('moe_loop')

            if profiler is not None:
                profiler.profile_start('moe_reverse')
                profiler.profile_end('moe_reverse')
        else:
            # === STEP 2: CREATE DISPOSE MAP ===
            flat_selected_experts = selected_experts.view(-1)
            flat_routing_weights = routing_weights.view(-1)
            token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)

            # === STEP 3: PERMUTE ===
            sorted_expert_indices, sorted_order = flat_selected_experts.sort(0)
            permuted_token_indices = token_indices[sorted_order]
            permuted_routing_weights = flat_routing_weights[sorted_order]
            dispatched_x = x[permuted_token_indices]

            # === STEP 4: VECTORIZED EXPERT PROCESSING ===
            tokens_per_expert = F.one_hot(sorted_expert_indices, num_classes=self.num_experts).sum(dim=0)

            # Compute start indices for each expert
            starts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
            starts[1:] = tokens_per_expert[:-1].cumsum(0)


            if profiler is not None:
                profiler.profile_end('moe_permute')

            if profiler is not None:
                profiler.profile_start('moe_loop')

            # Vectorized expert forward pass
            concatenated_outputs = self._vectorized_expert_forward(
                dispatched_x, sorted_expert_indices, tokens_per_expert, starts
            )

            if profiler is not None:
                profiler.profile_end('moe_loop')

            if profiler is not None:
                profiler.profile_start('moe_reverse')

            # === STEP 5: REVERSE PERMUTATION AND COMBINE RESULTS ===
            weighted_outputs = concatenated_outputs * permuted_routing_weights.unsqueeze(1)
            final_output = torch.zeros_like(x)
            inverse_sorted_order = sorted_order.argsort(0)
            unpermuted_outputs = weighted_outputs[inverse_sorted_order]
            scatter_indices = token_indices.unsqueeze(1).expand(-1, embed_dim)
            final_output.scatter_add_(0, scatter_indices, unpermuted_outputs.to(dtype=final_output.dtype))

            if profiler is not None:
                profiler.profile_end('moe_reverse')

        if profiler is not None:
            profiler.profile_start('moe_shared')

        # Add shared expert outputs (if any) - keep same as original
        if self.num_shared_experts > 0:
            if self.num_shared_experts == 1:
                shared_output = self.shared_experts[0](x)
                final_output = final_output + shared_output
            else:
                shared_gate_logits = self.shared_expert_gate(x)
                shared_weights = F.softmax(shared_gate_logits, dim=-1)
                shared_outputs = torch.stack([
                    expert(x) for expert in self.shared_experts
                ], dim=1)
                shared_combined = (shared_outputs * shared_weights.unsqueeze(-1)).sum(dim=1)
                final_output = final_output + shared_combined

        if profiler is not None:
            profiler.profile_end('moe_shared')

        return final_output.view(orig_shape)


class VectorizedGatedMoeFeedForward(VectorizedMoeFeedForward):
    """Vectorized Gated Mixture-of-Experts Feed-Forward layer - uses batched matmul with GLU for ~10-50x speedup."""

    def _init_weights(self):
        """Initialize stacked expert weights for gated feedforward."""
        # fc1 (GatedLinearUnit): embed_dim -> hidden_dim * 2, so weight is [hidden_dim * 2, embed_dim]
        # For BMM we need [num_experts, embed_dim, hidden_dim * 2] (transposed)
        self.w1 = nn.Parameter(torch.randn(self.num_experts, self.embed_dim, self.hidden_dim * 2) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim * 2))

        # fc2: hidden_dim -> embed_dim, so weight is [embed_dim, hidden_dim]
        # For BMM we need [num_experts, hidden_dim, embed_dim] (transposed)
        self.w2 = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, self.embed_dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(self.num_experts, self.embed_dim))

    def _init_legacy_experts(self, from_legacy: bool):
        """Initialize legacy ModuleList experts with GatedFeedForward."""
        if from_legacy:
            self.experts = nn.ModuleList([
                GatedFeedForward(self.embed_dim, self.hidden_dim, self.activation, self.dropout_p)
                for _ in range(self.num_experts)
            ])
        else:
            self.experts = None

    def _init_shared_experts(self):
        """Initialize shared experts with GatedFeedForward."""
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                GatedFeedForward(self.embed_dim, self.hidden_dim, self.activation, self.dropout_p)
                for _ in range(self.num_shared_experts)
            ])

            if self.num_shared_experts > 1:
                self.shared_expert_gate = nn.Linear(self.embed_dim, self.num_shared_experts, bias=False)

    def load_weights_from_legacy(self):
        """Transfer weights from legacy gated experts to stacked tensors, then free legacy memory."""
        if self.experts is None:
            raise ValueError("No legacy experts to load from. Was from_legacy=True during init?")

        with torch.no_grad():
            # Stack weights from legacy gated experts into new parameters
            # fc1.linear.weight is [hidden_dim * 2, embed_dim], we transpose to [embed_dim, hidden_dim * 2]
            self.w1.copy_(torch.stack([e.fc1.linear.weight.T for e in self.experts], dim=0))
            self.b1.copy_(torch.stack([e.fc1.linear.bias for e in self.experts], dim=0))
            # fc2.weight is [embed_dim, hidden_dim], we transpose to [hidden_dim, embed_dim]
            self.w2.copy_(torch.stack([e.fc2.weight.T for e in self.experts], dim=0))
            self.b2.copy_(torch.stack([e.fc2.bias for e in self.experts], dim=0))

        # Free legacy memory
        del self.experts
        self.experts = None
        torch.cuda.empty_cache()

    def _apply_expert_weights(self, padded_input: torch.Tensor) -> torch.Tensor:
        """
        Apply gated expert weights to padded input using batched matmul.

        Args:
            padded_input: [num_experts, max_tokens, embed_dim]

        Returns:
            output: [num_experts, max_tokens, embed_dim]
        """
        # First linear layer (GatedLinearUnit): [E, T, D] @ [E, D, 2H] -> [E, T, 2H]
        gated_hidden = torch.bmm(padded_input, self.w1)  # [num_experts, max_tokens, hidden_dim * 2]
        gated_hidden = gated_hidden + self.b1.unsqueeze(1)  # Add bias

        # Split into linear and gate components
        l, g = gated_hidden.chunk(2, dim=-1)  # Each: [num_experts, max_tokens, hidden_dim]

        # Apply gating: l * activation(g)
        hidden = l * self.activation(g)

        # Dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Second linear layer: [E, T, H] @ [E, H, D] -> [E, T, D]
        output = torch.bmm(hidden, self.w2)  # [num_experts, max_tokens, embed_dim]
        output = output + self.b2.unsqueeze(1)  # Add bias

        return output

    def _single_token_expert_forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward pass for a single token through top-k gated experts.

        Args:
            x: [1, embed_dim] - single token input
            selected_experts: [top_k] - indices of selected experts
            routing_weights: [top_k] - routing weights for each expert

        Returns:
            output: [1, embed_dim] - weighted sum of expert outputs
        """
        # Select weights and biases for active experts using advanced indexing
        w1_selected = self.w1[selected_experts]  # [top_k, embed_dim, hidden_dim * 2]
        b1_selected = self.b1[selected_experts]  # [top_k, hidden_dim * 2]
        w2_selected = self.w2[selected_experts]  # [top_k, hidden_dim, embed_dim]
        b2_selected = self.b2[selected_experts]  # [top_k, embed_dim]

        # Expand single token for batched computation: [1, embed_dim] -> [top_k, 1, embed_dim]
        x_expanded = x.unsqueeze(0).expand(self.top_k, -1, -1)  # [top_k, 1, embed_dim]

        # First linear layer (GatedLinearUnit): [top_k, 1, D] @ [top_k, D, 2H] -> [top_k, 1, 2H]
        gated_hidden = torch.bmm(x_expanded, w1_selected)  # [top_k, 1, hidden_dim * 2]
        gated_hidden = gated_hidden + b1_selected.unsqueeze(1)  # Add bias

        # Split into linear and gate components
        l, g = gated_hidden.chunk(2, dim=-1)  # Each: [top_k, 1, hidden_dim]

        # Apply gating: l * activation(g)
        hidden = l * self.activation(g)

        # Dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Second linear layer: [top_k, 1, H] @ [top_k, H, D] -> [top_k, 1, D]
        expert_outputs = torch.bmm(hidden, w2_selected)  # [top_k, 1, embed_dim]
        expert_outputs = expert_outputs + b2_selected.unsqueeze(1)  # Add bias

        # Apply routing weights: [top_k, 1, embed_dim] * [top_k, 1, 1] -> [top_k, 1, embed_dim]
        weighted_outputs = expert_outputs * routing_weights.view(-1, 1, 1)

        # Sum across experts: [top_k, 1, embed_dim] -> [1, embed_dim]
        final_output = weighted_outputs.sum(dim=0)

        return final_output

