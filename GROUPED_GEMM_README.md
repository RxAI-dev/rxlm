# Grouped GEMM Optimization for MoE Layers

This document explains the new `grouped_gemm` integration for optimizing Mixture-of-Experts (MoE) layers in RxLM.

## Overview

The vectorized MoE implementation now supports optional `grouped_gemm` library integration for faster expert computation with reduced memory usage. This eliminates the padding overhead from the batched matrix multiplication approach.

### Performance Improvements

- **Memory Reduction**: ~30-50% less memory usage by eliminating padding
- **Speed Improvement**: Expected 1.5-3x faster than padded BMM approach
- **Contiguous Layout**: No padding means more efficient memory access patterns

### Previous Implementation (Padded BMM)

The original vectorized implementation used padded tensors:
- Shape: `[num_experts, max_tokens_per_expert, embed_dim]`
- Problem: Padding wastes memory when experts have uneven token distribution
- Example: 80 experts with max 3.3k tokens → significant padding overhead

### New Implementation (Grouped GEMM)

The new grouped GEMM approach:
- Shape: `[total_tokens, embed_dim]` - no padding!
- Uses contiguous layout with `tokens_per_expert` count array
- Processes all experts in single optimized kernel
- Works with existing token permutation/sorting code

## Installation

Install the `grouped_gemm` library:

```bash
# Basic installation (cuBLAS mode)
pip install grouped_gemm

# OR: Optimized CUTLASS mode (recommended for best performance)
TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install grouped_gemm
```

**Note**: Replace `8.0` with your GPU's compute capability:
- Ampere (A100, RTX 30xx): 8.0 or 8.6
- Ada (RTX 40xx): 8.9
- Hopper (H100): 9.0

## Usage

### Enable grouped_gemm in Model Config

Simply add `use_grouped_gemm=True` when creating MoE layers:

```python
from rxlm.transformers.moe import VectorizedMoeFeedForward, VectorizedGatedMoeFeedForward

# Standard MoE with grouped_gemm
moe_layer = VectorizedMoeFeedForward(
    embed_dim=512,
    hidden_dim=2048,
    num_experts=80,
    activation=nn.GELU(),
    top_k=8,
    num_shared_experts=1,
    use_grouped_gemm=True,  # Enable grouped_gemm optimization
)

# Gated MoE with grouped_gemm
gated_moe = VectorizedGatedMoeFeedForward(
    embed_dim=512,
    hidden_dim=2048,
    num_experts=80,
    activation=nn.SiLU(),
    top_k=8,
    num_shared_experts=1,
    use_grouped_gemm=True,  # Enable grouped_gemm optimization
)
```

### Fallback Behavior

If `grouped_gemm` is not installed:
- A warning is printed
- Falls back to standard vectorized BMM implementation
- No errors or crashes

### Testing

Run the provided test script to verify correctness and measure performance:

```bash
# Basic test with default config (300M model size)
python test_grouped_gemm_moe.py

# Custom configuration
python test_grouped_gemm_moe.py \
    --batch-size 16 \
    --seq-len 128 \
    --embed-dim 512 \
    --hidden-dim 2048 \
    --num-experts 80 \
    --top-k 8 \
    --num-shared 1 \
    --dtype bfloat16

# Test gated MoE
python test_grouped_gemm_moe.py --gated

# Test with different model sizes
python test_grouped_gemm_moe.py --embed-dim 1024 --hidden-dim 4096  # Larger model
python test_grouped_gemm_moe.py --embed-dim 256 --hidden-dim 1024   # Smaller model
```

The test script compares:
- ✅ Correctness: Outputs should match (within numerical precision)
- ⚡ Speed: Time per forward pass
- 💾 Memory: Peak GPU memory usage

## Implementation Details

### Weight Format

Weights are stored in the same format for both implementations:
- `w1`: `[num_experts, embed_dim, hidden_dim]` for input → hidden
- `w2`: `[num_experts, hidden_dim, embed_dim]` for hidden → output

### Token Format

The existing token permutation is already compatible:
1. Tokens are sorted by expert ID
2. Contiguous layout: expert0 tokens, then expert1 tokens, etc.
3. `tokens_per_expert`: count array for each expert

### Grouped GEMM API

```python
import grouped_gemm as gg

# FC1: [total_tokens, embed_dim] @ [E, embed_dim, hidden_dim] -> [total_tokens, hidden_dim]
hidden = gg.ops.gmm(dispatched_x, self.w1, tokens_per_expert, trans_b=False)

# Add bias using repeat_interleave
hidden = hidden + torch.repeat_interleave(self.b1, tokens_per_expert, dim=0)
```

### Key Methods

- `_grouped_gemm_expert_forward()`: Main grouped GEMM computation
- `_expand_bias()`: Helper to expand bias tensors
- Both `VectorizedMoeFeedForward` and `VectorizedGatedMoeFeedForward` support it

## Troubleshooting

### Import Error

```
WARNING: grouped_gemm requested but not available.
Falling back to standard vectorized implementation.
```

**Solution**: Install grouped_gemm:
```bash
pip install grouped_gemm
```

### CUDA Errors

If you see CUDA kernel errors, ensure:
1. Your CUDA version is compatible (11.0+)
2. PyTorch CUDA version matches system CUDA
3. GPU compute capability is supported (7.0+)

### Performance Not Improving

If grouped_gemm is not faster:
1. Ensure CUTLASS mode is enabled during installation
2. Check that GPU utilization is high (use `nvidia-smi`)
3. Try larger batch sizes or longer sequences
4. Verify no other processes are using GPU

## References

- [grouped_gemm GitHub](https://github.com/tgale96/grouped_gemm)
- [NVIDIA Grouped GEMM Blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [PyTorch MoE Blog](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)

## Future Work

Potential improvements:
- FP8 quantization support
- Flash Attention integration
- Custom Triton kernels
- Dynamic expert capacity
