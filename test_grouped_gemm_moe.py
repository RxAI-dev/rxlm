#!/usr/bin/env python3
"""
Test script to compare vectorized MoE with and without grouped_gemm.

This script tests:
1. Correctness: Both implementations should produce similar outputs
2. Memory usage: grouped_gemm should use less memory (no padding)
3. Performance: grouped_gemm should be faster
"""

import torch
import torch.nn as nn
import time

# Add python/src to path
import sys
sys.path.insert(0, 'python/src')

from rxlm.transformers.moe import VectorizedMoeFeedForward, VectorizedGatedMoeFeedForward, GROUPED_GEMM_AVAILABLE

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def test_moe_implementation(
    use_grouped_gemm: bool,
    use_gated: bool = False,
    batch_size: int = 16,
    seq_len: int = 128,
    embed_dim: int = 512,
    hidden_dim: int = 2048,
    num_experts: int = 80,
    top_k: int = 8,
    num_shared_experts: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda',
    num_iters: int = 10,
):
    """Test MoE implementation with given configuration."""

    print(f"\n{'='*80}")
    print(f"Testing {'Gated' if use_gated else 'Standard'} MoE with grouped_gemm={use_grouped_gemm}")
    print(f"Config: batch={batch_size}, seq={seq_len}, embed={embed_dim}, hidden={hidden_dim}")
    print(f"        experts={num_experts}, top_k={top_k}, shared={num_shared_experts}, dtype={dtype}")
    print(f"{'='*80}\n")

    # Create model
    if use_gated:
        model = VectorizedGatedMoeFeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=nn.SiLU(),
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            use_grouped_gemm=use_grouped_gemm,
        )
    else:
        model = VectorizedMoeFeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            activation=nn.GELU(),
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            use_grouped_gemm=use_grouped_gemm,
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Warm up
    with torch.no_grad():
        _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure memory before
    mem_before = get_gpu_memory()

    # Forward pass
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iters):
            output = model(x)

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

    # Measure memory after
    mem_after = get_gpu_memory()
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3 if device == 'cuda' else 0

    # Results
    avg_time = elapsed / num_iters * 1000  # ms
    tokens_per_sec = (batch_size * seq_len * num_iters) / elapsed

    print(f"Results:")
    print(f"  Average forward time: {avg_time:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Memory allocated: {mem_after:.2f} GB")
    print(f"  Memory peak: {mem_peak:.2f} GB")
    print(f"  Memory delta: {mem_after - mem_before:.2f} GB")

    return {
        'output': output,
        'time': avg_time,
        'throughput': tokens_per_sec,
        'memory': mem_after,
        'memory_peak': mem_peak,
    }


def compare_implementations(use_gated: bool = False, **kwargs):
    """Compare standard vectorized vs grouped_gemm implementations."""

    print(f"\n{'#'*80}")
    print(f"# Comparing {'Gated' if use_gated else 'Standard'} MoE Implementations")
    print(f"{'#'*80}")

    # Test without grouped_gemm
    torch.manual_seed(42)
    results_standard = test_moe_implementation(
        use_grouped_gemm=False,
        use_gated=use_gated,
        **kwargs
    )

    # Test with grouped_gemm (if available)
    if GROUPED_GEMM_AVAILABLE:
        torch.manual_seed(42)
        results_grouped = test_moe_implementation(
            use_grouped_gemm=True,
            use_gated=use_gated,
            **kwargs
        )

        # Compare outputs
        output_diff = (results_standard['output'] - results_grouped['output']).abs().max().item()
        output_mean = results_standard['output'].abs().mean().item()
        relative_diff = output_diff / (output_mean + 1e-6)

        print(f"\n{'='*80}")
        print(f"Comparison Summary:")
        print(f"{'='*80}")
        print(f"Output difference (abs max): {output_diff:.6f}")
        print(f"Output relative diff: {relative_diff:.6f}")
        print(f"\nSpeedup: {results_standard['time'] / results_grouped['time']:.2f}x")
        print(f"Memory reduction: {results_standard['memory_peak'] - results_grouped['memory_peak']:.2f} GB")
        print(f"Memory ratio: {results_grouped['memory_peak'] / results_standard['memory_peak']:.2f}x")

        if output_diff < 0.01:
            print(f"\n✅ Correctness check PASSED (diff < 0.01)")
        else:
            print(f"\n❌ Correctness check FAILED (diff >= 0.01)")
    else:
        print(f"\n⚠️  grouped_gemm not available. Install with: pip install grouped_gemm")
        print(f"Only tested standard implementation.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test MoE with grouped_gemm')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--num-experts', type=int, default=80, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=8, help='Top-k experts per token')
    parser.add_argument('--num-shared', type=int, default=1, help='Number of shared experts')
    parser.add_argument('--gated', action='store_true', help='Use gated MoE')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'], help='Data type')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num-iters', type=int, default=10, help='Number of iterations for timing')

    args = parser.parse_args()

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    kwargs = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_experts': args.num_experts,
        'top_k': args.top_k,
        'num_shared_experts': args.num_shared,
        'dtype': dtype_map[args.dtype],
        'device': args.device,
        'num_iters': args.num_iters,
    }

    # Run comparison
    compare_implementations(use_gated=args.gated, **kwargs)
