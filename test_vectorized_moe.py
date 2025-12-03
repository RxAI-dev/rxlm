#!/usr/bin/env python3
"""Test script to verify vectorized MoE implementation matches original."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'python/src')

from rxlm.transformers.moe import MoeFeedForward, GatedMoeFeedForward
from rxlm.transformers.moe import VectorizedMoeFeedForward, VectorizedGatedMoeFeedForward


def test_basic_feedforward():
    """Test VectorizedMoeFeedForward matches MoeFeedForward."""
    print("\n=== Testing VectorizedMoeFeedForward ===")

    # Parameters
    embed_dim = 512
    hidden_dim = 128
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 16

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create original model
    original = MoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.ReLU(),
        top_k=top_k,
        dropout=0.0  # No dropout for testing
    )

    # Create vectorized model with from_legacy=True
    vectorized = VectorizedMoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.ReLU(),
        top_k=top_k,
        dropout=0.0,
        from_legacy=True
    )

    # Copy weights from original to vectorized (via legacy experts)
    vectorized.experts = original.experts
    vectorized.router = original.router
    vectorized.load_weights_from_legacy()

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Set to eval mode
    original.eval()
    vectorized.eval()

    # Forward pass
    with torch.no_grad():
        output_original = original(x)
        output_vectorized = vectorized(x)

    # Check outputs match
    max_diff = (output_original - output_vectorized).abs().max().item()
    mean_diff = (output_original - output_vectorized).abs().mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Test PASSED - outputs match within tolerance")
        return True
    else:
        print("✗ Test FAILED - outputs differ significantly")
        return False


def test_gated_feedforward():
    """Test VectorizedGatedMoeFeedForward matches GatedMoeFeedForward."""
    print("\n=== Testing VectorizedGatedMoeFeedForward ===")

    # Parameters
    embed_dim = 512
    hidden_dim = 128
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 16

    # Set seed for reproducibility
    torch.manual_seed(43)

    # Create original model
    original = GatedMoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.SiLU(),
        top_k=top_k,
        dropout=0.0  # No dropout for testing
    )

    # Create vectorized model with from_legacy=True
    vectorized = VectorizedGatedMoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.SiLU(),
        top_k=top_k,
        dropout=0.0,
        from_legacy=True
    )

    # Copy weights from original to vectorized (via legacy experts)
    vectorized.experts = original.experts
    vectorized.router = original.router
    vectorized.load_weights_from_legacy()

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Set to eval mode
    original.eval()
    vectorized.eval()

    # Forward pass
    with torch.no_grad():
        output_original = original(x)
        output_vectorized = vectorized(x)

    # Check outputs match
    max_diff = (output_original - output_vectorized).abs().max().item()
    mean_diff = (output_original - output_vectorized).abs().mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Test PASSED - outputs match within tolerance")
        return True
    else:
        print("✗ Test FAILED - outputs differ significantly")
        return False


def test_memory_usage():
    """Test memory usage of vectorized implementation."""
    print("\n=== Testing Memory Usage ===")

    # Larger scale parameters
    embed_dim = 512
    hidden_dim = 128
    num_experts = 80
    top_k = 8
    batch_size = 32
    seq_len = 1024

    torch.manual_seed(44)

    # Create vectorized model
    model = VectorizedMoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.ReLU(),
        top_k=top_k,
        dropout=0.0
    )

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Get memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device = torch.device('cuda')
        model = model.to(device)
        x = x.to(device)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB

    model.eval()
    with torch.no_grad():
        output = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        mem_after = torch.cuda.memory_allocated() / 1024**2  # MB

        print(f"Memory before: {mem_before:.2f} MB")
        print(f"Memory peak: {mem_peak:.2f} MB")
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Memory increase: {mem_peak - mem_before:.2f} MB")

        # Calculate expected padding memory
        total_tokens = batch_size * seq_len * top_k
        avg_tokens_per_expert = total_tokens / num_experts
        max_tokens_estimate = avg_tokens_per_expert * 1.5  # Rough estimate with imbalance
        padded_memory = num_experts * max_tokens_estimate * embed_dim * 2 / 1024**2  # FP16
        print(f"Estimated padded tensor memory: {padded_memory:.2f} MB")
    else:
        print("CUDA not available - skipping memory test")

    print("✓ Memory test completed")
    return True


def test_single_token_inference():
    """Test single-token inference path."""
    print("\n=== Testing Single Token Inference ===")

    embed_dim = 512
    hidden_dim = 128
    num_experts = 8
    top_k = 2

    torch.manual_seed(45)

    # Create model
    model = VectorizedMoeFeedForward(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=nn.ReLU(),
        top_k=top_k,
        dropout=0.0
    )

    # Create single token input
    x = torch.randn(1, 1, embed_dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    if output.shape == x.shape:
        print("✓ Single token test PASSED")
        return True
    else:
        print("✗ Single token test FAILED")
        return False


if __name__ == "__main__":
    print("Testing Vectorized MoE Implementation")
    print("=" * 50)

    results = []
    results.append(test_basic_feedforward())
    results.append(test_gated_feedforward())
    results.append(test_memory_usage())
    results.append(test_single_token_inference())

    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("✓ All tests PASSED!")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
