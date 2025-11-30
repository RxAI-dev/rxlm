# Memory-Driven Gated DeltaNet (MD-GDN)

## Overview

Memory-Driven Gated DeltaNet (MD-GDN) is an advanced linear attention mechanism that extends Kimi Delta Attention (KDA) with memory-based features specifically designed for Reactive Language Models (RxLM). MD-GDN integrates RxLM's Short-Term Memory (STM) system directly into the linear attention mechanism, enabling more efficient and context-aware processing.

## Key Innovations

### 1. Continuous State with STM Correction

Unlike standard linear attention mechanisms that reset their recurrent state at each interaction, MD-GDN maintains a **continuous recurrent state** across interactions. This state is intelligently corrected using information from the RxLM memory system:

- **Persistent State**: The recurrent state from linear attention is preserved between interactions
- **STM-Derived State**: Memory slots are compressed (mean pooled) and projected to the state space
- **Gated Fusion**: A learned gate interpolates between the persistent state and STM-derived state

This mechanism allows MD-GDN to:
- Leverage long-term information stored in STM
- Maintain continuity across conversation turns
- Adapt to context shifts when memory is updated

### 2. Memory-Conditioned Gating

Standard linear attention uses gates (α and β) conditioned only on the current sequence. MD-GDN extends this by conditioning gates on **both** the current sequence and the memory state:

- **Memory Context**: STM is compressed via mean pooling
- **Fusion Modes**:
  - **Additive**: Separate projections for sequence and memory, combined additively
  - **Concatenative**: Joint projection of concatenated sequence and memory features
- **Enhanced Expressiveness**: Gates can modulate attention based on stored conversational context

This allows MD-GDN to:
- Adjust attention patterns based on conversational history
- Prioritize or de-prioritize certain patterns based on memory
- Achieve better coherence in long conversations

## Architecture

### Core Components

```
Input Sequence (x) + Short-Term Memory (STM)
    ↓
┌──────────────────────────────────────┐
│   Query/Key/Value Projections       │
│   (with optional short convolutions) │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Memory-Conditioned Gate Computation│
│   • Compress STM → mem_context       │
│   • Fuse with sequence features      │
│   • Compute α, β gates               │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   STM-Corrected State Initialization │
│   • Project STM → state_space        │
│   • Gate(persistent_state, stm_state)│
│   • Initialize recurrent state       │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Chunked Linear Attention           │
│   • L2 normalized Q/K                │
│   • Delta rule state updates         │
│   • Gated recurrence                 │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Gated Output Projection            │
│   • RMSNorm with gating              │
│   • Final linear projection          │
└──────────────────────────────────────┘
    ↓
Output + Updated Recurrent State
```

### State Update Formula

The recurrent state update follows a gated delta rule:

```
h_t = β_t ⊙ h_{t-1} + α_t ⊙ (k_t^T ⊗ v_t)
```

Where:
- `h_t`: Recurrent state at time t
- `β_t`: Forget gate (memory-conditioned)
- `α_t`: Input gate (memory-conditioned)
- `k_t, v_t`: Keys and values at time t

For STM correction:

```
h_init = gate ⊙ h_persistent + (1 - gate) ⊙ h_stm
```

Where:
- `h_persistent`: State from previous interaction
- `h_stm`: STM-derived state
- `gate`: Learned interpolation weights

## Usage

### Configuration

```python
from rxlm.rxt.models import RxTComponentConfig

decoder_config = RxTComponentConfig(
    num_layers=24,
    embed_dim=1024,
    ff_dim=4096,
    att_heads=16,
    seq_len=8192,
    stm_size=512,

    # Enable MD-GDN for self-attention
    use_linear_self_attn=True,
    linear_attn_type='md_gdn',  # Memory-Driven Gated DeltaNet
    linear_attn_mode='chunk',
    linear_attn_expand_v=1.0,

    # Other standard parameters...
    use_flash_attention=True,
    use_gated=True,
    ff_activation='swish',
    use_rms_norm=True,
)
```

### Model Creation

```python
from rxlm.rxt.models import RxTAlpha

model = RxTAlpha(
    decoder_config=decoder_config,
    encoder_config=encoder_config,
    memory_attention_config=memory_attention_config,
    tokenizer_config=tokenizer_config,
)

model.share_components()
```

### Inference

MD-GDN works seamlessly with RxLM's interaction flow:

```python
# Initialize memory
stm_init_state = model.tokenize_full_interaction(
    "You are a helpful assistant.", '', max_seq_len=512
)
model.init_stm_state(**stm_init_state)

# Interact
for token_id in model.interact(**tokenized_query, max_seq_len=512):
    if token_id == -1:
        print('\n', end='')
    elif token_id == -2:
        print('[Memory updated]\n')
    else:
        print(model.stringify_token(token_id), end='')
```

## Advanced Configuration

### Memory Fusion Modes

MD-GDN supports two modes for fusing sequence and memory features:

#### Additive Fusion (Default)
```python
# In MemoryDrivenGatedDeltaNet initialization:
memory_fusion_mode='add'
```

Pros:
- More parameter efficient
- Faster computation
- Good for most use cases

#### Concatenative Fusion
```python
memory_fusion_mode='concat'
```

Pros:
- More expressive
- Better for complex memory dependencies
- Recommended for large models

### State Correction

Enable or disable STM state correction:

```python
use_stm_correction=True  # Default, recommended
```

When disabled, MD-GDN behaves more like standard KDA but retains memory-conditioned gating.

## Performance Characteristics

### Computational Complexity

- **Standard Attention**: O(n²) in sequence length
- **Linear Attention (KDA)**: O(n) in sequence length
- **MD-GDN**: O(n) in sequence length + O(m) for memory processing

Where n is sequence length and m is number of memory slots (typically m << n).

### Memory Usage

MD-GDN adds minimal memory overhead:
- Persistent recurrent state: `[batch, num_heads, head_dim_k, head_dim_v]`
- Memory conditioning projections: Additional linear layers

### Training Considerations

- **Chunk Mode**: Recommended for training, memory efficient
- **Gradient Flow**: Full backpropagation through memory correction and conditioning
- **Initialization**: STM correction helps with cold-start problem
- **Stability**: Gated updates prevent catastrophic forgetting

## Comparison with Other Mechanisms

| Feature | Standard Attention | KDA | MD-GDN |
|---------|-------------------|-----|---------|
| Complexity | O(n²) | O(n) | O(n) |
| Memory Integration | Via cross-attention | None | Native |
| State Continuity | None | Per-sequence | Cross-interaction |
| Memory Conditioning | None | None | Yes |
| Best For | Short sequences | Long sequences | RxLM conversations |

## Implementation Details

### Simplified Delta Rule

MD-GDN uses a simplified but efficient implementation of the gated delta rule:

```python
# For each time step t:
o_t = q_t @ h_{t-1}  # Output from current state
kv_t = k_t^T @ v_t    # Outer product update
h_t = β_t * h_{t-1} + α_t * kv_t  # Gated state update
```

### L2 Normalization

Query and key vectors are L2-normalized for stability:

```python
q = F.normalize(q, p=2, dim=-1)
k = F.normalize(k, p=2, dim=-1)
```

This ensures stable gradients and better convergence during training.

### Chunked Processing

For efficiency, sequences are processed in chunks (default size: 64 tokens):

```python
chunk_size = 64
for start in range(0, seq_len, chunk_size):
    end = min(start + chunk_size, seq_len)
    process_chunk(q[start:end], k[start:end], v[start:end])
```

## Limitations and Future Work

### Current Limitations

1. **Mode Support**: Currently only 'chunk' mode is implemented
2. **Kernel Optimization**: Uses PyTorch implementation; could benefit from Triton kernels
3. **Multi-Value Attention**: GVA (Grouped Value Attention) not yet supported

### Future Enhancements

1. **Fused Recurrent Mode**: For even faster inference on short sequences
2. **Optimized Kernels**: Custom Triton/CUDA kernels for memory operations
3. **Adaptive Fusion**: Learning when to rely more on memory vs. sequence
4. **Multi-Modal Memory**: Extending to handle multi-modal memory inputs

## Research Background

MD-GDN builds upon several key innovations:

1. **Linear Attention**: Efficient O(n) attention mechanisms
2. **Delta Rule**: Gated recurrence for stable state updates
3. **Kimi Delta Attention**: Per-channel gating for expressiveness
4. **RxLM Memory System**: Attention-based memory for conversational AI

## References

- [Gated Linear Attention (GLA)](https://arxiv.org/abs/2312.06635)
- [DeltaNet Paper](https://arxiv.org/abs/2102.11174)
- [Kimi Linear Attention](https://arxiv.org/abs/2501.00000) (2025)
- [Reactive Transformers (RxLM)](https://github.com/RxAI-dev/rxlm)
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

## Support

For issues or questions about MD-GDN:
- GitHub Issues: [RxAI-dev/rxlm](https://github.com/RxAI-dev/rxlm/issues)
- Email: support@rxai.dev

## License

MD-GDN is part of the RxLM framework and is licensed under the Reactive AI Framework License (RAFL) v1.0.
