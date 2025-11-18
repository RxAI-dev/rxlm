# Linear Attention Support in RxLM

RxLM now supports linear attention mechanisms for self-attention layers, enabling more efficient training and inference for long sequences. This feature uses the `flash-linear-attention` library to provide optimized implementations of Gated Linear Attention (GLA), DeltaNet, Gated DeltaNet, and Kimi Delta Attention (KDA).

## Overview

Linear attention mechanisms offer O(n) complexity instead of O(n²) for standard attention, making them particularly suitable for:
- Long sequence processing
- Memory-efficient training
- Faster inference on extended contexts

The implementation allows you to use linear attention for **self-attention** while keeping standard attention for **memory cross-attention**, maintaining the benefits of RxLM's memory system.

## Installation

First, install the `flash-linear-attention` library:

```bash
pip install fla-nightly
```

Or build from source for the latest features:

```bash
git clone https://github.com/sustcsonglin/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
```

## Supported Linear Attention Types

1. **GLA (Gated Linear Attention)**: Fast and efficient with gating mechanisms
2. **DeltaNet**: Parallelized linear transformers with delta rule
3. **Gated DeltaNet**: Enhanced DeltaNet with gating
4. **KDA (Kimi Delta Attention)**: Advanced per-channel gating variant
5. **MD-GDN (Memory-Driven Gated DeltaNet)**: KDA extended with memory features (recommended for RxLM)

## Usage

### Configuring a Model with Linear Attention

To use linear attention in your RxT models, add the following parameters to your component configuration:

```python
from rxlm.rxt.models import RxTComponentConfig

decoder_config = RxTComponentConfig(
    num_layers=12,
    vocab_size=32000,
    embed_dim=768,
    ff_dim=3072,
    att_heads=12,
    seq_len=8192,
    stm_size=512,
    use_flash_attention=True,
    use_gated=True,
    ff_activation='swish',
    ff_dropout=0.0,
    att_dropout=0.0,
    use_rms_norm=True,
    att_groups=4,
    use_moe=False,
    num_experts=8,
    moe_top_k=2,
    self_att_type='gqa',
    cross_att_type='mqa',
    att_experts=None,
    att_query_experts=None,
    att_query_groups=4,
    cross_att_groups=None,
    cross_att_query_groups=1,
    use_head_norm=True,
    init_identity_norm=False,

    # Linear attention parameters
    use_linear_self_attn=True,              # Enable linear attention for self-attention
    linear_attn_type='kda',                  # Options: 'gla', 'deltanet', 'gated_deltanet', 'kda'
    linear_attn_mode='chunk',                # Training mode: 'chunk' or 'fused'
    linear_attn_expand_k=0.5,                # Key expansion ratio (not used for KDA)
    linear_attn_expand_v=1.0,                # Value expansion ratio
)
```

### Parameter Details

- **use_linear_self_attn** (bool): Set to `True` to enable linear attention for self-attention layers
- **linear_attn_type** (str): Choose the linear attention variant:
  - `'gla'`: Gated Linear Attention - fast and efficient
  - `'deltanet'`: DeltaNet - parallelized linear transformers
  - `'gated_deltanet'`: Gated DeltaNet - enhanced with gating
  - `'kda'`: Kimi Delta Attention - advanced per-channel gating
  - `'md_gdn'`: Memory-Driven Gated DeltaNet - KDA with memory features (recommended for RxLM)
- **linear_attn_mode** (str): Training mode:
  - `'chunk'`: Chunk-based training (memory efficient, recommended for training)
  - `'fused'`: Fused kernel (faster but higher memory usage)
- **linear_attn_expand_k** (float): Key dimension expansion ratio (default: 0.5) - note: not used for KDA
- **linear_attn_expand_v** (float): Value dimension expansion ratio (default: 1.0)

### Example: Creating a Model with Kimi Delta Attention (Recommended)

```python
from rxlm.rxt.models import RxTAlpha, RxTComponentConfig

# Configure decoder with KDA for self-attention (best quality)
decoder_config = RxTComponentConfig(
    num_layers=24,
    embed_dim=1024,
    ff_dim=4096,
    att_heads=16,
    seq_len=16384,  # Longer sequences possible with linear attention
    use_linear_self_attn=True,
    linear_attn_type='kda',  # Kimi Delta Attention - per-channel gating
    linear_attn_mode='chunk',
    linear_attn_expand_v=1.0,  # Value expansion ratio
    # ... other parameters
)

# Configure encoder (can also use linear attention)
encoder_config = RxTComponentConfig(
    num_layers=6,
    embed_dim=1024,
    ff_dim=2048,
    att_heads=16,
    seq_len=2048,
    use_linear_self_attn=True,
    linear_attn_type='kda',  # Or use 'gla', 'deltanet', 'gated_deltanet'
    # ... other parameters
)

# Create model
model = RxTAlpha(
    decoder_config=decoder_config,
    encoder_config=encoder_config,
    # ... other parameters
)
```

## Training Considerations

### Memory Efficiency
Linear attention reduces memory requirements significantly, allowing:
- Longer training sequences
- Larger batch sizes
- More efficient gradient computation

### Compatibility
The linear attention layers are fully compatible with:
- RxLM's memory cross-attention system
- Mixed precision training
- Distributed training (DDP)
- Gradient checkpointing
- All RxLM training stages (Joint LM, SFT, Memory Pre-training, etc.)

### Performance Tips
1. Use `linear_attn_mode='chunk'` for training to save memory
2. Set appropriate `expand_k` and `expand_v` ratios based on your model size
3. **MD-GDN is recommended for RxLM** - it integrates memory features natively for better conversational coherence
4. **KDA is recommended for standalone use** due to its advanced per-channel gating
5. Gated DeltaNet is a good alternative with strong performance
6. Linear attention works best with longer sequences (>2048 tokens)
7. For KDA and MD-GDN, `expand_k` is not used - the model uses `head_dim` internally
8. MD-GDN maintains continuous state across interactions, improving multi-turn conversations

## Inference

Linear attention layers support both training and inference modes:

```python
# Standard inference (same as before)
model.eval()
with torch.no_grad():
    output = model.interact(**tokenized_query)
```

The linear attention layers handle caching automatically during generation, maintaining efficiency for autoregressive decoding.

## Architecture Details

When `use_linear_self_attn=True`:
- **Self-attention layers**: Use the specified linear attention mechanism (GLA/DeltaNet/Gated DeltaNet/KDA/MD-GDN)
- **Memory cross-attention layers**: Continue using standard attention (MHA/GQA/MQA/SQA)

This hybrid approach allows you to:
- Benefit from linear attention's efficiency for processing input sequences
- Maintain precise attention over the RxLM memory state
- (MD-GDN only) Integrate memory directly into self-attention for enhanced coherence

## Limitations

- Linear attention is only available for self-attention layers
- Memory cross-attention always uses standard attention mechanisms
- RoPE (Rotary Position Embeddings) are not used with linear attention (linear attention has its own position handling)

## References

- [flash-linear-attention GitHub](https://github.com/sustcsonglin/flash-linear-attention)
- [Gated Linear Attention Paper](https://arxiv.org/abs/2312.06635)
- [DeltaNet Paper](https://arxiv.org/abs/2102.11174)
- [Kimi Delta Attention Paper](https://arxiv.org/abs/2501.00000) - "Kimi Linear: An Expressive, Efficient Attention Architecture" (2025)
- [Memory-Driven Gated DeltaNet (MD-GDN) Documentation](md_gdn.md) - Extended documentation for MD-GDN

## Support

For issues or questions about linear attention in RxLM:
- GitHub Issues: [RxAI-dev/rxlm](https://github.com/RxAI-dev/rxlm/issues)
- Email: support@rxai.dev
