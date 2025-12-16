"""
Utilities for torch.compile optimization of RxT models.

Usage:
    from rxlm.compile import compile_model, CompileConfig

    # Compile with default settings
    compiled_model = compile_model(model)

    # Compile with custom config
    config = CompileConfig(mode='max-autotune', fullgraph=False)
    compiled_model = compile_model(model, config)

    # For training models
    compiled_training_model = compile_training_model(training_model)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class CompileConfig:
    """Configuration for torch.compile optimization."""
    mode: Literal['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'] = 'reduce-overhead'
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    backend: str = 'inductor'
    disable: bool = False


# Default configs for different use cases
TRAINING_CONFIG = CompileConfig(
    mode='reduce-overhead',
    fullgraph=False,
    dynamic=True,  # Dynamic shapes for varying batch sizes
)

INFERENCE_CONFIG = CompileConfig(
    mode='max-autotune',
    fullgraph=False,
    dynamic=False,  # Static shapes for inference
)


def compile_module(module: nn.Module, config: CompileConfig = None) -> nn.Module:
    """
    Compile a single module with torch.compile.

    Args:
        module: The module to compile
        config: CompileConfig instance (uses TRAINING_CONFIG by default)

    Returns:
        Compiled module
    """
    if config is None:
        config = TRAINING_CONFIG

    if config.disable:
        return module

    return torch.compile(
        module,
        mode=config.mode,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
        backend=config.backend,
    )


def compile_rope(rope_module: nn.Module, config: CompileConfig = None) -> nn.Module:
    """
    Compile the RoPE module's _rotate method for optimized rotation.

    The _rotate method is a hot path that benefits significantly from compilation.
    """
    if config is None:
        config = CompileConfig(mode='reduce-overhead', fullgraph=True, dynamic=True)

    if config.disable:
        return rope_module

    # Compile the _rotate method specifically
    rope_module._rotate = torch.compile(
        rope_module._rotate,
        mode=config.mode,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
    )
    return rope_module


def compile_ff_layers(model: nn.Module, config: CompileConfig = None) -> nn.Module:
    """
    Compile all GatedFeedForward and FeedForward layers in the model.

    These layers are compute-intensive and benefit from compilation.
    """
    if config is None:
        config = CompileConfig(mode='reduce-overhead', fullgraph=True, dynamic=True)

    if config.disable:
        return model

    from .transformers.ff import GatedFeedForward, FeedForward, GatedLinearUnit

    def _compile_ff(module):
        for name, child in module.named_children():
            if isinstance(child, (GatedFeedForward, FeedForward)):
                compiled = torch.compile(
                    child,
                    mode=config.mode,
                    fullgraph=config.fullgraph,
                    dynamic=config.dynamic,
                )
                setattr(module, name, compiled)
            elif isinstance(child, GatedLinearUnit):
                compiled = torch.compile(
                    child,
                    mode=config.mode,
                    fullgraph=config.fullgraph,
                    dynamic=config.dynamic,
                )
                setattr(module, name, compiled)
            else:
                _compile_ff(child)

    _compile_ff(model)
    return model


def compile_moe_expert_weights(model: nn.Module, config: CompileConfig = None) -> nn.Module:
    """
    Compile the _apply_expert_weights method in VectorizedMoE layers.

    Note: The router is intentionally excluded due to @torch._dynamo.disable decorator.
    """
    if config is None:
        config = CompileConfig(mode='reduce-overhead', fullgraph=True, dynamic=True)

    if config.disable:
        return model

    from .transformers.moe import VectorizedMoeFeedForward, VectorizedGatedMoeFeedForward

    def _compile_moe(module):
        for name, child in module.named_children():
            if isinstance(child, (VectorizedMoeFeedForward, VectorizedGatedMoeFeedForward)):
                child._apply_expert_weights = torch.compile(
                    child._apply_expert_weights,
                    mode=config.mode,
                    fullgraph=config.fullgraph,
                    dynamic=config.dynamic,
                )

                child._compute_shared_experts = torch.compile(
                    child._compute_shared_experts,
                    mode=config.mode,
                    fullgraph=config.fullgraph,
                    dynamic=config.dynamic,
                )
            else:
                _compile_moe(child)

    _compile_moe(model)
    return model


def compile_attention_layers(model: nn.Module, config: CompileConfig = None) -> nn.Module:
    """
    Compile attention computation (excluding Flash Attention which is already optimized).

    For non-Flash attention, compile the attention score computation.
    """
    if config is None:
        config = CompileConfig(mode='reduce-overhead', fullgraph=False, dynamic=True)

    if config.disable:
        return model

    from .transformers.positional import RotaryPositionalEmbedding

    def _compile_attention(module):
        for name, child in module.named_children():
            if isinstance(child, RotaryPositionalEmbedding):
                # Compile RoPE _rotate method
                child._rotate = torch.compile(
                    child._rotate,
                    mode=config.mode,
                    fullgraph=True,
                    dynamic=config.dynamic,
                )
            else:
                _compile_attention(child)

    _compile_attention(model)
    return model


def compile_model(
    model: nn.Module,
    config: CompileConfig = None,
    compile_ff: bool = True,
    compile_moe: bool = True,
    compile_attention: bool = True,
    compile_full_model: bool = False,
) -> nn.Module:
    """
    Apply torch.compile optimizations to an RxT model.

    Args:
        model: The model to compile (encoder, decoder, or training model)
        config: CompileConfig instance
        compile_ff: Whether to compile FeedForward layers
        compile_moe: Whether to compile MoE expert weight application
        compile_attention: Whether to compile attention (RoPE) layers
        compile_full_model: Whether to compile the entire model (not recommended for MoE)

    Returns:
        Model with compiled components
    """
    if config is None:
        config = TRAINING_CONFIG

    if config.disable:
        return model

    if compile_full_model:
        return compile_module(model, config)

    if compile_attention:
        model = compile_attention_layers(model, config)

    if compile_ff:
        model = compile_ff_layers(model, config)

    if compile_moe:
        model = compile_moe_expert_weights(model, config)

    return model


def compile_training_model(
    training_model: nn.Module,
    config: CompileConfig = None,
) -> nn.Module:
    """
    Apply torch.compile optimizations to a JointTrainingModel.

    This compiles the encoder, decoder, and MLM head separately
    to maintain compatibility with the MoE router.

    Args:
        training_model: JointTrainingModel instance
        config: CompileConfig instance

    Returns:
        Training model with compiled components
    """
    if config is None:
        config = TRAINING_CONFIG

    if config.disable:
        return training_model

    # Compile encoder
    if hasattr(training_model, 'encoder'):
        training_model.encoder = compile_model(
            training_model.encoder,
            config,
            compile_ff=True,
            compile_moe=False,  # Encoder typically doesn't have MoE
            compile_attention=True,
        )

    # Compile decoder
    if hasattr(training_model, 'decoder'):
        training_model.decoder = compile_model(
            training_model.decoder,
            config,
            compile_ff=True,
            compile_moe=True,
            compile_attention=True,
        )

    # Compile MLM head
    if hasattr(training_model, 'mlm_head'):
        training_model.mlm_head = compile_module(
            training_model.mlm_head,
            CompileConfig(mode=config.mode, fullgraph=True, dynamic=config.dynamic)
        )

    return training_model


# Convenience function to check if torch.compile is available
def is_compile_available() -> bool:
    """Check if torch.compile is available (PyTorch 2.0+)."""
    return hasattr(torch, 'compile')
