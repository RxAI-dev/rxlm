# Direct Memory and Preference Optimization (DMPO) and Supervised Memory-Aware Training (SMAT) Improvements

**Adam Filipek (adamfilipek@rxai.dev)**
*Reactive AI (https://rxai.dev)*
*January 2026*

## Abstract

This document describes the implementation of Direct Memory and Preference Optimization (DMPO), a novel training algorithm for Reactive Transformer that combines Direct Preference Optimization (DPO) with memory-aware training. Additionally, we detail improvements to the Supervised Memory-Aware Training (SMAT) pipeline to support the new RxT-Beta Interaction Template with hybrid reasoning and agentic tool usage capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [RxT-Beta Interaction Template](#rxt-beta-interaction-template)
3. [Dataset Improvements](#dataset-improvements)
4. [Direct Memory and Preference Optimization (DMPO)](#direct-memory-and-preference-optimization-dmpo)
5. [Anchored DMPO (APO-style)](#anchored-dmpo-apo-style)
6. [Implementation Details](#implementation-details)
7. [Usage Guide](#usage-guide)
8. [References](#references)

---

## Introduction

The original Reactive Transformer training curriculum consists of four supervised stages:
1. **Joint Language Model Pre-Training** - Co-trains decoder and encoder
2. **Joint Interaction Supervised Fine-Tuning (SFT)** - Adapts to conversational format
3. **Self-Supervised Memory Attention Pre-Training** - Trains memory attention network
4. **Supervised Memory-Aware Training (SMAT)** - Trains full memory-dependent interaction cycle

The original plan included a fifth stage: **Memory Reinforcement Learning (MRL)**, which would use reinforcement learning to further optimize memory updates and response quality. However, MRL proved to be slow and unstable in practice.

This document introduces **DMPO (Direct Memory and Preference Optimization)** as an alternative to MRL, combining the stability of direct preference optimization with memory-aware training. DMPO offers several advantages:

- **Stability**: Unlike RL-based approaches, DMPO doesn't require reward modeling or policy gradient estimation
- **Efficiency**: Single forward-backward pass per preference pair instead of multiple rollouts
- **Memory Integration**: Naturally integrates with RxT's memory system by updating memory only on accepted interactions
- **Preference Learning**: Directly optimizes the model to prefer high-quality responses over low-quality ones

---

## RxT-Beta Interaction Template

### Overview

The RxT-Beta model introduces a new Interaction Template that extends the simple query-answer format with support for:
- **Hybrid Reasoning**: Optional thinking/reasoning blocks before answers
- **Agentic Tool Usage**: Tool calls and tool result processing
- **Internal Instructions**: Per-interaction system prompts

### Special Tokens

| Token | Name | Description |
|-------|------|-------------|
| `[Q]` | Query | User's query/question |
| `[A]` | Answer | Model's response |
| `[T]` | Think | Thinking/reasoning block |
| `[C]` | Tool Call | Agentic tool call (JSON) |
| `[U]` | Tool Use | Tool result/usage |
| `[I]` | Internal | Internal instruction |

### Template Structures

**Fast Answer Mode:**
```
[BOS][I]internal[Q]query[A]answer[C]tool_call[EOS]
```

**Extended Thinking Mode:**
```
[BOS][I]internal[Q]query[T]thinking[A]answer[C]tool_call[EOS]
```

**Tool Usage Mode:**
```
[BOS][I]internal[U]tool_result[T]thinking[A]answer[C]tool_call[EOS]
```

### Hybrid Reasoning Control

The model's reasoning mode is controlled by the token at the end of the query:
- `[Q]query[A]` - Forces fast answer without reasoning
- `[Q]query[T]` - Activates extended thinking mode

### Agentic Tool Flow

1. **Tool Call**: Model generates `[C]tool_call_json` at the end of response
2. **Tool Execution**: External system executes the tool
3. **Tool Use**: Result is passed back as `[U]tool_result`
4. **Response**: Model processes result and generates summary/response

Tool calls can be chained, and answers can be optionally skipped for direct tool calls.

---

## Dataset Improvements

### HybridReasoningSftDataset

A new dataset class for Interaction SFT (Stage 2) that supports the full Interaction Template:

```python
from rxlm.training import HybridReasoningSftDataset

dataset = HybridReasoningSftDataset(
    interactions=data,
    tokenizer=tokenizer,
    max_seq_len=1024,
    # Field names
    query_field='query',
    answer_field='answer',
    think_field='think',
    tool_call_field='tool_call',
    tool_use_field='tool_use',
    internal_field='internal',
    # Special tokens
    query_token='[Q]',
    answer_token='[A]',
    think_token='[T]',
    tool_call_token='[C]',
    tool_use_token='[U]',
    internal_token='[I]',
)
```

**Dataset Format:**
```json
{
    "query": "What is the weather?",
    "think": "Let me check the weather API...",
    "answer": "I'll check the weather for you.",
    "tool_call": "{\"tool\": \"weather\", \"location\": \"NYC\"}",
    "internal": null,
    "tool_use": null
}
```

**Loss Masking:**
- **Masked (ignored)**: `internal`, `query`, `tool_use` tokens
- **Used for loss**: `think`, `answer`, `tool_call` tokens

### HybridReasoningSmatDataset

Extended SMAT dataset that uses a unified `interactions` list for all interactions (including the first one):

```python
from rxlm.training import HybridReasoningSmatDataset

dataset = HybridReasoningSmatDataset(
    episodes=data,
    tokenizer=tokenizer,
    max_seq_len=1024,
    interactions_field='interactions',
)
```

**Dataset Format:**
```json
{
    "interactions": [
        {
            "query": "Hello, how are you?",
            "answer": "I'm doing well, thank you!"
        },
        {
            "query": "What's the weather like?",
            "think": "Let me check...",
            "answer": "It's sunny today.",
            "tool_call": null
        }
    ]
}
```

Each interaction is tokenized into:
- **Query part**: `[BOS][I]internal[Q|U]query_or_tool_use`
- **Answer part**: `[T]think[A]answer[C]tool_call[EOS]`

---

## Direct Memory and Preference Optimization (DMPO)

### Theoretical Background

**Direct Preference Optimization (DPO)** [Rafailov et al., 2023] provides a simpler alternative to RLHF by directly optimizing the policy to prefer accepted over rejected responses. The standard DPO loss is:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]
$$

where:
- $y_w$: accepted (preferred) response
- $y_l$: rejected (dispreferred) response
- $\pi$: current policy
- $\pi_{\text{ref}}$: frozen reference policy
- $\beta$: temperature parameter controlling preference strength

### DMPO Extension for Reactive Transformer

DMPO adapts DPO for the memory-aware, multi-step setting of RxT:

#### 1. Memory-Conditioned Preferences

At each interaction step $t$, the model generates responses conditioned on the accumulated memory state $STM_{t-1}$. Preferences are learned in this memory-conditioned context:

$$
\pi(y|x) \rightarrow \pi(y|x, STM_{t-1})
$$

#### 2. Accepted-Only Memory Updates

Unlike standard SMAT where all interactions update memory, DMPO only updates memory based on accepted interactions:

$$
STM_t = \text{MemAttn}(STM_{t-1}, \text{Encoder}(\text{concat}(X_t, Y_t^{\text{accepted}})))
$$

This ensures the memory system learns to store high-quality information, not noise from rejected responses.

#### 3. Multi-Step Preference Propagation

Preference learning propagates through the memory update chain. The gradient flow is:

```
Loss ← Decoder ← Memory Cross-Attention ← STM_t ← Memory Attention ← Encoder
```

This allows the model to learn which memory states lead to better downstream responses.

### DMPO Algorithm

```
Algorithm: DMPO Training Step

Input: Batch of episodes with accepted/rejected pairs
Output: Updated model parameters

1. Initialize memory state STM_0 with random noise
2. For each interaction step t = 1 to N:
   a. Get query Q_t, accepted answer A_w, rejected answer A_l

   b. Clone memory state for reference: STM_ref = STM_{t-1}.detach()

   c. Update memory with ACCEPTED interaction only:
      ED_t = Encoder(concat(Q_t, A_w))
      STM_t = MemAttn(STM_{t-1}, ED_t)

   d. Compute policy log probs:
      logp_w = sum(log π(A_w | Q_t, STM_t))
      logp_l = sum(log π(A_l | Q_t, STM_t))

   e. Compute reference log probs (with frozen reference decoder):
      logp_w_ref = sum(log π_ref(A_w | Q_t, STM_ref))
      logp_l_ref = sum(log π_ref(A_l | Q_t, STM_ref))

   f. Compute DPO loss:
      L_t = -log σ(β * ((logp_w - logp_l) - (logp_w_ref - logp_l_ref)))

3. Total loss L = sum(L_t)
4. Backpropagate and update parameters
```

### Key Design Decisions

1. **Reference Model**: A frozen copy of the decoder is used as the reference policy. This provides a stable baseline for preference computation.

2. **Memory Alignment**: The reference model uses the same memory state as the policy model, ensuring fair comparison of responses.

3. **Query Masking**: Only answer tokens contribute to the log probability computation; query tokens are masked.

4. **Gradient Flow**: Gradients flow through the memory update when using accepted interactions, allowing end-to-end optimization of the memory system.

---

## Anchored DMPO (APO-style)

### Motivation

Standard DPO can sometimes lead to reward hacking, where the model learns to game the preference signal rather than genuinely improving response quality. **Anchored Preference Optimization (APO)** addresses this by adding a neutral reference point.

### Three-Way Preference

The anchored variant adds a neutral reference response:
- **Accepted**: Strongly preferred response ($y_w$)
- **Neutral**: Reference baseline ($y_n$) - neither good nor bad
- **Rejected**: Dispreferred response ($y_l$)

### Anchored Loss

The anchored loss combines standard DPO with regularization toward the neutral response:

$$
\mathcal{L}_{\text{APO}} = \mathcal{L}_{\text{DPO}} + \lambda \cdot \mathbb{E}\left[\left|\log \frac{\pi(y_n|x)}{\pi_{\text{ref}}(y_n|x)}\right|\right]
$$

where $\lambda$ is the anchor weight controlling regularization strength.

### Benefits

1. **Prevents Reward Hacking**: The neutral anchor prevents the model from drifting too far from reasonable behavior.

2. **Baseline Reference**: Provides a stable measurement point for preference strength.

3. **Safety Mechanism**: Maintains response quality even under strong preference optimization.

---

## Implementation Details

### DmpoDataset

```python
from rxlm.training import DmpoDataset

dataset = DmpoDataset(
    episodes=data,
    tokenizer=tokenizer,
    max_seq_len=1024,
    use_anchored=False,  # Set True for APO-style
)
```

**Dataset Format:**
```json
{
    "interactions": [
        {
            "query": "What is 2+2?",
            "accepted": {
                "think": "Simple arithmetic...",
                "answer": "The answer is 4.",
                "tool_call": null
            },
            "rejected": {
                "think": null,
                "answer": "I don't know.",
                "tool_call": null
            },
            "neutral": {  // Optional, for anchored variant
                "think": null,
                "answer": "Let me calculate that for you.",
                "tool_call": null
            }
        }
    ]
}
```

### DmpoTrainer

```python
from rxlm.training import DmpoTrainer, DmpoModel, DmpoDataset

# Create model
model = DmpoModel(
    encoder=encoder,
    decoder=decoder,
    memory_attention=memory_attention,
)

# Create trainer
trainer = DmpoTrainer(
    model=model,
    device=device,
    vocab_size=vocab_size,
    # DPO parameters
    beta=0.1,  # Temperature
    label_smoothing=0.0,
    # Training parameters
    max_seq_len=256,
    pad_token_id=0,
)

# Train
trainer(
    epochs=3,
    batch_size=8,
    dataset=train_dataset,
    validation_dataset=val_dataset,
    optimizer=optimizer,
)
```

### AnchoredDmpoTrainer

```python
from rxlm.training import AnchoredDmpoTrainer

trainer = AnchoredDmpoTrainer(
    model=model,
    device=device,
    vocab_size=vocab_size,
    beta=0.1,
    anchor_weight=0.1,  # Regularization strength
    max_seq_len=256,
    pad_token_id=0,
)
```

### Hyperparameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `beta` | DPO temperature | 0.05 - 0.5 | 0.1 |
| `label_smoothing` | Prevents overconfidence | 0.0 - 0.2 | 0.0 |
| `anchor_weight` | APO regularization | 0.05 - 0.2 | 0.1 |
| `max_seq_len` | Maximum sequence length | 256 - 2048 | 256 |

---

## Usage Guide

### Training Pipeline

The recommended training pipeline for RxT-Beta:

1. **Joint Pre-Training** (Stage 1) - Standard
2. **Interaction SFT** (Stage 2) - Use `HybridReasoningSftDataset`
3. **Memory Attention Pre-Training** (Stage 3) - Standard with `HybridReasoningSmatDataset`
4. **SMAT** (Stage 4) - Use `HybridReasoningSmatDataset`
5. **DMPO** (Stage 5) - Use `DmpoTrainer` with `DmpoDataset`

### Data Preparation

#### For Hybrid Reasoning SFT:
```python
# Each example is a single interaction
sft_data = [
    {
        "query": "What is Python?",
        "think": "Python is a programming language...",
        "answer": "Python is a high-level programming language.",
    },
    {
        "query": "Search for weather",
        "internal": "Use the weather tool",
        "answer": "Let me check the weather.",
        "tool_call": '{"tool": "weather", "query": "current"}',
    },
]
```

#### For SMAT:
```python
# Each example is an episode with multiple interactions
smat_data = [
    {
        "interactions": [
            {"query": "Hello!", "answer": "Hi there!"},
            {"query": "How are you?", "answer": "I'm doing well!"},
        ]
    }
]
```

#### For DMPO:
```python
# Each example has accepted/rejected pairs
dmpo_data = [
    {
        "interactions": [
            {
                "query": "Explain gravity",
                "accepted": {
                    "think": "Gravity is a fundamental force...",
                    "answer": "Gravity is the force that attracts objects with mass."
                },
                "rejected": {
                    "answer": "Gravity makes things fall down."
                },
            }
        ]
    }
]
```

### Best Practices

1. **Start with Lower Beta**: Begin with `beta=0.05` and increase if preferences aren't learned strongly enough.

2. **Use Label Smoothing**: Set `label_smoothing=0.1` to prevent overconfidence.

3. **Monitor Reward Margin**: Track the difference between accepted and rejected log probs during training.

4. **Validate Preference Accuracy**: Ensure the model correctly ranks accepted over rejected on validation data.

5. **Consider Anchored Variant**: If the model shows signs of reward hacking, switch to `AnchoredDmpoTrainer`.

---

## References

1. **DPO**: Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*.

2. **APO**: Ivison, H., et al. (2024). "Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment." *arXiv preprint*.

3. **RxT**: Filipek, A. (2025). "Reactive Transformer: Stateful Real-Time Processing for Event-Driven AI."

4. **RLHF**: Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS 2022*.

---

## Appendix: Mathematical Details

### Log Probability Computation

For a sequence of tokens $y = (y_1, ..., y_T)$ with attention mask $m = (m_1, ..., m_T)$:

$$
\log \pi(y|x) = \frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \cdot \log p(y_t | y_{<t}, x)
$$

### DPO Gradient

The gradient of the DPO loss with respect to model parameters $\theta$:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \sigma(-\beta \cdot r_\theta) \cdot \left(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right)
$$

where $r_\theta = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$

### Memory Update Gradient Flow

In DMPO, gradients flow through memory updates:

$$
\frac{\partial \mathcal{L}}{\partial \theta_{\text{enc}}} = \frac{\partial \mathcal{L}}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial STM_t} \cdot \frac{\partial STM_t}{\partial ED_t} \cdot \frac{\partial ED_t}{\partial \theta_{\text{enc}}}
$$

This enables end-to-end optimization of the encoder and memory attention network.
