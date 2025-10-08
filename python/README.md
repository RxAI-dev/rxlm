<span>
  <img src="https://raw.githubusercontent.com/RxAI-dev/rxlm/refs/heads/main/assets/logo/logo_rxai_v2.png" width="400" />
  <img src="https://raw.githubusercontent.com/RxAI-dev/rxlm/refs/heads/main/assets/logo/logo_rxnn_v2.png" width="400" />
</span>

# Reactive AI - RxLM: Reactive Language Models
## Part of Reactive Neural Networks (RxNN) Platform Ecosystem

> ### IMPORTANT LICENSE NOTICE - FREE & OPEN ONLY FOR NON-COMMERCIAL USAGE
> This framework (RxLM) and all the RxNN Platform Ecosystem is licensed under the **Reactive AI Framework License (RAFL) v1.0** (on bottom of the README),
> that's enabling free and open access only for non-commercial usage. Any commercial usage of the framework and derivatives require
> separate "Reactive AI Commercial License Agreement".
> 
> By accessing, using, downloading and copying/forking this framework, you're implicitly stating that you agree with the license terms.
> 
> Training and publishing the models using RxLM Framework and Reactive Transformer (RxT) is considered as usage and require
> licensing the model under the **Reactive AI Model & Architecture License (RAML) v1.0** (on bottom of the README).

RxNN is AI/Deep Learning development platform made for Reactive Neural Networks and Event-driven AI, introduced by Reactive AI.

> ### Documentation is still in progress

## Contents

- [A New Paradigm for Conversational AI: The Reactive Transformer](#a-new-paradigm-for-conversational-ai-the-reactive-transformer)
  - [The Limitations of Stateless Large Language Models](#the-limitations-of-stateless-large-language-models)
  - [Event-Driven AI: A Shift to Stateful, Real-Time Processing](#event-driven-ai-a-shift-to-stateful-real-time-processing)
  - [The Reactive Transformer (RxT) Architecture](#the-reactive-transformer-rxt-architecture)
  - [The Attention-Based Memory System (ABMS)](#the-attention-based-memory-system-abms)
  - [Sparse Query Attention (SQA): The Computational Engine of RxT](#sparse-query-attention-sqa-the-computational-engine-of-rxt)
- [Installation](#installation)
  -[Core dependencies](#core-dependencies)
    - [Installing FlashAttention for Maximum Performance](#installing-flashattention-for-maximum-performance)
- [HuggingFace Ecosystem Integration](#huggingface-ecosystem-integration)
- [Building Reactive Transformer Models](#building-reactive-transformer-models)
  - [Configuring the Core Components](#configuring-the-core-components)
  - [Assembling the Full RxT Model](#assembling-the-full-rxt-model)
- [The Reactive Transformer Training Curriculum](#the-reactive-transformer-training-curriculum)
  - [An Overview of the 6-Step Training Pipeline](#an-overview-of-the-6-step-training-pipeline)
  - [Curated Datasets for Each Training Stage](#curated-datasets-for-each-training-stage)
  - [Stage 0: Tokenizer Training](#stage-0-tokenizer-training)
  - [Stage 1: Joint Language Model Pre-Training](#stage-1-joint-language-model-pre-training)
  - [Stage 2: Joint Interaction Supervised Fine-Tuning (SFT)](#stage-2-joint-interaction-supervised-fine-tuning-sft)
  - [Stage 3: Self-Supervised Memory Attention Pre-Training](#stage-3-self-supervised-memory-attention-pre-training)
  - [Stage 4: Supervised Memory-Aware Training](#stage-4-supervised-memory-aware-training)
  - [Stage 5: Memory Reinforcement Learning (MRL)](#stage-5-memory-reinforcement-learning-mrl)
  - [Stage 6: Reactive RLHF (RxRLHF) (Future Work)](#stage-6-reactive-rlhf-rxrlhf-future-work)
- [Inference with Reactive Transformers](#inference-with-reactive-transformers)
  - [Basic Usage: Single, Turn-by-Turn Interactions](#basic-usage-single-turn-by-turn-interactions)
  - [Advanced Usage: High-Throughput Batch Processing](#advanced-usage-high-throughput-batch-processing)
- [Training Decoder-Only LLMs with Sparse Query Attention (SQA)](#training-decoder-only-llms-with-sparse-query-attention-sqa)
- [TensorBLEU: A GPU-Accelerated Metric for In-Training Evaluation](#tensorbleu-a-gpu-accelerated-metric-for-in-training-evaluation)
- [MRQbench: A Benchmark for Memory Retention Quality (Announcement)](#mrqbench-a-benchmark-for-memory-retention-quality-announcement)
- [API Reference](#api-reference)
- [Contact and Licensing](https://www.google.com/search?q=%23contact-and-licensing)

## A New Paradigm for Conversational AI: The Reactive Transformer

This repository provides the official implementation of the **Reactive Language Model (RxLM)** framework, centered around the **Reactive Transformer (RxT)**, a novel architecture designed for stateful, real-time, and computationally efficient conversational AI. RxLM represents a fundamental paradigm shift away from the dominant stateless approach of modern Large Language Models (LLMs).

### The Limitations of Stateless Large Language Models

The Transformer architecture has become the foundation for modern AI, yet its application to dialogue is constrained by a critical design flaw: it is inherently stateless. To maintain context in a conversation, models like GPT or Llama must re-ingest and re-process the entire history of the dialogue with every new turn. This "brute-force" context management leads to two debilitating problems:

1.  **Quadratic Computational Scaling:** The computational cost scales quadratically with the length of the conversation history. For a conversation with $N$ interactions of average length $T$, the total user-facing cost is on the order of $O(N^2 \cdot T)$. This makes long-running dialogues economically impractical and environmentally costly.
2.  **High Latency:** As the conversation grows, the time required to process the ever-expanding prompt increases, leading to noticeable delays in response generation.

This approach is not only inefficient but also fundamentally unnatural. Humans do not need to recall an entire day's history to remember a conversation from ten minutes prior. This constant reprocessing is a major source of hallucinations in long dialogues, as the model struggles to differentiate information from different time steps within a flattened sequence.

### Event-Driven AI: A Shift to Stateful, Real-Time Processing

RxLM is built on the **Event-Driven AI** paradigm, which reframes the problem of conversation. Instead of processing a monolithic block of data (the full chat history), the model operates in a continuous, stateful loop, treating each user query and model response as a discrete event. This approach is guided by the **"Reactivity Hypothesis,"** which posits that "real awareness and AGI models require continuous, stateful, real-time processing".

By maintaining an internal, fixed-size memory state and processing only the current interaction, the Reactive Transformer achieves **linear computational scaling**. The total cost of a conversation scales as $O(N \cdot T)$, making long, coherent, and economically viable dialogues possible.

### The Reactive Transformer (RxT) Architecture

The RxT is an encoder-decoder architecture designed specifically for the event-driven paradigm. Its operational flow is cyclical and features an asynchronous memory update mechanism that is key to its low-latency performance.

1.  **Response Generation (Synchronous):** At interaction step $t$, the **Generator-Decoder** receives the user query $X_t$. It generates a response $Y_t$ by conditioning on both the query and the memory state from the previous interaction, $STM_{t-1}$, which is accessed via Memory Cross-Attention. This is the only part of the cycle the user experiences directly.
2.  **Memory Update (Asynchronous):** After the response $Y_t$ has been streamed to the user, the **Memory Encoder** processes the complete interaction—a concatenation of the query and the generated answer, $concat(X_t, Y_t)$. It produces a rich semantic representation called the Encoded Data, $ED_t$.
3.  **Memory Consolidation (Asynchronous):** A dedicated **Memory Attention** network takes the previous memory state $STM_{t-1}$ and the new Encoded Data $ED_t$ as input to compute the updated memory state, $STM_t$. This new state is then ready for the next interaction, $t+1$.

This asynchronous design ensures that the computationally intensive memory update process happens "in the background" and does not add to the user-perceived latency, enabling truly real-time interaction.

### The Attention-Based Memory System (ABMS)

The core innovation of RxT is its integrated memory system. The Short-Term Memory (STM) is not a cache of past tokens but a collection of fixed-size, learnable vectors, or "memory slots," organized into layers.

  * **Memory Read:** During generation, the decoder accesses the conversational context stored in the STM via **Memory Cross-Attention**. In this operation, the hidden states of the decoder's input act as the Queries ($Q$), while the memory slots from the corresponding STM layer act as the Keys ($K$) and Values ($V$):
    $$RetrievedContext = Attention(Q=H_{dec}, K=STM_{t-1}, V=STM_{t-1})$$
  * **Memory Write:** The memory update process is handled by the Memory Attention network. Here, the memory slots from the previous state ($STM_{t-1}$) act as the Queries ($Q$), actively seeking out relevant information from the latest interaction, which is provided by the Encoded Data ($ED_t$) as Keys ($K$) and Values ($V$):
    $$Update = Attention(Q=STM_{t-1}, K=ED_t, V=ED_t)$$
  * **Memory Plasticity:** To control the rate of change and prevent catastrophic forgetting, the memory update is managed by **Residual Gates**. Instead of a simple addition, the gate computes a dynamic, weighted interpolation between the old and new states, providing stability during training and long-term interaction :
    $$G_{\text{elementwise}} = sigmoid(W_{\text{gate}} \odot (STM_{t-1} + Update))$$ or $$G_{\text{linear}} = sigmoid(W_{\text{gate}} \cdot (STM_{t-1} + Update) + b_{\text{gate}})$$
    $$STM_t = (1 - G) \odot STM_{t-1} + G \odot Update$$

### Sparse Query Attention (SQA): The Computational Engine of RxT

The architectural design of the Reactive Transformer creates specific computational requirements that are perfectly met by **Sparse Query Attention (SQA)**, another key innovation within the RxLM framework.

While optimizations like Multi-Query (MQA) and Grouped-Query Attention (GQA) are designed to reduce the memory bandwidth bottleneck during autoregressive decoding, they provide no speed advantage in compute-bound scenarios where a full sequence is processed in parallel. The RxT architecture contains two such components: the **Memory Encoder** and the **Memory Attention** network. Both of these process a complete interaction sequence in a single, non-autoregressive forward pass.

This is precisely the scenario where SQA excels. Instead of reducing Key/Value heads, SQA reduces the number of Query heads. This architectural change directly reduces the number of floating-point operations (FLOPs) required for the attention score computation by a factor of $H/H_q$, where $H$ is the total number of heads and $H_q$ is the number of query heads.

The use of SQA is not an incidental choice but a deliberate, synergistic co-design. The RxT architecture creates the ideal conditions for SQA to provide maximum benefit, making the entire memory update cycle significantly faster and more efficient than it would be with standard attention mechanisms. All RxT models in this framework leverage SQA for this reason.

## Installation

Setting up the `rxlm` environment requires installing the core library and its dependencies, with special attention given to the `flash-attn` package for optimal performance.

### Core Dependencies

Install the main packages using `pip install rxlm torch transformers tokenizers huggingface_hub datasets tensorboard`

### Installing FlashAttention for Maximum Performance

`flash-attn` provides a highly optimized, memory-efficient implementation of the attention mechanism and is crucial for achieving high performance with `rxlm` models. However, its installation can be complex. The following methods are ordered from simplest to most advanced.

#### Method 1: Direct Installation with Pip (Not recommended)

This is the main method for most Linux environments with a compatible NVIDIA GPU (Ampere, Ada, or Hopper architecture) and CUDA toolkit installed, but it's extremely slow, so it's better to use pre-compiled version..

First, ensure the necessary build tools are installed:
```bash
pip install packaging ninja
```

Then, install `flash-attn` using the `--no-build-isolation` flag:

```bash
pip install flash-attn --no-build-isolation
```

**Note:** If you are on a machine with many CPU cores but limited RAM (\< 96 GB), the compilation may fail by exhausting memory. To prevent this, limit the number of parallel compilation jobs by setting the `MAX_JOBS` environment variable [2, 3]:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

#### Method 2: Using Pre-compiled Wheels (Easiest for All Platforms)

Attempting to compile `flash-attn` from source can sometimes be a "trap," leading to very long build times or failures.[4] A more reliable alternative is to use the pre-compiled binaries ("wheels") provided with each release on the official [GitHub releases page](https://github.com/Dao-AILab/flash-attention/releases).

1.  **Identify the correct wheel file.** The filename contains information about the Python version (`cp310` for Python 3.10), CUDA version (`cu124` for CUDA 12.4), PyTorch version (`torch2.4`), and platform (`linux_x86_64` or `win_amd64`). Select the file that matches your environment.
2.  **Download and install the wheel.** Copy the URL to the file and install it using `pip`. For example:
    ```bash
    # Example for Linux, Python 3.10, CUDA 12.3, PyTorch 2.4
    wget [https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
    pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```

## HuggingFace Ecosystem Integration

The `rxlm` framework is designed to integrate seamlessly with the HuggingFace ecosystem, allowing you to leverage its powerful tools for model sharing, data handling, and tokenization.

  * **Hub:** Pre-trained RxT models, configurations, and tokenizers are connected to HuggingFace Hub. You can load models directly using the familiar `from_pretrained` method.
  * **Datasets:** The `datasets` library is the recommended tool for loading and pre-processing the large corpora required for pre-training and the structured conversational datasets needed for fine-tuning. RxLM is using it internally.
  * **Tokenizers:** You can load tokenizers trained specifically for RxT models directly from the Hub using the provided helper functions in the framework.

## Building Reactive Transformer Models

A Reactive Transformer model is constructed from three primary components: a Generator-Decoder, a Memory Encoder, and a Memory Attention network.

### Configuring the Core Components

Each component is defined by a configuration object that specifies its architecture.

  * **Generator-Decoder (`RxTDecoder`):** This is the component responsible for autoregressive text generation. To manage parameter counts effectively while maintaining high capacity, its feed-forward networks are implemented as Mixture-of-Experts (MoE) layers. Its configuration specifies the number of layers, hidden dimensions, attention heads, and MoE parameters.
  * **Memory Encoder (`RxTDecoder`):** The encoder's sole purpose is to create a condensed representation of a completed interaction. It is typically a smaller, dense Transformer (without MoE layers) that processes the concatenated query and response.
  * **Memory Attention Network:** This network is responsible for the memory update mechanism. Its configuration specifies the type of memory attention to use ('simple', 'self', 'interlayer', 'self-interlayer'), which determines how information is consolidated within the memory state. Interlayer variants are recommended.

## The Reactive Transformer Training Curriculum

Training a multi-component, asynchronous system like the Reactive Transformer is a complex task that cannot be solved with a naive, end-to-end approach. Such a strategy would fail due to an unstable learning signal, as the untrained decoder and memory system would corrupt each other's gradients.

To solve this, `rxlm` employs a carefully designed, multi-stage training curriculum. This curriculum acts as a **scaffolding process**, systematically pre-training and integrating each component to solve predictable failure modes before unifying the entire system. This structured approach is essential for successfully training a functional, stateful model.

### An Overview of the 6-Step Training Pipeline

The full training pipeline consists of several supervised and reinforcement learning stages designed to incrementally build the model's capabilities.

1.  **Joint LM Pre-Training:** Learn fundamental knowledge and establish a shared semantic space between the encoder and decoder.
2.  **Joint Interaction SFT:** Adapt the language components to the conversational format.
3.  **Self-Supervised Memory Attention Pre-Training:** Bootstrap the memory update network.
4.  **Supervised Memory-Aware Training:** Train the full, end-to-end memory-dependent interaction cycle.
5.  **Memory Reinforcement Learning (MRL):** Refine conversational quality with sequence-level rewards.
6.  **Reactive RLHF (RxRLHF):** Align the model with human preferences (future work).

The logic behind this sequence is to systematically de-risk the training process. By first teaching the decoder *how* to use a perfect memory signal (Stages 1-2) and independently teaching the memory network *how* to produce a coherent signal (Stage 3), the final integration (Stage 4) becomes a stable and solvable learning problem.

### Curated Datasets for Each Training Stage

Different stages of the curriculum require different types of data.

  * **Stages 1 & 2:** Require a large text corpus (e.g., FineWeb-edu) for general language understanding, and a dataset of structured query-response pairs for conversational fine-tuning.
  * **Stages 3 & 4:** Require datasets of multi-turn dialogues to train the memory update and retention mechanisms.

### Stage 0: Tokenizer Training

Before training begins, a tokenizer is trained on the target corpus. It must include special tokens to delineate conversational structure, such as `[Q]`, `[A]`, and  `[T]` (for "thinking").

### Stage 1: Joint Language Model Pre-Training

This initial stage co-trains the Generator-Decoder and Memory Encoder to establish a shared semantic foundation. The model is trained with a dual-objective function :

1.  **Encoder Task (MLM):** The encoder receives a randomly masked version of an input sequence and is trained to predict the masked tokens via a Masked Language Modeling loss, $\mathcal{L}_{MLM}$.
2.  **Decoder Task (Autoregressive):** The decoder receives the original, unmasked sequence. For its Memory Cross-Attention layers, it is given a "cheated" or "teacher-forced" context. This context is the hidden state output from the pre-trained encoder. Crucially, this encoder output is **detached from the computation graph**, so no gradients flow back from the decoder to the encoder. A small amount of noise is added to this context for regularization. The decoder is then trained with a standard autoregressive cross-entropy loss, $\mathcal{L}_{AR}$.

The total loss is a weighted sum: $\mathcal{L}_{Joint} = \alpha\mathcal{L}_{AR} + \beta\mathcal{L}_{MLM}$. This process forces the decoder to learn how to effectively use its memory cross-attention mechanism by providing it with a perfect, stable context signal.

### Stage 2: Joint Interaction Supervised Fine-Tuning (SFT)

This stage follows the **exact same algorithm** as Stage 1. The only difference is the training data, which is shifted from a general text corpus to a dataset of structured conversational turns formatted with the special tokens (e.g., `[Q]...[A]...`). This adapts the model to the specific turn-taking format of dialogue.

### Stage 3: Self-Supervised Memory Attention Pre-Training

This stage addresses the central challenge of training the Memory Attention network: there are no ground-truth labels for what an "ideal" memory state should look like. To solve this "cold start" problem, a self-supervised proxy task is used.

1.  A pseudo-label, $STM_{target}$, is generated via a dynamic weighted average of the previous memory state $STM_{t-1}$ and the new Encoded Data $ED_t$:
    $$STM_{target} = (1 - w_t) \cdot STM_{t-1} + w_t \cdot ED_t$$
2.  The Memory Attention network computes the actual updated state $STM_t = MemAttn(STM_{t-1}, ED_t)$.
3.  The loss function is the negative cosine similarity between the predicted state and the target state, which encourages semantic alignment:
    $$\mathcal{L}_{Mem} = -\text{cosine\_similarity}(STM_t, STM_{target})$$

This stage is critical. Without it, the randomly initialized memory network would feed noise to the decoder, which would quickly learn to ignore its memory cross-attention layers entirely, defeating the purpose of the architecture. This pre-training ensures the memory network produces semantically coherent outputs.

### Stage 4: Supervised Memory-Aware Training

This is the final supervised stage, where all pre-trained components are unified to train the model on its intended, event-driven operational cycle. This is the first time the decoder learns to rely on a meaningful, accumulated memory state from genuinely past interactions.

The algorithm proceeds over a curriculum of multi-step dialogues.

1.  For the first interaction in a dialogue, the memory state $STM_0$ is initialized (e.g., with random noise). The decoder generates a response conditioned on this blank state, and the autoregressive loss is computed.
2.  For each subsequent step $t$, the full interaction $I_t$ is encoded to produce $ED_t$. The memory is updated: $STM_t = MemAttn(STM_{t-1}, ED_t)$.
3.  The decoder then processes the next interaction, $I_{t+1}$, conditioned on the newly computed memory state $STM_t$, and the loss is calculated.
4.  The total loss is the sum of losses from all steps, and backpropagation is performed through the entire sequence of operations.

Upon completion, the model has a functional memory system and is prepared for refinement via reinforcement learning.

### Stage 5: Memory Reinforcement Learning (MRL)

Supervised training teaches the model fluency and style, but it optimizes for next-token prediction, which doesn't always align with conversational quality. The MRL stage refines the model by optimizing for sequence-level metrics that better capture coherence and relevance. Using a custom modification of PPO Reinforcement Learning algorithm - Implicit Memory Policy Optimization (IMPO), the model is trained to maximize a reward signal computed from its generated responses, such as a BLEU score against a reference. This stage is made computationally feasible by tools like **TensorBLEU** (described below). More details soon with planned research paper.

### Stage 6: Reactive RLHF (RxRLHF) (Future Work)

The final stage in the planned curriculum is Reactive Reinforcement Learning from Human Feedback. This will adapt the standard RLHF alignment technique for stateful models. Instead of just rating single responses, human feedback will be used to train a reward model that also considers conversational context and memory, allowing for alignment on long-term coherence and consistency.

## Inference with Reactive Transformers

Using a trained RxT model involves initializing its memory and then interacting with it on a turn-by-turn basis. The framework supports both simple single-stream interaction and high-throughput batch processing.

### Basic Usage: Single, Turn-by-Turn Interactions

The following example demonstrates a typical interaction loop with a trained `RxTAlpha` model.

```python
from rxlm.rxt.models import RxTAlpha
from rxlm.training.tokenizer import load_tokenizer_from_hf_hub

# Load a pre-trained model and its tokenizer
tokenizer = load_tokenizer_from_hf_hub('ReactiveAI/RxT-Alpha-Mini-Supervised')
model = RxTAlpha.from_pretrained('ReactiveAI/RxT-Alpha-Mini-Supervised', tokenizer=tokenizer)
model.share_components() # Connects shared embeddings and STM

seq_len = 256

# Initialize the Short-Term Memory (STM), can be used as a system prompt
stm_init_text = "You are a helpful and creative assistant."
stm_init_state = model.tokenize_full_interaction(stm_init_text, '', max_seq_len=seq_len)
model.init_stm_state(**stm_init_state)

# Helper function to handle the interaction loop
def interaction(query: str):
    tokenized_query = model.tokenize_query(query, max_seq_len=seq_len)
    print(f"User: {query}")
    print("Model: ", end='')
    
    # The model.interact method is a generator
    for token_id in model.interact(**tokenized_query, max_seq_len=seq_len, temperature=0.8):
        if token_id == -1:
            # Special token indicating the start of the asynchronous memory update
            print('\n', end='')
        elif token_id == -2:
            # Special token indicating the memory update is complete
            print('[Memory updated]\n')
        else:
            # Stream the generated text token by token
            txt_token = model.stringify_token(token_id)
            print(txt_token, end='')

# Process the first interaction
interaction("Tell me a short story about a brave knight.")

# Process a follow-up interaction; the model will use its memory
interaction("What was the knight's name?")
```

The special return values `-1` and `-2` from the `interact` generator directly correspond to the asynchronous memory update phase of the RxT architecture. They signal that the user-facing response is complete and the internal state consolidation is now occurring.

### Advanced Usage: High-Throughput Batch Processing

For production environments, `rxlm` supports batch processing of multiple independent conversations simultaneously. This is managed through methods like `batch_interact` and `batch_interactions`.
To use batch mode, you must first configure the model's internal STM to handle a batch of memory states:

```python
batch_size = 16
model.set_batch_mode(use_batch_mode=True, batch_size=batch_size)
```

The `batch_interact` method can then process a list of queries, that will be processed as batch and will return a list of generated tokens, managing the generation and memory updates for all conversations in parallel, offering significant throughput gains. This method handles padding, attention masking, and stopping criteria for each sequence in the batch.

## Training Decoder-Only LLMs with Sparse Query Attention (SQA)

While SQA is integral to the Reactive Transformer, it is also a powerful, standalone attention mechanism that can significantly accelerate the training of standard decoder-only LLMs.

As established, SQA reduces the computational complexity (FLOPs) of the attention mechanism, which is the primary bottleneck during training and other full-sequence processing tasks. In these scenarios, memory-bandwidth optimizations like GQA offer no performance benefit.

By replacing the standard `GQA` or `MHA` layers in a traditional LLM with an `SQA` implementation, you can achieve substantial throughput improvements with only a minor impact on final model quality, as demonstrated in small-scale experiments. This translates directly to faster training times and lower computational costs.

The table below, adapted from benchmarks in the SQA paper, illustrates the performance advantage of SQA over baseline attention mechanisms in a compute-bound forward pass on long sequences.

**Table 1: SQA Performance Benchmark on Long Sequences (Time per step in seconds)**

| Seq. Length | xSQA (Fastest) | SQA (Balanced) | GQA (Baseline) | MHA (Baseline) | Speedup (xSQA vs GQA) |
| :---------- | :------------- | :------------- | :------------- | :------------- | :-------------------- |
| 32,768 | 0.1348 | 0.1991 | 0.3637 | 0.3727 | **2.70x** |
| 131,072 | 0.3759 | 0.6308 | 1.2558 | 1.2648 | **3.34x** |
| 200,000 | 0.8194 | 1.4116 | 2.8596 | 2.8734 | **3.49x** |

For researchers and practitioners training LLMs from scratch, integrating SQA can lead to significant savings in both time and financial resources.

## TensorBLEU: A GPU-Accelerated Metric for In-Training Evaluation

The advanced training stages of the RxT curriculum, particularly Memory Reinforcement Learning (MRL), require a dense reward signal to be calculated for every generated sample in every training batch. Using standard, CPU-based metrics like NLTK's `sentence_bleu` for this purpose creates a severe performance bottleneck, as data must be moved from GPU to CPU and processed in a slow, serial loop.

**TensorBLEU** is a custom metric implementation designed to solve this problem. It is a critical piece of infrastructure that makes MRL computationally feasible. Its key features are :

  * **Fully Vectorized and GPU-Accelerated:** It operates directly on batches of token ID tensors within PyTorch, eliminating the GPU-CPU data transfer bottleneck.
  * **Per-Sentence Calculation:** It computes a separate BLEU score for each sample in the batch, providing the granular reward signal needed for effective RL.
  * **Memory-Efficient Counting:** A naive vectorization of n-gram counting would lead to a memory explosion. TensorBLEU uses a novel mechanism based on `torch.unique` to create a compact, batch-specific dictionary of n-grams. This keeps memory usage proportional to the number of unique n-grams present in the batch, not the entire vocabulary size.

Benchmarks show that TensorBLEU provides dramatic speedups over NLTK, ranging from **13x** on consumer-grade GPUs to over **40x** on data-center-class hardware. This transforms the reward calculation from a major training bottleneck into a negligible overhead.

While TensorBLEU is a "Token-ID BLEU" and is not suitable for final, publication-ready model evaluation (for which text-based tools like SacreBLEU should be used), it is the ideal tool for in-training optimization and relative performance tracking.

## MRQbench: A Benchmark for Memory Retention Quality (Announcement)

To facilitate the standardized evaluation of stateful language models, we are developing the **Memory Retention Quality Benchmark (MRQbench)**. Standard metrics like perplexity do not adequately capture a model's ability to maintain context and coherence over long conversations.

MRQbench will provide a suite of multi-turn dialogue tasks designed to specifically test a model's memory. Performance will be evaluated using a composite reward score, similar to the one used for MRL, which measures not only the fluency of a response but also its semantic relevance to both the immediate query and the broader history of the conversation. This tool will enable a fair and direct comparison of the memory capabilities of stateful architectures like RxT against stateless models, pushing research towards more coherent and context-aware conversational agents.

## API Reference

This section provides a brief overview of the main user-facing model classes.

### `rxlm.rxt.models.RxTAlpha`

The primary implementation of the Reactive Transformer.

```python
class RxTAlpha(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        decoder_config: RxTComponentConfig,
        encoder_config: RxTComponentConfig,
        memory_attention_config: RxTInterlayerMemoryAttentionConfig,
        tokenizer_config: RxTAlphaTokenizerConfig,
        tokenizer: Union = None,
        memory_attention_variant: Literal['interlayer', 'self-interlayer'] = 'interlayer',
        **kwargs
    ):
        #...
```

  * **Key Parameters:**
      * `decoder_config`: Configuration for the Generator-Decoder component.
      * `encoder_config`: Configuration for the Memory Encoder component.
      * `memory_attention_config`: Configuration for the Memory Attention network.
      * `tokenizer_config`: Dictionary containing special token IDs.
      * `memory_attention_variant`: Defaults to `'interlayer'`.
  * **Key Methods:**
      * `from_pretrained(model_id,...)`: Loads a pre-trained model from the HuggingFace Hub.
      * `share_components()`: Connects the shared embeddings and STM between components. Must be called after initialization.
      * `init_stm_state(input_ids, attention_mask)`: Initializes or resets the model's memory state using a given text.
      * `interact(input_ids,...)`: A generator that yields tokens for a single interaction and handles the memory update cycle.
      * `set_batch_mode(use_batch_mode, batch_size)`: Configures the STM to handle batches of conversations.
      * `batch_interactions(input_ids,...)`: Processes a batch of queries to generate a batch of responses.

### `rxlm.rxt.models.RxTBeta`

An alternative variant of the Reactive Transformer with a more expressive memory attention mechanism.

```python
class RxTBeta(RxTAlpha):
    def __init__(self,...):
        super(RxTBeta, self).__init__(
          ...,
            memory_attention_variant='self-interlayer',
            **kwargs
        )
```

The `RxTBeta` class inherits directly from `RxTAlpha`. Its only difference is that it defaults the `memory_attention_variant` to `'self-interlayer'`, which corresponds to the Gated Self/Interlayer Memory Attention variant described in the research paper. This variant may offer improved performance on complex dialogue tasks at the cost of slightly increased computational complexity in the memory update phase.

## Contact and Licensing

  * [**Website:** https://rxai.dev](https://rxai.dev)
  * [**HuggingFace**](https://huggingface.co/ReactiveAI)
  * **Commercial Licensing:** `licensing@rxai.dev`
  * **Support & Inquiries:** `contact@rxai.dev`
  * **Author:** `adamfilipek@rxai.dev`

