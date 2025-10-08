# **TensorBLEU: Vectorized GPU-based BLEU Score Implementation for Per-Sentence In-Training Evaluation**

*Adam Filipek (adamfilipek@rxai.dev)*  
*Reactive AI (https://rxai.dev)*  
*October 2025*  

## Abstract
> Modern natural language processing models have achieved unprecedented scale, yet the tools for their evaluation often remain a computational bottleneck, limiting the pace of research. This is particularly acute for in-training evaluation metrics, such as per-sentence reward signals in Reinforcement Learning, which must operate efficiently on batches of token IDs directly on the GPU. In this paper, we introduce TensorBLEU, a novel implementation of the BLEU metric designed from the ground up for this specific use case. Our approach is fully vectorized for GPU-accelerated, per-sentence computation within PyTorch and introduces a memory-efficient counting mechanism. By creating a compact, batch-specific dictionary of n-grams using `torch.unique`, our method avoids the prohibitive memory costs of traditional hashing-based vectorization, making it practical for large-vocabulary models. We benchmark TensorBLEU against NLTK, the standard library for token-ID-based BLEU calculation on the CPU. Experiments show that TensorBLEU provides speedups of over 13x on consumer-grade GPUs (NVIDIA T4) and exceeding 40x on data-center-class hardware (NVIDIA A100). This performance transforms a significant bottleneck into a negligible part of the training loop. By clearly defining its role as a "Token-ID BLEU" for development purposes and open-sourcing our implementation, we provide a powerful tool for accelerating research in areas like RL-based model fine-tuning.

## 1. Introduction

The advancement in Natural Language Processing (NLP) has been driven by the scaling of neural architectures and datasets. However, while models have become exponentially more powerful, the tools for their evaluation have often failed to keep pace. Metrics that cannot efficiently process batches of data in parallel on the GPU create an unnecessary bottleneck related to data transfer and sequential computation.

This problem is most critical for applications that require a metric to be computed repeatedly inside the training loop. A prime example is Reinforcement Learning (RL) for fine-tuning language models, where a dense reward signal is needed for each generated sample in a batch. A single "corpus" score for the entire batch provides a weak, averaged signal, whereas a per-sample reward is necessary for effective policy gradient updates. A slow, CPU-bound metric can dominate the computation time, making such approaches impractical. The BLEU score, a standard for assessing text generation quality, is a desirable candidate for such a reward, but traditional implementations are ill-suited for this role (Papineni et al., 2002). While libraries like NLTK can compute BLEU from integer token IDs, they require moving tensors from GPU to CPU, converting them to lists, and processing each sample in a Python loop, creating a severe performance bottleneck. On the other hand, the more efficient SacreBLEU (Post, 2018) requires inputs in text format, so in the case of RL, generated tokens must be decoded solely for reward calculation, which introduces another potential bottleneck.

In our case, the need to implement an efficient, GPU-based BLEU calculation arose during the training of our Reactive Transformer models (Filipek, 2025). During the memory reinforcement learning stage, BLEU calculation on the CPU was combined with cosine similarity calculation on the GPU, which required continuous data copying between devices.

#### Our Contribution: TensorBLEU
In this paper, we present TensorBLEU, a re-architected implementation of the BLEU metric designed specifically for batched, per-sentence, vectorized computation on token IDs within the PyTorch environment. Our work makes two key contributions:
1.  **A Memory-Efficient Vectorized Algorithm for Per-Sentence BLEU:** We introduce a novel method for n-gram counting that avoids the memory explosion of naive vectorization. Instead of using a large hash space, we use `torch.unique` on the n-gram tensors themselves to create a compact, batch-specific dictionary. This allows for efficient, parallel per-sentence counting via a "batched bincount" technique in a memory space proportional to the number of unique n-grams in the batch, not the vocabulary size.
2.  **A High-Performance In-Training Metric with Demonstrated Scalability:** We benchmark TensorBLEU on both consumer (NVIDIA T4) and data-center (NVIDIA A100) GPUs, demonstrating speedups that scale with hardware capabilities and effectively remove the evaluation bottleneck across different research environments.

## 2. Background and Related Work

### 2.1. The Original BLEU Metric
The BLEU (Bilingual Evaluation Understudy) metric was introduced by Papineni et al. (2002) to address the slow and expensive process of human evaluation for machine translation. It is based on two components: modified n-gram precision and a brevity penalty. Modified n-gram precision uses a "clipping" mechanism to prevent systems from over-generating common, correct words. The brevity penalty (BP) penalizes candidate translations that are shorter than their references. The final score is the geometric mean of the precisions (typically for n=1 to 4), multiplied by the BP:
$$
\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

### 2.2. The Need for Standardization: SacreBLEU
As BLEU became the dominant metric, a reproducibility crisis emerged. Scores varied wildly between papers due to undisclosed differences in preprocessing and tokenization, making fair comparison impossible (Post, 2018). To solve this, Post (2018) introduced SacreBLEU, a standardized implementation that manages the entire evaluation pipeline, including a canonical tokenization scheme. This ensures that reported scores are comparable and reproducible, establishing SacreBLEU as the gold standard for final, publication-ready evaluation.

### 2.3. From Evaluation to Optimization: BLEU as an RL Reward
Standard training of sequence models via word-level cross-entropy suffers from "exposure bias": the model is only trained on ground-truth prefixes, not its own, often imperfect, predictions (Ranzato et al., 2015). A solution is to train at the sequence level by directly optimizing a metric like BLEU. Since BLEU is non-differentiable, Reinforcement Learning (RL) provides a framework for this optimization. The model acts as a policy, generation is a sequence of actions, and the final BLEU score serves as the reward.

The seminal work by Ranzato et al. (2015) introduced this sequence-level training paradigm to NLP, using the REINFORCE algorithm to directly optimize for metrics like BLEU. Subsequent work, such as Li et al. (2016), successfully applied this technique to dialogue generation, demonstrating its broader utility. However, the computational cost of calculating the BLEU reward for every batch on the CPU has remained a major barrier to the widespread adoption of this powerful technique. TensorBLEU is designed to remove this barrier.

### 2.4. Token-ID BLEU vs. Linguistic BLEU
It is crucial to distinguish between two modes of BLEU calculation.

#### Linguistic BLEU
This is the standard for final model evaluation (e.g., SacreBLEU). It operates on detokenized text and applies its own standardized tokenization to ensure scores are reproducible.

#### Token-ID BLEU
In contrast, this operates directly on the integer outputs of a model's tokenizer. The n-grams are sequences of subword IDs. While unsuitable for final reporting, this metric is perfectly suited for *internal, relative* evaluation during the development cycle (e.g., as an RL reward), where the tokenizer is held constant. TensorBLEU is a high-performance implementation of Token-ID BLEU.

## 3. TensorBLEU: A Memory-Efficient Vectorized Implementation
This section details the core algorithm of TensorBLEU, which is designed to compute a separate BLEU score for each candidate-reference pair in a batch in a fully vectorized manner, without resorting to Python loops.

### 3.1. Vectorized n-gram Extraction with `unfold`
The first step is to extract all n-grams from the entire batch of sentences in parallel. We use the `Tensor.unfold` method in PyTorch. For a batch of token sequences with shape `(batch_size, seq_len)`, applying `tensor.unfold(dimension=1, size=n, step=1)` returns a view of the original tensor containing all n-gram slices, with a shape of `(batch_size, num_ngrams, n)`. This operation is highly efficient as it avoids data copying and processes all sentences simultaneously.

### 3.2. Memory-Efficient Counting via a Unified n-gram Dictionary
A naive approach to vectorizing n-gram counting involves hashing each n-gram slice into a unique integer and using a counting tensor of size `(batch_size, V^n)`, where V is the vocabulary size. This leads to a memory explosion for modern vocabularies.

To solve this, we developed a memory-efficient method that operates in a compact space. The algorithm proceeds as follows:
1.  **Unified N-gram Collection:** For a given order $n$, we extract all valid n-grams from both the candidate and all reference sentences in the batch and flatten them into a single large tensor of shape `(total_ngrams, n)`.
2.  **Compact Dictionary Creation:** We apply `torch.unique(all_ngrams, dim=0, return_inverse=True)`. This is the key step. It returns two tensors:
    * `unique_ngrams`: A tensor containing only the unique n-grams that actually appear in the current batch.
    * `inverse_indices`: A 1D tensor mapping each original n-gram to its new, compact ID (i.e., its index in `unique_ngrams`).
    
    This step effectively creates a batch-specific "dictionary" of n-grams, where the memory required is proportional to the number of unique n-grams present, not the theoretical maximum.

### 3.3. Batched Counting and Clipping with Offset Bincounting
With the compact IDs from the previous step, we can now perform counting for each sentence in parallel. We use a novel "batched bincount" technique.
1.  **The Offset Mechanism:** For each sentence $i$ in the batch, we add a unique offset, calculated as $i \times \text{num\_unique\_ngrams}$, to its compact n-gram IDs. This ensures that the IDs for each sentence occupy a unique, non-overlapping range.
2.  **Single Bincount Operation:** These offset IDs are then flattened into a single 1D tensor. A single call to `torch.bincount` on this tensor computes the n-gram counts for all sentences simultaneously. The resulting flat tensor is then reshaped to `(batch_size, num_unique_ngrams)`.
3.  **Reference Counts and Clipping:** The same process is applied to the reference n-grams. To obtain the final reference counts for clipping, a `torch.maximum` operation is taken over the count tensors derived from each reference set. The final clipping is a simple, vectorized `torch.minimum(candidate_counts, reference_max_counts)`.

### 3.4. Final Score Aggregation and The `tensor_corpus_bleu` Variant
With the clipped counts (numerators) and total candidate n-gram counts (denominators) computed for each sentence and each n-gram order, the rest of the calculation is performed element-wise across the batch dimension. The modified precisions $p_n$, brevity penalty, and final geometric mean are assembled using standard PyTorch functions. Our implementation also includes standard smoothing methods ('floor', 'add-k', 'exp') as described by Chen and Cherry (2014).

In addition to the per-sentence function, we provide a `tensor_corpus_bleu` variant that computes a single score for the entire batch by aggregating statistics before calculating precision. The performance of this variant is nearly identical to the per-sentence version. This is because the most computationally expensive steps—n-gram extraction with `unfold` and the creation of the compact dictionary with `torch.unique`—are performed on the entire batch's n-grams in both cases and dominate the runtime. The additional complexity in the per-sentence version (offset calculation, per-sentence aggregation) consists of lightweight arithmetic operations that are negligible on a GPU. This allows researchers to obtain a more granular, per-sentence reward signal at virtually no performance penalty compared to a less useful corpus-level score.

## 4. Experimental Design

### 4.1. Correctness Verification
To prove that TensorBLEU correctly implements the BLEU algorithm for token IDs, we verified that it produces numerically identical results to NLTK's `sentence_bleu` function when given the same lists of token IDs, weights, and smoothing function. Our implementation consistently achieved equivalence within the margin of floating-point precision ($< 10^{-6}$).

### 4.2. Performance Benchmarking

#### Objective
To quantify the speedup of TensorBLEU over the standard CPU-based method for calculating Token-ID BLEU.

#### Reference Implementation
We compare against NLTK's `sentence_bleu`, which is the standard library for this task. The NLTK implementation is run on the CPU and involves iterating through the batch in a Python loop.

#### Hardware and Data
Experiments were conducted across two distinct hardware tiers:
* **Consumer-Grade:** A Google Colab environment featuring an NVIDIA T4 GPU (16GB VRAM) and an Intel Xeon CPU (2 cores, 13GB RAM).
* **Data-Center-Grade:** A Novita.ai cloud instance featuring an NVIDIA A100 GPU (80GB HBM2e VRAM) and a 14 vCPU instance (240GB RAM).

We used batches of token sequences of two lengths: 256 tokens (typical for many tasks) and 1024 tokens (representing longer-form generation). All computations for TensorBLEU were performed on the GPU using float32 precision.

#### Variables
We measured the wall-clock execution time (mean of 5 runs) while varying the batch size, using values of 16, 32, 64, 128, 256, and 512.

## 5. Results and Analysis

### 5.1. Performance Results
The performance results are presented in Table 1 for the consumer-grade NVIDIA T4 GPU and Table 2 for the data-center-grade NVIDIA A100 GPU.

**Table 1:** Performance on NVIDIA T4 GPU (mean execution time in seconds). Lower is better.
| **Batch Size** | **NLTK (CPU)** | **TensorBLEU (GPU)** | **Speedup Factor** |
| :--- | :---: | :---: | :---: |
| ***Sequence Length: 256 tokens*** |
| 32 | 0.042s | 0.011s | 3.8x |
| 64 | 0.079s | 0.015s | 5.3x |
| 128 | 0.163s | 0.016s | 10.2x |
| 256 | 0.333s | 0.035s | 9.5x |
| 512 | 0.702s | 0.085s | 8.3x |
| ***Sequence Length: 1024 tokens*** |
| 16 | 0.072s | 0.011s | 6.5x |
| 32 | 0.131s | 0.015s | 8.7x |
| 64 | 0.252s | 0.019s | 13.3x |
| 128 | 0.482s | 0.036s | 13.4x |
| 256 | 0.974s | 0.084s | 11.6x |

**Table 2:** Performance on NVIDIA A100 GPU for 1024-token sequences. Lower is better.
| **Batch Size** | **NLTK (CPU)** | **TensorBLEU (GPU)** | **Speedup Factor** |
| :--- | :---: | :---: | :---: |
| 32 | 0.107s | 0.009s | 11.9x |
| 64 | 0.200s | 0.010s | 20.0x |
| 128 | 0.380s | 0.013s | 29.2x |
| 256 | 0.764s | 0.019s | 40.2x |
| 512 | 1.525s | 0.041s | 37.2x |

### 5.2. Analysis of Results
The data clearly demonstrate the profound performance advantage of our vectorized, GPU-based approach. The NLTK implementation, being a serial loop over sentences, exhibits near-linear time complexity with respect to batch size. In contrast, TensorBLEU shows sub-linear scaling, as the fixed costs of launching GPU kernels are amortized over an increasing number of parallel computations.

#### Impact of Sequence Length
Comparing the results on the T4 GPU (Table 1), we observe that the speedup advantage of TensorBLEU grows with sequence length. For a batch size of 128, the speedup increases from 10.2x for 256-token sequences to 13.4x for 1024-token sequences. This is because the number of n-grams to process grows significantly with sequence length, heavily penalizing the iterative CPU approach, while the parallel GPU architecture can absorb this increased workload much more efficiently.

#### Impact of Hardware Tier
The comparison between the T4 (Table 1) and A100 (Table 2) results for 1024-token sequences highlights the algorithm's scalability. On the A100, the speedup factor reaches a remarkable 40.2x at a batch size of 256. This demonstrates that the algorithm's design effectively leverages the superior memory bandwidth and computational power of high-end GPUs. The performance gains are super-linear, indicating that the implementation is not limited by its own logic but by the underlying hardware, which is a hallmark of a well-designed, scalable algorithm. The A100's massive memory bandwidth is particularly beneficial for the `torch.unique` operation on a very large tensor of n-grams.

#### The Bottleneck Vanishes
For a typical training scenario (e.g., batch size 256, 1024 tokens), TensorBLEU reduces the evaluation time from a significant fraction of a second (764ms on a capable CPU) to just 19ms on an A100 GPU. This effectively transforms the metric calculation from a potential training bottleneck into a negligible overhead.

## 6. Discussion and Future Work

### 6.1. Implications for NLP Research
The primary implication of our work is that it makes large-scale RL fine-tuning of language models using BLEU as a dense reward signal computationally cheap and practical. By reducing the computation time for a batch from hundreds of milliseconds to just a few, TensorBLEU removes a critical bottleneck that previously made such approaches prohibitively slow. This accelerates the research and development cycle, allowing for more extensive experimentation, hyperparameter sweeps, and application to larger models where every second of training time counts.

### 6.2. Limitations and Proper Usage
It is essential to use TensorBLEU correctly. As a "Token-ID BLEU" metric, its results are dependent on the tokenizer and are not directly comparable to scores from other models using different tokenizers. It is an **internal development and optimization tool**, ideal for measuring relative improvements during training. For final, publication-ready results that are comparable across the field, researchers must continue to use standardized, text-based tools like SacreBLEU (Post, 2018), which ensure consistent tokenization and processing.

### 6.3. Future Work
The core contribution of this paper extends beyond a fast BLEU implementation to a generalizable methodology for vectorizing n-gram-based metrics on GPUs using `torch.unique` as a memory-efficient hashing mechanism. Based on this, we propose several directions for future work:
1.  **Generalizing the Technique:** The vectorized counting methodology can be extended to other n-gram-based metrics like ROUGE and METEOR. Developing a suite of high-performance "TensorMetrics" would provide the community with a powerful toolkit for GPU-accelerated evaluation.
2.  **Integration with RL Libraries:** To facilitate adoption, we plan to develop official integrations and tutorials for popular RL libraries like Hugging Face's TRL and AllenAI's RL4LMs, making it trivial for researchers to use TensorBLEU as a reward function.
3.  **Exploring Further Optimizations:** We plan to investigate the performance impact of lower-precision data types like `bfloat16` and explore the use of custom CUDA kernels for the counting mechanism to potentially achieve even greater speedups on specific hardware.

## 7. Conclusion
In this paper, we presented TensorBLEU, a fully vectorized, memory-efficient, and GPU-accelerated implementation of the BLEU metric designed for per-sentence, in-training evaluation on token IDs. By leveraging a novel counting mechanism based on `torch.unique`, our method is practical for large-vocabulary models and avoids the memory explosion of naive vectorization. It achieves speedups of over 13x on consumer-grade hardware and over 40x on data-center GPUs compared to the standard CPU-based NLTK implementation. This performance demonstrates excellent scalability and effectively eliminates the evaluation bottleneck for in-training use cases. By clearly defining its scope and open-sourcing the code, we provide the NLP community with a critical piece of infrastructure to accelerate research in computationally intensive paradigms like Reinforcement Learning.

## Acknowledgements
The article was created in cooperation with Google Gemini 2.5 Pro (Google DeepMind, 2025) in "Deep Research" mode, which helped with the analysis of the algorithm and formatting the final version of the research paper.

## Code and Documentation
Implementation and usage documentation for TensorBLEU in both sentence and corpus modes is publicly available as "Free Component" in our RxLM framework (https://github.com/RxAI-dev/rxlm) as `rxlm.metrics.tensorbleu` module. "Free Components" in **Reactive AI Framework License (RAFL) v1.0** are available under **Apache-2.0** license terms.

## References
1.  Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). Bleu: a method for automatic evaluation of machine translation. In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics* (pp. 311-318).
2.  Post, M. (2018). A call for clarity in reporting bleu scores. In *Proceedings of the Third Conference on Machine Translation: Research Papers* (pp. 186-191).
3.  Ranzato, M. A., Chopra, S., Auli, M., & Zaremba, W. (2015). Sequence level training with recurrent neural networks. *arXiv preprint arXiv:1511.06732*.
4.  Li, J., Monroe, W., Ritter, A., Jurafsky, D., Galley, M., & Gao, J. (2016). Deep reinforcement learning for dialogue generation. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing* (pp. 1192-1202).
5.  Chen, B., & Cherry, C. (2014). A systematic comparison of smoothing techniques for sentence-level bleu. In *Proceedings of the Ninth Workshop on Statistical Machine Translation* (pp. 362-367).
6.  Filipek, A. (2025). Reactive Transformer (RxT) - Stateful Real-Time Processing for Event-Driven Reactive Language Models. *arXiv preprint arXiv:2510.03561*.
7.  Google DeepMind. (2025). Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. *arXiv preprint arXiv:2507.06261*.
