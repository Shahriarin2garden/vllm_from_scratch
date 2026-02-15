# Untitled

Status: Not started

# Lab 0.1: From Text to Tensors – The Inference Computational Graph

**Navigation:** [← Main](https://www.notion.so/README.md) | [Next: Lab 0.2 →](https://www.notion.so/lab0.2/README.md)

## Introduction

This lab dissects the complete computational pipeline that transforms a raw text prompt into a generated token. You will examine every core data structure and operation in the inference graph, from tokenization through the KV cache to final sampling. Unlike traditional tutorials that present code as a black box, this lab requires you to predict, fill in blanks, and experiment with deliberate failures. The goal is to build a mental model that serves as the foundation for understanding high‑performance inference engines like vLLM.

The significance of this understanding extends beyond academic interest. As noted in recent research, "speeding up each inference request is instrumental in achieving high throughput and latency at scale" . Production systems must manage the KV cache—the primary dynamic memory consumer—under concurrent request loads, making efficient memory management essential for performance.

**Prerequisites:** Basic Python knowledge, familiarity with neural network concepts, and access to a machine with Python 3.8+ and PyTorch installed (see Environment Setup).

---

## Learning Objectives

By the end of this lab, you will be able to:

1. **Tokenize** raw text and explain the byte‑pair encoding (BPE) algorithm with reference to its original compression origins .
2. **Contrast** the training computational graph with the inference graph, identifying why the KV cache is the only dynamic state during inference and how this shift changes system bottlenecks from compute to memory bandwidth .
3. **Calculate** the memory footprint of the KV cache for a given model and sequence length, understanding why this dominates GPU memory under concurrency .
4. **Implement** a simplified paged KV cache manager based on the vLLM PagedAttention design to eliminate fragmentation .
5. **Orchestrate** a minimal inference engine that handles multiple requests with continuous batching, recognizing the distinct characteristics of prefill and decode phases .

---

## Prologue: The Challenge

You are a performance engineer at a startup that has just fine‑tuned a 13B parameter language model. Your team wants to offer it as a real‑time API. Early tests show that the model runs, but latency is high and memory usage grows unpredictably. The CTO asks you to investigate: "Why does inference memory blow up after a few requests? How can we serve hundreds of concurrent users?"

To answer these questions, you must understand exactly what happens inside the inference engine. Your first task is to map the pipeline from raw text to output tokens, identify the bottlenecks, and propose a memory management strategy. This lab builds that understanding step by step.

As you progress, consider this observation from systems research: "The KV cache grows linearly with respect to the length and number of reasoning chains, as the new key–value representations are appended to it. Hence, it can easily exhaust the memory of the accelerator and slow down each generation step, as attention is memory-bound: its cost is dominated by the time needed to retrieve the cache from memory" . This memory-bound nature is the central challenge you must address.

---

## Environment Setup

Create a Python virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers numpy
```

Create a working directory:

```bash
mkdir -p lab0.1
cd lab0.1
touch tokenizer_demo.py kv_cache_demo.py scheduler_demo.py
```

All code examples assume you are working inside this directory.

---

## Chapter 1: Text to Tokens – The Tokenization Pipeline

### 1.1 Why Tokenization Matters

A language model does not read characters; it reads integers that represent subword units. The choice of tokenizer directly affects:

- **Sequence length** – A larger vocabulary compresses text but increases embedding matrix size.
- **Latency** – Longer sequences mean more computation and memory.
- **Cost** – More tokens per request increase inference cost.

Byte Pair Encoding (BPE), originally developed as a data compression algorithm in 1994 , was adapted for neural machine translation to handle rare words by breaking them into subword units . Modern LLMs use a modified version that builds a fixed-size vocabulary by iteratively merging the most frequent adjacent byte pairs in a training corpus.

The diagram below shows the transformation from raw text to token IDs.

![1000091400.png](1000091400.png)

### 1.2 Think First: Vocabulary Trade‑offs

**Question:** If a model has a 4096‑token context window, how many English words can it roughly process in one request? How many Japanese characters? (Assume English averages 1.3 tokens/word, Japanese 2–3 tokens/character.)

<details>
<summary>Click to review</summary>

- English: 4096 / 1.3 ≈ 3150 words.
- Japanese: 4096 / 2.5 ≈ 1638 characters.

This variability affects system design: an English‑heavy workload can fit longer prompts than a Japanese‑heavy one, even with the same model. The tokenizer's byte-level BPE variant, used in models like GPT-2 and RoBERTa, converts text to UTF-8 first, ensuring any Unicode character can be represented .

</details>

### 1.3 Anatomy of a Tokenizer

Tokenizers use algorithms like Byte Pair Encoding (BPE) to find a balance between vocabulary size and sequence length. BPE iteratively merges the most frequent adjacent byte pairs in a training corpus.

```mermaid
flowchart TD
    A["Raw User Prompt<br/>'The capital of France is'"] --> B_Node[Text Normalization];
    B_Node --> C_Node[Subword Segmentation];
    C_Node --> D_Node[Vocabulary ID Lookup];

    subgraph B_Subgraph [" NORMALIZATION "]
        B1[Lowercasing?]
        B2[Unicode Normalization<br/>NFKC/NFD]
        B3[Whitespace/Accent Handling]
    end

    subgraph C_Subgraph [" BPE ALGORITHM APPLIED "]
        C1["Find most frequent pair: 'Th' + 'e' -> 'The'"]
        C2["Merge 'cap' + 'ital' -> 'capital'"]
        C3["Remainder: 'of', 'France', 'is'"]
    end

    subgraph D_Subgraph [" MAP TO INDICES "]
        D1["'The' -> 1"]
        D2["'capital' -> 307"]
        D3["'of' -> 2647"]
        D4["'France' -> 310"]
        D5["'is' -> 278"]
    end

    D_Node --> E["Output Tensor<br/>Shape: [seq_len=5]<br/>[1, 307, 2647, 310, 278]"];

    linkStyle default stroke:#333,stroke-width:3px
```

### 1.4 BPE Algorithm Implementation

The following code implements the core logic of BPE. Some parts are missing—fill them in.

```python
# tokenizer_demo.py
from collections import Counter
from typing import List, Dict, Tuple

def build_bpe_vocab(text_corpus: List[str], vocab_size: int) -> Tuple[Dict, Dict]:
    """
    Build BPE vocabulary from text corpus.
    Based on the original compression algorithm described by Gage (1994) .
    """
    # Initialize with byte-level tokens (0-255)
    vocab = {chr(i): i for i in range(256)}
    merges = {}

    for _ in range(vocab_size - 256):
        # Count adjacent pairs
        pairs = Counter()
        for text in text_corpus:
            # Convert text to list of token IDs (using current vocab)
            # In practice, this would use the current merge rules
            tokens = list(text.encode('utf-8'))
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i+1])] += 1

        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]

        # Q1: What should new_token_id be?
        new_token_id = ___  # Fill in

        # Q2: Add the pair to vocab. What key should be used?
        vocab[___] = new_token_id
        merges[best_pair] = new_token_id

        # Apply merge to corpus (simplified)
        # In a real implementation, you would update the corpus representation

    return vocab, merges
```

**Hints:**

- Q1: The new token ID should be the next available index.
- Q2: The key should be the pair itself (e.g., (97, 98) for 'a','b') so that later we can look up merged tokens.

<details>
<summary>Click to see solution</summary>

```python
new_token_id = len(vocab)
vocab[best_pair] = new_token_id
```

The vocabulary maps either a single byte (as a character) or a tuple of two token IDs to a new token ID. This allows the BPE algorithm to build a hierarchy of merges. Modern implementations like SentencePiece and HuggingFace tokenizers extend this concept with byte-level BPE to handle any Unicode character .

</details>

### 1.5 From Discrete IDs to Continuous Embeddings

Once token IDs are obtained, they are passed through an embedding layer—a simple lookup table that returns a dense vector for each ID. This matrix is one of the largest parameters in the model.

![1000091401.png](1000091401.png)

**Mathematical Formulation:**

Given token IDs $X \in \mathbb{Z}^{B \times S}$ and embedding matrix $W \in \mathbb{R}^{V \times D}$, the operation is:

$$E = W[X] \in \mathbb{R}^{B \times S \times D}$$

This indexing operation is a pure, parallelizable memory read. In production systems, the embedding matrix is typically stored in GPU memory and accessed with high bandwidth.

### 1.6 Complete Initial Pipeline

The following class combines tokenization and embedding. Fill in the missing parts.

```python
# text_to_tensor.py
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class TextToTensorPipeline:
    """End-to-end pipeline from raw text to model-ready embeddings."""

    def __init__(self, tokenizer, embedding_layer: nn.Embedding, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.embedding = embedding_layer.to(device)
        self.device = device

    def __call__(self, prompts: List[str], max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batch of prompts to embeddings.

        Args:
            prompts: List of text prompts
            max_length: Optional truncation/padding length

        Returns:
            embeddings: Float tensor of shape [batch, seq_len, hidden_dim]
            attention_mask: Bool tensor of shape [batch, seq_len]
        """
        # 1. Tokenization (CPU-bound)
        tokenized = self.tokenizer(
            prompts,
            padding=True if max_length else 'longest',
            truncation=True if max_length else False,
            max_length=max_length,
            return_tensors="pt"
        )

        # 2. Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # 3. Embedding lookup (GPU memory-bound)
        # Q1: Complete the embedding lookup
        embeddings = self.embedding(___)

        # 4. Apply positional encoding scaling (common in transformer models)
        hidden_dim = embeddings.size(-1)
        embeddings = embeddings * (hidden_dim ** 0.5)

        return embeddings, attention_mask
```

**Hints:**

- Q1: `self.embedding` expects token IDs.

<details>
<summary>Click to see solution</summary>

```python
embeddings = self.embedding(input_ids)
```

</details>

### 1.7 Test and Verify

Run the following code to see tokenization in practice with a real tokenizer (GPT‑2). First, predict what the output will look like.

```python
# test_tokenizer.py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The capital of France is"
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)
print("Decoded:", tokenizer.decode(tokens))
```

**Predict:** What will `decode(tokens)` return? Will it exactly match the input string?

<details>
<summary>Click to verify</summary>

```
Token IDs: [464, 2506, 286, 3260, 318]
Decoded: The capital of France is
```

Note: The tokenizer adds a leading space before "The" because GPT‑2 was trained with a space before each word. This is an artifact of its BPE vocabulary and byte-level encoding .

</details>

### 1.8 Tokenization Pipeline Architecture

The diagram below shows how tokenization fits into the broader inference pipeline in production systems like NVIDIA's TensorRT-LLM .

```mermaid
graph TB
    subgraph Tokenization_Pipeline
        direction LR
        T1[Raw Text] --> T2[Pre-tokenization<br/>Whitespace/Punctuation]
        T2 --> T3[BPE Merge Rules<br/>Apply learned merges]
        T3 --> T4[Vocabulary Lookup<br/>Map to IDs]
        T4 --> T5[Input IDs Tensor]
    end

    subgraph Model_Input
        T5 --> E1[Embedding Layer<br/>Lookup Table]
        E1 --> E2[Positional Encoding<br/>Add position info]
        E2 --> E3[Final Embeddings<br/>to Transformer]
    end

    style Tokenization_Pipeline fill:#e3f2fd
    style Model_Input fill:#e8f5e9
```

### 1.9 Checkpoint

**Self-Assessment:**

- [ ]  I can explain why tokenization affects the effective context window.
- [ ]  I can describe the BPE merge process in one sentence.
- [ ]  I understand that the embedding layer is a simple lookup table.
- [ ]  I can predict the output of `tokenizer.decode(tokenizer.encode(...))` for a simple sentence.

### 1.10 Experiment: Vocabulary Size Impact

1. Load a different tokenizer, e.g., `"bert-base-uncased"`, and tokenize the same sentence.
2. Compare the number of tokens.
3. Which tokenizer produces more tokens? Why?

<details>
<summary>Experiment Guide</summary>

BERT uses WordPiece tokenization, which differs from BPE. You'll likely find that BERT produces more tokens for the same text because its vocabulary size (∼30,000) is smaller than GPT-2's (∼50,000). This trade-off between vocabulary size and sequence length is fundamental to tokenizer design .

</details>

---

## Chapter 2: The Inference Graph vs Training Graph

### 2.1 Think First: What's Different in Inference?

During training, the model computes gradients and updates weights. During inference, you only need the forward pass to produce logits.

**Question:** Which components of the training pipeline can be completely eliminated during inference? Which must remain?

<details>
<summary>Click to review</summary>

- **Eliminated:** Backward pass, gradient storage, optimizer states (momentum, variance).
- **Remain:** Forward pass, weights, and the KV cache (for autoregressive generation).

As noted in the vLLM paper analysis, "the training graph's memory footprint is dominated by optimizer states and gradients needed for the backward pass, consuming 3-4x the memory of the weights alone. The inference graph eliminates all this, leaving only static weights and the dynamic KV Cache" .

</details>

### 2.2 Contrasting Computational Graphs

The diagram below highlights the divergence.

```mermaid
flowchart TD
    subgraph Training_Graph [" TRAINING COMPUTATIONAL GRAPH "]
        direction TB
        TG_Weights["Model Weights<br/>(FP16)"] --> TG_FP["Forward Pass<br/>Compute Activations"];
        TG_FP --> TG_Loss["Loss Calculation<br/>(Cross-Entropy)"];
        TG_Loss --> TG_BP["Backward Pass<br/>Compute Gradients"];
        TG_BP --> TG_Grad["Gradients<br/>(FP16)"];
        TG_Grad --> TG_Opt["Optimizer Step<br/>(AdamW/Adam)"];
        TG_Opt --> TG_Update["Update Weights"];
        TG_Update ===> TG_Weights;

        subgraph TG_Mem [" Peak Memory Includes "]
            M1["• Weights: 2 bytes/param"]
            M2["• Gradients: 2 bytes/param"]
            M3["• Optimizer States<br/>  (Momentum, Variance):<br/>  8 bytes/param"]
            M4["• Activations: 20-100 bytes/token"]
        end
    end

    subgraph Inference_Graph [" INFERENCE COMPUTATIONAL GRAPH "]
        direction TB
        IG_Weights["Model Weights<br/>(FP16/INT8/INT4)"] --> IG_FP["Forward Pass Only"];
        IG_KV[["Dynamic KV Cache"]] ==> IG_FP;
        IG_FP --> IG_Sample["Sample Next Token"];
        IG_FP --> IG_UpdateKV["Append to KV Cache"];
        IG_UpdateKV ===> IG_KV;

        subgraph IG_Mem [" Peak Memory Dominated By "]
            M5["• Weights: 0.5-2 bytes/param"]
            M6["• KV Cache: Variable<br/>  Batch × Seq × Layers ×<br/>  Heads × Dim × 2 × dtype"]
            M7["• Activations: Minimal<br/>  (Forward pass only)"]
        end
    end

    Training_Graph ===>|"Remove Gradients,<br/>Optimizer, Backward Pass"| Inference_Graph;
```

### 2.3 Memory Breakdown for a 13B Model

Research shows that "GPU memory is primarily consumed by model weights and the remainder is used by the KV cache. Thus, the free GPU space available to the KV cache is a scarce resource that needs to be managed in an efficient way" .

| Component | Training (Adam, FP16) | Inference (FP16 + KV Cache) | Inference (INT4 + KV Cache) |
| --- | --- | --- | --- |
| **Weights** | 26 GB | 26 GB | **~6.5 GB** |
| **Gradients** | 26 GB | **0 GB** | **0 GB** |
| **Optimizer States** | 52 GB | **0 GB** | **0 GB** |
| **Activations** | ~40‑80 GB | <1 GB | <1 GB |
| **KV Cache** | N/A | **Dynamic** | **Dynamic** |
| **Approx. Total** | **~144+ GB** | **~26 GB + Cache** | **~6.5 GB + Cache** |
| **Hardware Required** | 8×H100 (80GB) | 1×H100 (80GB) | 1×RTX 4090 (24GB) |

### 2.4 Quantifying the Gap

The table shows that inference can run on hardware an order of magnitude smaller. The primary bottleneck shifts from compute (FLOPs) to **memory bandwidth**, specifically for the KV cache. As vLLM's analysis reveals, "the system becomes memory-bound, not compute-bound" when using fine-grained batching .

### 2.5 Training vs Inference Memory Footprint Visualization

This diagram illustrates why inference enables aggressive optimizations like quantization.

```mermaid
graph TD
    subgraph Training_Footprint
        direction LR
        T1[Weights<br/>26 GB] --> T2[Gradients<br/>26 GB]
        T2 --> T3[Optimizer States<br/>52 GB]
        T3 --> T4[Activations<br/>40-80 GB]
    end

    subgraph Inference_Footprint
        direction LR
        I1[Weights<br/>26 GB] --> I2[KV Cache<br/>Dynamic]
        I2 --> I3[Activations<br/>< 1 GB]
    end

    Training_Footprint -->|"3-4x memory<br/>reduction"| Inference_Footprint

    style Training_Footprint fill:#ffebee
    style Inference_Footprint fill:#e8f5e9
```

### 2.6 Checkpoint

**Self-Assessment:**

- [ ]  I can list three components present in training but absent in inference.
- [ ]  I can explain why inference memory is dominated by weights + KV cache.
- [ ]  I understand why quantization (INT4) is feasible for inference but difficult during training.

---

## Chapter 3: The KV Cache – Core of Autoregression

### 3.1 Think First: Why Can't We Recompute?

In an autoregressive model, each new token attends to all previous tokens. Without caching, generating token $t_n$ would require recomputing the key and value vectors for tokens $t_1…t_{n-1}$—an $O(n^2)$ cost.

**Question:** If you generate 1000 tokens and each forward pass takes 10 ms without cache, how long would it take with recomputation? With caching?

<details>
<summary>Click to review</summary>

- Without cache: first token (prefill) processes prompt, then each subsequent token recomputes all previous keys/values → total time ~ O(n²). For n=1000, ~500k operations, likely seconds to minutes.
- With cache: prefill O(prompt_len), each decode O(1) new work → linear time. For 1000 tokens, ~1000 * (decode time) which is typically < 0.1 ms per token → <100 ms.

As the IEEE characterization paper notes, the KV cache exists "to avoid redundant recomputation in each decode iteration" .

</details>

### 3.2 Anatomy of the Cache in a Single Attention Head

During forward pass, each token's hidden state is projected into query (Q), key (K), and value (V). The KV cache stores K and V from all previous tokens.

```mermaid
flowchart TD
    subgraph Layer_L [" TRANSFORMER LAYER L "]
        direction TB
        subgraph Attention_Head [" Single Attention Head "]
            direction LR

            subgraph Current_Token [" Processing Token t_n "]
                H["Hidden State<br/>h_n"] --> Q_Proj["Q Projection"];
                H --> K_Proj["K Projection"];
                H --> V_Proj["V Projection"];

                Q_Proj --> Q["q_n<br/>(Query)"];
                K_Proj --> K["k_n<br/>(Key)"];
                V_Proj --> V["v_n<br/>(Value)"];

                Q --> Attention["Attention<br/>Module"];
            end

            subgraph KV_Cache [" KV Cache for Layer L, Head H "]
                KC["Keys:<br/>k₁, k₂, ..., k_{n-1}"];
                VC["Values:<br/>v₁, v₂, ..., v_{n-1}"];
            end

            KV_Cache ==> Attention;
            Attention --> O["Output<br/>o_n"];
        end

        O ==>|"Append to cache"| KV_Cache_Update["Update<br/>KV Cache"];
        K ==> KV_Cache_Update;
        V ==> KV_Cache_Update;
    end
```

### 3.3 KV Cache Growth per Token (Iterative View)

The sequence diagram shows how the cache expands and why fragmentation appears without paging.

```mermaid
sequenceDiagram
    participant GPU_Mem as GPU Memory
    participant ReqA as Request A
    participant ReqB as Request B

    Note over ReqA: Prefill: 6 tokens
    ReqA->>GPU_Mem: Allocate KV rows t₁–t₆ (contiguous)
    Note over ReqA: Decode step 1
    ReqA->>GPU_Mem: Append KV row t₇
    Note over ReqA: Decode step 2
    ReqA->>GPU_Mem: Append KV row t₈

    Note over ReqB: Prefill: 3 tokens
    ReqB->>GPU_Mem: Allocate KV rows t₁–t₃ (finds gap or tail)

    Note over GPU_Mem: ⚠️ Without paging:<br/>Mixed-length gaps appear as<br/>requests finish; new requests<br/>struggle to find contiguous chunks
```

### 3.4 Cache Memory Calculation

Fill in the blanks to compute the cache size for Llama-3 8B:

```python
# Configuration
seq_len = 2048
num_layers = 32
num_kv_heads = 8   # Grouped Query Attention
head_dim = 128
dtype_bytes = 2    # FP16

# Q1: Calculate bytes per token (all layers, all kv heads)
cache_per_token = 2 * ___ * ___ * ___ * ___
print(f"Cache per token: {cache_per_token} bytes = {cache_per_token / 1024:.2f} KB")

# Q2: Calculate total cache for seq_len
total_cache_bytes = cache_per_token * ___
print(f"Total cache: {total_cache_bytes / (1024**3):.2f} GB")
```

**Hints:**

- Q1: 2 (for K and V) × layers × kv_heads × head_dim × dtype_bytes.
- Q2: Multiply by seq_len.

<details>
<summary>Click to see solution</summary>

```python
cache_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
# = 2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KB

total_cache_bytes = cache_per_token * seq_len
# = 131072 * 2048 = 268435456 bytes ≈ 0.25 GB
```

</details>

### 3.5 KV Cache Scaling Visualization

This diagram shows how cache size scales with sequence length and batch size.

```mermaid
graph TD
    subgraph Cache_Sizing [" KV CACHE MEMORY BREAKDOWN "]
        direction LR
        Input_Node["Input<br/>Parameters"] --> Formula["Cache Size<br/>Formula"];

        subgraph Input_Subgraph [" Model Configuration "]
            L["L = 32 layers"]
            H["H_kv = 8 heads"]
            Dh["D_h = 128 dim"]
            Bytes["2 bytes (FP16)"]
        end

        Formula --> PerToken["Per Token:<br/>128 KB"];
        PerToken --> SeqLen["Sequence<br/>Length"];

        subgraph SeqLen [" Varying Sequence Lengths "]
            S1["S = 512<br/>→ 64 MB"]
            S2["S = 1024<br/>→ 128 MB"]
            S3["S = 2048<br/>→ 256 MB"]
            S4["S = 4096<br/>→ 512 MB"]
        end

        SeqLen --> Batch["Batch<br/>Scaling"];

        subgraph Batch [" Batch Size Impact "]
            B1["Batch = 1<br/>→ 256 MB"]
            B2["Batch = 4<br/>→ 1 GB"]
            B3["Batch = 16<br/>→ 4 GB"]
            B4["Batch = 64<br/>→ 16 GB"]
        end

        Batch --> Conclusion["Conclusion:<br/>Cache dominates<br/>memory at scale"];
    end
```

### 3.6 Checkpoint

**Self-Assessment:**

- [ ]  I can explain why the KV cache is necessary.
- [ ]  I can compute the cache size for any model given its configuration.
- [ ]  I understand that the cache grows linearly with sequence length.

### 3.7 Experiment: Simulate Cache Growth

1. Write a simple Python script that simulates the KV cache growth for multiple requests with random sequence lengths.
2. Measure how much memory would be wasted if you allocated contiguous blocks for each request (i.e., internal fragmentation due to different lengths).

<details>
<summary>Experiment Guide</summary>

The vLLM paper found that "only 20–38% of the allocated KV cache memory is actually used in existing systems" due to fragmentation . Your simulation should demonstrate that contiguous allocation leads to significant waste, motivating the need for paged attention.

</details>

---

## Chapter 4: PagedAttention – Solving Fragmentation

### 4.1 Think First: The Fragmentation Problem

When requests finish at different times, they leave holes in GPU memory. A new request may not fit into a single contiguous hole, even though total free memory is sufficient.

**Question:** How do operating systems solve a similar problem with RAM fragmentation?

<details>
<summary>Click to review</summary>

Operating systems use **paging**: physical memory is divided into fixed‑size frames, and each process's virtual address space is mapped to these frames via a page table. Non‑contiguous physical pages can be mapped to a contiguous virtual view.

This is exactly the insight behind PagedAttention: "Why not treat the KV cache like an operating system treats virtual memory?" .

</details>

### 4.2 Contiguous Allocation vs PagedAttention

The diagram below illustrates the problem and solution.

![page.png](page.png)

The vLLM paper identifies three distinct kinds of waste with contiguous allocation :

1. **Reserved but Unused Slots:** Allocating space for maximum possible tokens, but request uses far fewer.
2. **Internal Fragmentation:** Over-allocation within a request's reserved slab.
3. **External Fragmentation:** Free memory scattered in non-contiguous chunks.

### 4.3 PagedAttention Core Data Structures

Implement the core classes with missing parts.

```python
# kv_cache.py
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Dict
import torch

@dataclass
class KVCacheBlock:
    """A fixed-size block of KV cache holding multiple tokens."""
    block_id: int
    key_cache: torch.Tensor   # shape: [num_layers, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor
    ref_count: int = 0

    def get_kv(self, layer_idx: int, head_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV tensors for specific layer and head."""
        return (
            self.key_cache[layer_idx, head_idx],
            self.value_cache[layer_idx, head_idx]
        )

class PagedKVCache:
    """Manages KV cache using paged attention scheme as described in vLLM paper ."""

    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre‑allocate all blocks
        self.blocks: List[KVCacheBlock] = []
        for block_id in range(num_blocks):
            key = torch.zeros(num_layers, num_kv_heads, block_size, head_dim, dtype=dtype, device='cuda')
            value = torch.zeros_like(key)
            self.blocks.append(KVCacheBlock(block_id, key, value))

        self.free_blocks: deque = deque(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}   # request_id -> list of block_ids

    def allocate(self, request_id: str, seq_len: int) -> Optional[List[int]]:
        """Allocate blocks for a request. Returns list of block IDs or None if insufficient."""
        blocks_needed = (seq_len + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < blocks_needed:
            return None

        allocated = []
        for _ in range(blocks_needed):
            # Q1: Pop from free_blocks and append to allocated
            block_id = self.free_blocks.___
            allocated.___
            self.blocks[block_id].ref_count += 1

        self.block_tables[request_id] = allocated
        return allocated

    def free(self, request_id: str):
        """Free blocks for a finished request."""
        if request_id not in self.block_tables:
            return
        for block_id in self.block_tables[request_id]:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                # Q2: Return block to free pool
                self.free_blocks.___
        del self.block_tables[request_id]

    def get_required_blocks(self, seq_len: int) -> int:
        """Calculate number of blocks needed for a sequence length."""
        return (seq_len + self.block_size - 1) // self.block_size
```

**Hints:**

- Q1: Use `popleft()` and `append()`.
- Q2: Use `append()`.

<details>
<summary>Click to see solution</summary>

```python
block_id = self.free_blocks.popleft()
allocated.append(block_id)
...
self.free_blocks.append(block_id)
```

</details>

### 4.4 Block Table and Physical Memory

The block table maps logical token positions to physical block IDs.

```mermaid
flowchart LR
    subgraph LogicalView [" LOGICAL SEQUENCE (Request X) "]
        L1["Block 0<br/>tokens 0-15"] --> L2["Block 1<br/>tokens 16-31"];
        L2 --> L3["Block 2<br/>tokens 32-47"];
    end

    subgraph PhysicalGPU [" PHYSICAL GPU MEMORY "]
        P1["Block 7"]:::free
        P2["Block 2"]:::free
        P3["Block 19"]:::used
        P4["Block 4"]:::free
    end

    L1 -."block_table[0] → 7".-> P1
    L2 -."block_table[1] → 19".-> P3
    L3 -."block_table[2] → 4".-> P4

    classDef used fill:#a5d6a7
    classDef free fill:#ffcdd2
    class P3 used;
    class P1,P2,P4 free;
```

### 4.5 PagedAttention Kernel Architecture

The diagram below shows how the PagedAttention kernel accesses blocks during computation, based on the vLLM implementation .

```mermaid
graph TD
    subgraph GPU_Thread_Block ["GPU Thread Block (per head per sequence)"]
        direction TB
        TB1[Thread Group 0<br/>Loads Query Vectors] --> TB2[Load Block Table]
        TB2 --> TB3{Warp 0<br/>Process Block 0}
        TB2 --> TB4{Warp 1<br/>Process Block 1}
        TB2 --> TB5{Warp 2<br/>Process Block 2}

        subgraph Warp_Processing ["Warp Processing a Block"]
            W1[Each thread group<br/>loads Key vectors] --> W2[Compute Q·K dot products]
            W2 --> W3[Accumulate in<br/>shared memory]
            W3 --> W4[Load Value vectors]
            W4 --> W5[Compute weighted sum]
        end

        TB3 --> TB6[Aggregate results<br/>across warps]
        TB4 --> TB6
        TB5 --> TB6
        TB6 --> TB7[Write output]
    end

    style GPU_Thread_Block fill:#e3f2fd
    style Warp_Processing fill:#e8f5e9
```

### 4.6 Block Size Trade-offs

Block size is a critical tuning parameter. The vLLM paper notes: "A smaller block size reduces internal fragmentation but increases kernel overhead; a larger block size improves kernel efficiency but wastes more memory" .

```mermaid
graph TD
    subgraph Block_Size_Impact ["Block Size Trade-offs"]
        direction LR
        BS["Block Size (tokens/block)"] --> Small["Small Block (e.g., 8)"]
        BS --> Medium["Medium Block (e.g., 16)"]
        BS --> Large["Large Block (e.g., 32)"]

        Small --> S1["✓ Less internal waste"]
        Small --> S2["✗ More kernel calls"]
        Small --> S3["✗ Larger block table"]

        Medium --> M1["Balanced approach<br/>vLLM default"]

        Large --> L1["✗ More internal waste"]
        Large --> L2["✓ Fewer kernel calls"]
        Large --> L3["✓ Smaller block table"]
    end

    style Block_Size_Impact fill:#f3e5f5
```

### 4.7 Checkpoint

**Self-Assessment:**

- [ ]  I can explain why contiguous allocation leads to fragmentation.
- [ ]  I understand how PagedAttention uses a block table to decouple logical and physical memory.
- [ ]  I can implement a basic block allocator with reference counting.

### 4.8 Experiment: Simulate Fragmentation

1. Write a simulation that allocates and frees blocks for random requests.
2. Track the number of free blocks and compare the fragmentation in contiguous vs paged allocation.

<details>
<summary>Experiment Guide</summary>

The vLLM paper reports that with PagedAttention, "internal fragmentation collapses" to at most block_size - 1 tokens per sequence, and external fragmentation is eliminated entirely . Your simulation should show that paged allocation achieves near-100% utilization while contiguous allocation wastes 60-80% of memory.

</details>

---

## Chapter 5: Building the Inference Pipeline

### 5.1 Think First: Orchestrating Requests

A real inference engine must handle multiple requests concurrently. Requests arrive at different times, have different prompt lengths, and generate different numbers of output tokens.

**Question:** What challenges arise when mixing a short prompt (e.g., 10 tokens) with a long prompt (e.g., 2000 tokens) in the same batch?

<details>
<summary>Click to review</summary>

- **Prefill phase:** The short prompt finishes quickly and moves to decode, while the long prompt is still in prefill. The batch must handle both phases simultaneously.
- **Memory:** The long prompt consumes more KV cache blocks.
- **Scheduling:** The engine must decide when to admit new requests and when to preempt running ones.

As the IEEE characterization study notes, "the decision of what batching strategy, what KV cache eviction policy to apply and how the KV cache impacts the inference performance is non-trivial" .

</details>

### 5.2 Two-Phase Inference Architecture

Production systems like NVIDIA's TensorRT-LLM explicitly separate prefill and generation phases .

```mermaid
graph LR
    INPUT_PROMPT(Input<br/>Prompt) --> TOKENIZER(Tokenize)

    subgraph VIT_BOX ["Optional"]
        VIT_PROCESS(ViT<br>Processing)
    end

    TOKENIZER --> VIT_PROCESS
    VIT_PROCESS --> PREFILL_ENGINE(Prefill<br/>Phase)

    PREFILL_ENGINE --> GENERATE_KV[Generate<br/>KV-Cache]
    GENERATE_KV --> SAMPLE_FIRST(Sample First<br/>Token)

    SAMPLE_FIRST --> GENERATION_ENGINE[Generation<br/>Phase]

    GENERATION_ENGINE --> UPDATE_KV(Update<br>KV Cache)
    UPDATE_KV --> SAMPLE_TOKEN(Sample Next<br/>Token)

    SAMPLE_TOKEN --> STOP_CHECK{Stop<br/>Condition?}
    STOP_CHECK -->|N| GENERATION_ENGINE
    STOP_CHECK -->|Y| OUTPUT_SEQUENCE(Generated<br/>Sequence)

    subgraph PHASE1 ["Phase 1: Prefill"]
        PREFILL_ENGINE
        GENERATE_KV
        SAMPLE_FIRST
    end

    subgraph PHASE2 ["Phase 2: Generation"]
        GENERATION_ENGINE
        UPDATE_KV
        SAMPLE_TOKEN
        STOP_CHECK
        OUTPUT_SEQUENCE
    end
```

### 5.3 Core Data Structures

The scheduler tracks request state and allocates resources.

```python
# scheduler.py
from enum import Enum
from dataclasses import dataclass
import time
from typing import List, Optional, Dict, Any

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

@dataclass
class InferenceRequest:
    """A single user request with engine-managed state."""
    request_id: str
    prompt_token_ids: List[int]
    output_token_ids: List[int] = None
    status: RequestStatus = RequestStatus.WAITING
    block_table: List[int] = None
    created_at: float = None
    started_at: float = None

    def __post_init__(self):
        if self.output_token_ids is None:
            self.output_token_ids = []
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        return len(self.output_token_ids) == 0

class BatchScheduler:
    """Implements continuous batching as described in vLLM paper ."""

    def __init__(self, max_batch_size=64, max_seq_len=4096):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.waiting_queue = []
        self.running_queue = []
        self.finished_queue = []

    def add_request(self, request: InferenceRequest):
        self.waiting_queue.append(request)

    def schedule(self, kv_cache: PagedKVCache) -> List[InferenceRequest]:
        # Sort waiting by prompt length (smaller first for efficiency)
        self.waiting_queue.sort(key=lambda r: len(r.prompt_token_ids))

        scheduled = []
        for request in list(self.waiting_queue):
            if len(self.running_queue) + len(scheduled) >= self.max_batch_size:
                break
            # Q1: Compute required blocks for this request
            required_blocks = kv_cache.get_required_blocks(request.total_tokens)
            if len(kv_cache.free_blocks) < required_blocks:
                continue
            block_table = kv_cache.allocate(request.request_id, request.total_tokens)
            if block_table:
                request.block_table = block_table
                request.status = RequestStatus.RUNNING
                request.started_at = time.time()
                self.waiting_queue.remove(request)
                self.running_queue.append(request)
                scheduled.append(request)
        return scheduled
```

### 5.4 The Engine's Execution Loop

The following diagram shows the flow of one engine step.

```mermaid
flowchart TD
    Start["Start<br/>Inference Step"] ==> Schedule{Scheduler};

    subgraph Schedule [" SCHEDULING PHASE "]
        direction TB
        S1["Check<br/>Waiting Queue"] --> S2["Prioritize<br/>Requests"];
        S2 --> S3["Check Cache<br/>Availability"];
        S3 --> S4["Allocate KV<br/>Cache Blocks"];
        S4 --> S5["Move to<br/>Running Queue"];
    end

    Schedule ==> Batch{"Request<br/>Types?"};

    Batch ==>|"Prefill Only"| Prefill["Prefill<br/>Batch"];
    Batch ==>|"Decode Only"| Decode["Decode<br/>Batch"];
    Batch ==>|"Mixed"| Mixed["Mixed<br/>Batch"];

    Prefill ==> Model["Model Forward Pass"];
    Decode ==> Model;
    Mixed ==> Model;

    subgraph Model [" MODEL EXECUTION "]
        M1["Input<br/>Embeddings"] --> M2["Transformer<br/>Layers"];

        subgraph M2 [" Paged Attention in Layer L "]
            M3["Block Table<br/>Lookup"];
            M4["Gather<br/>KV Cache"];
            M5["Compute<br/>Attention"];
            M6["Update<br/>Cache"];
        end

        M2 --> M7["Output<br/>Logits"];
    end

    Model ==> Sample[Sampling];

    subgraph Sample [" NEXT TOKEN SELECTION "]
        S6["Apply<br/>Temperature"];
        S7["Top-k/Top-p<br/>Filtering"];
        S8["Random<br/>Sampling"];
    end

    Sample ==> Update["Update Requests"];

    subgraph Update [" STATE UPDATE "]
        U1["Append<br/>New Tokens"];
        U2["Update<br/>KV Cache"];
        U3["Check<br/>Completion"];
        U4["Update<br/>Metrics"];
    end

    Update ==> Cleanup["Cleanup<br/>Finished"];
    Cleanup ==> End["End<br/>Step"];
```

### 5.5 Continuous Batching Timeline

This Gantt chart shows how prefill and decode phases interleave for multiple requests.

```mermaid
gantt
    dateFormat  X
    axisFormat  %L ms
    title Continuous Batching Timeline: Overlapping Requests

    section Request A (512 tokens)
    Prefill A (45ms)     :done, a1, 0, 45
    Decode A (100 tokens):active, a2, 45, 145

    section Request B (64 tokens)
    Waiting             :done, b0, 10, 15
    Prefill B (6ms)     :done, b1, 15, 21
    Decode B (50 tokens):active, b2, 21, 71

    section Request C (256 tokens)
    Waiting             :done, c0, 20, 25
    Prefill C (22ms)    :done, c1, 25, 47
    Decode C (150 tokens):active, c2, 47, 197
```

### 5.6 Simplified Engine Step Implementation

Fill in the missing parts of the engine's main step.

```python
# engine.py
class NanoVLLMEngine:
    """Minimal inference engine implementing the full pipeline."""

    def __init__(self, model, tokenizer, scheduler, kv_cache):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.kv_cache = kv_cache
        self.requests: Dict[str, InferenceRequest] = {}

    def add_request(self, prompt: str, **params) -> str:
        """Add a new inference request."""
        token_ids = self.tokenizer.encode(prompt)
        request_id = f"req_{len(self.requests)}"
        request = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=token_ids
        )
        self.requests[request_id] = request
        self.scheduler.add_request(request)
        return request_id

    def step(self):
        # 1. Schedule new requests
        scheduled = self.scheduler.schedule(self.kv_cache)
        if not self.scheduler.running_queue:
            return {}

        # 2. Prepare batch
        batch = self._prepare_batch(self.scheduler.running_queue)

        # 3. Forward pass
        if batch['is_prefill']:
            logits = self.model.prefill(batch['embeddings'], batch['block_tables'])
        else:
            logits = self.model.decode(batch['embeddings'], batch['block_tables'], batch['seq_lengths'])

        # 4. Sample next tokens
        next_tokens = self._sample(logits, batch['requests'])

        # 5. Update each request
        completed = []
        for i, req in enumerate(batch['requests']):
            req.output_token_ids.append(next_tokens[i])
            # Q1: Update KV cache (append new K/V for this token)
            # (Assume model has method to get new K/V from last token)
            new_k, new_v = self.model.get_last_kv(req, next_tokens[i])
            # In a full implementation, you would call kv_cache.append
            # self.kv_cache.append(req.request_id, new_k, new_v)

            if self._is_finished(req):
                req.status = RequestStatus.FINISHED
                self.scheduler.running_queue.remove(req)
                self.scheduler.finished_queue.append(req)
                self.kv_cache.free(req.request_id)
                completed.append(req.request_id)

        return {'completed': completed}
```

**Hints:**

- Q1: The exact implementation of `append` depends on the cache design. In a paged cache, you would write into the current block and allocate a new block if needed.

<details>
<summary>Click to see discussion</summary>

In a real implementation, `append` would:

- Determine the current block and offset from the block table and sequence length.
- Write the new K and V tensors into the appropriate slot in that block.
- If the block is full, allocate a new block from the free pool and update the block table.

The vLLM kernel documentation describes this process in detail: each thread group loads query vectors, then iterates over blocks, computing attention scores and accumulating results .

</details>

### 5.7 Request State Machine

This diagram shows the lifecycle of a request through the engine.

```mermaid
stateDiagram-v2
    [*] --> WAITING: New request arrives
    WAITING --> RUNNING: Blocks allocated + batched
    WAITING --> CANCELLED: User/timeout

    RUNNING --> RUNNING: Decode token
    RUNNING --> FINISHED: EOS / max_tokens
    RUNNING --> CANCELLED: User stop

    FINISHED --> [*]: Free blocks, return output
    CANCELLED --> [*]: Free blocks, return error

    note right of RUNNING
        Has block_table
        Lives in batch mix
        KV grows linearly
    end note
```

### 5.8 Checkpoint

**Self-Assessment:**

- [ ]  I can explain the role of the scheduler in continuous batching.
- [ ]  I understand how the engine interleaves prefill and decode.
- [ ]  I can trace the data flow from request admission to token output.

---

## Epilogue: The Complete System

You have now built (conceptually) all components of a minimal inference engine:

| Component | Purpose | Source |
| --- | --- | --- |
| Tokenizer | Converts text to token IDs | BPE algorithm |
| Embedding layer | Maps IDs to dense vectors | Standard transformer component |
| Transformer layers | Process embeddings with cached K/V | Attention mechanism |
| KV cache | Stores keys/values for previous tokens | Avoids recomputation |
| PagedAttention allocator | Manages cache blocks without fragmentation | vLLM design |
| Scheduler | Orchestrates multiple requests | Continuous batching |
| Sampler | Selects next token | Top-k, top-p, temperature |

To verify your understanding, run through the full flow for a single request:

```python
# Simulate one inference step (pseudocode)
request = InferenceRequest(request_id="1", prompt_token_ids=[1,2,3,4])
engine.add_request(request)
while engine.has_running():
    output = engine.step()
    if output['completed']:
        print(f"Completed: {output['completed']}")
```

---

## The Principles

1. **Inference is a subset of training** – Remove backward pass, gradients, and optimizer states; only forward pass and KV cache remain. This shifts bottlenecks from compute to memory bandwidth .
2. **The KV cache is the only dynamic state** – Its size grows linearly with sequence length and batch size, dominating memory at scale. As research shows, "the KV cache grows linearly with respect to the length and number of reasoning chains" .
3. **Paging eliminates fragmentation** – Decouple logical token order from physical memory using fixed‑size blocks and a block table, exactly as operating systems have done for decades .
4. **Two phases have different characteristics** – Prefill is compute‑bound, decode is memory‑bound; treat them separately in scheduling .
5. **Continuous batching maximizes utilization** – Interleave prefill and decode to keep GPU busy while respecting memory limits .

---

## Troubleshooting

### Error: CUDA out of memory

**Cause:** KV cache has grown too large or too many requests are running concurrently.

**Solutions:**

- Reduce `max_batch_size` in the scheduler.
- Use a smaller block size (but may increase block table overhead).
- Enable request preemption (swap KV cache to CPU) as described in the vLLM paper .

### Error: Tokenizer produces unexpected tokens

**Cause:** The tokenizer may add special tokens (e.g., `<|endoftext|>`) or handle spaces differently.

**Solution:** Inspect the tokenizer's configuration. Byte-level BPE tokenizers like GPT-2's add spaces differently from WordPiece tokenizers like BERT's .

### Error: Model forward pass fails due to shape mismatch

**Cause:** Block table indices out of range or incorrect tensor shapes.

**Solution:** Verify that the block table length matches the number of blocks allocated. Check that `head_dim` and `num_kv_heads` match the model's configuration.

---

## Next Steps

Now that you understand the inference pipeline, proceed to [**Lab 0.2: Prefill vs. Decode – The Two‑Phase Engine**](https://www.notion.so/lab0.2/README.md), where you will dive deeper into the compute‑bound prefill phase and memory‑bound decode phase, and implement a continuous batching scheduler that handles both.

**Suggested extensions:**

- Implement a simple simulator that models cache allocation and measures throughput under different scheduling policies.
- Add support for **speculative decoding** to reduce latency.
- Explore **quantization** (INT8, INT4) to shrink the weight memory footprint.
- Investigate KV cache compression techniques like Dynamic Memory Sparsification .

---

## Additional Resources

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.readthedocs.io/) – Production inference engine with PagedAttention
- [FlashAttention: Fast and Memory‑Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) – I/O-aware attention algorithm
- [Byte-Pair Encoding tokenization](https://huggingface.co/docs/transformers/tokenizer_summary) – Hugging Face's tokenizer documentation
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) – NVIDIA's optimized inference runtime
- [Characterizing KV Caching Under Concurrency](https://ieeexplore.ieee.org/document/11078403) – IEEE study on KV cache behavior

---

**Navigation:** [← Main](https://www.notion.so/README.md) | [Next: Lab 0.2 →](https://www.notion.so/lab0.2/README.md)