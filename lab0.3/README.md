# Lab 0.3: The Heart of the Matter – KV Cache & Attention

**Navigation:** [← Lab 0.2](../lab0.2/README.md) | [Main](../README.md)

## Introduction

Large Language Model (LLM) inference spends 40–70% of its dynamic memory on the Key-Value (KV) cache. The way you manage this cache determines how many users you can serve concurrently, how fast each token is generated, and whether your system fits into a single GPU. This lab walks you through the evolution of KV cache management—from naive contiguous allocation to the production‑ready PagedAttention. You will implement each stage, measure its memory efficiency, and understand why modern inference engines adopted virtual memory techniques for neural networks.

## Learning Objectives

By the end of this lab, you will be able to:

1. **Explain** the mathematical necessity of caching in autoregressive attention using complexity analysis.
2. **Implement** a naive contiguous KV cache and quantify its internal fragmentation.
3. **Design** a dynamic block‑based allocator that reduces waste but introduces external fragmentation.
4. **Build** a complete PagedAttention cache with block tables and slot mapping.
5. **Extend** the cache with block‑sparse attention and prefix caching for advanced memory savings.
6. **Analyze** performance trade‑offs using simulated memory usage and latency metrics.

**Prerequisites:**  
- Basic Python and PyTorch (tensor operations)  
- Understanding of transformer attention mechanism (Q, K, V)  
- Familiarity with command line and virtual environments

## Prologue: The Challenge

You join the inference optimisation team at a fast‑growing AI startup. The product uses a large language model to generate chat responses, documentation, and code. Initial load tests reveal a worrying trend: as soon as more than eight users chat simultaneously, the GPU runs out of memory and the service crashes.

Your senior engineer hands you the profiling report:

- **Naive KV cache per sequence:** 512 MB reserved (max length 2048 tokens)  
- **Actual average usage:** 128 tokens → 32 MB  
- **Batch of 16 sequences:** 8.2 GB allocated, but only 1.2 GB truly used  
- **Result:** 88% internal fragmentation, GPU underutilised, and users queued.

Your mission: redesign the KV cache subsystem to support **at least 64 concurrent users** with the same GPU memory budget. You must eliminate fragmentation, allow sequences to grow arbitrarily, and keep the attention kernel fast.

This lab guides you through the exact steps that turned vLLM into the industry standard. You will build each component yourself, starting from the simplest cache and ending with a fully functional PagedAttention manager.

## Environment Setup

Create a dedicated directory and virtual environment for this lab.

**For Linux/macOS:**

```bash
# Create project folder
mkdir ~/kv-cache-lab
cd ~/kv-cache-lab

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch numpy matplotlib
```

**For Windows (PowerShell):**

```powershell
# Create project folder
New-Item -ItemType Directory -Path ~\kv-cache-lab
Set-Location ~\kv-cache-lab

# Set up Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required packages
pip install torch numpy matplotlib
```

We will simulate GPU memory using CPU tensors, but the code is written to be device‑agnostic. If you have a CUDA‑capable GPU, you can move tensors to `cuda` by changing the device argument.

Create the initial project structure:

**Linux/macOS:**
```bash
mkdir -p cache_implementations tests
touch cache_implementations/__init__.py
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path cache_implementations, tests
New-Item -ItemType File -Path cache_implementations\__init__.py
```

All code you write will be placed in the `cache_implementations/` directory or directly in a Jupyter notebook if you prefer interactive exploration.


![visual-1.png](visual-1.png)
---

## Chapter 1: Foundations – Why Caching?

### Opening Context

The attention mechanism is the computational heart of transformers. Without caching, generating a single new token would require recomputing the key and value vectors for every previous token—a cost that grows quadratically with sequence length. This chapter derives the mathematical necessity of caching and sets the stage for every optimisation that follows.

### 1.1 What You Will Build

You will implement two versions of attention:
- **Without cache:** Recompute all keys and values for every token.
- **With cache:** Store keys and values and reuse them.

By comparing their theoretical complexity and actual runtime (simulated), you will see why caching is non‑negotiable.

### 1.2 Think First: The Quadratic Trap

Consider a sequence of length `n`. For each new token `t`, the attention mechanism computes:

- Query for token `t`: `Q_t = W_q · h_t`
- Keys for all previous tokens: `K_{1:t} = W_k · [h_1, ..., h_t]`
- Values for all previous tokens: `V_{1:t} = W_v · [h_1, ..., h_t]`
- Attention output: `softmax(Q_t · K_{1:t}ᵀ/√d) · V_{1:t}`

**Question:** How many key and value vectors are computed **in total** when generating all `n` tokens (assuming we start with a prompt of length `p` and generate `n-p` new tokens)?

<details>
<summary>Click to review</summary>

Without cache, each step recomputes all keys and values up to that point.  
If the prompt length is `p` and we generate `m = n-p` tokens:

- Prompt processing: `p` K,V vectors computed once.
- Generation step 1: recompute all `p+1` K,V → `p+1` vectors.
- Generation step 2: recompute all `p+2` K,V → `p+2` vectors.
- ...
- Generation step m: recompute all `p+m = n` K,V → `n` vectors.

Total K,V computations = `p + (p+1) + ... + n` = sum from `p` to `n` of `i`.  
For large `n` this is approximately `(n² - p²)/2`, i.e. **quadratic** in sequence length.

With cache, each step computes **only the current token’s K,V** → `n` total computations (linear).

</details>

### 1.3 Implementation: Simulate the Cost

We will not implement real attention kernels; instead we will count operations to see the quadratic explosion.

Create a file `cache_implementations/attention_sim.py` and complete the code:

```python
import time

def attention_without_cache(prompt_len, gen_len):
    """Simulate attention without KV cache by counting recomputations."""
    total_kv_computations = ___  # Q1: Initialize counter
    # prompt processing
    total_kv_computations += ___  # Q2: How many K,V for prompt?
    for step in range(1, gen_len + 1):
        # at step i, we have prompt_len + i tokens
        total_kv_computations += ___ + ___  # Q3: What computation happens each step?
    return total_kv_computations

def attention_with_cache(prompt_len, gen_len):
    """With cache: compute once per token."""
    return ___ + ___  # Q4: Total tokens processed

# Example: prompt=128, generate 100 tokens
p, g = 128, 100
print(f"Without cache: {attention_without_cache(p, g)} operations")
print(f"With cache:    {attention_with_cache(p, g)} operations")
```

**Hints:**
- Q1: What value should a counter start at?
- Q2: In the first pass, how many tokens in the prompt need K,V computed?
- Q3: At each generation step, we recompute K,V for all previous tokens
- Q4: How many total tokens are there (prompt + generated)?

<details>
<summary>Click to see solution</summary>

```python
import time

def attention_without_cache(prompt_len, gen_len):
    """Simulate attention without KV cache by counting recomputations."""
    total_kv_computations = 0  # Q1: Initialize counter
    # prompt processing
    total_kv_computations += prompt_len  # Q2: How many K,V for prompt?
    for step in range(1, gen_len + 1):
        # at step i, we have prompt_len + i tokens
        total_kv_computations += prompt_len + step  # Q3: What computation happens each step?
    return total_kv_computations

def attention_with_cache(prompt_len, gen_len):
    """With cache: compute once per token."""
    return prompt_len + gen_len  # Q4: Total tokens processed

# Example: prompt=128, generate 100 tokens
p, g = 128, 100
print(f"Without cache: {attention_without_cache(p, g)} operations")
print(f"With cache:    {attention_with_cache(p, g)} operations")
```

**Answers:**
- Q1: `0` – counters start at zero
- Q2: `prompt_len` – each prompt token needs K,V computed
- Q3: `prompt_len + step` – recompute K,V for all tokens up to current position
- Q4: `prompt_len + gen_len` – total number of tokens processed

</details>

**Predict:** Run the code. What ratio do you observe between the two numbers?

<details>
<summary>Click to verify</summary>

```
Without cache: 16576 operations
With cache:    228 operations
```

The cached version is about **73×** cheaper for this modest sequence. For longer sequences the gap becomes astronomical.

</details>

### 1.4 Understanding the Complexity

The diagram below illustrates the difference in computational load between the two approaches.

```mermaid
graph TD
    subgraph NO [" WITHOUT KV CACHE O(n²) "]
        NC1["Token 1<br/>Process 1 token<br/>Ops: 1"] --> NC2["Token 2<br/>Reprocess: 1,2<br/>Ops: 2"]
        NC2 --> NC3["Token 3<br/>Reprocess: 1,2,3<br/>Ops: 3"]
        NC3 --> NC4["..."]
        NC4 --> NC5["Token n<br/>Reprocess all<br/>Ops: n"]
    end
    
    subgraph YES [" WITH KV CACHE O(n) "]
        WC1["Token 1<br/>Process & Cache<br/>Ops: 1"] --> WC2["Token 2<br/>Use cache + new<br/>Ops: 1"]
        WC2 --> WC3["Token 3<br/>Use cache + new<br/>Ops: 1"]
        WC3 --> WC4["..."]
        WC4 --> WC5["Token n<br/>Use cache + new<br/>Ops: 1"]
    end
    
    NC5 -->|"n=1000<br/>~500K ops"| Result1["Too Slow<br/>Prohibitive for Serving"]
    WC5 -->|"n=1000<br/>~1K ops"| Result2["Practical<br/>Enable Real-time"]
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 1.1: Comparison of total operations with and without KV cache. Adapted from concepts in the vLLM paper (Kwon et al., 2023).*

The table below summarises the asymptotic behaviour:

| Approach | Time per token | Total time for n tokens |
|----------|----------------|-------------------------|
| No cache | O(t)           | O(n²)                   |
| With cache| O(t) (attention) + O(1) (projection) | O(n²) total, but O(n) new projections |

Even with cache, the attention dot‑product still grows linearly with sequence length, but the expensive projection of K and V happens only once.

### 1.5 Matching Exercise: Complexity Analysis

Match each operation to its computational complexity:

| Operation | Complexity (A-E) |
|-----------|------------------|
| Computing K,V for all tokens without cache (n tokens) | ___ |
| Computing K,V with cache (n tokens) | ___ |
| Attention dot-product Q · Kᵀ at position t | ___ |
| Memory lookup for cached K,V | ___ |
| Pre-allocating cache tensor | ___ |

**Options:**
- A: O(1) – constant time
- B: O(n) – linear in sequence length
- C: O(n²) – quadratic in sequence length
- D: O(t) – linear in current position
- E: O(log n) – logarithmic

<details>
<summary>Click to verify answers</summary>

| Operation | Complexity | Explanation |
|-----------|------------|-------------|
| Computing K,V for all tokens without cache (n tokens) | **C: O(n²)** | Sum of 1+2+...+n = n(n+1)/2 |
| Computing K,V with cache (n tokens) | **B: O(n)** | Each token computed once |
| Attention dot-product Q · Kᵀ at position t | **D: O(t)** | Dot product with all previous tokens |
| Memory lookup for cached K,V | **A: O(1)** | Direct tensor indexing |
| Pre-allocating cache tensor | **A: O(1)** | Single allocation operation |

</details>

### 1.6 Checkpoint

**Self-Assessment:**
- [ ] You can explain why the attention equation without cache leads to quadratic complexity
- [ ] You have run the simulation and verified the operation count
- [ ] You completed the fill-in-the-blank code and got the correct output
- [ ] You can match operations to their complexity classes
- [ ] You understand that caching reduces the computational burden from quadratic to linear for key/value projections

---

## Chapter 2: Naive Implementation – Contiguous Cache

### Opening Context

The simplest way to cache keys and values is to pre‑allocate a contiguous tensor large enough for the maximum possible sequence length. This is what many tutorials show. It works for a single sequence, but when many sequences share the GPU, the waste becomes unacceptable.

### 2.1 What You Will Build

You will implement the `NaiveKVCache` class, then simulate a batch of sequences with varying lengths to observe internal fragmentation.

### 2.2 Think First: Memory Waste

Assume a GPU with 40 GB memory, a model with 32 attention heads and head dimension 128. Each token’s KV cache requires `2 * 32 * 128 * 2 bytes = 16 KB` (using `bfloat16`).  
If you set `max_seq_len = 4096`, each sequence reserves `4096 * 16 KB = 64 MB`.

**Question:** With 40 GB total, how many sequences can you fit **if you pre‑allocate the maximum for each**? How many could you fit if sequences used only 256 tokens on average?

<details>
<summary>Click to review</summary>

- Max allocation per sequence: 64 MB → 40 GB / 64 MB = **640 sequences** (theoretical maximum).  
- If each sequence actually uses 256 tokens, they need only `256 * 16 KB = 4 MB` each.  
- With the naive pre‑allocation you are **wasting 60 MB per sequence** (94% waste).  
- In reality you would hit OOM long before 640 sequences because the reserved memory is not yet used but is considered allocated by the allocator. This is internal fragmentation.

</details>

### 2.3 Implementation: NaiveKVCache

Create `cache_implementations/naive.py` and complete the implementation:

```python
import torch
from typing import Tuple

class NaiveKVCache:
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.max_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Q1: Pre-allocate cache for maximum possible length
        self.k_cache = torch.zeros(___, ___, ___, dtype=dtype)  # shape?
        self.v_cache = torch.zeros(___, ___, ___, dtype=dtype)  # shape?
        self.current_len = ___  # Q2: How many tokens are currently stored?

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full cache."""
        pos = ___  # Q3: Where to write the new token?
        self.k_cache[pos] = new_k.squeeze()  # assume shape [num_heads, head_dim]
        self.v_cache[pos] = new_v.squeeze()
        self.current_len += ___  # Q4: Increment by how much?
        return self.k_cache[:___], self.v_cache[:___]  # Q5: Return how many positions?

    def reset(self):
        self.current_len = ___  # Q6: Reset to what value?

    def allocated_bytes(self):
        """Return total allocated memory in bytes."""
        element_size = self.k_cache.element_size()
        return (self.k_cache.numel() + self.v_cache.numel()) * element_size

    def used_bytes(self):
        """Return memory actually used by current tokens."""
        element_size = self.k_cache.element_size()
        used_slots = ___ * ___ * ___  # Q7: How many elements used?
        return used_slots * element_size * ___  # Q8: Multiply by what for K and V?
```

**Hints:**
- Q1: Pre-allocate for `max_seq_len` tokens, each with `num_heads` and `head_dim`
- Q2: Initially, how many tokens are stored?
- Q3: Use `current_len` as the write position
- Q4: Each update adds how many tokens?
- Q5: Return only the occupied portion of the cache
- Q6: When resetting, set current length to?
- Q7: How many elements are used? (current_len * num_heads * head_dim)
- Q8: We have both K and V caches

<details>
<summary>Click to see solution</summary>

```python
import torch
from typing import Tuple

class NaiveKVCache:
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.max_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Allocate maximum possible cache upfront [max_len, num_heads, head_dim]
        self.k_cache = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype)
        self.current_len = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full cache."""
        pos = self.current_len
        self.k_cache[pos] = new_k.squeeze()  # assume shape [num_heads, head_dim]
        self.v_cache[pos] = new_v.squeeze()
        self.current_len += 1
        return self.k_cache[:self.current_len], self.v_cache[:self.current_len]

    def reset(self):
        self.current_len = 0

    def allocated_bytes(self):
        """Return total allocated memory in bytes."""
        element_size = self.k_cache.element_size()
        return (self.k_cache.numel() + self.v_cache.numel()) * element_size

    def used_bytes(self):
        """Return memory actually used by current tokens."""
        element_size = self.k_cache.element_size()
        used_slots = self.current_len * self.num_heads * self.head_dim
        return used_slots * element_size * 2  # K and V
```

</details>

Now simulate a batch of 16 sequences with random lengths between 100 and 500 tokens, but all pre‑allocated for `max_len=2048`.

```python
# simulation.py
from cache_implementations.naive import NaiveKVCache
import random

num_heads = 32
head_dim = 128
max_len = 2048
batch_size = 16

caches = []
for i in range(batch_size):
    cache = NaiveKVCache(max_len, num_heads, head_dim)
    # Simulate a sequence of random length
    seq_len = random.randint(100, 500)
    for _ in range(seq_len):
        # dummy K,V tensors
        k = torch.randn(num_heads, head_dim)
        v = torch.randn(num_heads, head_dim)
        cache.update(k, v)
    caches.append(cache)

total_alloc = sum(c.allocated_bytes() for c in caches) / (1024**3)
total_used = sum(c.used_bytes() for c in caches) / (1024**3)
print(f"Total allocated: {total_alloc:.2f} GB")
print(f"Total used:      {total_used:.2f} GB")
print(f"Fragmentation:   {(1 - total_used/total_alloc)*100:.1f}%")
```

**Predict:** What fragmentation percentage do you expect?

<details>
<summary>Click to verify</summary>

With `max_len=2048` and actual lengths 100‑500, you should see fragmentation around 80‑90%. Example output:

```
Total allocated: 2.00 GB
Total used:      0.24 GB
Fragmentation:   88.0%
```

</details>

### 2.4 Understanding the Code

The diagram below visualises the memory waste in naive allocation.

```mermaid
graph TD
    subgraph VRAM [" GPU VRAM LAYOUT - 4K TOKENS PER SEQUENCE "]
        A1["Seq A<br/>Allocated: 4K slots<br/>Actual: 512 tokens<br/>3.5K WASTED"]
        A2["Seq B<br/>Allocated: 4K slots<br/>Actual: 128 tokens<br/>3.9K WASTED"]
        A3["Seq C<br/>Allocated: 4K slots<br/>Actual: 2K tokens<br/>2K WASTED"]
        A4["... 97 more sequences ...<br/>Allocated: 388K slots<br/>No space for new requests"]
    end
    
    METRICS["MEMORY CRISIS<br/>Allocated: 400K slots<br/>Used: 50K tokens<br/>FREE: 0GB GPU RAM<br/>87.5% INTERNAL FRAGMENTATION"]
    
    A1 --> A2 --> A3 --> A4 --> METRICS
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 2.1: Internal fragmentation in a naive contiguous cache. Adapted from concepts in the vLLM paper.*

- The cache pre‑allocates `max_seq_len` slots, regardless of how many tokens are actually stored.
- `allocated_bytes` counts the full tensor memory, while `used_bytes` counts only the occupied positions.
- The waste is **internal fragmentation**: allocated but unused memory inside each cache.

### 2.5 Matching Exercise: Memory Concepts

Match each memory concept to its definition:

| Concept | Definition (A-F) |
|---------|------------------|
| Internal fragmentation | ___ |
| External fragmentation | ___ |
| Memory allocation | ___ |
| Memory utilization | ___ |
| Pre-allocation | ___ |
| Wasted memory | ___ |

**Options:**
- A: Ratio of used memory to allocated memory
- B: Unused space within an allocated block
- C: Free memory scattered across non-contiguous regions
- D: Allocating maximum capacity before knowing actual needs
- E: Requesting memory from a pool or system
- F: Allocated but unused memory that could serve other requests

<details>
<summary>Click to verify answers</summary>

| Concept | Definition | Explanation |
|---------|------------|-------------|
| Internal fragmentation | **B** | Unused space within an allocated block (e.g., allocated 2048 but use 128) |
| External fragmentation | **C** | Free memory scattered and non-contiguous (prevents large allocations) |
| Memory allocation | **E** | Requesting memory from a pool or system |
| Memory utilization | **A** | Ratio of used to allocated memory (100% - fragmentation%) |
| Pre-allocation | **D** | Allocating maximum capacity upfront |
| Wasted memory | **F** | Allocated but unused memory that could serve others |

</details>

### 2.6 Experiment: Breaking Point Analysis

This experiment demonstrates how naive allocation limits concurrency.

**Task:** Find the maximum number of concurrent sequences before running out of memory.

Create `experiments/naive_limit.py`:

```python
from cache_implementations.naive import NaiveKVCache
import random

MAX_GPU_MEMORY = 16 * (1024**3)  # Simulate 16 GB GPU
max_len = 2048
num_heads = 32
head_dim = 128

caches = []
total_allocated = 0
seq_count = 0

try:
    while total_allocated < MAX_GPU_MEMORY:
        cache = NaiveKVCache(max_len, num_heads, head_dim)
        # Simulate actual usage (100-500 tokens)
        actual_len = random.randint(100, 500)
        for _ in range(actual_len):
            cache.update(torch.randn(num_heads, head_dim), 
                        torch.randn(num_heads, head_dim))
        
        total_allocated += cache.allocated_bytes()
        caches.append(cache)
        seq_count += 1
        
        if seq_count % 10 == 0:
            total_used = sum(c.used_bytes() for c in caches)
            print(f"Sequences: {seq_count}, Allocated: {total_allocated/(1024**3):.2f} GB, "
                  f"Used: {total_used/(1024**3):.2f} GB, "
                  f"Waste: {(1-total_used/total_allocated)*100:.1f}%")
except:
    pass

print(f"\nMaximum concurrent sequences: {seq_count}")
print(f"This is {'below' if seq_count < 64 else 'above'} the target of 64 users")
```

**Predict:** How many sequences can you fit before hitting 16 GB? Will it meet the 64-user requirement from the prologue?

<details>
<summary>Click to see analysis</summary>

With `max_len=2048`, each cache allocates ~32 MB. Theoretical maximum: 16 GB / 32 MB = **512 sequences**.  
However, with only 100–500 tokens actually used, you are wasting 75–90% of memory.  
Effective capacity: around **100–150 sequences** before hitting memory pressure.

This demonstrates why naive allocation fails the 64-user target when accounting for safety margins and OS overhead.

</details>

### 2.7 Checkpoint

**Self-Assessment:**
- [ ] You can implement a contiguous KV cache with proper initialization
- [ ] You completed the fill-in-the-blank code and understand each component
- [ ] You can measure allocated vs used memory accurately
- [ ] You matched memory concepts to their definitions correctly
- [ ] You have observed that internal fragmentation can exceed 80% with realistic length distributions
- [ ] You ran the breaking point experiment and understand the concurrency limits

---

## Chapter 3: Dynamic Allocation – Solving Internal Fragmentation

### Opening Context

Internal fragmentation arises because we pre‑allocate a fixed maximum. The natural fix is to allocate memory **on demand** as the sequence grows. This is analogous to using a dynamic array or linked list: you start with a small block and add more blocks when needed.

### 3.1 What You Will Build

You will implement a `DynamicKVCache` that allocates fixed‑size blocks (e.g., 16 tokens per block) and chains them together. This eliminates internal fragmentation because you never allocate more than you actually use.

### 3.2 Think First: External Fragmentation

While dynamic allocation solves internal waste, it introduces a new problem: **external fragmentation**. After many sequences finish and release blocks, the free blocks may be scattered across the memory pool. A new sequence that needs several contiguous blocks may not find them, even though the total free memory is sufficient.

Draw a mental picture: suppose you have free blocks at indices 2,3,4 (contiguous) and 7,8,9 (contiguous). A request for three contiguous blocks can be satisfied. But if free blocks are 2,4,6,8,10, you have five free blocks but no three in a row. This is external fragmentation.

### 3.3 Implementation: DynamicKVCache with Blocks

We will simulate physical memory as a list of blocks, each a small tensor. The `DynamicKVCache` per sequence holds a list of block indices, and the global allocator manages a free list.

Create `cache_implementations/dynamic.py` and complete the implementation:

```python
import torch
from collections import deque
from math import ceil

class Block:
    def __init__(self, block_id, block_size, num_heads, head_dim, dtype=torch.bfloat16):
        self.block_id = block_id
        self.size = block_size
        self.k_data = torch.zeros(___, ___, ___, dtype=dtype)  # Q1: Shape?
        self.v_data = torch.zeros(___, ___, ___, dtype=dtype)  # Q2: Shape?
        self.occupied = ___  # Q3: Initially how many tokens?

    def add(self, k, v):
        if self.occupied >= ___:  # Q4: When is block full?
            raise RuntimeError("Block full")
        self.k_data[___] = k  # Q5: Where to write?
        self.v_data[___] = v  # Q6: Where to write?
        self.occupied += ___  # Q7: Increment by?

    def is_full(self):
        return self.occupied == ___  # Q8: Full when?

class GlobalAllocator:
    """Simulates a global pool of physical blocks."""
    def __init__(self, total_blocks, block_size, num_heads, head_dim):
        self.blocks = [Block(i, block_size, num_heads, head_dim) for i in range(___)]
        self.free_blocks = deque(range(___))

    def alloc_blocks(self, num_blocks):
        if len(self.free_blocks) < ___:  # Q9: When to fail?
            raise MemoryError("Not enough free blocks")
        return [self.free_blocks.popleft() for _ in range(___)]

    def free_blocks_ids(self, block_ids):
        for bid in block_ids:
            self.free_blocks.append(___)

class DynamicKVCache:
    def __init__(self, allocator, seq_id):
        self.allocator = allocator
        self.seq_id = seq_id
        self.blocks = ___  # Q10: Initialize as what?
        self.current_len = ___  # Q11: Initially?

    def allocate_for_prompt(self, prompt_len):
        num_blocks = ceil(prompt_len / self.allocator.blocks[0].size)
        block_ids = self.allocator.alloc_blocks(___)
        self.blocks.extend(___)

    def append_token(self, k, v):
        if not self.blocks:
            # need at least one block
            block_id = self.allocator.alloc_blocks(1)[0]
            self.blocks.append(___)

        last_block = self.allocator.blocks[self.blocks[___]]  # Q12: Index?
        if last_block.is_full():
            # allocate new block
            new_id = self.allocator.alloc_blocks(___)[0]
            self.blocks.append(___)
            last_block = self.allocator.blocks[___]

        last_block.add(k, v)
        self.current_len += ___

    def free(self):
        self.allocator.free_blocks_ids(___)
        self.blocks = ___
        self.current_len = ___
```

**Hints:**
- Block storage: [block_size, num_heads, head_dim]
- Track occupied slots in each block
- Free list managed by GlobalAllocator
- Append to last block if not full, otherwise allocate new

<details>
<summary>Click to see solution</summary>

```python
import torch
from collections import deque
from math import ceil

class Block:
    def __init__(self, block_id, block_size, num_heads, head_dim, dtype=torch.bfloat16):
        self.block_id = block_id
        self.size = block_size
        self.k_data = torch.zeros(block_size, num_heads, head_dim, dtype=dtype)
        self.v_data = torch.zeros(block_size, num_heads, head_dim, dtype=dtype)
        self.occupied = 0   # number of tokens stored

    def add(self, k, v):
        if self.occupied >= self.size:
            raise RuntimeError("Block full")
        self.k_data[self.occupied] = k
        self.v_data[self.occupied] = v
        self.occupied += 1

    def is_full(self):
        return self.occupied == self.size

class GlobalAllocator:
    """Simulates a global pool of physical blocks."""
    def __init__(self, total_blocks, block_size, num_heads, head_dim):
        self.blocks = [Block(i, block_size, num_heads, head_dim) for i in range(total_blocks)]
        self.free_blocks = deque(range(total_blocks))

    def alloc_blocks(self, num_blocks):
        if len(self.free_blocks) < num_blocks:
            raise MemoryError("Not enough free blocks")
        return [self.free_blocks.popleft() for _ in range(num_blocks)]

    def free_blocks_ids(self, block_ids):
        for bid in block_ids:
            self.free_blocks.append(bid)

class DynamicKVCache:
    def __init__(self, allocator, seq_id):
        self.allocator = allocator
        self.seq_id = seq_id
        self.blocks = []          # list of block IDs
        self.current_len = 0

    def allocate_for_prompt(self, prompt_len):
        num_blocks = ceil(prompt_len / self.allocator.blocks[0].size)
        block_ids = self.allocator.alloc_blocks(num_blocks)
        self.blocks.extend(block_ids)

    def append_token(self, k, v):
        if not self.blocks:
            # need at least one block
            block_id = self.allocator.alloc_blocks(1)[0]
            self.blocks.append(block_id)

        last_block = self.allocator.blocks[self.blocks[-1]]
        if last_block.is_full():
            # allocate new block
            new_id = self.allocator.alloc_blocks(1)[0]
            self.blocks.append(new_id)
            last_block = self.allocator.blocks[new_id]

        last_block.add(k, v)
        self.current_len += 1

    def free(self):
        self.allocator.free_blocks_ids(self.blocks)
        self.blocks = []
        self.current_len = 0
```

</details>

Now simulate multiple sequences with random lengths, then free some and try to allocate a large contiguous chunk.

```python
# simulation_dynamic.py
from cache_implementations.dynamic import GlobalAllocator, DynamicKVCache
import random

allocator = GlobalAllocator(total_blocks=1000, block_size=16, num_heads=32, head_dim=128)

# Create 20 sequences with random lengths
sequences = []
for i in range(20):
    seq = DynamicKVCache(allocator, f"seq{i}")
    prompt_len = random.randint(20, 300)
    seq.allocate_for_prompt(prompt_len)
    # simulate generation by appending more tokens
    for _ in range(random.randint(0, 50)):
        k = torch.randn(32,128)
        v = torch.randn(32,128)
        seq.append_token(k,v)
    sequences.append(seq)

# Free half of them
for i in range(10):
    sequences[i].free()

# Try to allocate a new sequence that needs 5 blocks (80 tokens)
try:
    new_seq = DynamicKVCache(allocator, "new")
    new_seq.allocate_for_prompt(80)
    print("Allocation succeeded")
except MemoryError:
    print("Allocation failed due to fragmentation")

# Check free blocks distribution
free_indices = list(allocator.free_blocks)
print(f"Free blocks: {free_indices[:20]}... (total {len(free_indices)})")
```

**Predict:** Will the allocation for 80 tokens succeed? Run the simulation multiple times; do you ever see failures even though total free blocks > needed?

<details>
<summary>Click to discuss</summary>

Because we only check `len(free_blocks) >= num_blocks`, the allocator can succeed even if blocks are scattered. In this simple allocator we are **not** requiring contiguity in physical memory because we store each block separately. However, if the attention kernel requires that the blocks of a sequence be contiguous in physical memory (which is not the case here), external fragmentation would prevent allocation. In our simulation, blocks are independent, so we don't see external fragmentation yet. The next chapter introduces PagedAttention, which uses a block table to decouple logical contiguity from physical scattering, thereby eliminating fragmentation completely.

</details>

### 3.4 Understanding the Code

The diagram below contrasts static and dynamic allocation.

```mermaid
graph TD
    subgraph STATIC [" STATIC ALLOCATION "]
        S1["Seq A<br/>4K pre-allocated<br/>Used: 512<br/>Waste: 3.5K"]
        S2["Seq B<br/>4K pre-allocated<br/>Used: 128<br/>Waste: 3.9K"]
        S3["Overhead<br/>7.5K slots unused<br/>80% FRAGMENTED"]
    end
    
    subgraph DYNAMIC [" DYNAMIC ALLOCATION "]
        D1["Seq A<br/>2 blocks (32 tokens)<br/>Exact fit"]
        D2["Seq B<br/>1 block (16 tokens)<br/>Exact fit"]
        D3["Free<br/>Many blocks<br/>Available"]
    end
    
    STATS["100 Concurrent Sequences<br/><br/>Static: 400K slots, 320K wasted (80%)<br/>Dynamic: 80K slots, 10K wasted (12.5%)"]
    
    S1 --> S3
    S2 --> S3
    D1 --> D3
    D2 --> D3
    S3 --> STATS
    D3 --> STATS
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 3.1: Static vs dynamic allocation. Dynamic allocation eliminates internal fragmentation but may still suffer from external fragmentation if contiguity is required. Adapted from concepts in the vLLM paper.*

The next diagram illustrates how external fragmentation can occur over time.

```mermaid
graph TD
    subgraph TIMELINE [" MEMORY FRAGMENTATION TIMELINE "]
        T0["T=0: All Active<br/>SeqA: B0 B1<br/>SeqB: B2 B3<br/>SeqC: B4 B5<br/>SeqD: B6 B7"]
        T1["T=1: SeqA & C Leave<br/>FREE: B0 B1, B4 B5<br/>USED: B2 B3 (SeqB)<br/>USED: B6 B7 (SeqD)"]
        T2["T=2: SeqE Needs 3 Blocks<br/>Request: Contiguous 3"]
    end
    
    FRAGSTAT["Memory State<br/>Free Blocks: 4 total<br/>B0-B1 (island 1)<br/>B4-B5 (island 2)<br/>Gap: B2-B3 (used)"]
    
    ATTEMPT["Allocation Attempt<br/>Max Contiguous: 2 blocks<br/>Needed: 3 blocks<br/>FAILS"]
    
    PROBLEM["EXTERNAL FRAGMENTATION<br/>4 FREE BLOCKS<br/>1 REQUEST REJECTED<br/>47% effective utilization"]
    
    T0 --> T1 --> T2 --> FRAGSTAT
    FRAGSTAT --> ATTEMPT --> PROBLEM
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 3.2: External fragmentation after sequences finish. Free blocks are not contiguous, so a request for three contiguous blocks fails even though enough total free blocks exist. Adapted from concepts in the vLLM paper.*

### 3.5 Scenario: Fragmentation in Production

**Scenario:** A production inference server handles chat requests with highly variable lengths. The distribution is:
- 40% short (50-100 tokens)
- 40% medium (200-500 tokens)
- 20% long (1000-2000 tokens)

Requests arrive and complete randomly. After 2 hours of operation with dynamic block allocation (block_size=16), the memory looks like this:

```
Free blocks: [5, 12, 18, 23, 27, 31, 35, 40, 42, 47, 51, 55, ...]
Total free: 280 blocks
Largest contiguous free region: 8 blocks
```

A new request arrives that needs 150 tokens (10 blocks).

**Question 1:** Can the allocator satisfy this request if it requires physical contiguity?

**Question 2:** Can the allocator satisfy this request if blocks can be non-contiguous (like in PagedAttention)?

**Question 3:** What percentage of the free pool is wasted due to fragmentation if we require contiguity?

<details>
<summary>Click to analyze</summary>

**Answer 1:** No. The request needs 10 contiguous blocks, but the largest contiguous region is only 8 blocks. This is external fragmentation.

**Answer 2:** Yes. With PagedAttention, blocks do not need to be contiguous. The 280 free blocks can serve any request needing ≤280 blocks, regardless of allocation pattern.

**Answer 3:** The request needs 10 blocks out of 280 available. Without PagedAttention, it fails. Effective utilization = (280 - 10) / 280 = 96.4% of memory is wasted for this request. In aggregate, if all new requests need >8 blocks, **72% of free memory becomes unusable** (280 - 8) / 280.

This demonstrates why production systems adopted virtual memory techniques.

</details>

### 3.6 Checkpoint

**Self-Assessment:**
- [ ] You can implement a block-based dynamic allocator with proper initialization
- [ ] You completed the fill-in-the-blank code for Block, GlobalAllocator, and DynamicKVCache
- [ ] You understand that dynamic allocation removes internal fragmentation
- [ ] You analyzed the scenario and understand how external fragmentation occurs
- [ ] You can explain why non-contiguous allocation (paging) solves external fragmentation
- [ ] You can identify that external fragmentation can still occur if physical contiguity is required

---

## Chapter 4: PagedAttention – Virtual Memory for LLMs

### Opening Context

Dynamic block allocation still ties a sequence to specific physical blocks, but it does not require them to be contiguous. However, the attention kernel must be able to gather the scattered blocks efficiently. PagedAttention introduces a **block table** that maps logical token positions to physical block IDs, exactly like a page table in an operating system. This abstraction completely eliminates fragmentation and enables near‑100% memory utilisation.

### 4.1 What You Will Build

You will implement a `PagedKVCache` that manages a global physical memory pool of blocks, maintains per‑sequence block tables, and provides methods to read/write tokens by logical index.

### 4.2 Think First: Page Tables

In virtual memory, each process has a page table that translates virtual addresses to physical frames. The physical frames can be anywhere in RAM. How does this help with fragmentation?

**Question:** What would happen if a process’s pages were scattered across physical memory? Would the process notice?

<details>
<summary>Click to review</summary>

The process does not see physical addresses; it works with virtual addresses that appear contiguous. The hardware MMU translates each access. Scattering is invisible to the process and **eliminates external fragmentation** because any free frame can be used to satisfy any page request.

</details>

### 4.3 Implementation: PagedKVCache

We will now build the core of this lab: a complete paged cache with block tables.

Create `cache_implementations/paged.py` and complete the implementation:

```python
import torch
from collections import deque
from math import ceil
from typing import List, Tuple, Dict

class PagedKVCache:
    def __init__(self, total_blocks: int, block_size: int,
                 num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Q1: Physical storage shape: [total_blocks, 2, num_heads, block_size, head_dim]
        # 2 stands for key (0) and value (1)
        self.kv_data = torch.zeros(
            ___, ___, ___, ___, ___,  # Complete the shape
            dtype=dtype
        )

        # Free block management
        self.free_blocks = deque(range(___))

        # Block tables: sequence_id -> list of physical block indices
        self.block_tables: Dict[str, List[int]] = {}

    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for num_tokens, return list of physical block indices."""
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        blocks_needed = ceil(num_tokens / ___)  # Q2: Divide by what?
        if len(self.free_blocks) < ___:  # Q3: Check condition
            raise MemoryError(f"Not enough free blocks: need {blocks_needed}, have {len(self.free_blocks)}")
        allocated = [self.free_blocks.popleft() for _ in range(___)]
        self.block_tables[___] = ___  # Q4: Store mapping
        return allocated

    def write_token(self, seq_id: str, token_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write K and V for the token at logical position token_idx."""
        # Find block and offset
        block_idx_in_seq = token_idx // ___  # Q5: Divide by what?
        offset = token_idx % ___  # Q6: Modulo what?

        # Get physical block
        physical_block = self.block_tables[seq_id][___]  # Q7: Index?

        # Write - k/v shape expected: [num_heads, head_dim]
        self.kv_data[physical_block, ___, :, offset, :] = k  # Q8: 0 or 1 for K?
        self.kv_data[physical_block, ___, :, offset, :] = v  # Q9: 0 or 1 for V?

    def read_blocks(self, seq_id: str, start_token: int, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a contiguous range of tokens (start_token inclusive, num_tokens long)."""
        # This is simplified; real implementation would gather efficiently.
        # We'll return concatenated K and V tensors of shape [num_heads, num_tokens, head_dim].
        blocks = self.block_tables[seq_id]
        token_pos = start_token
        tokens_read = 0
        k_parts, v_parts = [], []
        while tokens_read < num_tokens:
            block_idx_in_seq = token_pos // self.block_size
            offset = token_pos % self.block_size
            physical = blocks[block_idx_in_seq]
            # How many tokens we can read from this block
            remaining_in_block = self.block_size - offset
            take = min(remaining_in_block, num_tokens - tokens_read)
            # Slice: [num_heads, take, head_dim]
            k_part = self.kv_data[physical, 0, :, offset:offset+take, :]
            v_part = self.kv_data[physical, 1, :, offset:offset+take, :]
            k_parts.append(k_part)
            v_parts.append(v_part)
            tokens_read += take
            token_pos += take
        k_all = torch.cat(k_parts, dim=1)
        v_all = torch.cat(v_parts, dim=1)
        return k_all, v_all

    def free_sequence(self, seq_id: str):
        """Return all blocks of a sequence to the free pool."""
        if seq_id not in self.block_tables:
            return
        for block in self.block_tables[seq_id]:
            self.free_blocks.append(___)  # Q10: What to append?
        del self.block_tables[___]  # Q11: Delete what?
```

**Hints:**
- Q1: 5D tensor [total_blocks, 2 (K/V), num_heads, block_size, head_dim]
- Q2-3: Ceiling division by block_size to find blocks needed
- Q4: Store block list in table
- Q5-6: Convert logical token index to block index and offset
- Q7: Use block_idx_in_seq to index into the sequence's block table
- Q8-9: K is index 0, V is index 1 in dimension 1
- Q10-11: Append block ID and delete sequence entry

<details>
<summary>Click to see solution</summary>

```python
import torch
from collections import deque
from math import ceil
from typing import List, Tuple, Dict

class PagedKVCache:
    def __init__(self, total_blocks: int, block_size: int,
                 num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Physical storage: [total_blocks, 2, num_heads, block_size, head_dim]
        # 2 stands for key (0) and value (1)
        self.kv_data = torch.zeros(
            total_blocks, 2, num_heads, block_size, head_dim,
            dtype=dtype
        )

        # Free block management
        self.free_blocks = deque(range(total_blocks))

        # Block tables: sequence_id -> list of physical block indices
        self.block_tables: Dict[str, List[int]] = {}

    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for num_tokens, return list of physical block indices."""
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        blocks_needed = ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError(f"Not enough free blocks: need {blocks_needed}, have {len(self.free_blocks)}")
        allocated = [self.free_blocks.popleft() for _ in range(blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated

    def write_token(self, seq_id: str, token_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write K and V for the token at logical position token_idx."""
        # Find block and offset
        block_idx_in_seq = token_idx // self.block_size
        offset = token_idx % self.block_size

        # Get physical block
        physical_block = self.block_tables[seq_id][block_idx_in_seq]

        # Write
        # k/v shape expected: [num_heads, head_dim]
        self.kv_data[physical_block, 0, :, offset, :] = k
        self.kv_data[physical_block, 1, :, offset, :] = v

    def read_blocks(self, seq_id: str, start_token: int, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a contiguous range of tokens (start_token inclusive, num_tokens long)."""
        # This is simplified; real implementation would gather efficiently.
        # We'll return concatenated K and V tensors of shape [num_heads, num_tokens, head_dim].
        blocks = self.block_tables[seq_id]
        token_pos = start_token
        tokens_read = 0
        k_parts, v_parts = [], []
        while tokens_read < num_tokens:
            block_idx_in_seq = token_pos // self.block_size
            offset = token_pos % self.block_size
            physical = blocks[block_idx_in_seq]
            # How many tokens we can read from this block
            remaining_in_block = self.block_size - offset
            take = min(remaining_in_block, num_tokens - tokens_read)
            # Slice: [num_heads, take, head_dim]
            k_part = self.kv_data[physical, 0, :, offset:offset+take, :]
            v_part = self.kv_data[physical, 1, :, offset:offset+take, :]
            k_parts.append(k_part)
            v_parts.append(v_part)
            tokens_read += take
            token_pos += take
        k_all = torch.cat(k_parts, dim=1)
        v_all = torch.cat(v_parts, dim=1)
        return k_all, v_all

    def free_sequence(self, seq_id: str):
        """Return all blocks of a sequence to the free pool."""
        if seq_id not in self.block_tables:
            return
        for block in self.block_tables[seq_id]:
            self.free_blocks.append(block)
        del self.block_tables[seq_id]
```

</details>

Now let's test with a simple scenario.

```python
# test_paged.py
from cache_implementations.paged import PagedKVCache
import torch

cache = PagedKVCache(total_blocks=100, block_size=16, num_heads=32, head_dim=128)

# Allocate for sequence A (needs 30 tokens -> 2 blocks)
blocks_A = cache.allocate("A", 30)
print(f"Seq A blocks: {blocks_A}")

# Write token 0, then token 25
k0 = torch.randn(32,128)
v0 = torch.randn(32,128)
cache.write_token("A", 0, k0, v0)

k25 = torch.randn(32,128)
v25 = torch.randn(32,128)
cache.write_token("A", 25, k25, v25)

# Read tokens 20 to 29 (10 tokens)
k_read, v_read = cache.read_blocks("A", 20, 10)
print(f"Read K shape: {k_read.shape}")  # expected [32, 10, 128]
```

**Predict before running:**

**Question 1:** How many blocks will be allocated for 30 tokens with block_size=16?

**Question 2:** Token 25 will be stored in which block (0 or 1) and at which offset within that block?

**Question 3:** When reading tokens 20-29, which physical blocks will be accessed?

**Question 4:** What will be the shape of `k_read`?

<details>
<summary>Click to verify predictions</summary>

**Answer 1:** ceil(30 / 16) = 2 blocks  
**Answer 2:** Block index: 25 // 16 = 1, Offset: 25 % 16 = 9  
**Answer 3:** Tokens 20-29 span from block 1 offset 4 to block 1 offset 13 (all in block 1)  
**Answer 4:** Shape will be [32, 10, 128] (num_heads, num_tokens, head_dim)

Actual output:
```
Seq A blocks: [0, 1]
Read K shape: torch.Size([32, 10, 128])
```

Token 20 is in block index 1 (since block_size=16, tokens 0-15: block0, 16-31: block1).  
Token 25 is also in block1. So `read_blocks` will read from block1 only, offset 4 to 13 (10 tokens).

</details>

### 4.4 Understanding the Code

The diagram below shows the architecture of PagedAttention.

```mermaid
graph TB
    subgraph LOGICAL [" LOGICAL ADDRESS SPACE - SEQUENCE PERSPECTIVE "]
        VA1["<b>Seq A</b><br/>Virtual Blk 0<br/>Pos 0-15"]
        VA2["<b>Seq A</b><br/>Virtual Blk 1<br/>Pos 16-31"]
        VA3["<b>Seq A</b><br/>Virtual Blk 2<br/>Pos 32-47"]
        
        VB1["<b>Seq B</b><br/>Virtual Blk 0<br/>Pos 0-15"]
        VB2["<b>Seq B</b><br/>Virtual Blk 1<br/>Pos 16-31"]
    end
    
    subgraph MAPPING [" BLOCK TABLE - TRANSLATION LAYER "]
        PT1["<b>Seq A Table</b><br/>→ [42, 17, 85]"]
        PT2["<b>Seq B Table</b><br/>→ [23, 64]"]
    end
    
    subgraph PHYSICAL [" PHYSICAL VRAM - GPU MEMORY "]
        P42["<b>Block 42</b><br/>Seq A Data"]
        P17["<b>Block 17</b><br/>Seq A Data"]
        P85["<b>Block 85</b><br/>Seq A Data"]
        P23["<b>Block 23</b><br/>Seq B Data"]
        P64["<b>Block 64</b><br/>Seq B Data"]
        P87["<b>Block 87</b><br/>FREE"]
        P88["<b>Block 88</b><br/>FREE"]
    end
    
    VA1 --> PT1
    VA2 --> PT1
    VA3 --> PT1
    VB1 --> PT2
    VB2 --> PT2
    
    PT1 --> P42
    PT1 --> P17
    PT1 --> P85
    PT2 --> P23
    PT2 --> P64
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 4.1: PagedAttention architecture with logical blocks mapped to physical blocks via a block table. Adapted from the vLLM paper (Kwon et al., 2023).*

The next diagram details the memory access process.

```mermaid
graph TD
    subgraph REQUEST [" TOKEN ARRIVAL & INDEXING "]
        T1["Token Arrives<br/>Position: 42<br/>Sequence ID: chat_123"]
    end
    
    subgraph TRANSLATION [" ADDRESS TRANSLATION "]
        BT["Step 1: Block Table Lookup<br/>block_tables['chat_123']<br/>→ [15, 87, 42, 91]"]
        
        Calc["Step 2: Calculate Address<br/>block_idx = 42 ÷ 16 = 2<br/>offset = 42 % 16 = 10"]
        
        Map["Step 3: Map to Physical<br/>physical_block = table[2]<br/>→ Block 42"]
    end
    
    subgraph MEMORY [" MEMORY ACCESS "]
        PM["Step 4: Physical Memory<br/>Address: Block 42, Slot 10"]
        
        Read["Step 5: Read/Write K/V<br/>Store token data at (42, 10)"]
    end
    
    subgraph KERNEL [" GPU KERNEL EXECUTION "]
        Kernel["Step 6: PagedAttention Kernel<br/>Inputs: Q, block_tables, seq_lens<br/>Process 100s tokens in parallel<br/>using pre-compiled SlotMapping"]
    end
    
    BENEFITS["KEY ADVANTAGES<br/>Zero fragmentation<br/>O(1) allocation<br/>Non-contiguous storage"]
    
    T1 --> BT --> Calc --> Map --> PM --> Read --> Kernel --> BENEFITS
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 4.2: Memory access flow using block tables. Adapted from concepts in the vLLM paper.*

### 4.5 Experiment: Fragmentation‑Free Allocation

Simulate many sequences with varying lengths, free some, and check if you ever fail to allocate a new sequence due to fragmentation.

```python
# experiment_fragmentation.py
import random
from cache_implementations.paged import PagedKVCache

cache = PagedKVCache(total_blocks=200, block_size=16, num_heads=32, head_dim=128)
seqs = []

# Create 15 sequences
for i in range(15):
    seq_id = f"seq{i}"
    length = random.randint(50, 300)
    cache.allocate(seq_id, length)
    seqs.append((seq_id, length))

# Free 7 random sequences
for _ in range(7):
    idx = random.randint(0, len(seqs)-1)
    seq_id, _ = seqs.pop(idx)
    cache.free_sequence(seq_id)

# Try to allocate a new sequence needing 100 blocks (1600 tokens)
try:
    cache.allocate("new_big", 1600)
    print("Allocation succeeded even after fragmentation!")
except MemoryError:
    print("Allocation failed (should not happen if enough free blocks)")
```

Run this multiple times. Do you ever see a failure when the total free blocks are sufficient? Why not?

<details>
<summary>Click to discuss</summary>

The allocation only checks that the number of free blocks is at least the required number. Because blocks are independent, there is no need for contiguity. Thus, **external fragmentation is eliminated**. This is the key advantage of PagedAttention.

</details>

### 4.6 Checkpoint

- [ ] You can explain how a block table decouples logical and physical addresses.
- [ ] You have implemented a basic PagedKVCache that reads/writes tokens by logical index.
- [ ] You have verified that allocation never fails due to fragmentation as long as total free blocks suffice.

---

## Chapter 5: Slot Mapping – Optimising GPU Access

### Opening Context

The block table indirection is great for memory management, but it adds overhead if the GPU kernel has to perform a table lookup for every token. In practice, we pre‑compute a **slot mapping** that tells the kernel exactly where each token’s K and V reside in physical memory. This flattens the per‑token access into a simple indexed load, enabling massive parallelism.

### 5.1 What You Will Build

You will extend the `PagedKVCache` with a method to build a slot mapping for a batch of sequences. You will then simulate how this mapping is used in a fused attention kernel.

### 5.2 Think First: Kernel Launches

When the GPU processes a batch, it launches a grid of threads. Each thread might handle one token. If each thread had to compute `block_idx = token_idx // block_size`, look up the block table in global memory, then compute the physical address, the latency would be high. How can we pre‑compute these addresses so that each thread just loads from a known offset?

<details>
<summary>Click to review</summary>

We can build a tensor of the same length as the total number of tokens in the batch, where each element is the **physical slot index** (a flat address). This tensor is transferred to the GPU once per batch. Then each thread can directly index into the KV cache using that slot. This technique is called **slot mapping** or **flattened indexing**.

</details>

### 5.3 Implementation: Slot Mapping Builder

Add the following method to `PagedKVCache`. Complete the implementation:

```python
def build_flat_slot_mapping(self, sequences: List[Tuple[str, int]]) -> torch.Tensor:
    """Build a flat slot mapping for K (same for V) that directly indexes into a flattened cache.
    Returns tensor of shape [total_tokens] with indices.
    """
    slot_mapping = []
    for seq_id, ctx_len in sequences:
        blocks = self.block_tables[___]  # Q1: Look up which table?
        for pos in range(___):  # Q2: Iterate over how many positions?
            block_idx = pos // ___  # Q3: Calculate block index
            offset = pos % ___  # Q4: Calculate offset
            physical_block = blocks[___]  # Q5: Get physical block
            slot = ___ * self.block_size + ___  # Q6: Calculate flat slot index
            slot_mapping.append(___)
    return torch.tensor(slot_mapping, dtype=torch.long)
```

**Hints:**
- Q1: Use seq_id to look up the block table
- Q2: Iterate for ctx_len positions
- Q3-4: Same logic as write_token
- Q5: Index into blocks list
- Q6: Flatten 2D (block, offset) to 1D index

<details>
<summary>Click to see solution</summary>

```python
def build_flat_slot_mapping(self, sequences: List[Tuple[str, int]]) -> torch.Tensor:
    """Build a flat slot mapping for K (same for V) that directly indexes into a flattened cache.
    Returns tensor of shape [total_tokens] with indices.
    """
    slot_mapping = []
    for seq_id, ctx_len in sequences:
        blocks = self.block_tables[seq_id]
        for pos in range(ctx_len):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            physical_block = blocks[block_idx]
            slot = physical_block * self.block_size + offset
            slot_mapping.append(slot)
    return torch.tensor(slot_mapping, dtype=torch.long)
```

</details>

Now we can simulate how a kernel would use this mapping.

```python
# simulate_kernel.py
# Assume we have a flattened K cache (just for simulation)
total_blocks = 100
block_size = 16
num_heads = 32
head_dim = 128
k_flat = torch.randn(total_blocks * block_size, num_heads * head_dim)  # flattened

# Build slot mapping for a batch
cache = PagedKVCache(total_blocks, block_size, num_heads, head_dim)
cache.allocate("A", 30)
# ... (populate with some writes)
slot_map = cache.build_flat_slot_mapping([("A", 30)])

# Simulate kernel gathering all K values for the batch
gathered_k = k_flat[slot_map]  # shape [30, num_heads*head_dim]
```

**Question:** What does `gathered_k` represent? How would you reshape it to `[num_heads, 30, head_dim]`?

<details>
<summary>Click to review</summary>

`gathered_k` is a tensor where each row is the flattened K vector for one token. To get the original shape, you can do `gathered_k.view(30, num_heads, head_dim).permute(1,0,2)`.

</details>

### 5.4 Understanding the Code

The diagram below illustrates how slot mapping is constructed from a batch of sequences.

```mermaid
graph TD
    subgraph BATCH [" BATCH FLATTENING "]
        S1["<b>Sequence A</b><br/>5 tokens<br/>Blocks: 42, 15"]
        S2["<b>Sequence B</b><br/>10 tokens<br/>Blocks: 23, 64, 87"]
        S3["<b>Sequence C</b><br/>3 tokens<br/>Blocks: 91"]
    end
    
    subgraph FLAT [" FLATTEN TO TOKEN STREAM "]
        TP["<b>Token Array</b><br/>A0 A1 A2 A3 A4 | B0 B1 ... B9 | C0 C1 C2<br/><b>18 tokens total</b>"]
    end
    
    subgraph CALC [" SLOT CALCULATION "]
        EX["<b>Example: Seq B, Token 25</b><br/>block_idx = 25 ÷ 16 = 1<br/>offset = 25 % 16 = 9<br/>physical_block = blocks[1] = 64<br/>slot = 64 × 32 + 9 × 2 = 2050"]
    end
    
    subgraph MAPPING [" SLOT MAPPING CONSTRUCTION "]
        SM["Slot Mapping Tensor<br/>K_slots: [1344, 1346, 1348, ...]<br/>V_slots: [1345, 1347, 1349, ...]<br/>One tensor for all tokens"]
    end
    
    EXECUTION["GPU KERNEL BENEFITS<br/>Parallel random access<br/>No per-sequence logic<br/>Coalesced memory reads"]
    
    S1 --> TP
    S2 --> TP
    S3 --> TP
    TP --> EX
    EX --> SM
    SM --> EXECUTION
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 5.1: Slot mapping construction from a batch of sequences. Adapted from concepts in the vLLM paper.*

The next diagram shows how prefix sharing can be implemented with block tables and reference counting.

```mermaid
flowchart TD
    subgraph LOGICAL [" LOGICAL VIEW " ]
        LA["Seq A<br/>Prefix 0-7<br/>Tail 8-15"]
        LB["Seq B<br/>Prefix 0-7<br/>Tail 8-20"]
    end

    subgraph TABLES [" BLOCK TABLES " ]
        TA["Seq A: [P0, P1, A2]"]
        TB["Seq B: [P0, P1, B2, B3]"]
    end

    subgraph PHYSICAL [" PHYSICAL BLOCKS " ]
        P0["Block P0<br/>Prefix<br/>ref=2"]
        P1["Block P1<br/>Prefix<br/>ref=2"]
        A2["Block A2<br/>Tail A<br/>ref=1"]
        B2["Block B2<br/>Tail B<br/>ref=1"]
        B3["Block B3<br/>Tail B<br/>ref=1"]
    end

    subgraph SAFETY [" SAFETY & COW " ]
        G1["Shared blocks marked RO"]
        G2["On write: copy-on-write<br/>new block + ref--"]
        G3["Scheduler: co-batch same prefix"]
    end

    LA --> TA
    LB --> TB
    TA --> P0
    TA --> P1
    TA --> A2
    TB --> P0
    TB --> P1
    TB --> B2
    TB --> B3
    P0 --> G1
    P1 --> G1
    A2 --> G2
    B2 --> G2
    B3 --> G2
    G1 --> G3

    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 5.2: Cross-sequence prefix sharing with reference counting and copy-on-write. Adapted from concepts in the vLLM paper.*

### 5.5 Experiment: Compare Lookup Overhead

Simulate a batch of 1000 tokens. Measure the time to gather K vectors using:
1. The block table indirection (loop over tokens, compute block/offset, read from 5D tensor).
2. The slot mapping (build the map once, then use advanced indexing).

You can use `timeit` to compare. Which is faster?

<details>
<summary>Click to see discussion</summary>

Advanced indexing with a pre‑built mapping is typically much faster because it can be implemented as a single contiguous memory copy (scatter/gather) that is heavily optimised in PyTorch and CUDA. The block‑table method would require many small tensor slices and loops, which are inefficient on GPU.

</details>

### 5.6 Checkpoint

- [ ] You can explain why slot mapping is essential for high‑performance GPU kernels.
- [ ] You have built a slot mapping from a batch of sequences.
- [ ] You understand how the kernel uses the mapping to gather K/V values.

---

## Chapter 6: Advanced Optimizations

### Opening Context

With the basic PagedAttention in place, we can add two advanced features that further improve memory efficiency and speed: block‑sparse attention and prefix caching. These are used in production systems to handle long contexts and shared prompts.

### 6.1 What You Will Build

You will extend your `PagedKVCache` to support:
- **Block‑sparse attention:** Load only a sliding window of recent blocks.
- **Prefix caching:** Share identical prompt prefixes across sequences.

### 6.2 Think First: Locality in Attention

In practice, a token’s attention often focuses on nearby tokens (locality of reference). Do you really need to attend to the very first token when generating token 5000? Research shows that a sliding window works nearly as well as full attention for many tasks. How could you exploit this to reduce memory bandwidth?

<details>
<summary>Click to review</summary>

If you only need the last `W` blocks of context for each token, you can avoid loading older blocks from HBM, reducing memory traffic. This is called **sparse attention** or **sliding window attention**. Implemented at the block level, it becomes block‑sparse attention.

</details>

### 6.3 Implementation: Block‑Sparse Attention

Add a method to read only a window of blocks:

```python
def read_sparse(self, seq_id: str, token_idx: int, window_blocks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read only the blocks within a window ending at the block containing token_idx."""
    blocks = self.block_tables[seq_id]
    current_block_idx = token_idx // self.block_size
    start_block_idx = max(0, current_block_idx - window_blocks + 1)
    # Collect the physical blocks in the window
    k_parts, v_parts = [], []
    for i in range(start_block_idx, current_block_idx + 1):
        physical = blocks[i]
        # For simplicity, read the whole block (but could read only needed offsets)
        k_parts.append(self.kv_data[physical, 0])  # [num_heads, block_size, head_dim]
        v_parts.append(self.kv_data[physical, 1])
    k_window = torch.cat(k_parts, dim=1)
    v_window = torch.cat(v_parts, dim=1)
    return k_window, v_window
```

This is a simplified version; a real implementation would handle partial blocks at the edges.

Now simulate the reduction in memory traffic:

```python
# simulate_sparse.py
cache = PagedKVCache(total_blocks=100, block_size=16, num_heads=32, head_dim=128)
# ... populate sequence with many blocks
token_pos = 500  # token 500 is in block 31 (since 16*31=496)
k_full, _ = cache.read_blocks("A", 0, 500)  # read all
k_sparse, _ = cache.read_sparse("A", 500, window_blocks=4)  # read last 4 blocks

print(f"Full K shape: {k_full.shape}")    # [32, 500, 128]
print(f"Sparse K shape: {k_sparse.shape}")# [32, 64, 128] if 4*16=64
```

**Question:** What is the memory traffic reduction ratio? Does it depend on the total sequence length?

<details>
<summary>Click to answer</summary>

If window size is fixed (e.g., 4 blocks = 64 tokens), the traffic for each decode step is constant, while full attention grows linearly with sequence length. For long sequences, the reduction approaches `window_size / total_len` → 0, meaning you save almost all memory bandwidth.

</details>

### 6.4 Prefix Caching

Many requests share a common prefix, such as a system prompt. If you can detect identical prefixes, you can reuse the same physical blocks across multiple sequences.

Extend your cache with a prefix cache dictionary that maps a hash of the prefix to a list of block IDs. Use reference counting to free blocks only when no sequence uses them.

```python
class PrefixCachingKVCache(PagedKVCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_cache = {}  # hash -> list of block IDs
        self.block_refcount = {} # block_id -> int

    def get_or_create_prefix(self, prefix_hash: str, num_tokens: int) -> List[int]:
        if prefix_hash in self.prefix_cache:
            blocks = self.prefix_cache[prefix_hash]
            for b in blocks:
                self.block_refcount[b] = self.block_refcount.get(b, 0) + 1
            return blocks
        else:
            blocks = self.allocate(prefix_hash, num_tokens)  # temporary seq_id
            self.prefix_cache[prefix_hash] = blocks
            for b in blocks:
                self.block_refcount[b] = 1
            return blocks

    def allocate_with_prefix(self, seq_id: str, prefix_hash: str, total_tokens: int) -> List[int]:
        # Assume prefix_len is known from hash or passed separately
        # For simplicity, we store prefix_len in prefix_cache
        # This method would first get the prefix blocks, then allocate remaining.
        pass
```

**Important:** When freeing a sequence that shares prefix blocks, you must decrement the refcounts and only free blocks when refcount reaches zero.

### 6.5 Understanding the Code

The diagram below contrasts full attention with block‑sparse attention.

```mermaid
graph TD
    subgraph FULL [" FULL ATTENTION - EVERY TOKEN ALL BLOCKS "]
        FA1["Block 0<br/>All 8 blocks"]
        FA2["Block 1<br/>All 8 blocks"]
        FA3["Block 2<br/>All 8 blocks"]
        FA4["..."]
        FA5["Block 7<br/>All 8 blocks"]
    end
    
    subgraph SPARSE [" SPARSE ATTENTION - WINDOW=3 LOCAL "]
        SA1["Block 0<br/>Block 0"]
        SA2["Block 1<br/>Blocks 0-1"]
        SA3["Block 2<br/>Blocks 0-2"]
        SA4["Block 3<br/>Blocks 1-3"]
        SA5["Block 4<br/>Blocks 2-4"]
        SA6["Block 5<br/>Blocks 3-5"]
        SA7["Block 6<br/>Blocks 4-6"]
        SA8["Block 7<br/>Blocks 5-7"]
    end
    
    subgraph METRICS [" PERFORMANCE IMPACT "]
        PERF["For 8 Blocks<br/><br/>Full Attention: 64 block-pairs<br/>Sparse (W=3): 24 block-pairs<br/><br/>62.5% Memory Traffic Reduction<br/>~2x Decode Speedup"]
    end
    
    FA1 --> FA2 --> FA3 --> FA4 --> FA5
    SA1 --> SA2 --> SA3 --> SA4 --> SA5 --> SA6 --> SA7 --> SA8
    FA5 --> PERF
    SA8 --> PERF
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 6.1: Full attention vs block‑sparse attention with window size 3. Adapted from concepts in the vLLM paper.*

The next diagram illustrates prefix sharing in action.

```mermaid
graph TD
    subgraph SYSTEM [" SYSTEM PROMPT - SHARED PREFIX "]
        SP["Prefix<br/>You are a helpful assistant...<br/>Hash: abc123<br/>Length: 12 tokens"]
    end
    
    subgraph USERS [" CONCURRENT REQUESTS "]
        UR1["User 1<br/>Explain quantum physics<br/>Total: 15 tokens"]
        UR2["User 2<br/>Write a poem about AI<br/>Total: 17 tokens"]
        UR3["User 3<br/>Debug this Python code<br/>Total: 18 tokens"]
    end
    
    subgraph TABLES [" BLOCK TABLES "]
        BT1["User 1 Table<br/>42, 87, 15, 23"]
        BT2["User 2 Table<br/>42, 87, 15, 64, 91"]
        BT3["User 3 Table<br/>42, 87, 15, 33, 78"]
    end
    
    subgraph BLOCKS [" PHYSICAL BLOCKS "]
        P1["SHARED PREFIX<br/>B42, B87, B15<br/>ref_count=3"]
        P2["User 1 unique<br/>B23"]
        P3["User 2 unique<br/>B64, B91"]
        P4["User 3 unique<br/>B33, B78"]
    end
    
    subgraph SAVINGS [" MEMORY SAVINGS "]
        MEM["Without Sharing: 9 blocks x 3 = 27 allocations<br/>With Sharing: 3 shared + 6 unique = 9 blocks<br/><br/>66% Memory Reduction"]
    end
    
    SP --> UR1
    SP --> UR2
    SP --> UR3
    UR1 --> BT1
    UR2 --> BT2
    UR3 --> BT3
    BT1 --> P1
    BT2 --> P1
    BT3 --> P1
    BT1 --> P2
    BT2 --> P3
    BT3 --> P4
    P1 --> MEM
    
    linkStyle default stroke:#333,stroke-width:3px
```
*Diagram 6.2: Prefix sharing reduces memory allocation by reusing common prefix blocks. Adapted from concepts in the vLLM paper.*

### 6.6 Checkpoint

- [ ] You have implemented a block‑sparse read method.
- [ ] You understand how prefix caching can save memory for common prompts.
- [ ] You can explain the role of reference counting in safely sharing blocks.

---

## Final Conceptual Assessment

Before completing the lab, verify your understanding with these comprehensive questions.

### Question 1: Architecture Comparison

Match each architecture to its primary limitation:

| Architecture | Primary Limitation (A-D) |
|--------------|---------------------------|
| Naive contiguous cache | ___ |
| Dynamic block allocation (contiguous required) | ___ |
| PagedAttention (basic) | ___ |
| PagedAttention + prefix caching | ___ |

**Options:**
- A: Complex reference counting and potential memory leaks
- B: Internal fragmentation (80-90% waste)
- C: External fragmentation limits large allocations
- D: Minimal limitations, suitable for production

<details>
<summary>Click to verify</summary>

| Architecture | Limitation | Explanation |
|--------------|------------|-------------|
| Naive contiguous cache | **B** | Pre-allocates max_len, wastes 80-90% when actual usage is lower |
| Dynamic block allocation | **C** | Blocks scattered, can't satisfy large contiguous requests |
| PagedAttention (basic) | **D** | Minimal limitations, near-zero fragmentation |
| PagedAttention + prefix caching | **A** | Reference counting adds complexity, bugs can leak memory |

</details>

### Question 2: Production Scenario

**Scenario:** You are deploying an LLM inference service with these requirements:
- 40 GB GPU memory available
- Model: 32 attention heads, head_dim=128, bfloat16
- Average request: 300 tokens (prompt + generation)
- Peak request: 2000 tokens
- Target: 128 concurrent users

**Calculate:**

**Part A:** With naive contiguous cache (max_len=2048), how much memory per sequence?

**Part B:** How many sequences can fit in 40 GB with naive allocation?

**Part C:** With PagedAttention (block_size=16), how much memory needed for 128 sequences averaging 300 tokens each?

**Part D:** What is the memory utilization improvement from naive to PagedAttention?

<details>
<summary>Click to see analysis</summary>

**Part A:**  
Each token needs: 2 (K+V) × 32 heads × 128 dim × 2 bytes = 16 KB  
Per sequence: 2048 tokens × 16 KB = **32 MB**

**Part B:**  
40 GB / 32 MB = **1,250 sequences** (theoretical maximum, ignoring OS and model weights)

**Part C:**  
128 sequences × 300 tokens × 16 KB = **614 MB**  
(Plus small overhead for block tables)

**Part D:**  
Naive (128 sequences): 128 × 32 MB = 4.1 GB allocated, ~500 MB used = **88% waste**  
PagedAttention: 614 MB allocated, 614 MB used = **~0% waste**

Memory savings: (4.1 GB - 0.614 GB) / 4.1 GB = **85% reduction** in memory footprint.

This allows serving **6-7× more concurrent users** with the same hardware.

</details>

### Question 3: Debugging Challenge

A developer implements PagedAttention but sees this error:

```
RuntimeError: index 5 is out of bounds for dimension 0 with size 3
```

The error occurs in `write_token` at this line:
```python
physical_block = self.block_tables[seq_id][block_idx_in_seq]
```

Context:
- Sequence allocated with `allocate("test", 40)`
- block_size = 16
- Trying to write token at position 85

**Question:** What is the root cause and how should it be fixed?

<details>
<summary>Click to see answer</summary>

**Root Cause:**  
Sequence was allocated for 40 tokens, which requires ceil(40/16) = **3 blocks** (indices 0, 1, 2).  
Trying to write token 85 requires block_idx_in_seq = 85 // 16 = **5**, which doesn't exist.

**Fix Options:**

1. **Pre-allocate correctly:** Allocate for maximum expected length:
   ```python
   cache.allocate("test", 100)  # Allocate for at least 100 tokens
   ```

2. **Dynamic growth:** Add a method to allocate additional blocks on demand:
   ```python
   def grow_sequence(self, seq_id: str, new_tokens: int):
       current_blocks = len(self.block_tables[seq_id])
       current_capacity = current_blocks * self.block_size
       if new_tokens <= current_capacity:
           return
       additional_blocks = ceil((new_tokens - current_capacity) / self.block_size)
       new_blocks = self.alloc_blocks(additional_blocks)
       self.block_tables[seq_id].extend(new_blocks)
   ```

3. **Validation:** Add bounds checking in `write_token`:
   ```python
   required_blocks = ceil((token_idx + 1) / self.block_size)
   if required_blocks > len(self.block_tables[seq_id]):
       raise ValueError(f"Token {token_idx} requires {required_blocks} blocks, "
                       f"but only {len(self.block_tables[seq_id])} allocated")
   ```

</details>

### Question 4: Optimization Strategy

You profile your PagedAttention implementation and find:
- Memory utilization: 95% (excellent)
- Attention kernel: 60% of total latency
- Block table lookups: 15% of total latency
- Memory bandwidth: 85% saturated

**Rank these optimizations by potential impact (1=highest, 4=lowest):**

- [ ] Implement block-sparse attention with window_size=512
- [ ] Use FlashAttention kernel
- [ ] Add prefix caching for common system prompts
- [ ] Optimize block table data structure

<details>
<summary>Click to see ranking</summary>

**Ranking:**

1. **Use FlashAttention kernel** (Rank 1)  
   Addresses the largest bottleneck (60% of latency). Can reduce attention compute by 2-4×.

2. **Implement block-sparse attention** (Rank 2)  
   Reduces memory bandwidth (currently 85% saturated) and attention compute. Effective for long contexts.

3. **Add prefix caching** (Rank 3)  
   Saves memory and compute for redundant prefix processing, but only helps when prefixes are shared. Impact depends on workload.

4. **Optimize block table data structure** (Rank 4)  
   Only 15% of latency. Premature optimization given other bottlenecks.

**General Principle:** Optimize the largest bottleneck first (Amdahl's Law). Attention kernel optimization gives maximum return.

</details>

### Question 5: System Design

**Scenario:** Design a scheduler for a PagedAttention-based inference server.

Given:
- 1000 total blocks, block_size=16
- Requests arrive with varying prompt lengths (50-1000 tokens)
- Generation lengths unknown (could be 10-2000 tokens)

**Design Questions:**

**Part A:** When should you reject a new request?

**Part B:** If you must evict a running sequence due to memory pressure, which should you evict?

**Part C:** How would you handle a sequence that grows beyond its initial allocation?

<details>
<summary>Click to see design considerations</summary>

**Part A: Admission Control**

Reject when:
```python
free_blocks < required_blocks + safety_margin
```

Where:
- `required_blocks = ceil(prompt_len / block_size)`
- `safety_margin` accounts for unknown generation length (e.g., 10-20% of total blocks)

Alternatively, use **speculative allocation**: allocate only for the prompt, reserve some blocks for generation, evict if generation exceeds estimate.

**Part B: Eviction Policy**

Options ranked by fairness/efficiency:

1. **Longest sequence first (LRU-variant)**: Evict sequence with most tokens generated  
   *Rationale:* It has consumed most resources, others may finish soon

2. **Lowest priority**: If sequences have priority levels (e.g., premium users)  
   *Rationale:* Business logic

3. **Random**: Simple, but unfair  
   *Rationale:* Easy to implement, no starvation

4. **Shortest sequence first**: Keep long sequences  
   *Rationale:* Avoid wasting progress on long sequences

Production systems like vLLM use **preemptive priority scheduling** with priority queues.

**Part C: Dynamic Growth**

Implement "grow on demand":
```python
def append_token(self, seq_id, k, v):
    # Check if we need more blocks
    required_blocks = ceil((self.current_len[seq_id] + 1) / self.block_size)
    if required_blocks > len(self.block_tables[seq_id]):
        # Try to allocate one more block
        if self.free_blocks:
            new_block = self.free_blocks.popleft()
            self.block_tables[seq_id].append(new_block)
        else:
            # Out of memory, trigger eviction or reject
            raise MemoryError("Cannot grow sequence")
    # Proceed with write
    self.write_token(seq_id, self.current_len[seq_id], k, v)
    self.current_len[seq_id] += 1
```

This allows sequences to grow dynamically without pre-allocating for worst case.

</details>

---

## Epilogue: The Complete System

You have now built a production-ready KV cache manager. Starting from a naive contiguous cache, you progressed through dynamic allocation and finally implemented PagedAttention with block tables, slot mapping, sparse attention, and prefix sharing.

The final `PagedKVCache` class (with all extensions) can support hundreds of concurrent users with near-zero fragmentation and high GPU utilization.

### Implementation Progress Summary

| Component | Status | Memory Efficiency | Complexity |
|-----------|--------|-------------------|------------|
| Naive Contiguous Cache | ✓ Complete | 10-20% | Simple |
| Dynamic Block Allocation | ✓ Complete | 30-50% | Moderate |
| PagedAttention Core | ✓ Complete | 85-95% | Complex |
| Slot Mapping | ✓ Complete | Same | Moderate |
| Block-Sparse Attention | ✓ Complete | 90-98%* | Complex |
| Prefix Caching | ✓ Complete | 95-99%* | Very Complex |

*When applicable to workload

### End-to-End Verification

Run a comprehensive test that simulates a realistic workload. Create `tests/comprehensive_test.py`:

```python
from cache_implementations.paged import PagedKVCache
import random
import torch

def test_comprehensive_workload():
    """Simulate realistic production workload."""
    cache = PagedKVCache(total_blocks=500, block_size=16, num_heads=32, head_dim=128)
    
    print("=" * 60)
    print("COMPREHENSIVE KV CACHE TEST")
    print("=" * 60)
    
    # Phase 1: Initial allocation
    print("\nPhase 1: Allocating 50 sequences...")
    sequences = []
    for i in range(50):
        seq_id = f"seq_{i}"
        length = random.randint(100, 500)
        try:
            blocks = cache.allocate(seq_id, length)
            sequences.append((seq_id, length, blocks))
        except MemoryError:
            print(f"  Failed to allocate sequence {i} (out of memory)")
            break
    
    print(f"  Successfully allocated: {len(sequences)} sequences")
    print(f"  Free blocks remaining: {len(cache.free_blocks)}")
    
    # Phase 2: Write tokens
    print("\nPhase 2: Writing tokens...")
    for seq_id, length, _ in sequences[:10]:  # Write to first 10
        for pos in range(min(length, 50)):  # Write first 50 tokens
            k = torch.randn(32, 128)
            v = torch.randn(32, 128)
            cache.write_token(seq_id, pos, k, v)
    print("  Wrote tokens to 10 sequences")
    
    # Phase 3: Read and verify
    print("\nPhase 3: Reading tokens...")
    seq_id, length, _ = sequences[0]
    k_read, v_read = cache.read_blocks(seq_id, 0, min(length, 50))
    print(f"  Read {k_read.shape[1]} tokens from {seq_id}")
    print(f"  Shape: {k_read.shape}")
    
    # Phase 4: Free half
    print("\nPhase 4: Freeing 25 sequences...")
    for seq_id, _, _ in sequences[25:50]:
        cache.free_sequence(seq_id)
    print(f"  Free blocks after cleanup: {len(cache.free_blocks)}")
    
    # Phase 5: Allocate new (tests defragmentation)
    print("\nPhase 5: Allocating new sequences (no fragmentation)...")
    new_sequences = 0
    for i in range(30):
        try:
            cache.allocate(f"new_{i}", random.randint(100, 300))
            new_sequences += 1
        except MemoryError:
            break
    print(f"  Allocated {new_sequences} new sequences")
    print(f"  Free blocks remaining: {len(cache.free_blocks)}")
    
    # Phase 6: Slot mapping
    print("\nPhase 6: Building slot mapping...")
    batch = [(seq_id, min(length, 50)) for seq_id, length, _ in sequences[:5]]
    slot_map = cache.build_flat_slot_mapping(batch)
    print(f"  Slot mapping shape: {slot_map.shape}")
    print(f"  Total tokens in batch: {slot_map.shape[0]}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    active_sequences = len(cache.block_tables)
    total_blocks_used = sum(len(blocks) for blocks in cache.block_tables.values())
    utilization = (total_blocks_used / 500) * 100
    print(f"  Active sequences: {active_sequences}")
    print(f"  Total blocks used: {total_blocks_used}/500")
    print(f"  Memory utilization: {utilization:.1f}%")
    print(f"  Free blocks: {len(cache.free_blocks)}")
    print(f"  Fragmentation: 0% (PagedAttention eliminates fragmentation)")
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_comprehensive_workload()
```

**Predict:** Before running, estimate:
1. How many sequences will successfully allocate in Phase 1?
2. Will Phase 5 succeed in allocating 30 new sequences after freeing 25?

<details>
<summary>Click to verify predictions</summary>

**Prediction 1:**  
With 500 blocks and sequences needing 100-500 tokens (7-32 blocks each):  
Average: ~15 blocks per sequence  
Expected: 500 / 15 ≈ **33 sequences** (actual will be ~30-40 due to randomness)

**Prediction 2:**  
After freeing 25 sequences (~375 blocks freed), you have plenty of free blocks.  
30 new sequences needing ~150 blocks total: **Yes, will succeed**

PagedAttention makes this deterministic: allocation succeeds if and only if `free_blocks >= required_blocks`.

</details>

### Learning Completion Checklist

Verify you have mastered all learning objectives:

**Foundational Understanding:**
- [ ] You can explain why caching is necessary using complexity analysis (O(n²) vs O(n))
- [ ] You can calculate memory requirements for different cache architectures
- [ ] You understand the difference between internal and external fragmentation

**Implementation Skills:**
- [ ] You implemented a naive contiguous KV cache with proper memory tracking
- [ ] You implemented a dynamic block-based allocator with free list management
- [ ] You implemented PagedAttention with block tables and address translation
- [ ] You implemented slot mapping for efficient GPU kernel access

**Analysis Capabilities:**
- [ ] You can measure memory utilization and identify fragmentation
- [ ] You can predict allocation success/failure based on memory state
- [ ] You can compare different architectures quantitatively
- [ ] You analyzed production scenarios and designed solutions

**Advanced Topics:**
- [ ] You understand how block-sparse attention reduces memory bandwidth
- [ ] You can explain prefix caching with reference counting
- [ ] You completed the final conceptual assessment questions
- [ ] You ran the comprehensive end-to-end test successfully

If you checked all boxes, you are ready for production-level inference optimization work.

### Summary Table of Implementations

| **Aspect** | **Naive Contiguous** | **Dynamic Allocation** | **PagedAttention** | **Advanced Paged** |
|------------|----------------------|------------------------|--------------------|--------------------|
| Memory Fragmentation | Internal (High) | External (High) | None | None |
| Allocation Speed | O(1) | O(n) for search | O(1) | O(1) |
| Max Concurrent Users | Low | Medium | High | Very High |
| Memory Utilization | 10-20% | 30-50% | 85-95% | 90-98% |
| Implementation Complexity | Simple | Moderate | Complex | Very Complex |
| Prefix Sharing | Impossible | Difficult | Possible | Efficient |
| Best For | Prototyping | Low-concurrency | Production | Large-scale |

---

## The Principles

1. **Decouple logical and physical addresses** – By using a block table, you eliminate external fragmentation and allow any free block to serve any request.
2. **Pre‑compute access patterns** – Slot mapping flattens per‑token addressing, enabling GPU kernels to run at peak efficiency.
3. **Exploit locality** – Sparse attention reduces memory bandwidth by loading only relevant blocks; it is especially effective for long sequences.
4. **Share read‑only data** – Prefix caching with reference counting avoids duplicate storage of common prompts, multiplying memory efficiency.
5. **Measure, then optimise** – The evolution from naive to PagedAttention was driven by profiling and understanding where memory and time were wasted.

---

## Troubleshooting

### Error: `IndexError: index out of range` in `write_token`

**Cause:** The `token_idx` exceeds the allocated number of blocks for that sequence.  
**Solution:** Ensure you allocated enough tokens for the sequence, or dynamically allocate new blocks when appending beyond current allocation.

### Error: `MemoryError: Not enough free blocks`

**Cause:** The global free pool does not have enough blocks to satisfy the allocation.  
**Solution:** Increase `total_blocks` in the constructor, or implement eviction (LRU) to free cold blocks. In a real system, you would also have a scheduler that pre‑empts sequences when memory is low.

### Error: CUDA out of memory when using `kv_data` on GPU

**Cause:** `total_blocks * block_size * num_heads * head_dim * 2` is too large for GPU memory.  
**Solution:** Reduce `total_blocks` or `block_size`, or use a smaller model. This lab simulates on CPU, so you shouldn't face this unless you move to GPU.

### Error: Inconsistent reference counts after freeing

**Cause:** Not all sequences that shared a prefix were freed properly, or `free_sequence` did not decrement refcounts correctly.  
**Solution:** Implement a proper reference counting mechanism and ensure `free_sequence` only frees blocks when refcount reaches zero.

---

## Next Steps

Now that you understand the internals of KV cache management, you can:

1. **Integrate** your `PagedKVCache` into a simple inference loop (see Lab 0.4).
2. **Profile** the performance using PyTorch Profiler or NVIDIA Nsight to identify remaining bottlenecks.
3. **Add multi‑GPU support** by sharding the KV cache across devices.
4. **Implement a scheduler** that uses the block table to pre‑empt and resume sequences, enabling advanced batching strategies.
5. **Explore other memory optimisations** such as quantised KV cache or offloading to CPU.

Continue to **Lab 0.4: The Profiling Baseline – Measurement & Optimization**, where you will apply data‑driven techniques to further improve your implementation.

---

## Additional Resources

- [vLLM: PagedAttention Paper (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [PyTorch Documentation on Scatter/Gather](https://pytorch.org/docs/stable/generated/torch.gather.html)

---

*This lab was developed following the Poridhi Labs Development Guide to ensure active learning and professional standards.*