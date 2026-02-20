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

```bash
# Create project folder
mkdir ~/kv-cache-lab
cd ~/kv-cache-lab

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch numpy
```

We will simulate GPU memory using CPU tensors, but the code is written to be device‑agnostic. If you have a CUDA‑capable GPU, you can move tensors to `cuda` by changing the device argument.

Create the initial project structure:

```bash
mkdir -p cache_implementations tests
touch cache_implementations/__init__.py
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

Create a file `cache_implementations/attention_sim.py`:

```python
import time

def attention_without_cache(prompt_len, gen_len):
    """Simulate attention without KV cache by counting recomputations."""
    total_kv_computations = 0
    # prompt processing
    total_kv_computations += prompt_len
    for step in range(1, gen_len + 1):
        # at step i, we have prompt_len + i tokens
        total_kv_computations += prompt_len + step
    return total_kv_computations

def attention_with_cache(prompt_len, gen_len):
    """With cache: compute once per token."""
    return prompt_len + gen_len

# Example: prompt=128, generate 100 tokens
p, g = 128, 100
print(f"Without cache: {attention_without_cache(p, g)} operations")
print(f"With cache:    {attention_with_cache(p, g)} operations")
```

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

### 1.5 Checkpoint

- [ ] You can explain why the attention equation without cache leads to quadratic complexity.
- [ ] You have run the simulation and verified the operation count.
- [ ] You understand that caching reduces the computational burden from quadratic to linear for key/value projections.

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

Create `cache_implementations/naive.py`:

```python
import torch
from typing import Tuple

class NaiveKVCache:
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.max_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Allocate maximum possible cache upfront [2, max_len, num_heads, head_dim]
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

### 2.5 Experiment: Vary `max_seq_len`

Run the simulation with `max_len = 512` and `max_len = 4096`. How does fragmentation change? What does this tell you about the relationship between max_len and average usage?

<details>
<summary>Click to see discussion</summary>

- If `max_len` is closer to the actual lengths (e.g., 512 vs 100‑500), fragmentation decreases because the allocated tensor is a better fit.  
- However, in production you cannot know the exact maximum length a conversation will reach. If you set it too low, long conversations will crash. This forces you to pick a large upper bound, causing high fragmentation.

</details>

### 2.6 Checkpoint

- [ ] You can implement a contiguous KV cache.
- [ ] You can measure allocated vs used memory.
- [ ] You have observed that internal fragmentation can exceed 80% with realistic length distributions.

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

Create `cache_implementations/dynamic.py`:

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

### 3.5 Checkpoint

- [ ] You can implement a block‑based dynamic allocator.
- [ ] You understand that dynamic allocation removes internal fragmentation.
- [ ] You can identify that external fragmentation can still occur if physical contiguity is required.

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

Create `cache_implementations/paged.py`:

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

Now let's test with a simple scenario.

```python
# test_paged.py
from cache_implementations.paged import PagedKVCache

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

**Predict:** What will be the shape of `k_read`? Which physical blocks are involved?

<details>
<summary>Click to verify</summary>

Token 20 is in block index 1 (since block_size=16, tokens 0‑15: block0, 16‑31: block1).  
Token 25 is also in block1. So `read_blocks` will read from block1 only, offset 4 to 13 (10 tokens).  
Output shape: `[32, 10, 128]`.

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

Add the following method to `PagedKVCache`:

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

## Epilogue: The Complete System

You have now built a production‑ready KV cache manager. Starting from a naive contiguous cache, you progressed through dynamic allocation and finally implemented PagedAttention with block tables, slot mapping, sparse attention, and prefix sharing.

The final `PagedKVCache` class (with all extensions) can support hundreds of concurrent users with near‑zero fragmentation and high GPU utilisation.

### End‑to‑End Verification

Run a comprehensive test that simulates a realistic workload:

1. Pre‑fill the cache with several long sequences.
2. Free some sequences randomly.
3. Allocate new sequences with shared prefixes.
4. Perform sparse attention reads.
5. Measure total allocated vs used memory.

```python
# final_test.py
from cache_implementations.paged import PrefixCachingKVCache
import random

cache = PrefixCachingKVCache(total_blocks=1000, block_size=16, num_heads=32, head_dim=128)

# Create 50 sequences with a common system prompt of 100 tokens
system_prompt_hash = "sys_v1"
# We need to allocate the prefix first (as a separate "sequence")
prefix_blocks = cache.get_or_create_prefix(system_prompt_hash, 100)

seqs = []
for i in range(50):
    seq_id = f"user{i}"
    # total tokens = 100 (prefix) + random tail (0-200)
    tail_len = random.randint(0,200)
    # For simplicity, we allocate tail separately and combine block lists
    # In a real system, you'd have a method allocate_with_prefix
    # Here we just simulate by creating a new entry with combined blocks
    cache.block_tables[seq_id] = prefix_blocks.copy()
    if tail_len > 0:
        tail_blocks = cache.allocate(seq_id, tail_len)
        cache.block_tables[seq_id].extend(tail_blocks)
    seqs.append(seq_id)

# Now free 20 sequences
for _ in range(20):
    sid = seqs.pop(random.randint(0, len(seqs)-1))
    cache.free_sequence(sid)  # this should decrement refcounts appropriately

# Check that prefix blocks still exist if refcount>0
print(f"Prefix blocks still allocated: {prefix_blocks[0] in cache.free_blocks}")
```

**Expected outcome:** Prefix blocks remain in use until all sequences that share them are freed.

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