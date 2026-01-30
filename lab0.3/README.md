# lab 0.3

Status: Not started

**Navigation:** [← Lab 0.2](../lab0.2/README.md) | [Main](../README.md)

# **Lab 0.3: The Heart of the Matter – KV Cache & Attention**

## **📚 Lab Objective**

- **Primary Goal**: To master the complete evolution of KV cache management from naive contiguous allocation to sophisticated PagedAttention. You will build understanding iteratively, from basic attention mathematics to production-ready memory optimization.
- **Learning Outcome**: You will be able to implement, step-by-step, a full KV cache management system, understanding the trade-offs at each stage and why PagedAttention is the optimal solution for high-throughput serving.

## **🎯 Architectural Significance**

The KV cache represents 40-70% of dynamic memory usage during inference. Its management dictates system throughput, latency, and maximum concurrent users. This lab moves from theory to implementation—showing you exactly how to build the memory subsystem that makes nano-vLLM possible.

### **Conceptual Overview: "Tetris" for AI Memory**
Before we write code, let's understand the core problem:
*   **The Problem:** LLMs don't generate text in predictable blocks. One user asks a question and gets a 10-word answer; another gets a 2000-word essay.
*   **Old Solution (Pre-vLLM):** Reserve a huge parking spot (VRAM buffer) for every car (request), assuming every car is a limousine (max length). If a generic sedan parks there, 90% of the space is wasted.
*   **New Solution (PagedAttention):** Break the parking spot into tiny slots (Pages/Blocks). A limousine takes 50 slots scattered around the lot; a sedan takes 5. This is non-contiguous memory management, similar to how your computer's RAM works (Virtual Memory).

---

![visual-1.png](visual-1.png)



## **Iterative Learning Path**

### **Phase 1: Foundations – Understanding the Core Problem**

### **Step 1.1: The Mathematical Necessity of Caching**

**Concept Intuition:**
Why do we need a "Cache" at all?
Imagine translating a book. To translate page 100, you need context from pages 1-99.
*   **Without Cache:** You re-read pages 1-99 every time you translate a single new word on page 100.
*   **With Cache:** You take notes (Key/Value vectors) on pages 1-99 once. To translate a new word, you just glance at your notes.

**The Attention Equation Without Cache**:

```
For generating token t:
    Q_t = W_q · h_t
    K_{1:t} = W_k · [h_1, h_2, ..., h_t]  # Recompute all keys
    V_{1:t} = W_v · [h_1, h_2, ..., h_t]  # Recompute all values
    Attention = softmax(Q_t · K_{1:t}ᵀ/√d) · V_{1:t}

```

**Complexity**: O(t) operations per token → O(n²) for sequence of length n

**The Attention Equation With Cache**:

```
# First time (prefill):
    K_{1:n}, V_{1:n} = compute_and_cache(prompt_tokens)

# Each decode step:
    Q_t = W_q · h_t
    K_t, V_t = compute_current(t)           # O(1)
    K_cache.append(K_t), V_cache.append(V_t) # O(1)
    Attention = softmax(Q_t · K_cacheᵀ/√d) · V_cache  # O(t)

```

**Complexity**: O(n²) total → **97-99% computation saved** for 1000-token sequence

**Visual 1.1: Computation Without vs With KV Cache**

**Detailed Explanation for Researchers:**
*   **Quadratic vs. Linear:** The "No Cache" subgraph demonstrates the $O(n^2)$ catastrophe. For token 3, we process 3 items. For token 1000, we process 1000 items. The total work is the area of the triangle: $\sum_{i=1}^N i \approx \frac{N^2}{2}$.
*   **The Cache Invariant:** The "With Cache" subgraph shows constant time complexity *per new token* (regarding projections) and linear time complexity for attention dot products. This shift from Triangle to Line is what makes real-time generation possible.

```mermaid
graph TD
    NC1[Token 1: Process 1 token] --> NC2[Token 2: Re-process 2 tokens]
    NC2 --> NC3[Token 3: Re-process 3 tokens]
    NC3 --> NC4[...]
    NC4 --> NC5[Token n: Re-process n tokens]
    
    WC1[Token 1: Process & Cache] --> WC2[Token 2: Read cache + 1 new]
    WC2 --> WC3[Token 3: Read cache + 1 new]
    WC3 --> WC4[...]
    WC4 --> WC5[Token n: Read cache + 1 new]
    
    NC5 -->|n=1000 → ~500k ops| Result1[Prohibitive for Serving]
    WC5 -->|n=1000 → ~1k ops| Result2[Practical for Real-time]
    
    style NC5 fill:#fca5a5
    style WC2 fill:#bbf7d0
    style WC3 fill:#bbf7d0
    style WC5 fill:#bbf7d0
```

### **Step 1.2: Naive Implementation - Contiguous Cache**

**Code Explanation for Researchers:**
This is the "standard" implementation found in most HuggingFace tutorials.
*   **Pre-allocation:** `torch.zeros(max_seq_len, ...)` locks the memory immediately.
*   **The Waste:** If `max_seq_len=2048` but the actual conversation is only 50 tokens, we have allocated (and blocked) memory for 1998 tokens that contain nothing but zeros. Multipy this by batch size (e.g., 64), and we are wasting gigabytes of VRAM.

**Basic Implementation**:

```python
class NaiveKVCache:
    def __init__(self, max_seq_len: int, hidden_size: int, num_heads: int):
        self.max_len = max_seq_len
        # Allocate maximum possible cache upfront [2, max_len, num_heads, head_dim]
        self.k_cache = torch.zeros(max_seq_len, num_heads, hidden_size)
        self.v_cache = torch.zeros(max_seq_len, num_heads, hidden_size)
        self.current_len = 0

    def update(self, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        """Append new K/V and return full cache."""
        pos = self.current_len
        self.k_cache[pos] = new_k
        self.v_cache[pos] = new_v
        self.current_len += 1
        return self.k_cache[:self.current_len], self.v_cache[:self.current_len]

    def reset(self):
        """Reset for new sequence."""
        self.current_len = 0

```

**The Critical Limitation**: Each sequence gets its own `max_seq_len` allocation. For 4K context with 1000 concurrent users:

- **Memory**: 1000 × 4K × 2 × 32 heads × 128 dim × 2 bytes = **65.5 GB**
- **Utilization**: Most sequences use <10% of allocation → **~90% wasted**

**Visual 1.2: Memory Waste in Naive Allocation**

**Detailed Explanation for Researchers:**
*   **The "Tetris" Problem (Internal Fragmentation):** This diagram visualizes the "Internal Fragmentation" pathology. We see massive blocks of Red (Allocated but Unused) memory.
*   **The System Impact:** Even if the physical GPU has free space, the *allocator* thinks it's full because it reserved `MaxSeqLen` for everyone. This artifically caps the batch size.
*   **Why `MaxSeqLen`?** We are forced to guess `MaxSeqLen` upfront because standard tensor operations require contiguous memory. We can't easily resize a tensor in the middle of a CUDA kernel without expensive memory copies.

```mermaid
graph LR
    A1[Seq A: 4K slots<br/>Actual: 512 tokens] --> A2[Seq B: 4K slots<br/>Actual: 128 tokens]
    A2 --> A3[Seq C: 4K slots<br/>Actual: 2K tokens]
    A3 --> A4[... 100s more ...]
    
    U1[Actually Used: 6GB] --> U2[512 + 128 + 2K tokens]
    
    F1[Free: 0GB<br/>No space for new requests<br/>Despite apparent waste]
    
    A4 --> Summary[18GB Wasted<br/>75% Fragmentation]
    
    style A1 fill:#fca5a5
    style A2 fill:#fca5a5
    style A3 fill:#fca5a5
    style U1 fill:#bbf7d0
    style F1 fill:#ef4444
```

**Key Insight**: This is **internal fragmentation** - memory allocated but unused within each allocation.

---

### **Phase 2: Evolution – Solving the Fragmentation Problem**

### **Step 2.1: Dynamic Allocation with Variable Blocks**

**Concept Intuition:**
This is the "Linked List" approach to memory.
Instead of an array `[1...1000]`, we say "Here is a block of 16. If you need more, I'll give you another block of 16 and note down where it is."
The memory is no longer physically contiguous (one big block), but logically connected.

**First Improvement**: Allocate only what's needed, grow as needed.

```python
class DynamicKVCache:
    def __init__(self):
        self.blocks = {}  # sequence_id -> list of (k_block, v_block)
        self.block_size = 16  # Tokens per block

    def allocate_for_sequence(self, seq_id: str, prompt_len: int) -> List[int]:
        """Dynamically allocate blocks for a new sequence."""
        num_blocks = ceil(prompt_len / self.block_size)
        allocated = []

        for _ in range(num_blocks):
            # Allocate new tensor block
            block = self._alloc_block()
            allocated.append(block)

        self.blocks[seq_id] = allocated
        return allocated

    def append_token(self, seq_id: str, k: Tensor, v: Tensor) -> bool:
        """Append token to sequence, allocate new block if needed."""
        seq_blocks = self.blocks[seq_id]
        last_block = seq_blocks[-1]

        if last_block.tokens < self.block_size:
            # Add to existing block
            last_block.add(k, v)
            return True
        else:
            # Allocate new block
            new_block = self._alloc_block()
            new_block.add(k, v)
            seq_blocks.append(new_block)
            return True

```

**Visual 2.1: Dynamic vs Static Allocation**

**Detailed Explanation for Researchers:**
*   **Efficiency Gain:** The diagram contrasts the rigid "Static" containers with the flexible "Dynamic" blocks. We move from ~80% waste to ~12% waste.
*   **The Trade-off:** We traded *memory efficiency* for *complexity*. Now, the Attention kernel can't just read `base_pointer + offset`. It has to traverse a list of blocks. This is where standard PyTorch kernels fail and custom CUDA kernels become necessary.

```mermaid
graph TD
    S1[Static: Seq A<br/>4K pre-allocated] --> S2[Static: Seq B<br/>4K pre-allocated]
    S2 --> S3[Wasted: 7.5K slots]
    
    D1[Dynamic: Seq A<br/>2 blocks 32 tokens] --> D2[Dynamic: Seq B<br/>1 block 16 tokens]
    D2 --> D3[Free: Many blocks available]
    
    S3 -->|Fixed overhead| Problem[Cannot serve many users]
    D3 -->|On-demand| Solution[More concurrent users]
    
    Metrics[100 concurrent sequences:<br/>Static: 400K slots, 320K wasted 80%<br/>Dynamic: 80K slots, 10K wasted 12.5%]
    
    style S3 fill:#fca5a5
    style D3 fill:#bbf7d0
    style Problem fill:#ef4444
    style Solution fill:#bbf7d0
```

### **Step 2.2: The Fragmentation Problem Emerges**

**Concept Intuition:**
This is the "Swiss Cheese" memory problem (External Fragmentation).
You have plenty of free memory overall (e.g., 2GB), but it's scattered in tiny 10MB holes between active programs.
If a new user arrives and needs a contiguous 100MB block, you have to say "Out of Memory" even though you have 2GB free.

**The New Problem**: When sequences finish at different times, memory becomes fragmented.

```python
def demonstrate_fragmentation():
    """Show how fragmentation occurs with dynamic allocation."""
    # Start with 4 sequences
    seqs = {
        'A': ['B0', 'B1', 'B2', 'B3'],  # 4 blocks
        'B': ['B4', 'B5'],             # 2 blocks
        'C': ['B6', 'B7', 'B8'],       # 3 blocks
        'D': ['B9', 'B10', 'B11']      # 3 blocks
    }

    # Sequences A and C finish
    free_blocks = ['B0', 'B1', 'B2', 'B3', 'B6', 'B7', 'B8']

    # New sequence needs 4 contiguous blocks
    # Free blocks: [0-3, 6-8] - NOT CONTIGUOUS!
    # Cannot allocate despite having 7 free blocks

    return "Fragmentation prevents allocation!"

```

**Visual 2.2: External Fragmentation Demonstrated**

**Detailed Explanation for Researchers:**
*   **The Scenario:** At T1, users A and C leave. This frees up blocks `B0-B1` and `B4-B5`.
*   **The Paradox:** We have **4 blocks free** total (100% of what is needed). Sequence E enters needing **3 blocks**.
*   **The Failure:** Because the free memory is split into two small islands (`[..][BB][..][DD]`), we cannot fit a contiguous chunk of size 3. The allocator throws an OOM error despite having 4 blocks available.

```mermaid
graph TD
    T0[T=0: All sequences active<br/>A: B0 B1, B: B2 B3, C: B4 B5, D: B6 B7]
    
    T1[T=1: A & C finish<br/>Free: B0 B1, B: B2 B3, Free: B4 B5, D: B6 B7]
    
    T2[T=2: New Seq E needs 3 blocks]
    
    FB[Free Blocks: 4 total<br/>B0 B1 and B4 B5<br/>Gap: B2 B3 used]
    
    Attempt[Attempt Allocation]
    
    Result[FAILS!<br/>Max contiguous is 2<br/>External Fragmentation]
    
    T0 --> T1
    T1 --> T2
    T2 --> FB
    FB --> Attempt
    Attempt --> Result
    
    style Result fill:#ef4444
    style FB fill:#fca5a5
```

**Key Insight**: We've traded internal fragmentation for **external fragmentation** - free memory exists but isn't contiguous.

---

### **Phase 3: Revolution – PagedAttention Solution**

### **Step 3.1: The Virtual Memory Analogy**

PagedAttention borrows from OS virtual memory:

- **Page/Block**: Fixed-size memory unit (e.g., 16 tokens)
- **Page Table/Block Table**: Maps virtual positions → physical blocks
- **Physical Memory**: Flat array of blocks shared by all sequences

**Visual 3.1: PagedAttention Architecture Overview**

**Detailed Explanation for Researchers:**
*   **Decoupling:** This diagram shows the complete separation of Logical (Sequence) and Physical (VRAM) address spaces.
*   **Non-Contiguous:** Notice `SeqA` uses physical blocks `42`, `17`, `85`. These are wildly out of order. This means we can fill *any* hole in VRAM with *any* part of *any* sequence.
*   **Zero Fragmentation:** As long as there is 1 free physical block, we can store 1 more chunk of tokens. External fragmentation is effectively eliminated.

```mermaid
graph TB
    VA1[Seq A: Virt Block 0<br/>Positions 0-15]
    VA2[Seq A: Virt Block 1<br/>Positions 16-31]
    VA3[Seq A: Virt Block 2<br/>Positions 32-47]
    
    VB1[Seq B: Virt Block 0<br/>Positions 0-15]
    VB2[Seq B: Virt Block 1<br/>Positions 16-31]
    
    PT1[A Table: 42, 17, 85]
    PT2[B Table: 23, 64]
    
    P42[Physical Block 42]
    P17[Physical Block 17]
    P85[Physical Block 85]
    P23[Physical Block 23]
    P64[Physical Block 64]
    P87[Free Block 87]
    P88[Free Block 88]
    
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
    
    style P87 fill:#bbf7d0
    style P88 fill:#bbf7d0
```

### **Step 3.2: Core Data Structures Implementation**

**Code Explanation for Researchers:**
This class `PagedKVCache` is the heart of the lab.
*   **`self.kv_data`:** This is the massive pre-allocated GPU tensor. It reserves *all* VRAM at startup.
*   **`block_tables`:** This Python dictionary replaces the simple list/pointer. It maps `seq_id` to a list of integers (block indices).
*   **`write_token`**: This method performs the critical translation: `Logical Token Index -> Physical Block Index`. This math `block_idx = token_idx // block_size` is exactly how MMUs (Memory Management Units) work in hardware.

```python
class PagedKVCache:
    """Complete PagedAttention implementation."""

    def __init__(self, total_blocks: int, block_size: int = 16,
                 num_heads: int = 32, head_dim: int = 128):
        # Free block management
        self.free_blocks = deque(range(total_blocks))

        # Block tables: sequence_id -> list of physical block indices
        self.block_tables: Dict[str, List[int]] = {}

        # Physical storage: [total_blocks, 2, num_heads, block_size, head_dim]
        # Index: [physical_block, 0=key/1=value, head, position, dimension]
        self.kv_data = torch.zeros(
            total_blocks, 2, num_heads, block_size, head_dim,
            dtype=torch.bfloat16, device='cuda'
        )

        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads

    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for tokens, return physical block indices."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        blocks_needed = ceil(num_tokens / self.block_size)

        if len(self.free_blocks) < blocks_needed:
            raise MemoryError("Not enough free blocks")

        # Allocate from free pool
        allocated = [self.free_blocks.popleft() for _ in range(blocks_needed)]
        self.block_tables[seq_id] = allocated

        return allocated

    def write_token(self, seq_id: str, token_idx: int,
                    k_data: Tensor, v_data: Tensor):
        """Write token's K/V to cache at logical position token_idx."""
        # 1. Find which block and offset
        block_idx = token_idx // self.block_size
        offset = token_idx % self.block_size

        # 2. Get physical block from block table
        physical_block = self.block_tables[seq_id][block_idx]

        # 3. Write to physical memory
        self.kv_data[physical_block, 0, :, offset, :] = k_data  # Keys
        self.kv_data[physical_block, 1, :, offset, :] = v_data  # Values

    def read_for_attention(self, seq_id: str, context_len: int) -> Tuple[Tensor, Tensor]:
        """Read K/V for entire context length of sequence."""
        # This is simplified - real implementation uses slot_mapping
        blocks = self.block_tables[seq_id]
        num_blocks_needed = ceil(context_len / self.block_size)

        # Gather all needed blocks
        k_blocks = []
        v_blocks = []

        for i in range(num_blocks_needed):
            if i >= len(blocks):
                raise IndexError(f"Block index {i} out of range for blocks list of length {len(blocks)}")
            block = blocks[i]
            # Slice this block's K and V
            tokens_in_block = min(self.block_size, context_len - i * self.block_size)

            k_block = self.kv_data[block, 0, :, :tokens_in_block, :]
            v_block = self.kv_data[block, 1, :, :tokens_in_block, :]

            k_blocks.append(k_block)
            v_blocks.append(v_block)

        # Concatenate along sequence dimension
        k_all = torch.cat(k_blocks, dim=1)  # [num_heads, context_len, head_dim]
        v_all = torch.cat(v_blocks, dim=1)

        return k_all, v_all

```

**Visual 3.2: Memory Access with Block Tables**

**Detailed Explanation for Researchers:**
*   **The Indirection Cost:** This diagram highlights the trade-off. We pay for memory efficiency with *pointers*.
*   **The Translation Layer:** In "Step 2", notice the math `Block idx = 42 // 16`. This computation is trivial for a CPU, but inside a massive GPU kernel running on 100,000 threads, managing these lookups efficiently requires coalesced memory access patterns to avoid "Memory Divergence".
*   **Kernel Complexity:** Step 4 implies the complexity. The PagedAttention kernel is *not* just a standard MatMul. It is a gather-compute-scatter operation.

```mermaid
graph TD
    T1[1. Token Arrives<br/>Position 42<br/>Sequence: chat_123]
    
    BT[2. Block Table Lookup<br/>block_tables chat_123 = 15, 87, 42, 91]
    
    Calc[3. Calculate Address<br/>Block idx = 42 ÷ 16 = 2<br/>Offset = 42 % 16 = 10]
    
    Map[4. Map to Physical<br/>Physical block = table 2 = 42]
    
    PM[5. Physical Memory Access<br/>Block 42, Slot 10]
    
    Read[6. Read/Write K/V<br/>at block=42, slot=10]
    
    Kernel[7. PagedAttention Kernel<br/>Input: Q, block_tables, seq_lens<br/>Processes 100s tokens parallel]
    
    T1 --> BT --> Calc --> Map --> PM --> Read --> Kernel
    
    Efficiency[Key Advantages:<br/>Zero fragmentation<br/>O1 allocation<br/>Efficient sharing]
    
    Kernel --> Efficiency
```

### **Step 3.3: The Slot Mapping Optimization**

**Concept Intuition:**
This is the "Compiler" step for memory.
Python logic (class`PagedKVCache`) is slow. The GPU is fast.
We don't want the GPU asking Python "Where is block 5?" for every token.
Instead, we compile a massive cheatsheet (`SlotMapping`) in one go. We hand this tensor to the GPU and say: "Here is the exact address of every single byte you will need for the next millisecond. Go."

**Code Explanation for Researchers:**
The function `build_slot_mapping` constructs the **Flattened State**.
Typical PyTorch works on `[Batch, Seq, Dim]`.
vLLM works on `[Total_Tokens_In_Batch, Dim]`.
This function flattens the hierarchical structure (Batch -> Sequence -> Block -> Token) into a flat list of physical pointers (`slot_k`, `slot_v`) so the CUDA kernel can trust blindly.

The real performance comes from preparing data for the GPU kernel.

```python
def build_slot_mapping(sequences: List[SequenceInfo]) -> SlotMapping:
    """Build the slot_mapping tensor for efficient GPU execution.

    The kernel needs to know for each token in the batch:
    - Which physical block it's in
    - Which offset within that block
    - Whether it's K or V
    """

    slot_mapping = []
    block_tables_flat = []
    sequence_indices = []

    for seq_idx, seq in enumerate(sequences):
        # Get this sequence's block table
        blocks = seq.block_table

        # For each token position in this sequence
        for pos in range(seq.current_len):
            # Calculate block and offset
            block_idx = pos // BLOCK_SIZE
            offset = pos % BLOCK_SIZE

            # Get physical block
            physical_block = blocks[block_idx]

            # Calculate slot index
            # Even indices for K, odd for V in interleaved storage
            slot_k = physical_block * (BLOCK_SIZE * 2) + (offset * 2)
            slot_v = slot_k + 1

            slot_mapping.append((slot_k, slot_v))
            block_tables_flat.append(physical_block)
            sequence_indices.append(seq_idx)

    return SlotMapping(
        k_slots=torch.tensor([s[0] for s in slot_mapping], device='cuda'),
        v_slots=torch.tensor([s[1] for s in slot_mapping], device='cuda'),
        block_indices=torch.tensor(block_tables_flat, device='cuda'),
        seq_indices=torch.tensor(sequence_indices, device='cuda')
    )

def paged_attention_forward(q: Tensor, slot_mapping: SlotMapping,
                           kv_cache: PagedKVCache) -> Tensor:
    """Simplified forward pass showing slot_mapping usage."""
    # In real implementation, this is a CUDA kernel
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Gather K and V using slot mapping (simplified CPU version)
    k_gathered = torch.zeros_like(q)
    v_gathered = torch.zeros_like(q)

    for i in range(batch_size):
        for h in range(num_heads):
            for s in range(seq_len):
                # Get slot indices from mapping
                k_slot = slot_mapping.k_slots[i * seq_len + s]
                v_slot = slot_mapping.v_slots[i * seq_len + s]

                # Convert slot index to block/offset
                k_block = k_slot // (BLOCK_SIZE * 2)
                k_offset = (k_slot % (BLOCK_SIZE * 2)) // 2

                # Read from cache (simplified)
                k_gathered[i, h, s] = kv_cache.kv_data[k_block, 0, h, k_offset]
                v_gathered[i, h, s] = kv_cache.kv_data[k_block, 1, h, k_offset]

    # Standard attention computation
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k_gathered.transpose(-2, -1)) * scale

    # Apply attention mask (not shown)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_gathered)

    return output

```

**Visual 3.3: Slot Mapping Construction**

**Detailed Explanation for Researchers:**
*   **Sequential vs. Parallel:** This diagram shows how we break the sequential dependency of "Batch Processing".
*   **The Flattening:** By flattening all tokens into a single array (18 tokens), we allow the GPU to launch 18 parallel threads (or thread blocks) that are completely independent. Thread #17 knows exactly where to find its data without asking Thread #0.
*   **Coalescing:** The `SlotMapping` tensor itself is stored contiguously in memory. This ensures that when the GPU Warps load the mapping, they do so efficiently, before scattering to read the actual KV data.

```mermaid
graph TD
    S1[Sequence A: 5 tokens<br/>Block Table: 42, 15]
    S2[Sequence B: 10 tokens<br/>Block Table: 23, 64, 87]
    S3[Sequence C: 3 tokens<br/>Block Table: 91]
    
    TP[Flattened Tokens<br/>A0 A1 A2 A3 A4 B0...B9 C0 C1 C2<br/>Total: 18 tokens]
    
    SC1[Slot Calculation:<br/>1. Get block table<br/>2. block_idx = pos ÷ 16<br/>3. offset = pos % 16<br/>4. Get physical block<br/>5. slot = block × 32 + offset × 2]
    
    Example[Example: Seq B token 25<br/>block_idx = 25 ÷ 16 = 1<br/>offset = 25 % 16 = 9<br/>physical_block = 64<br/>slot = 64 × 32 + 9 × 2 = 2050]
    
    SM[Slot Mapping Tensor<br/>1344,1345 for A0<br/>1346,1347 for A1<br/>2050,2051 for B25]
    
    Note[Single tensor for all tokens<br/>Parallel random access<br/>No per-sequence logic]
    
    S1 --> TP
    S2 --> TP
    S3 --> TP
    TP --> SC1
    SC1 --> Example
    Example --> SM
    SM --> Note
```

---

### **Phase 4: Advanced Optimizations**

### **Step 4.1: Block-Sparse Attention**

**Concept Intuition:**
Imagine you are reading a 1000-page history book. To understand page 1000, do you really need to remember every single detail from page 5? Probably not. You mostly need the recent chapters (Local Attention) and maybe a few key events from the start (Sink Tokens).
Block-Sparse Attention says: "Let's not load the irrelevant pages into the extremeley expensive GPU L1 cache."

**Code Explanation for Researchers:**
The `BlockSparsePagedAttention` class introduces a `sparsity_pattern` mask.
Instead of the standard `read_for_attention` which grabs *everything*, `sparse_read` filters the block list.
This reduces the memory bandwidth requirement from $O(N)$ to $O(\text{window\_size})$, which is a massive speedup for long-context models.

Not all tokens need attention to all previous tokens. We can skip loading irrelevant blocks.

```python
class BlockSparsePagedAttention(PagedKVCache):
    """Extend with block-sparse attention support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_pattern = {}  # sequence_id -> attention mask

    def build_sparse_mask(self, seq_id: str, context_len: int,
                         window_size: int = 4) -> Tensor:
        """Build block-sparse attention mask.

        Only attends to last 'window_size' blocks for each position.
        """
        num_blocks = ceil(context_len / self.block_size)
        mask = torch.full((num_blocks, num_blocks), float('-inf'))

        for i in range(num_blocks):
            # Only attend to last window_size blocks
            start = max(0, i - window_size + 1)
            for j in range(start, i + 1):
                mask[i, j] = 0  # Allow attention

        self.sparsity_pattern[seq_id] = mask
        return mask

    def sparse_read(self, seq_id: str, query_block_idx: int) -> Tuple[Tensor, Tensor]:
        """Read only blocks needed for sparse attention."""
        mask = self.sparsity_pattern[seq_id]
        blocks = self.block_tables[seq_id]

        # Get which blocks to load based on mask
        blocks_to_load = []
        for j in range(len(blocks)):
            if mask[query_block_idx, j] == 0:
                blocks_to_load.append(blocks[j])

        # Load only those blocks
        k_blocks = [self.kv_data[block, 0] for block in blocks_to_load]
        v_blocks = [self.kv_data[block, 1] for block in blocks_to_load]

        return torch.cat(k_blocks, dim=1), torch.cat(v_blocks, dim=1)

```

**Visual 4.1: Block-Sparse vs Full Attention**

**Detailed Explanation for Researchers:**
*   **The Traffic Reduction:** The metric "62.5% reduction in memory traffic" is the key takeaway. In a bandwidth-bound regime (Decode phase), this directly translates to a ~2x speedup.
*   **Coarse-Grained Sparsity:** Notice we drop *entire blocks*, not individual tokens. Dropping tokens creates irregular memory access (bad for GPU). Dropping blocks keeps the access patterns effectively dense (compresses well), while skipping large chunks of HBM reads.

```mermaid
graph TD
    FA1[Full Attention:<br/>Block 0 → All 8 blocks]
    FA2[Block 1 → All 8 blocks]
    FA3[Block 2 → All 8 blocks]
    FA4[Block 7 → All 8 blocks]
    
    SA1[Sparse Window=3:<br/>Block 0 → Block 0]
    SA2[Block 1 → Blocks 0-1]
    SA3[Block 2 → Blocks 0-2]
    SA4[Block 3 → Blocks 1-3]
    SA5[Block 4 → Blocks 2-4]
    SA6[Block 5 → Blocks 3-5]
    SA7[Block 6 → Blocks 4-6]
    SA8[Block 7 → Blocks 5-7]
    
    Metrics[Performance 8 blocks:<br/>Full: 64 block-pairs<br/>Sparse: 24 block-pairs<br/>62.5% reduction]
    
    FA1 --> FA2 --> FA3 --> FA4
    SA1 --> SA2 --> SA3 --> SA4 --> SA5 --> SA6 --> SA7 --> SA8
    SA8 --> Metrics
    
    style FA1 fill:#f3f4f6
    style FA2 fill:#f3f4f6
    style FA3 fill:#f3f4f6
    style FA4 fill:#f3f4f6
    style SA1 fill:#bbf7d0
    style SA2 fill:#bbf7d0
    style SA3 fill:#bbf7d0
    style SA4 fill:#bbf7d0
    style SA5 fill:#bbf7d0
    style SA6 fill:#bbf7d0
    style SA7 fill:#bbf7d0
    style SA8 fill:#bbf7d0
```

### **Step 4.2: Prefix Caching & Sharing**

**Concept Intuition:**
This is "Deduplication" for prompts.
In a chatbot, almost every request starts with "You are a helpful assistant...".
In RAG (Retrieval Augmented Generation), multiple questions might share the same massive wikipedia article as context.
Instead of storing "You are a helpful assistant..." 1000 times (wasting 1GB), we store it once and have 1000 users point to it. This uses Reference Counting.

Common prefixes (system prompts) can be shared across sequences.

**Code Explanation for Researchers:**
The `PrefixCachingKVCache` class demonstrates the Copy-on-Write mechanism.
*   **`prefix_cache`:** A hash map acting as the "deduplication table".
*   **`block_refcount`:** The critical safety mechanism. Notice in `free_sequence`, we *decrement* but only *free* if count reaches zero. This prevents "User A finishes and deletes the system prompt while User B is still using it."

```python
class PrefixCachingKVCache(PagedKVCache):
    """Extend with prefix sharing support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_cache = {}  # hash(prompt) -> list of block ids
        self.block_refcount = defaultdict(int)  # block_id -> reference count

    def get_or_create_prefix(self, prompt_hash: str, prompt_len: int) -> List[int]:
        """Get cached prefix or create and cache it."""
        if prompt_hash in self.prefix_cache:
            # Reuse existing blocks
            blocks = self.prefix_cache[prompt_hash]
            for block in blocks:
                self.block_refcount[block] += 1
            return blocks.copy()
        else:
            # Create new prefix
            blocks = self.allocate_for_prompt(prompt_len)
            self.prefix_cache[prompt_hash] = blocks
            for block in blocks:
                self.block_refcount[block] = 1
            return blocks

    def allocate_with_prefix(self, seq_id: str, prefix_hash: str,
                           total_tokens: int) -> List[int]:
        """Allocate for sequence with potential prefix sharing."""
        prefix_blocks = self.get_or_create_prefix(prefix_hash, prefix_len)

        # Allocate new blocks for the non-prefix part
        remaining_tokens = total_tokens - prefix_len
        if remaining_tokens > 0:
            new_blocks = self.allocate(seq_id, remaining_tokens)
            all_blocks = prefix_blocks + new_blocks
        else:
            all_blocks = prefix_blocks

        self.block_tables[seq_id] = all_blocks
        return all_blocks

    def free_sequence(self, seq_id: str):
        """Free sequence, handling shared blocks carefully."""
        blocks = self.block_tables.pop(seq_id, [])
        for block in blocks:
            self.block_refcount[block] -= 1
            if self.block_refcount[block] == 0:
                # Actually free the block
                self.free_blocks.append(block)

```

**Visual 4.2: Prefix Sharing in Action**

**Detailed Explanation for Researchers:**
*   **Copy-on-Write (CoW):** This mechanism is identical to how OS processes fork(). The child process shares the parent's memory pages until it writes to one.
*   **The RefCount:** The key operational detail (implicit in the diagram) is Reference Counting. Block `B42` has `ref_count=3`. We cannot free `B42` until *all three* users finish their requests.
*   **Radix Tree:** In production vLLM, this lookup (system prompt -> hash -> blocks) is implemented via a Radix Tree (Trie), allowing automatic prefix matching even if the user didn't explicitly flag it as a "system prompt".

```mermaid
graph TD
    SP[System Prompt<br/>You are a helpful assistant<br/>Hash: abc123<br/>Length: 12 tokens]
    
    UR1[User 1: Explain quantum physics<br/>Total: 15 tokens]
    UR2[User 2: Write a poem about AI<br/>Total: 17 tokens]
    UR3[User 3: Debug this Python code<br/>Total: 18 tokens]
    
    BT1[User 1 Table: 42, 87, 15, 23]
    BT2[User 2 Table: 42, 87, 15, 64, 91]
    BT3[User 3 Table: 42, 87, 15, 33, 78]
    
    P1[Prefix Blocks SHARED:<br/>B42, B87, B15]
    P2[User 1 unique: B23]
    P3[User 2 unique: B64, B91]
    P4[User 3 unique: B33, B78]
    
    Metrics[Memory Savings:<br/>Without: 9 blocks 3 each<br/>With: 3 blocks shared<br/>66% reduction]
    
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
    P1 --> Metrics
    
    style P1 fill:#fef3c7
    style Metrics fill:#bbf7d0
```

---

## **📊 Performance Comparison Table**

| **Aspect** | **Naive Contiguous** | **Dynamic Allocation** | **PagedAttention** | **Advanced Paged** |
| --- | --- | --- | --- | --- |
| **Memory Fragmentation** | Internal (High) | External (High) | None | None |
| **Allocation Speed** | O(1) | O(n) for search | O(1) | O(1) |
| **Max Concurrent Users** | Low | Medium | High | Very High |
| **Memory Utilization** | 10-20% | 30-50% | 85-95% | 90-98% |
| **Implementation Complexity** | Simple | Moderate | Complex | Very Complex |
| **Prefix Sharing** | Impossible | Difficult | Possible | Efficient |
| **Best For** | Prototyping | Low-concurrency | Production | Large-scale |

## **🔬 Experimental Results Simulation**

```python
def simulate_performance():
    """Simulate performance differences across approaches."""
    configs = [
        ("Naive", NaiveKVCache(max_seq_len=4096)),
        ("Dynamic", DynamicKVCache()),
        ("Paged", PagedKVCache(total_blocks=1000)),
        ("Advanced", PrefixCachingKVCache(total_blocks=1000))
    ]

    results = []
    for name, cache in configs:
        # Simulate 100 sequences of varying lengths
        start_mem = get_memory_usage()

        for i in range(100):
            seq_len = random.randint(100, 3000)
            cache.allocate(f"seq_{i}", seq_len)

        end_mem = get_memory_usage()
        utilization = calculate_utilization(cache)

        results.append({
            "approach": name,
            "memory_used": end_mem - start_mem,
            "utilization": utilization,
            "fragmentation": calculate_fragmentation(cache)
        })

    return results

# Expected Results:
# 1. Naive: 40GB used, 15% utilization, 85% fragmentation
# 2. Dynamic: 25GB used, 40% utilization, 60% fragmentation
# 3. Paged: 8GB used, 88% utilization, 0% fragmentation
# 4. Advanced: 6GB used, 92% utilization, 0% fragmentation

```

## **📝 Lab Summary & Key Takeaways**

### **Iterative Learning Progression**:

1. **Phase 1**: Understood why caching is mathematically necessary and the limitations of naive approaches
2. **Phase 2**: Saw how dynamic allocation helps but introduces external fragmentation
3. **Phase 3**: Learned how PagedAttention eliminates fragmentation via block tables and slot mapping
4. **Phase 4**: Explored advanced optimizations like sparse attention and prefix sharing

### **Core Principles**:

1. **PagedAttention = Virtual Memory for LLMs**: Fixed-size blocks + block tables solve fragmentation
2. **Slot Mapping Enables Parallelism**: Single tensor tells kernel where every token lives
3. **Sharing Multiplies Efficiency**: Prefix caching can reduce memory by 60%+ for common prompts
4. **Sparsity Exploits Locality**: Most tokens only need attention to recent context

### **Implementation Checklist for nano-vLLM**:

- [ ]  Implement basic block allocation with `free_blocks` deque
- [ ]  Add `block_tables` dictionary for sequence → block mapping
- [ ]  Create `slot_mapping` construction in executor
- [ ]  Implement prefix detection and sharing
- [ ]  Add block-sparse attention for long contexts
- [ ]  Optimize with CUDA kernels for production

## **➡️ Next Steps: Lab 0.4**

Now that you understand **how** PagedAttention works at the algorithmic level, we need to measure its real-world performance. In **Lab 0.4: The Profiling Baseline – Measurement & Optimization**, you will:

1. Learn GPU profiling tools (Nsight, PyTorch profiler)
2. Establish performance metrics and SLOs
3. Identify bottlenecks in your implementation
4. Apply data-driven optimizations to achieve production readiness

> Senior Architect Insight: The journey from naive caching to PagedAttention mirrors the evolution of operating systems from simple memory models to virtual memory. Each step solves a specific problem but introduces new complexity. The key is understanding these trade-offs to choose the right architecture for your scale and requirements.
>