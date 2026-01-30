# Lab 0.2 Enhancements - Additional Educational Content

## Performance Benchmarks and Real-World Data

### Prefill vs Decode: Actual Performance Metrics

Based on vLLM paper (Kwon et al., 2023) and production deployments:

| Model | Prefill (512 tokens) | Decode (per token) | Ratio |
|-------|---------------------|-------------------|-------|
| GPT-3 (175B) | 2.3s | 23ms | 100:1 |
| LLaMA-2 70B | 1.1s | 15ms | 73:1 |
| LLaMA-2 13B | 180ms | 3.2ms | 56:1 |
| LLaMA-2 7B | 95ms | 1.8ms | 53:1 |

**Key Insights:**
- Prefill is 50-100× slower than decode per token
- Larger models have higher prefill/decode ratios
- Batch size dramatically affects these numbers

### GPU Utilization Patterns

**Without Continuous Batching:**
```
GPU Utilization: 30-40%
- Long prefills block the GPU
- Decode phases underutilize compute
- Sequential processing wastes resources
```

**With Continuous Batching (vLLM):**
```
GPU Utilization: 80-95%
- Prefills and decodes interleaved
- Multiple requests processed simultaneously
- Near-optimal resource usage
```

## Detailed Roofline Analysis

### Prefill Phase Roofline

```
Arithmetic Intensity (AI) = FLOPs / Bytes Moved

For Prefill (sequence length S, model size N):
- FLOPs ≈ 2 × N × S (matrix multiplications)
- Bytes ≈ 2 × N (model weights in FP16)
- AI = S (linear in sequence length)

Example (LLaMA-2 7B, S=512):
- FLOPs = 2 × 7B × 512 ≈ 7.2 TFLOPs
- Bytes = 2 × 7B ≈ 14 GB
- AI = 512 FLOPs/Byte

A100 GPU:
- Peak Compute: 312 TFLOPs (FP16)
- Peak Bandwidth: 1.5 TB/s
- Compute Bound Threshold: 312/1.5 = 208 FLOPs/Byte

Since 512 > 208, prefill is COMPUTE BOUND ✓
```

### Decode Phase Roofline

```
For Decode (single token, model size N):
- FLOPs ≈ 2 × N (one token through model)
- Bytes ≈ 2 × N (model weights) + KV cache reads
- AI ≈ 1 FLOPs/Byte (constant!)

Example (LLaMA-2 7B, context=2048):
- FLOPs = 2 × 7B ≈ 14 GFLOPs
- Bytes = 14 GB (weights) + 2 GB (KV cache) = 16 GB
- AI = 14/16 ≈ 0.875 FLOPs/Byte

Since 0.875 << 208, decode is MEMORY BOUND ✓
```

## Advanced Scheduling Algorithms

### Continuous Batching: The vLLM Innovation

**Traditional Static Batching:**
```python
# Pseudocode
batch = wait_until_full(max_batch_size=32, timeout=100ms)
results = process_batch(batch)  # All finish together
return results
```

**Problems:**
- Head-of-line blocking
- Wasted GPU cycles waiting
- Poor latency for small requests

**Continuous Batching:**
```python
# Pseudocode
running_batch = []
while True:
    # Add new requests
    while has_capacity() and has_waiting():
        running_batch.append(get_next_request())
    
    # Process one iteration
    step_results = model.forward(running_batch)
    
    # Remove finished requests
    for req in running_batch:
        if req.is_complete():
            running_batch.remove(req)
            yield req.result
```

**Benefits:**
- No waiting for batch to fill
- Requests join/leave dynamically
- 2-3× better throughput
- Lower latency variance

### Iteration-Level Scheduling

vLLM schedules at the **iteration level** (per forward pass), not request level:

```
Iteration 1: [Req1_prefill, Req2_decode, Req3_decode]
Iteration 2: [Req1_decode, Req2_decode, Req4_prefill]
Iteration 3: [Req1_decode, Req4_decode, Req5_prefill]
```

**Key Innovation**: Mix prefill and decode in same batch!
- Prefill: Process chunk of prompt
- Decode: Generate one token
- Both use same GPU efficiently

## Memory Management Deep Dive

### KV Cache Memory Calculation

For a single request:
```
KV_cache_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes

Example (LLaMA-2 7B, seq_len=2048, FP16):
= 2 × 32 × 32 × 128 × 2048 × 2
= 1,073,741,824 bytes
= 1 GB per request!
```

For 100 concurrent requests:
```
Total = 100 GB just for KV cache!
```

This is why PagedAttention is critical.

### PagedAttention Memory Savings

**Scenario**: 100 requests, average length 512 tokens, max length 4096

**Without PagedAttention (pre-allocated):**
```
Memory = 100 × 4096 × cache_per_token
       = 100 × 4096 × 128 KB
       = 52.4 GB
Utilization = (100 × 512) / (100 × 4096) = 12.5%
Wasted = 45.9 GB (87.5%)
```

**With PagedAttention (block_size=16):**
```
Blocks needed = 100 × ceil(512/16) = 3,200 blocks
Memory = 3,200 × 16 × 128 KB = 6.5 GB
Utilization = 100%
Saved = 45.9 GB (87.5% reduction!)
```

## FlashAttention Explained

### The Memory Problem

Standard attention materializes the full attention matrix:
```
S = QK^T  # Shape: [batch, heads, seq_len, seq_len]
P = softmax(S)
O = PV
```

For seq_len=2048:
```
Attention matrix size = 2048 × 2048 × 4 bytes = 16 MB per head
For 32 heads = 512 MB per layer
For 32 layers = 16 GB total!
```

### FlashAttention Solution

**Key Idea**: Never materialize the full attention matrix

**Algorithm**:
1. Divide Q, K, V into blocks that fit in SRAM (~20MB)
2. Load one block of Q and one block of K/V
3. Compute attention for this block
4. Update running statistics (max, sum)
5. Write output incrementally

**Benefits**:
- 3-5× faster (fewer HBM accesses)
- 10-20× less memory
- Enables longer sequences

**Pseudocode**:
```python
# Simplified FlashAttention
def flash_attention(Q, K, V, block_size=256):
    O = zeros_like(Q)
    l = zeros(Q.shape[0])  # row sums
    m = full(Q.shape[0], -inf)  # row maxes
    
    for i in range(0, Q.shape[0], block_size):
        Q_block = Q[i:i+block_size]  # Load to SRAM
        
        for j in range(0, K.shape[0], block_size):
            K_block = K[j:j+block_size]  # Load to SRAM
            V_block = V[j:j+block_size]
            
            # Compute attention for this block
            S_block = Q_block @ K_block.T
            
            # Online softmax with running statistics
            m_new = max(m[i:i+block_size], S_block.max(dim=1))
            l_new = exp(m - m_new) * l + exp(S_block - m_new).sum(dim=1)
            
            # Update output
            O[i:i+block_size] = (
                exp(m - m_new) * O[i:i+block_size] +
                exp(S_block - m_new) @ V_block
            ) / l_new
            
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new
    
    return O
```

> **Research Reference**: \"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\" (Dao et al., 2022)

## Chunked Prefill Strategy

### Why Chunk Prefills?

**Problem**: Long prompts (>8K tokens) block the GPU for seconds

**Example Timeline Without Chunking**:
```
Time: 0s    1s    2s    3s    4s    5s
GPU:  [====== Req1 Prefill (10K tokens) ======]
      [Req2 waiting...] [Req3 waiting...] [Req4 waiting...]
```

**With Chunking (chunk_size=512)**:
```
Time: 0s    1s    2s    3s    4s    5s
GPU:  [R1_c1][R1_c2][R1_c3][R1_c4][R1_c5]...
      [R2_d] [R3_d] [R2_d] [R3_d] [R2_d]...
```

### Optimal Chunk Size Selection

**Trade-offs**:
- **Small chunks** (128-256): Better fairness, higher overhead
- **Large chunks** (1024-2048): Lower overhead, worse fairness

**Adaptive Strategy**:
```python
def calculate_chunk_size(prompt_len, system_load):
    base_size = 512
    
    if system_load > 0.8:  # High load
        return base_size // 2  # Smaller chunks for fairness
    elif prompt_len > 10000:  # Very long prompt
        return base_size // 2  # Avoid blocking
    elif prompt_len < 1000:  # Short prompt
        return prompt_len  # Process all at once
    else:
        return base_size
```

## Production Deployment Patterns

### Disaggregated Architecture Benefits

**Cost Analysis**:

Traditional (Unified):
```
- 10× H100 GPUs (80GB each)
- Cost: $30,000/month
- Utilization: 40% (prefill-decode imbalance)
- Effective cost: $75,000/month per utilized GPU
```

Disaggregated:
```
Prefill Cluster:
- 3× H100 GPUs (compute-optimized)
- Cost: $9,000/month
- Utilization: 90%

Decode Cluster:
- 12× A100 GPUs (memory-optimized)
- Cost: $18,000/month
- Utilization: 85%

Total: $27,000/month (10% savings)
Better utilization: 87% average
Effective cost: $31,000/month per utilized GPU
```

**Savings**: 59% cost reduction per utilized GPU!

### Failure Recovery Strategies

**Checkpoint Frequency**:
```python
# Trade-off between recovery time and overhead
checkpoint_interval = calculate_interval(
    avg_generation_length=100,  # tokens
    checkpoint_cost=50ms,  # overhead per checkpoint
    recovery_cost=2s,  # cost to restart from beginning
)

# Optimal: checkpoint every ~20 tokens
# Recovery cost: 20 × 20ms = 400ms (vs 2s restart)
# Overhead: 5 × 50ms = 250ms per 100 tokens (2.5%)
```

## Performance Optimization Checklist

### Prefill Optimization
- [ ] Use FlashAttention or equivalent
- [ ] Enable tensor parallelism for large models
- [ ] Optimize batch size for compute utilization
- [ ] Use FP16 or BF16 (not FP32)
- [ ] Enable CUDA graphs for kernel fusion
- [ ] Profile and optimize attention kernel

### Decode Optimization
- [ ] Implement PagedAttention
- [ ] Optimize KV cache layout
- [ ] Use continuous batching
- [ ] Enable speculative decoding (if applicable)
- [ ] Minimize memory transfers
- [ ] Use quantization (INT8/INT4) for weights

### Scheduling Optimization
- [ ] Implement iteration-level scheduling
- [ ] Enable chunked prefill
- [ ] Set appropriate batch size limits
- [ ] Configure memory watermarks
- [ ] Implement priority queues
- [ ] Monitor and adjust dynamically

## Research Papers and References

### Foundational Papers
1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer architecture
   - https://arxiv.org/abs/1706.03762

2. **Efficient Transformers: A Survey** (Tay et al., 2020)
   - Comprehensive survey of attention optimizations
   - https://arxiv.org/abs/2009.06732

### Inference Optimization
3. **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
   - PagedAttention and continuous batching
   - https://arxiv.org/abs/2309.06180

4. **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)
   - IO-aware attention algorithm
   - https://arxiv.org/abs/2205.14135

5. **FlashAttention-2: Faster Attention with Better Parallelism** (Dao, 2023)
   - Improved FlashAttention with better GPU utilization
   - https://arxiv.org/abs/2307.08691

### System Design
6. **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., 2022)
   - Iteration-level scheduling
   - https://www.usenix.org/conference/osdi22/presentation/yu

7. **AlpaServe: Statistical Multiplexing with Model Parallelism** (Li et al., 2023)
   - Advanced scheduling and model parallelism
   - https://arxiv.org/abs/2302.11665

### Hardware Optimization
8. **TensorRT-LLM: Optimizing Inference Performance** (NVIDIA, 2023)
   - Production inference optimization
   - https://github.com/NVIDIA/TensorRT-LLM

9. **DeepSpeed Inference: Enabling Efficient Inference** (Microsoft, 2022)
   - Inference optimization techniques
   - https://www.deepspeed.ai/inference/

### Quantization and Compression
10. **GPTQ: Accurate Post-Training Quantization** (Frantar et al., 2022)
    - Weight quantization for inference
    - https://arxiv.org/abs/2210.17323

11. **AWQ: Activation-aware Weight Quantization** (Lin et al., 2023)
    - Improved quantization preserving accuracy
    - https://arxiv.org/abs/2306.00978

## Glossary of Terms

**Arithmetic Intensity (AI)**: Ratio of FLOPs to bytes moved. Higher AI means more compute-bound.

**Continuous Batching**: Dynamic batching where requests join/leave at iteration boundaries.

**Decode Phase**: Sequential token generation phase after prefill.

**Head-of-Line Blocking**: When slow requests block faster ones in a queue.

**Iteration-Level Scheduling**: Scheduling decisions made per forward pass, not per request.

**KV Cache**: Cached Key and Value tensors from previous tokens.

**PagedAttention**: Memory management technique using fixed-size blocks for KV cache.

**Prefill Phase**: Initial parallel processing of the input prompt.

**Roofline Model**: Performance model showing compute vs memory bounds.

**TTFT (Time-To-First-Token)**: Latency from request to first generated token.

**TPOT (Time-Per-Output-Token)**: Average time to generate each subsequent token.

## Hands-On Exercises

### Exercise 1: Calculate KV Cache Size
Given:
- Model: LLaMA-2 13B
- Layers: 40
- Heads: 40
- Head dimension: 128
- Sequence length: 4096
- Data type: FP16

Calculate total KV cache size.

<details>
<summary>Solution</summary>

```
KV_cache = 2 × layers × heads × head_dim × seq_len × dtype_bytes
         = 2 × 40 × 40 × 128 × 4096 × 2
         = 4,194,304,000 bytes
         = 4 GB per request
```
</details>

### Exercise 2: Roofline Analysis
Given:
- GPU: A100 (312 TFLOPs FP16, 1.5 TB/s bandwidth)
- Model: 7B parameters
- Sequence length: 1024

Determine if prefill is compute-bound or memory-bound.

<details>
<summary>Solution</summary>

```
Compute bound threshold = Peak FLOPs / Peak Bandwidth
                        = 312×10^12 FLOPs / 1.5×10^12 bytes/s
                        = 208 FLOPs/Byte

Prefill AI = sequence_length = 1024 FLOPs/Byte

Since 1024 > 208, prefill is COMPUTE BOUND.
```
</details>

### Exercise 3: Batch Size Optimization
You have:
- 24 GB GPU memory
- Model weights: 14 GB
- KV cache per request: 256 MB
- Activation memory per request: 100 MB

What's the maximum batch size?

<details>
<summary>Solution</summary>

```
Available memory = 24 GB - 14 GB = 10 GB
Memory per request = 256 MB + 100 MB = 356 MB

Max batch size = 10 GB / 356 MB ≈ 28 requests

Practical batch size (with safety margin): 24 requests
```
</details>

## Conclusion

This lab has covered the fundamental two-phase architecture of LLM inference:

1. **Prefill Phase**: Compute-bound, parallel processing of prompts
2. **Decode Phase**: Memory-bound, sequential token generation
3. **Scheduling**: Continuous batching for optimal GPU utilization
4. **Memory Management**: PagedAttention for efficient KV cache
5. **Optimizations**: FlashAttention, chunked prefill, disaggregation

Understanding these concepts is crucial for:
- Building efficient inference systems
- Optimizing production deployments
- Debugging performance issues
- Making architectural decisions

The next lab (0.3) will dive deeper into KV cache management and PagedAttention implementation.
