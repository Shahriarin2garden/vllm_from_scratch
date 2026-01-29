# vllm_from_scratch
[Nano-vllm from scratch](https://www.notion.so/2deb2d43e8cb80b59d50d8a9fd0054b2?pvs=21)

nano‑vLLM: Build‑from‑Scratch Architectural Labs
A Complete, Step‑by‑Step Guide with Enhanced Visual Learning

Why vLLM Matters

The Problem
Traditional LLM serving systems suffer from fundamental inefficiencies:
• GPU Underutilization: GPUs idle 60‑70% of the time
• Memory Fragmentation: Variable‑length sequences waste 30‑50% of memory
• Poor Throughput: Sequential processing limits to 1 request/iteration
• High Latency: Long requests delay all subsequent requests

The vLLM Revolution
vLLM introduces three breakthrough innovations achieving 10‑23× throughput improvements:

1. Continuous Batching: Process multiple requests simultaneously

2. Paged KV Cache: Manage memory in fixed‑size blocks

3. Hybrid Scheduling: Intelligently prioritize requests

Our Learning Journey
We build "nano‑vLLM" through  structured labs, each focusing on one core concept with clear, modular visualizations. This guide adopts a brick‑by‑brick approach, showing exactly how each component fits into the growing system.
