# Phase 1: Keypoints Extraction

## Original Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Innovation Points

### 1. Large Expert Parallelism (EP ≥ 16)
- Definition: Configurations with 16 or more experts per parallel group
- Core principle: Deploy at most one expert per GPU
- Goal: Maximize expert-level parallelism and minimize contention

### 2. Cross-Node Expert Distribution
- Topology-aware placement considering bandwidth, latency, and GPU memory
- Minimizes hotspotting on single nodes
- Ensures balanced token routing patterns

### 3. Communication-Compute Overlap
- Asynchronous token routing
- Pipeline scheduling for multi-layer MoE networks
- CUDA streams/NCCL for non-blocking transfers

### 4. Load Balancing Strategy
- Dynamic gating probability adjustment
- Token batching by destination expert
- Per-expert load monitoring

## Critical Dimensions and Parameters
- Model: 4-layer MoE, 16 experts per layer
- Expert type: MLP with hidden size 32768
- Token dimension: 8192
- MHA: 16 heads, 512 dimensions per head
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- Precision: FP16

## Performance Metrics
- TPS (Tokens per Second): 450,000 (proposed) vs 120,000 (baseline)
- TPOT (Time per Output Token): 2.2ms (proposed) vs 8.3ms (baseline)
- Improvement: 3.75× higher throughput, 3.8× lower latency

## Deployment Configuration
- 16 H100 GPUs
- One expert per GPU per layer
- Baseline: TP=8, PP=2 with 8 experts per GPU
- Proposed: EP=16 with 1 expert per GPU