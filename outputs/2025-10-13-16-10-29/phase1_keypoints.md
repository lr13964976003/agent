# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### 1. Problem Statement
- Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Need to shift from reducing communication to maximizing compute concurrency

### 2. Solution Overview
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- **Cross-node distribution**: Distribute experts across nodes to exploit all compute resources
- **Three key components**:
  1. Expert Placement Strategy (one expert per GPU)
  2. Routing and Load Balancing
  3. Communication Overlap and Scheduling

### 3. Technical Innovations
- **Single-expert-per-GPU deployment**: Each GPU hosts exactly one expert
- **Topology-aware placement**: Considers bandwidth, latency, memory capacity
- **Asynchronous token routing**: Overlaps communication with computation
- **Pipeline scheduling**: Tokens flow immediately to next layer experts

### 4. Experimental Results
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens
- **Dimensions**: 8192 token dim, 16 heads × 512 dim, 32768 MLP hidden
- **Performance**: 3.75× higher TPS, 3.8× lower TPOT vs baseline

### 5. Deployment Comparison
- **Baseline**: TP=8, PP=2, 16 GPUs, 8 experts/GPU
- **Proposed**: EP=16, 16 GPUs, 1 expert/GPU
- **Results**: 450,000 TPS vs 120,000 TPS, 2.2ms vs 8.3ms TPOT

### 6. Scalability Features
- Compatible with tensor parallelism (TP) for large experts
- Compatible with data parallelism (DP) for training
- Near-linear scaling with 16+ GPUs
- Effective for HPC environments with high-bandwidth interconnects