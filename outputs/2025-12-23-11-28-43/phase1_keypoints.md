# Phase 1: Key Points Extraction

## Core Innovation
- **Single expert per GPU deployment**: Unlike traditional approaches that colocate multiple experts on the same device, this method deploys at most one expert per GPU
- **Large Expert Parallelism (EP ≥ 16)**: Defines "large EP" as configurations with 16 or more experts per parallel group
- **Cross-node distribution**: Experts are distributed across nodes to minimize hotspotting and maximize computational parallelism

## Problem Statement
- Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication
- This creates computational bottlenecks and limits the degree of true expert parallelism
- As model and cluster sizes grow, this trade-off becomes increasingly suboptimal

## Technical Approach
1. **Expert Placement Strategy**: Assigns experts across GPUs and nodes with topology-aware placement
2. **Routing and Load Balancing**: Ensures balanced input distribution to experts through token batching and asynchronous routing
3. **Communication Overlap and Scheduling**: Minimizes cross-node data transfer impact through compute-communication overlap

## Key Results
- **Throughput improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Model configuration**: 16-layer MoE, 64 experts per layer, FP8 precision
- **Deployment**: 16 H100 GPUs with 4 experts per GPU (64 experts ÷ 16 GPUs = 4 experts per GPU)

## Deployment Configuration
- **Baseline**: TP=8, PP=2, experts colocated on GPUs
- **Proposed**: Cross-node expert parallelism with distributed expert placement
- **Hardware**: 16 H100 GPUs minimum for large EP regime
- **Communication**: Asynchronous token routing with topology-aware placement