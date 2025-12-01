# Phase 1: Keypoints Extraction

## **Core Problem Addressed**
Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

## **Proposed Solution**
Large-scale cross-node expert parallelism strategy that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond (defined as "large EP" regime).

## **Key Innovation**
Shifts optimization focus from reducing communication to maximizing compute concurrency by fully exploiting distributed resources, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## **Technical Approach**
1. **Expert Placement Strategy**: Single-expert-per-GPU deployment with topology-aware cross-node distribution
2. **Routing and Load Balancing**: Token batching by destination expert, asynchronous routing, and dynamic load balancing
3. **Communication Overlap**: Interleaving expert computation with token transfers using CUDA streams and asynchronous communication

## **Large EP Regime Definition**
Configurations where EP ≥ 16, enabling distributing experts across as many devices as possible (ideally one per GPU) to minimize resource contention and maximize expert-level parallel execution.

## **Performance Claims**
- ~3.75× higher throughput compared to baseline
- ~3.8× lower latency compared to baseline
- Near-linear scaling for EP ≥ 16
- Shifts bottleneck from intra-GPU contention to manageable communication overhead

## **Experimental Validation**
Tested on 16-layer MoE with 64 experts per layer using 16 H100 GPUs, demonstrating significant improvements in TPS (120,000 → 450,000) and TPOT (8.3ms → 2.2ms).