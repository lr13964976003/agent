# Phase 1: Key Points Extraction

## Large EP Definition
- **Large Expert Parallelism (Large EP)**: Defined as EP ≥ 16, qualifying as "large EP"
- This represents a significant departure from conventional approaches that use moderate EP degrees

## Core Problem Statement
- Traditional MoE implementations colocate multiple experts per GPU to minimize communication
- This creates computational bottlenecks and limits true expert parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

## Proposed Solution
- **Cross-node expert parallelism** strategy distributing at most one expert per GPU
- Prioritizes maximizing compute concurrency over minimizing communication
- Shifts optimization focus from communication reduction to compute saturation

## Key Architectural Decisions
1. **Single-expert-per-GPU deployment**: Each GPU hosts at most one expert
2. **Cross-node distribution**: Experts distributed across nodes using topology-aware placement
3. **Large EP regime**: Leveraging EP ≥ 16 for maximum parallel computation
4. **Communication-computation overlap**: Using asynchronous routing and pipeline scheduling

## Critical Mathematical Relationships
- For E experts and G GPUs: 
  - If E ≤ G: each expert assigned to distinct GPU
  - If E > G: experts replicated across GPUs while maximizing independent expert concurrency
- Token dimension: 7168
- MHA: 128 heads × 128 dimensions per head = 16,384 total attention dimensions
- MLP hidden size: 2048

## Deployment Context
- **Inference-only setting** (critical missing detail from previous submission)
- H100 GPU cluster with advanced interconnects (NVLink, InfiniBand, NVSwitch)
- Maximum GPU utilization through one-expert-per-GPU policy
- Network bandwidth becomes primary limiting factor in large EP regime

## Integration with Parallelism Strategies
- Compatible with tensor parallelism (TP) for experts exceeding single-GPU memory
- Works with data parallelism (DP) for synchronized weight updates
- Pipeline parallelism for multi-layer networks with fine-grained scheduling