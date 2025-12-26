# Latency vs Throughput Trade-offs

## Scope

This document defines the **fundamental trade-offs between latency and throughput** in decoder-only Transformer inference on multi-device systems.

It focuses on:
- How different parallel primitives affect latency and throughput
- Phase-specific behavior (prefill vs decode)
- System-level decision rules for parallel strategy selection

**Out of scope**:
- Training-time optimization objectives
- Application-level SLA definitions

---

## Definitions

- **Latency**: Time required to produce the next token for a single sequence
- **Throughput**: Number of tokens or sequences processed per unit time across the system

In autoregressive inference, latency and throughput are often **in tension** due to strict causal dependencies.

---

## Phase-Specific Characteristics

### Prefill Phase

- Multiple tokens per sequence are available
- High inherent parallelism across tokens
- Throughput-oriented optimizations are highly effective
- Latency is amortized across many tokens

### Decode Phase

- Single token per sequence per step
- Strong temporal dependency between steps
- Latency dominates user-perceived performance
- Throughput optimizations must not increase per-step latency excessively

---

## Impact of Parallel Primitives

### Tensor Parallelism (TP)

- **Latency**: Can reduce per-layer compute latency if communication cost is low
- **Throughput**: Improves throughput by scaling compute
- **Risk**: Communication overhead can dominate in decode, especially cross-node

### Sequence Parallelism (SP)

- **Latency**: Improves prefill latency for long sequences
- **Throughput**: High throughput gains in prefill
- **Constraint**: Not applicable in decode

### Pipeline Parallelism (PP)

- **Latency**: Increases per-token latency due to stage synchronization
- **Throughput**: High throughput when stages are balanced and microbatched
- **Risk**: Decode latency bounded by slowest stage

### Expert Parallelism (EP)

- **Latency**: Per-token latency depends on routing and expert placement
- **Throughput**: Enables scaling via sparse activation
- **Risk**: Load imbalance can harm both latency and throughput

### Data Parallelism (DP)

- **Latency**: Does not reduce single-sequence latency
- **Throughput**: Strong scaling across sequences
- **Use Case**: Best for batch-oriented workloads

---

## Microbatching Effects

- Increases throughput by keeping devices busy
- Can increase tail latency if microbatches queue
- Essential for PP efficiency
- Decode microbatching trades latency for throughput across sequences

---

## Topology and System Constraints

- Low-latency interconnect favors latency-sensitive strategies (TP within node)
- High-latency interconnect favors throughput-oriented strategies (DP, coarse PP)
- Memory constraints may force microbatching, impacting latency

---

## Strategy Selection Guidelines

### Latency-Optimized Inference

- Minimize PP depth
- Keep TP within low-latency interconnect
- Avoid cross-node synchronization in decode
- Use small or no microbatching

### Throughput-Optimized Inference

- Maximize DP across sequences
- Use PP with sufficient microbatching
- Apply TP and EP to scale compute
- Aggressively overlap compute and communication

---

## Failure-Prone Assumptions

- Assuming throughput optimizations always improve latency
- Ignoring decode phase sensitivity to communication
- Over-microbatching latency-critical workloads
- Misjudging topology impact on synchronization cost

---

## Summary

Latency and throughput trade-offs are central to inference strategy design:
- Prefill favors throughput-oriented parallelism
- Decode favors latency-sensitive execution
- TP, PP, EP, DP, and microbatching each shift the balance differently

This module provides **decision rules for balancing latency and throughput** when generating parallel strategy DAGs.