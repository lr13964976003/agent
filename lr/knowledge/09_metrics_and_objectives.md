# Metrics and Objectives

## Purpose

This document defines the **evaluation metrics and optimization objectives** used to:

- Compare candidate inference DAGs
- Guide automatic parallel strategy selection
- Enforce correctness–performance trade-offs

All metrics are evaluated **in the context of decoder-only Transformer inference**.

---

## Objective Taxonomy

Inference optimization is inherently **multi-objective**. Objectives are grouped into:

1. Latency-oriented objectives
2. Throughput-oriented objectives
3. Resource-efficiency objectives
4. Stability and correctness objectives

DAG generation agents must explicitly select or weight objectives.

---

## Latency Metrics

### End-to-End Latency

- Definition: Time from request arrival to final token emission
- Units: milliseconds

Decomposed into:
- Prefill latency
- First-token latency (TTFT)
- Per-token decode latency

Critical path length in DAG is the dominant factor.

---

### Decode Step Latency

- Measures single-token decode iteration
- Sensitive to:
  - TP communication
  - KV cache access
  - Synchronization points

Primary metric for **interactive workloads**.

---

## Throughput Metrics

### Tokens per Second (TPS)

- Definition: Total generated tokens per second
- Measured at steady state

Strongly influenced by:
- Batch size
- Microbatching
- Pipeline fill ratio

---

### Requests per Second (RPS)

- Definition: Completed requests per second
- Sensitive to request length distribution

More relevant for **short-prompt workloads**.

---

## Resource Efficiency Metrics

### Memory Footprint

- Model parameters
- KV cache
- Activation buffers

Constraints:
- Per-device HBM limit
- Fragmentation risk

---

### Compute Utilization

- FLOPs utilization
- SM / core occupancy

Low utilization often indicates:
- Over-parallelization
- Excessive synchronization

---

### Communication Overhead

- Collective operation volume
- Latency of synchronization

Measured as:
- Bytes transferred per token
- Time spent in comm nodes

---

## Scalability Metrics

### Strong Scaling Efficiency

- Fixed problem size
- Increasing device count

Indicates communication dominance.

---

### Weak Scaling Efficiency

- Proportional increase in workload
- Increasing device count

Relevant for large-batch serving.

---

## Stability and Correctness Metrics

### Determinism

- Identical outputs across runs
- No race conditions in KV cache

---

### Failure Rate

- Deadlocks
- Timeouts
- OOM events

Invalid DAGs are rejected regardless of performance.

---

## Composite Objective Functions

Typical objective formulations:

- **Latency-first**:
  - Minimize decode latency
  - Hard constraint on memory

- **Throughput-first**:
  - Maximize TPS
  - Allow higher latency

- **Balanced**:
  - Pareto-optimal frontier

Agents may use weighted sums or lexicographic ordering.

---

## Metric–DAG Mapping

- Critical path ↔ latency
- Parallel subgraph width ↔ throughput
- Node memory annotations ↔ footprint
- Communication nodes ↔ overhead

Metrics must be **derivable directly from DAG structure**.

---

## Summary

Metrics and objectives provide:

- Quantitative evaluation of inference DAGs
- Guidance for automated strategy selection
- A common optimization language across systems

This module is the **decision layer** above DAG construction.