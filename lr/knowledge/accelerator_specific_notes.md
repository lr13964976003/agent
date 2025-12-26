# Accelerator-Specific Notes

## Scope

This document captures **accelerator-specific considerations** that affect decoder-only Transformer inference execution, parallelism choices, and scheduling behavior.

It focuses on:
- Architectural characteristics of common AI accelerators
- How these characteristics constrain or favor specific parallel primitives
- Execution and scheduling implications for inference DAG generation

**Out of scope**:
- Vendor marketing specifications
- Training-time accelerator optimizations

---

## Common Accelerator Characteristics

Key hardware attributes influencing inference behavior:

- **Compute throughput** (FLOPs, matrix engine utilization)
- **On-device memory capacity and bandwidth**
- **Interconnect bandwidth and latency**
- **Kernel launch and synchronization overhead**
- **Support for collective communication primitives**

Parallel strategies must align with these characteristics to be effective.

---

## Tensor Parallelism (TP) Considerations

- TP benefits accelerators with:
  - High-bandwidth intra-node interconnects
  - Efficient collective operations (all-reduce, all-gather)
- Excessive TP degrees increase synchronization overhead
- Small batch or decode-heavy workloads are sensitive to TP latency

Guideline:
- Prefer TP within a single node or tightly coupled accelerator group

---

## Pipeline Parallelism (PP) Considerations

- PP effectiveness depends on:
  - Balanced stage compute times
  - Low-latency stage-to-stage communication
- Accelerators with high kernel launch overhead may suffer from fine-grained PP
- Coarse-grained PP stages are more robust across diverse hardware

Guideline:
- Use PP to scale model size, not to reduce single-token latency

---

## Sequence Parallelism (SP) Considerations

- SP is prefill-only and compute-heavy
- Best suited for accelerators with:
  - High memory bandwidth
  - Efficient parallel token processing
- Communication cost for token aggregation must be amortized

Guideline:
- Apply SP only when sequence length is large enough to justify overhead

---

## Expert Parallelism (EP) Considerations

- EP relies on sparse activation and routing efficiency
- Accelerators must handle:
  - Irregular communication patterns
  - Dynamic workload distribution
- Load imbalance can reduce utilization

Guideline:
- Co-locate frequently used experts and minimize cross-node routing

---

## Data Parallelism (DP) Considerations

- DP is the most hardware-agnostic primitive
- Requires minimal synchronization during inference
- Scales well across heterogeneous accelerator clusters

Guideline:
- Use DP to scale throughput across nodes and accelerator types

---

## Decode-Specific Accelerator Sensitivities

- Decode phase is latency-critical
- Accelerators with:
  - High synchronization latency
  - Slow collective operations
  - Limited kernel fusion

may underperform despite high peak FLOPs

Guideline:
- Optimize for low-latency execution paths in decode
- Avoid excessive cross-device communication

---

## Failure-Prone Assumptions

- Assuming peak FLOPs directly translate to inference speed
- Ignoring synchronization and kernel launch overhead
- Applying training-optimized parallelism to inference workloads
- Overestimating benefits of TP or PP on latency-sensitive hardware

---

## Summary

Accelerator-specific characteristics shape **which parallel strategies are viable and efficient**:

- TP favors tightly coupled, low-latency accelerators
- PP scales model size but increases latency
- SP benefits long-sequence prefill
- EP requires careful expert placement
- DP provides robust throughput scaling

This module ensures **parallel strategy DAGs remain realistic under real accelerator constraints**.