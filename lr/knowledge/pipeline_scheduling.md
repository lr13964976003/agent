# Pipeline Scheduling

## Scope

This document defines the **execution and scheduling semantics of Pipeline Parallelism (PP) across layers and devices** in decoder-only Transformer inference.

It focuses on:
- Layer assignment to pipeline stages
- Inter-stage communication and synchronization
- Phase-specific scheduling rules (prefill vs decode)
- Integration with TP, SP, EP, and DP

**Out of scope**:
- Hardware-specific kernel scheduling
- Training-time pipeline scheduling

---

## Pipeline Stage Assignment

- Each layer is assigned to a single pipeline stage
- Each stage is mapped to one or more devices
- TP, SP, EP can be applied within a stage as allowed by phase
- DP replicates the entire pipeline across sequences

Constraints:
- Stage boundaries must be respected
- No layer can execute in multiple stages simultaneously

---

## Prefill Phase Scheduling

- Multiple tokens are available, enabling **inter-layer pipelining**
- Each stage computes its assigned layers while downstream stages begin computation as soon as previous layer outputs are ready
- Communication of intermediate activations occurs asynchronously, overlapping with computation where possible
- SP can be applied across tokens within a stage
- EP applies per token routing as needed

Characteristics:
- Stage-wise pipelining maximizes device utilization
- Latency is reduced by overlapping computation and communication
- TP within a stage further parallelizes tensor computations

---

## Decode Phase Scheduling

- Only one token per sequence per step is available
- PP can pipeline layers **within the same token step only**
- No cross-step overlap is allowed due to autoregressive dependency
- Communication of intermediate activations is synchronous between stages
- TP and EP apply within a stage as allowed
- SP is not allowed

Consequences:
- Per-step latency is limited by **the slowest stage**
- Correctness depends on strict stage-to-stage synchronization

---

## Communication and Synchronization

- Inter-stage communication occurs after a stage completes its layers for the current token(s)
- Synchronization points are required at each stage boundary
- DP does not interfere with intra-stage pipeline scheduling
- KV Cache updates occur after attention layers in each stage, respecting phase semantics

---

## Failure-Prone Assumptions

- Overlapping decode steps across stages is invalid
- Ignoring stage boundaries when applying TP/SP/EP leads to incorrect computation
- Inter-stage communication must respect tensor shapes and batch/token alignment
- DP does not alter stage dependencies

---

## Summary

Pipeline scheduling defines the **temporal and device-level ordering of layer execution**:
- Prefill: multi-token pipelining, overlapping computation and communication
- Decode: single-token, per-step pipeline, strict synchronization
- Works in combination with TP, SP (prefill only), EP, and DP

This module ensures **correct and efficient execution of pipeline parallelism** in Transformer inference DAGs.