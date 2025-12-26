# DAG Construction Primitives

## Scope

This document defines the **primitive building blocks used to construct execution DAGs** for decoder-only Transformer inference.

It focuses on:
- Canonical node types
- Dependency semantics between nodes
- How parallel primitives (TP/SP/PP/EP/DP) are expressed in DAG form

This document serves as the **formal vocabulary** for DAG generation agents.

---

## Core DAG Concepts

- **Node**: An atomic executable unit (compute or communication)
- **Edge**: A strict dependency relationship (must-complete-before)
- **Subgraph**: A reusable pattern representing a module or phase
- **Critical Path**: Longest dependency chain determining latency

All DAGs must be **acyclic, deterministic, and phase-consistent**.

---

## Canonical Node Types

### Compute Nodes

- `Linear`
- `Attention`
- `Softmax`
- `Activation`
- `Embedding`
- `LMHead`
- `ExpertFFN`

Properties:
- Bound to a device or device group
- Consume and produce tensors

---

### Communication Nodes

- `AllReduce`
- `AllGather`
- `Scatter`
- `Gather`
- `Send`
- `Recv`

Properties:
- Represent explicit synchronization
- Must specify source and destination device sets

---

### Cache Mutation Nodes

- `KVWrite`
- `KVRead`

Properties:
- Strict ordering constraints
- Phase-sensitive semantics

---

## Phase Control Nodes

- `PrefillBegin / PrefillEnd`
- `DecodeStepBegin / DecodeStepEnd`

Used to:
- Prevent illegal cross-phase edges
- Enforce decode step serialization

---

## Parallelism Encoding Rules

### Tensor Parallelism (TP)

- Compute nodes replicated across TP ranks
- Communication nodes inserted for all-reduce/all-gather
- Edges enforce collective completion

### Sequence Parallelism (SP)

- Compute nodes partitioned along token dimension
- Aggregation nodes inserted before attention/KV usage
- SP nodes forbidden in decode subgraph

### Pipeline Parallelism (PP)

- Layers grouped into stage subgraphs
- `Send/Recv` nodes inserted between stages
- Stage boundaries define hard synchronization points

### Expert Parallelism (EP)

- `Scatter` nodes route tokens to experts
- Expert compute nodes executed conditionally
- `Gather` nodes merge expert outputs

### Data Parallelism (DP)

- Entire subgraphs replicated per DP rank
- Only output aggregation nodes added if required
- No internal DAG coupling between DP replicas

---

## Dependency Semantics

- Data dependencies must be explicit edges
- Communication nodes cannot be bypassed
- KVRead must depend on prior KVWrite
- DecodeStep(n+1) must depend on DecodeStep(n)

---

## Invalid DAG Patterns

- Cycles introduced by overlapping decode steps
- Missing synchronization after collective ops
- Cross-phase edges (prefill → decode without boundary)
- Shared KV cache access without ordering

---

## Summary

DAG construction primitives provide:
- A minimal, explicit vocabulary for inference execution
- Clear encoding of TP/SP/PP/EP/DP semantics
- Deterministic dependency enforcement

This module enables **automatic, correct-by-construction inference DAG generation**.