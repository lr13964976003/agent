# 05 Parallel Strategy DAG Rules

## Scope

This document defines the **rules and constraints for constructing valid parallel strategy DAGs** for decoder-only Transformer inference, combining all primitives (TP, SP, PP, EP) with execution phase semantics.

It focuses on:
- Composition legality of parallel primitives
- Phase-dependent constraints
- Communication and synchronization points
- DAG node and edge semantics

**Out of scope**:
- Hardware-specific scheduling
- Low-level kernel optimization
- Training-time parallelism

The goal is to provide **Agent-readable constraints** for automated DAG generation.

---

## DAG Node Definition

Each node in the DAG represents:
- A **computational unit**: a layer, a TP shard, an SP token segment, a pipeline stage, or an expert submodule
- Input dependencies: other nodes whose outputs are required before execution
- Phase tag: `prefill` or `decode`

Node properties:
- Deterministic execution
- Immutable KV Cache state per step
- Tagged with applicable parallel primitives

---

## DAG Edge Definition

Edges represent **dependency constraints**:
- **Intra-layer**: linear, attention, FFN, expert computation
- **Inter-layer**: PP layer-to-layer outputs
- **Token-to-token**: SP in prefill, causal attention
- **KV Cache dependencies**: decode step t+1 depends on step t

Edge rules:
1. **Preserve temporal order**: decode steps strictly sequential
2. **Preserve causal dependencies**: attention mask constraints enforced
3. **Preserve data integrity**: KV Cache writes must complete before read

---

## Phase-Specific DAG Rules

### Prefill Phase

- SP is allowed across tokens
- TP can be applied within layers
- PP can pipeline layers
- EP can route tokens to experts
- All edges respect causal masking and intra-step ordering

### Decode Phase

- SP is **not allowed** across steps
- TP can be applied within layer for the single token
- PP can pipeline layers **within a single token step only**
- EP applies per token per step
- Edges must preserve KV Cache and autoregressive dependencies

---

## Primitive Combination Rules

| Primitive | Prefill | Decode | Combination Constraints |
|-----------|---------|--------|-----------------------|
| TP        | Allowed | Allowed | Can be nested with SP, PP, EP within node
| SP        | Allowed | Not Allowed | Only within prefill
| PP        | Allowed | Layer-only | Must respect step boundaries
| EP        | Allowed | Allowed | Sparse per token, respects phase and routing

Rules:
- TP/SP/EP can coexist within a node if phase permits
- PP spans multiple nodes, edges must connect across stages
- Any combination violating phase or dependency is invalid

---

## Communication and Synchronization Rules

- **TP**: All-reduce/all-gather at operator boundaries
- **SP**: Communication of K/V state across sequence segments
- **PP**: Activation transfer between pipeline stages
- **EP**: Scatter/gather for expert routing

Synchronization points must be included in DAG edges.

---

## DAG Construction Guidelines

1. **Start with nodes per layer**
2. **Apply TP within node**
3. **Apply SP across token dimension (prefill only)**
4. **Assign EP routing per node**
5. **Connect nodes respecting PP stage boundaries**
6. **Add edges for KV Cache and causal dependencies**
7. **Label each node with phase tag**
8. **Validate DAG against phase-specific rules**

---

## Failure-Prone Scenarios

- Cross-step SP in decode
- TP node overlapping multiple decode steps incorrectly
- EP routing ignoring KV Cache or phase constraints
- PP overlapping decode steps incorrectly
- DAG edges violating causal attention or layer order

All such scenarios must be detected and rejected.

---

## Summary

A valid Parallel Strategy DAG must:
- Encode computation nodes with phase, primitive, and KV Cache information
- Include edges representing all temporal, causal, and layer dependencies
- Respect phase-specific legality rules for SP, PP, TP, and EP
- Include all necessary communication and synchronization nodes

This ensures that any DAG generated using these rules is **semantically correct, reproducible, and safe for inference across multiple devices**.