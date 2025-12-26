# 04 Expert Parallelism (EP)

## Scope

This document defines the **semantic foundations, legality conditions, and execution constraints of Expert Parallelism (EP)** in decoder-only Transformer inference, typically used in Mixture-of-Experts (MoE) layers.

It focuses on:
- Expert assignment and routing semantics
- Interaction with KV Cache, TP, SP, and PP
- Phase-specific validity

**Out of scope**:
- Specific MoE algorithms or gating heuristics
- Hardware-specific implementation details
- Training-time routing or load balancing

EP is treated as a **sparse cross-device parallelism primitive**.

---

## Definition

Expert Parallelism partitions computation **across multiple experts** where each expert is a separate submodule (e.g., a feed-forward network). 

Key properties:
- Only a subset of experts is active per token
- Expert assignment is deterministic for a given input and gating function
- Multiple devices can own distinct experts

---

## Routing Semantics

### Deterministic Assignment

- For a given token, the gating function selects one or more experts
- The selected experts receive the token input and produce outputs
- Non-selected experts do not receive or process the token

### Batch and Sequence Considerations

- Tokens in a batch may route to different experts
- Within a sequence, token routing is independent, except for shared randomness if used (e.g., stochastic gates)

---

## Phase Sensitivity

### Prefill Phase

- All prompt tokens are available
- Routing decisions can be applied in parallel across tokens
- KV Cache is updated for all tokens post-expert computation
- EP can be combined with TP and SP within a stage

### Decode Phase

- Only one token per sequence per step
- Routing decision applies per token per step
- KV Cache grows one token at a time
- EP cannot introduce cross-step token dependencies

Consequences:
- Decode remains strictly autoregressive
- Sparse expert routing must respect single-token constraint

---

## Interaction with Other Primitives

- **TP:** Each expert can be tensor-parallelized independently
- **SP:** Only applicable in prefill, respecting causal masking
- **PP:** Each pipeline stage may contain multiple experts, requiring intra-stage routing communication

All combinations must preserve the correctness of KV Cache and causal dependencies.

---

## Communication Semantics

- Tokens routed to different devices require **all-to-all or selective scatter/gather** operations
- Communication is synchronous within a step
- Communication cost scales with **number of active tokens × number of experts per token**

---

## Failure-Prone Assumptions

Invalid assumptions include:
- EP can parallelize across decode steps
- Tokens can be processed by unassigned experts
- Routing can ignore phase semantics (prefill vs decode)
- EP changes KV Cache or attention semantics

EP only affects **which expert modules process a token**, not the fundamental temporal or layer-wise dependencies.

---

## Role in Parallel Strategy Composition

Expert Parallelism:
- Introduces sparse, selective computation
- Interacts with TP/SP/PP to maximize device utilization
- Must respect all phase and step constraints

It represents the **most complex sparse dimension** in parallel inference DAGs.

---

## Summary

EP semantics can be summarized as follows:
- Partitions computation across experts, sparse per token
- Phase-sensitive: full parallelism in prefill, restricted in decode
- Must respect causal and KV Cache dependencies
- Combined with TP/SP/PP within constraints

EP is essential for **MoE-based inference DAG reasoning** and forms a critical component for safe, multi-device, sparse parallel inference.