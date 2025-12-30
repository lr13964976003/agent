# Hard Constraints: LLM Inference Parallelism Rules

> **Purpose**  
> This document defines **non-negotiable hard constraints** for an Agent that generates **LLM inference parallelism deployment plans**.  
> The Agent **must** follow these rules to avoid invalid, non-physical, or mechanically multiplicative parallel strategies.

---

## 1. GPU Count Is **Not** a Simple Product

You **MUST NOT** assume:

```text
GPU_total = DP × TP × PP × EP × SP
```

Parallel strategies operate on **different structural domains** of the model and are **not multiplicative by default**.

GPU allocation **MUST** be derived from **model–structure mapping**, not arithmetic composition.

Only the **parallel dimensions that take effect simultaneously within the same operator phase** need to be multiplied.

---

## 2. Structural Scope of Each Parallel Strategy

You **MUST** respect the following semantic scopes:

### 2.1 Data Parallel (DP)

- Operates *outside* the model
- Parallelizes **requests or batches**
- Does **NOT** split model parameters
- Does **NOT** reduce single-request latency

Inference role:
- Throughput / QPS scaling only

---

### 2.2 Tensor Parallel (TP)

- Operates *inside a single layer*
- Primarily applies to:
  - **Attention (QKV / Heads)**
  - **FFN (Hidden Dimension)**
- Does **NOT** cover:
  - Embeddings
  - LayerNorm
  - Routing
  - Entire model graph

Key rule:
- TP is **operator-level parallelism**, not model-level parallelism

---

### 2.3 Pipeline Parallel (PP)

- Operates on the **layer dimension**
- Splits the model into sequential pipeline stages
- Each stage owns a contiguous block of layers

Key rule:
- PP is an **outer structural parallelism**

---

### 2.4 Expert Parallel (EP / MoE Parallel)

- Operates on **Experts**, not layers or tensors
- Experts have **independent parameters**
- Common inference deployment:

```text
EP ≈ GPU_total
```

Meaning:
- One GPU hosts exactly one Expert
- No tensor parallelism inside an Expert by default
- EP directly consumes GPU resources

---

### 2.5 Sequence Parallel (SP)

- Operates on the **sequence-length dimension**
- Applies **only inside Attention**
- Typically coupled with TP or specialized attention algorithms

---

## 3. Mandatory MoE Inference Assumptions

For inference workloads with MoE, you **MUST** assume:

```text
EP ≈ GPU_total
```

Consequences:
- GPU allocation is **dominated by EP**
- EP is **not multiplied** with TP or PP
- Experts are mapped directly to GPUs

---

## 4. TP and EP Are **Not** Multiplicative

You **MUST NOT** multiply TP and EP.

Reason:
- TP applies to **Attention / FFN operators**
- EP applies to **Expert instances**
- They occur in **different submodules**, not the same computational region

Correct reasoning:
- TP groups execute Attention/FFN
- EP assigns Experts to GPUs during MoE layers

---

## 5. PP Is an Outer Structure, Not a Replication Factor

Pipeline Parallelism rules:

- PP splits layers into pipeline stages
- Each stage may internally use TP or EP
- PP does **NOT** replicate TP or EP groups

Correct hierarchy:

```text
PP (layer split)
  └── TP / EP (inside each stage)
```

---

## 6. Mandatory Reasoning Order (Non-Negotiable)

When generating any deployment plan, the Agent **MUST** follow this exact order:

1. **Identify model structure**
   - Number of layers
   - Attention configuration
   - Presence of MoE

2. **Decide structural parallelism**
   - PP for memory constraints
   - EP for MoE inference

3. **Decide operator-level parallelism**
   - TP / SP for Attention and FFN

4. **Decide DP last**
   - Only for request-level concurrency

Any plan violating this order is **invalid**.

---

## 7. Invalid Plan Patterns (Must Be Rejected)

The Agent **MUST** reject plans that:

- Mechanically multiply all parallel degrees
- Apply TP to the entire model
- Apply TP inside individual Experts by default
- Treat DP as a latency optimization
- Use PP without respecting layer structure

---

## 8. Core Principle (Do Not Violate)

Parallel strategy design is a **structural mapping problem**, not a numerical optimization problem.

- TP ≈ Attention / FFN operator parallelism
- PP ≈ Layer-level structural partitioning
- EP ≈ Expert-to-GPU mapping
- DP ≈ Request-level concurrency

> **GPU count is determined by the outermost structural parallelism, not by multiplying all parallel degrees.**

---

**End of Hard Constraints**

