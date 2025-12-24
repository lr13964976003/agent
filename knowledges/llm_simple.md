# LLM Inference Parallelism — Minimal Knowledge Core

> **Purpose**: Minimal, machine-consumable knowledge base for an **LLM Inference Parallel Strategy DAG Generation Agent**.
>
> **Scope**: Inference only. Training semantics are excluded.
>
> **Design Goal**: Prevent structural and semantic errors in automatically generated DAGs.

---

## 1. Inference Phase (Top-Level Split)

Inference **must** be split into two semantically independent branches in the DAG:

### 1.1 Prefill

- Long sequence input
- GEMM-dominated, high arithmetic intensity
- Full self-attention
- KV Cache is fully constructed
- Effective parallelism:
  - TP / PP / EP / SP

### 1.2 Decode

- Single token (or very small token batch)
- Strong temporal dependency
- Memory- and communication-bound
- Incremental causal attention
- Parallelism constraints:
  - Strict serialization
  - SP is largely ineffective

> DAG generation must explicitly distinguish **Prefill** and **Decode**.

---

## 2. Parallelism Primitives (Inference Only)

| Strategy | Partition Dimension | DAG Semantics |
|---------|--------------------|---------------|
| TP | Tensor / hidden dimension | Operator-level parallelism |
| SP | Sequence / tokens | Intra-layer token sharding |
| PP | Transformer layers | Stage-level serialization |
| EP | MoE experts | Sparse dispatch + aggregation |
| DP* | Requests / batches | DAG outer wrapper only |

> **DP Note**: DP is request-level concurrency and is not modeled as explicit communication inside the DAG.

---

## 3. Core Semantics (Hard Rules)

### 3.1 Tensor Parallelism (TP)

- Scope: Linear, Attention, Expert-internal ops
- Communication: All-Reduce, All-Gather
- DAG rule:
  - One logical compute node
  - Implicit or explicit synchronization inside the node

### 3.2 Sequence Parallelism (SP)

- Scope: Attention, normalization
- Effective mainly in Prefill
- Communication: All-Gather, Reduce-Scatter
- DAG rule:
  - Compute node is split into token shards
  - Explicit synchronization edges are required

### 3.3 Pipeline Parallelism (PP)

- Scope: Transformer blocks
- Enforces strict execution order
- Decode stage has severe pipeline bubbles
- DAG rule:
  - Explicit directed edges between stages
  - No cross-stage shortcuts

### 3.4 Expert Parallelism (EP / MoE)

- Sparse token-to-expert execution
- Communication-heavy
- DAG must decompose EP into:
  1. Routing
  2. Expert Compute (fan-out parallelism)
  3. Combine
- Routing and Combine must be explicit communication nodes

---

## 4. Parallelism Composition Hierarchy

Parallel dimensions must obey the following **non-invertible hierarchy**:

```
PP  →  SP  →  EP  →  TP
```

- PP defines global layer order
- SP partitions tokens within a layer
- EP routes tokens to experts
- TP accelerates inner-kernel computation

---

## 5. Valid Strategy Combinations (Closed Set)

The DAG Generation Agent **must only** use the following combinations:

### Single
- TP
- SP
- PP
- EP

### Two-Way
- TP × SP
- TP × PP
- TP × EP
- SP × PP
- SP × EP
- PP × EP

### Three-Way
- TP × SP × PP
- TP × SP × EP
- TP × PP × EP
- SP × PP × EP

### Four-Way (Full)
- TP × SP × PP × EP

> Introducing undefined parallel dimensions is not allowed.

---

## 6. Decode Phase — Mandatory Constraints

A Decode DAG **must** satisfy all of the following:

1. Strict temporal ordering
   - Token *t+1* depends on completion of token *t*
2. Explicit KV Cache dependencies
3. No implicit communication assumptions
4. No cross-token parallelism

> A Decode DAG represents **execution of a single token**.

---

## 7. DAG Primitive Set (Minimal Complete)

### Node Types

- Compute: Linear, Attention, Expert
- Communication: All-Reduce, All-Gather, All-to-All
- Control / State: Routing, KV Cache Update

### Edge Types

- Data dependency
- Pipeline dependency
- Synchronization dependency

---

## 8. One-Line Rule (Agent Anchor)

> **Inference DAG = strictly ordered execution graph with nested parallelism, where all communication is explicit and Decode never assumes token-level parallelism.**
