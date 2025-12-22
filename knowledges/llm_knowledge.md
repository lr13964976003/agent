# LLM Inference Parallelism Knowledge Base (for Parallel Strategy DAG Generation Agent)

> This document serves as the **knowledge input** for an **Agent that automatically generates Parallel Strategy DAGs** for Large Language Model (LLM) inference.
>
> **Scope limitation**: *Inference stage only*. Training-related parallelism semantics are explicitly excluded.
>
> **Primary objective**: provide the Agent with **clear conceptual boundaries, dependency rules, communication semantics, and temporal constraints** of inference parallelism, enabling generation of **structurally correct, semantically valid, and system-realistic DAGs**.

---

## 1. Inference Phase Decomposition

LLM inference is conventionally decomposed into two sub-phases:

1. **Prefill (Context Processing)**
2. **Decode (Autoregressive Generation)**

These phases differ fundamentally in **compute intensity, communication patterns, and parallelism suitability**, and therefore must be treated as **top-level branches** in any parallel-strategy DAG.

### 1.1 Prefill Phase Characteristics

- Input: full prompt sequence (long sequence length)
- Compute characteristics:
  - High arithmetic intensity (GEMM-dominated)
  - Full self-attention
- KV cache behavior:
  - KV cache is fully constructed in this phase
- Parallelism affinity:
  - TP / PP / EP scale efficiently

### 1.2 Decode Phase Characteristics

- Input: single token or very small token batch
- Compute characteristics:
  - Low arithmetic intensity
  - Memory- and communication-bound
  - Incremental (causal) attention
- KV cache behavior:
  - Frequent KV cache reads and updates
- Parallelism affinity:
  - Latency-dominated
  - TP / PP / EP must be carefully balanced

---

## 2. Parallelism Taxonomy (Inference Only)

Common parallel strategies used during inference include:

| Strategy | Full Name | Partition Dimension | Primary Objective |
|--------|-----------|---------------------|-------------------|
| TP | Tensor Parallelism | Tensor / matrix dimensions | Accelerate single-layer compute |
| PP | Pipeline Parallelism | Transformer layers | Improve device utilization |
| EP | Expert Parallelism | MoE experts | Scale expert capacity |
| DP* | Data / Request Parallelism | Requests / batches | Increase throughput |

> Note: During inference, DP typically manifests as **request-level or batch-level parallelism** without explicit gradient synchronization. In DAGs, it is usually modeled as the **outermost parallel dimension**, not as communication-heavy nodes.

---

## 3. Tensor Parallelism (TP)

### 3.1 Definition

In inference, Tensor Parallelism refers to:

- **Intra-layer parallelism**
- Partitioning large matrix operations across multiple devices

### 3.2 TP Application Scope in Inference

- Attention modules
  - QKV projections
  - Output projection
- FFN / MLP modules
  - Linear → Activation → Linear
- MoE expert internal linear layers

### 3.3 TP Communication Semantics

- Typical collectives:
  - All-Reduce
  - All-Gather
- Communication boundaries:
  - Linear layer outputs
  - Attention output aggregation

### 3.4 TP Modeling in DAGs

- TP is **operator-level parallelism**, not layer-level
- In DAG representation:
  - A single logical compute node
  - Mapped to multiple TP ranks
  - Implicit synchronization edges inside the node

---

## 4. Pipeline Parallelism (PP)

### 4.1 Definition

Pipeline Parallelism in inference refers to:

- Partitioning **Transformer blocks** across devices
- Passing activations sequentially between pipeline stages

### 4.2 PP Forms in Inference

#### 4.2.1 Prefill Phase PP

- Micro-batching is possible
- Throughput-oriented
- Pipeline bubbles are tolerable

#### 4.2.2 Decode Phase PP (Critical Case)

- Single-token strict serialization
- Strong sequential dependency
- Prominent pipeline bubbles
- Typical optimizations:
  - Reduce number of stages
  - Combine with TP / EP to shrink per-stage latency

### 4.3 PP Modeling in DAGs

- PP introduces **explicit directed edges** between stages
- DAG constraints:
  - Strict layer order
  - No backward or cross-stage dependencies

---

## 5. Expert Parallelism (EP / MoE Parallelism)

### 5.1 Definition

Expert Parallelism is the core parallel strategy for MoE models during inference:

- Experts are distributed across devices
- Each token activates only a small subset of experts (Top-K)

### 5.2 Key EP Characteristics in Inference

- Sparse computation
- Communication-intensive
  - Token dispatch
  - Output combine

### 5.3 EP Communication Semantics

- All-to-All or logically equivalent patterns
- Communication occurs:
  - After routing
  - Before and after expert execution

### 5.4 EP Modeling in DAGs

- EP must be decomposed into three logical stages:
  1. Routing
  2. Expert Compute
  3. Combine
- Expert Compute nodes may fan out in parallel
- Dispatch and Combine must be modeled as **explicit communication nodes**

---

## 6. Parallel Strategy Composition Rules

### 6.1 TP × PP

- Most common composition
- Semantics:
  - PP defines layer ownership
  - TP defines intra-layer parallel execution

### 6.2 TP × EP

- Common in MoE inference
- TP is applied inside experts
- Routing / dispatch typically remain non-TP

### 6.3 PP × EP

- MoE layers act as pipeline stages
- EP communication is contained within the stage

### 6.4 TP × PP × EP

- Standard configuration for large-scale MoE inference
- DAG must simultaneously represent:
  - Layer ordering (PP)
  - Operator parallelism (TP)
  - Expert fan-out and fan-in (EP)

---

## 7. Decode-Phase DAG Constraints (Mandatory)

For the **Decode phase**, the generated DAG must satisfy:

1. **Strict temporal ordering**
   - Token *t+1* depends on the full completion of token *t*

2. **Explicit KV cache dependencies**
   - Attention nodes must depend on historical KV

3. **No hidden communication assumptions**
   - All-Reduce / All-to-All must be explicit nodes or edges

4. **No cross-token parallelism assumptions**
   - DAG represents the execution of a single token

---

## 8. Recommended DAG Abstraction for the Agent

### 8.1 Node Types

- Compute Nodes
  - Linear
  - Attention
  - Expert
- Communication Nodes
  - All-Reduce
  - All-Gather
  - All-to-All
- Control / State Nodes
  - Routing
  - KV Cache Update

### 8.2 Edge Semantics

- Data Dependency
- Pipeline Dependency
- Synchronization Dependency

---

## 9. Concluding Notes

- Inference parallelism is **latency-first**, not throughput-first
- A valid DAG must preserve:
  - Sequential correctness
  - Explicit communication
  - Clear parallel hierarchy

> This document is intended to serve as a **stable knowledge baseline** for Parallel Strategy DAG Generation Agents, ensuring generated DAGs align with real inference system semantics.