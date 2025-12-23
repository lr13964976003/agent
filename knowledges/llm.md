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
| SP | Sequence Parallelism | Sequence length / tokens | Reduce activation & KV memory |
| DP* | Data / Request Parallelism | Requests / batches | Increase throughput |

> Note: During inference, DP typically manifests as **request-level or batch-level parallelism** without explicit gradient synchronization. In DAGs, it is usually modeled as the **outermost parallel dimension**, not as communication-heavy nodes.

---
-----|-----------|---------------------|-------------------|
| TP | Tensor Parallelism | Tensor / matrix dimensions | Accelerate single-layer compute |
| PP | Pipeline Parallelism | Transformer layers | Improve device utilization |
| EP | Expert Parallelism | MoE experts | Scale expert capacity |
| DP* | Data / Request Parallelism | Requests / batches | Increase throughput |

> Note: During inference, DP typically manifests as **request-level or batch-level parallelism** without explicit gradient synchronization. In DAGs, it is usually modeled as the **outermost parallel dimension**, not as communication-heavy nodes.

---

## 3. Sequence Parallelism (SP)

### 3.1 Definition

Sequence Parallelism in inference refers to:

- Partitioning computation along the **sequence length / token dimension**
- Different devices process **disjoint subsets of tokens** within the same layer

SP is conceptually orthogonal to TP and PP, and is primarily introduced to **reduce activation memory footprint and KV cache pressure** rather than raw compute latency.

### 3.2 SP Applicability in Inference

- Most effective in:
  - Prefill phase with long sequences
  - Attention and normalization layers
- Limited effectiveness in:
  - Decode phase (single-token execution)

### 3.3 SP Communication Semantics

- Typical communication patterns:
  - All-Gather (to assemble full sequence representations)
  - Reduce-Scatter (for backward-equivalent aggregation semantics; inference variant may simplify)
- Communication boundaries:
  - Attention score computation
  - LayerNorm / RMSNorm boundaries

### 3.4 SP Modeling in DAGs

- SP is **intra-layer, sequence-dimension parallelism**
- In DAG representation:
  - A logical compute node is split into multiple sequence shards
  - Shards may execute in parallel
  - Synchronization edges are required where full-sequence visibility is needed

---

## 4. Tensor Parallelism (TP)


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

## 6. Parallel Strategy Composition Rules (Complete Enumeration)

This section provides a **complete and explicit enumeration of inference-time parallel strategy combinations**, including cases **with and without Sequence Parallelism (SP)**.

The purpose is to enable the DAG Generation Agent to:
- Reason about **which combinations are valid**
- Understand **semantic hierarchy and constraints** for each combination
- Avoid implicit assumptions or missing parallel dimensions

The parallel strategy set considered is:

- TP: Tensor Parallelism
- SP: Sequence Parallelism
- PP: Pipeline Parallelism
- EP: Expert Parallelism (MoE)

---

### 6.1 Single-Strategy Configurations

These configurations involve exactly one parallel strategy.

#### 6.1.1 TP

- Scope: intra-layer tensor partitioning
- Typical usage: dense Transformer inference
- DAG characteristics:
  - Compute nodes mapped to multiple TP ranks
  - Implicit synchronization via All-Reduce / All-Gather

#### 6.1.2 SP

- Scope: sequence / token dimension partitioning
- Typical usage: long-context prefill
- DAG characteristics:
  - Parallel token shards
  - Explicit synchronization when full-sequence visibility is required

#### 6.1.3 PP

- Scope: layer-wise partitioning
- Typical usage: large models exceeding single-device memory
- DAG characteristics:
  - Strictly ordered pipeline stages
  - Directed inter-stage edges

#### 6.1.4 EP

- Scope: expert-wise partitioning (MoE)
- Typical usage: sparse expert models
- DAG characteristics:
  - Routing → Expert Compute → Combine
  - Explicit All-to-All communication

---

### 6.2 Two-Strategy Combinations

All valid pairwise combinations are listed below.

#### 6.2.1 TP × SP

- Semantics:
  - TP splits hidden / tensor dimensions
  - SP splits sequence / tokens
- Typical usage:
  - Attention-heavy prefill
- DAG notes:
  - Orthogonal parallel dimensions
  - Synchronization required at attention boundaries

#### 6.2.2 TP × PP

- Semantics:
  - PP assigns layers to stages
  - TP accelerates per-layer compute
- Typical usage:
  - Standard large-model inference
- DAG notes:
  - Hierarchy: PP → TP

#### 6.2.3 TP × EP

- Semantics:
  - EP distributes experts
  - TP applies inside each expert
- Typical usage:
  - MoE inference
- DAG notes:
  - Routing and dispatch remain non-TP

#### 6.2.4 SP × PP

- Semantics:
  - PP splits layers
  - SP partitions tokens within each stage
- Typical usage:
  - Long-context pipeline prefill
- DAG notes:
  - SP synchronization is stage-local

#### 6.2.5 SP × EP

- Semantics:
  - SP partitions tokens
  - EP routes tokens to experts
- Typical usage:
  - Long-sequence MoE models
- DAG notes:
  - Routing requires full or stage-local token visibility

#### 6.2.6 PP × EP

- Semantics:
  - MoE layers form pipeline stages
- Typical usage:
  - Large MoE models with pipeline parallelism
- DAG notes:
  - EP communication is contained within a PP stage

---

### 6.3 Three-Strategy Combinations

All valid three-way combinations are listed below.

#### 6.3.1 TP × SP × PP

- Semantics:
  - PP: layer partitioning
  - SP + TP: per-layer parallel execution
- Typical usage:
  - Long-context dense models
- DAG notes:
  - Explicit hierarchy: PP → (SP × TP)

#### 6.3.2 TP × SP × EP

- Semantics:
  - SP: token partitioning
  - EP: expert distribution
  - TP: expert-internal acceleration
- Typical usage:
  - Large MoE prefill
- DAG notes:
  - SP synchronization before routing and after combine

#### 6.3.3 TP × PP × EP

- Semantics:
  - PP: layer order
  - EP: sparse experts
  - TP: per-expert compute
- Typical usage:
  - Production MoE inference
- DAG notes:
  - EP is stage-local within PP

#### 6.3.4 SP × PP × EP

- Semantics:
  - PP splits layers
  - SP partitions tokens
  - EP routes tokens
- Typical usage:
  - Long-context MoE models
- DAG notes:
  - Careful ordering of SP sync and routing required

---

### 6.4 Four-Strategy Combination (Full)

#### 6.4.1 TP × SP × PP × EP

- Semantics:
  - PP: global layer ordering
  - SP: token-level partitioning
  - TP: tensor-level partitioning
  - EP: expert-level partitioning
- Typical usage:
  - Extremely large-scale MoE serving systems
- DAG constraints:
  - Strict hierarchy: PP → SP → EP → TP
  - Decode phase severely limits SP effectiveness
  - All communication must be explicit

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

