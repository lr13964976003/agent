# LLM Parallelism Knowledge Base

> This document serves as the **core knowledge base for an Agent that generates DAGs (Directed Acyclic Graphs) of LLM parallelization strategies**. It provides a structured, composable, and graph-oriented description of mainstream parallel strategies used in large language model (LLM) training and inference.

---

## 1. Objectives and Constraints of Parallelism

### 1.1 Primary Objectives

- **Scale model size** beyond single-device memory limits
- **Reduce end-to-end latency**, especially in inference (Decode phase)
- **Increase system throughput** (tokens / requests per unit time)
- **Improve hardware utilization** via compute–communication overlap

### 1.2 Key Constraints

- Device memory (Parameters / Activations / KV Cache)
- Interconnect bandwidth and topology (NVLink / HCCS / PCIe / IB)
- Synchronization cost (All-Reduce / All-Gather / All-to-All)
- Sequential dependencies (layer-wise and token-wise)

---

## 2. LLM Execution Phases (DAG Foundation)

### 2.1 High-Level Phase Decomposition

```text
Request
  ├─ Prefill Phase (Context Encoding)
  └─ Decode Phase (Autoregressive Generation)
```

- **Prefill**: sequence-level parallelism, compute intensive
- **Decode**: token-by-token generation, latency and communication sensitive

### 2.2 Transformer Layer Structure

```text
Input
 ├─ Attention
 │   ├─ QKV Projection
 │   ├─ Attention Score
 │   └─ Context Projection
 ├─ MLP / MoE
 └─ Residual + Norm
```

This structure is the atomic unit for parallel decomposition.

---

## 3. Data Parallelism (DP)

### 3.1 Definition

- Different devices process **different data samples**
- Model parameters are **fully replicated**

### 3.2 Characteristics

- Parallel granularity: batch / request level
- Primary communication: gradient All-Reduce (training)
- In inference, commonly used for **request-level parallelism**

### 3.3 DAG Representation

- Multiple replicated subgraphs
- Explicit gradient synchronization nodes (training)

---

## 4. Tensor Parallelism (TP)

### 4.1 Definition

- Partition tensors **within a single operator**
- Common split dimensions:
  - Linear weight matrices (column / row parallel)
  - Attention heads or hidden dimensions

### 4.2 Typical TP Partitioning

| Module | Partition | Communication |
|------|-----------|---------------|
| QKV Projection | Head / Hidden split | All-Gather / Reduce-Scatter |
| Attention Output | Row parallel | All-Reduce |
| FFN / MLP | Column + Row | All-Reduce |

### 4.3 DAG Features

- Operator-level compute nodes
- Explicit collective communication nodes
- Communication depends on compute completion

---

## 5. Pipeline Parallelism (PP)

### 5.1 Definition

- Partition the model along the **layer dimension**
- Each device is responsible for a stage of layers

### 5.2 Prefill vs Decode Behavior

| Phase | Characteristics |
|------|-----------------|
| Prefill | Micro-batch pipelining possible |
| Decode | Strong sequential dependency, limited pipelining |

### 5.3 DAG Representation

- Sequential stage subgraphs
- Explicit activation Send/Recv nodes
- Decode DAG is strictly time-ordered

---

## 6. Expert Parallelism (EP, MoE)

### 6.1 Definition

- Different devices host **different experts**
- Each token activates Top-K experts

### 6.2 Core Operators

```text
Input
 ├─ Router (Top-K Selection)
 ├─ Dispatch (All-to-All)
 ├─ Expert FFN
 └─ Combine (All-to-All)
```

### 6.3 DAG Features

- Dynamic execution paths determined by Router
- Dispatch / Combine as dominant communication nodes
- Highly parallel expert subgraphs

---

## 7. Sequence Parallelism (SP)

### 7.1 Definition

- Partition computation along the **sequence length dimension**
- Commonly applied in Attention to reduce activation and KV pressure

### 7.2 Typical Use Cases

- Long-context inference
- Combined with TP (e.g., Ring Attention)

### 7.3 DAG Representation

- Token-block subgraphs
- Local attention compute + ring-based communication

---

## 8. Hybrid Parallelism

### 8.1 Common Combinations

| Combination | Description |
|------------|-------------|
| DP + TP | Baseline scalable setup |
| TP + PP | Mainstream large-model inference |
| TP + EP | Mandatory for MoE |
| DP + TP + PP + EP | Ultra-large-scale training |

### 8.2 DAG Modeling Principles

- **Orthogonal dimensions**: each strategy splits a different axis
- **Explicit communication**: communication must be nodes
- **Phase separation**: Prefill and Decode modeled independently

---

## 9. Inference-Oriented DAG Differences

### 9.1 Prefill DAG

- High parallelism
- Batched operators
- Communication often hidden by computation

### 9.2 Decode DAG

- Token-level temporal dependency
- Explicit KV Cache read/write nodes
- Communication lies on the critical path

---

## 10. Key Knowledge for DAG-Generating Agents

### 10.1 Node Types

- Compute Nodes (MatMul, Attention, FFN)
- Communication Nodes (All-Reduce, All-Gather, All-to-All, Send/Recv)
- Control Nodes (Router, Barrier)
- Memory Nodes (KV Cache Read / Write)

### 10.2 Edge Types

- Data Dependency
- Temporal Dependency
- Communication Dependency

### 10.3 Suggested Generation Rules

1. Split subgraphs by **phase** (Prefill / Decode)
2. Expand hierarchically: **Layer → Module → Operator**
3. Parallel strategies define **partitioning rules over node sets**
4. All cross-device dependencies must introduce communication nodes

---

## 11. Applicability

- LLM training and inference parallel modeling
- Performance analysis and bottleneck identification
- Automated parallel strategy search (Auto-Parallel)
- Visualization, education, and documentation

---

**End of Knowledge Base**

