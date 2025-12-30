# LLM 并行策略 DAG 表达知识库（Inference Only）

> **Purpose**  
> This document defines how **LLM inference parallel strategies** (DP / TP / PP / EP / SP) MUST be **represented, connected, and constrained** when generating a **Parallel Strategy DAG**.
>
> It serves as **knowledge input** for a *DAG-generation Agent*, ensuring that:
> - Each parallel strategy is mapped to the **correct DAG abstraction**
> - Strategy combinations are **structurally valid**
> - Invalid multiplicative or cross-scope mappings are avoided

---

## 1. Core Design Principle

Parallel strategy DAGs MUST represent **structural execution relationships**, not mathematical parallel degrees.

> A valid DAG encodes *where* and *how* parallelism applies in the model execution graph, not how many times it is multiplied.

---

## 2. Canonical DAG Abstraction Layers

A Parallel Strategy DAG MUST be constructed using **three conceptual layers**:

```
[Request Level]
      ↓
[Model Structural Level]
      ↓
[Operator / Expert Level]
```

Each parallel strategy is restricted to exactly one layer.

---

## 3. Mapping Parallel Strategies to DAG Layers

### 3.1 Data Parallel (DP) → Request-Level Fan-Out

**DAG Representation**:
- DP is represented as a **fan-out** at the DAG root
- Each branch corresponds to an independent request or batch replica

```
Request
  ├── Replica 0
  ├── Replica 1
  └── Replica N
```

**Rules**:
- DP nodes MUST NOT split model layers or operators
- DP branches MUST be independent sub-DAGs
- DP MUST appear only at the DAG root

---

### 3.2 Pipeline Parallel (PP) → Sequential Stage Chain

**DAG Representation**:
- PP is represented as a **linear chain of stages**
- Each stage owns a contiguous block of layers

```
Stage 0 → Stage 1 → Stage 2 → Stage K
```

**Rules**:
- PP nodes MUST be sequentially connected
- PP stages MUST NOT form parallel branches
- Each PP stage may contain TP / EP subgraphs

---

### 3.3 Tensor Parallel (TP) → Intra-Operator Parallel Group

**DAG Representation**:
- TP is represented as a **parallel subgraph inside an operator node**
- All TP branches must re-converge

```
Attention
  ├── TP shard 0
  ├── TP shard 1
  └── TP shard k
        ↓
     AllReduce / Sync
```

**Rules**:
- TP MUST be scoped inside Attention or FFN nodes
- TP branches MUST rejoin before the next layer
- TP MUST NOT span across layers or PP stages

---

### 3.4 Expert Parallel (EP) → Expert Selection Fan-Out

**DAG Representation**:
- EP is represented as a **router-driven conditional fan-out**
- Each branch maps to exactly one Expert node

```
Router
  ├── Expert 0 (GPU 0)
  ├── Expert 1 (GPU 1)
  └── Expert N (GPU N)
```

**Inference Assumption (Mandatory)**:
```
Number of Expert nodes = Number of GPUs
```

**Rules**:
- EP branches MUST NOT rejoin until after expert computation
- TP inside Expert nodes is DISALLOWED by default
- EP MUST appear only inside MoE layers

---

### 3.5 Sequence Parallel (SP) → Sequence-Dimension Split

**DAG Representation**:
- SP is represented as a **parallel split on the sequence dimension**
- Typically embedded inside Attention nodes

```
Attention
  ├── Seq shard 0
  ├── Seq shard 1
  └── Seq shard k
```

**Rules**:
- SP MUST be scoped to Attention
- SP MUST NOT exist independently of Attention
- SP may coexist with TP inside the same operator

---

## 4. Valid Strategy Composition Patterns in DAGs

### 4.1 DP + PP

```
DP Fan-Out
  └── Stage 0 → Stage 1 → Stage 2
```

- DP at root
- PP inside each DP replica

---

### 4.2 PP + TP

```
Stage i
  └── Attention
        ├── TP shard 0
        ├── TP shard 1
        └── TP shard k
```

- TP fully contained inside a PP stage

---

### 4.3 PP + EP (MoE)

```
Stage j
  └── MoE Layer
        └── Router
              ├── Expert 0
              ├── Expert 1
              └── Expert N
```

- EP scoped to MoE layers
- Experts mapped 1:1 to GPUs

---

### 4.4 TP + SP (Attention Only)

```
Attention
  ├── (TP, Seq) shard 0
  ├── (TP, Seq) shard 1
  └── (TP, Seq) shard k
```

- Both operate at operator level
- Must rejoin before next layer

---

## 5. Invalid DAG Patterns (Must Be Rejected)

The DAG-generation Agent MUST reject DAGs that:

- Place TP or SP at the DAG root
- Apply TP across multiple layers
- Nest EP inside TP shards
- Treat EP as a replication factor
- Branch PP stages in parallel

---

## 6. Mandatory DAG Construction Order

When generating a DAG, the Agent MUST follow this order:

1. Create DP fan-out (if any)
2. Build PP stage chain
3. Insert operator-level parallelism (TP / SP)
4. Insert EP subgraphs inside MoE layers only

Any DAG violating this order is invalid.

---

## 7. Canonical Node Types for DAG Generation

The Agent SHOULD use the following node abstractions:

- `RequestNode`
- `PipelineStageNode`
- `AttentionNode`
- `FFNNode`
- `MoENode`
- `RouterNode`
- `ExpertNode`
- `AllReduceNode`

---

## 8. Summary Constraints (Hard Rules)

- DP = request-level fan-out
- PP = sequential stage chain
- TP = intra-operator parallel subgraph
- EP = expert fan-out mapped to GPUs
- SP = sequence-dimension split inside Attention

> A correct DAG reflects **where parallelism happens**, not how many times it is multiplied.

---

**End of DAG Knowledge File**