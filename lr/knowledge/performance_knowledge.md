# LLM 并行策略性能评估知识库（Inference Only）

> **Purpose**  
> This document defines how an **LLM Parallel Strategy Performance Evaluation Agent** MUST estimate and reason about **real inference performance**.
>
> It focuses on **latency, throughput, and KV cache memory**, under combinations of DP / TP / PP / EP / SP.
>
> The goal is to prevent naive or misleading estimates and ensure performance evaluation is **structurally correct, hardware-aware, and inference-realistic**.

---

## 1. Fundamental Principle

Inference performance is determined by **critical-path execution**, not by theoretical FLOPs or parallel degrees.

> Performance evaluation MUST be based on:
> - Execution structure (DAG)
> - Communication patterns
> - Memory residency (especially KV cache)

---

## 2. Core Performance Metrics

The Agent MUST evaluate performance using **all** of the following metrics.

### 2.1 End-to-End Latency

Latency MUST be decomposed into:

```
Latency = Prefill Latency + Decode Latency
```

Where:
- **Prefill**: full-sequence attention, compute-heavy
- **Decode**: autoregressive step-by-step, communication- and memory-bound

---

### 2.2 Throughput

Throughput MUST be measured as:

```
Throughput = Tokens / Second
```

And analyzed separately for:
- Single request
- Batched requests
- DP-scaled concurrent requests

---

### 2.3 KV Cache Memory

KV cache memory MUST be explicitly estimated and constrained.

```
KV_cache_size =
  num_layers × num_heads × head_dim × 2 (K+V)
  × sequence_length × dtype_size
```

KV cache is often the **dominant memory consumer** during inference.

---

## 3. Baseline Performance Model (No Parallelism)

For a single GPU, no parallelism:

### 3.1 Prefill Latency

```
T_prefill ≈ FLOPs_prefill / GPU_compute
```

Dominated by:
- Attention O(L²)
- FFN O(L)

---

### 3.2 Decode Latency (Per Token)

```
T_decode_step ≈ T_attention + T_ffn + T_memory
```

Decode is typically:
- Memory-bandwidth bound
- KV-cache access dominated

---

## 4. Impact of Each Parallel Strategy

### 4.1 Data Parallel (DP)

**Latency**:
- No reduction in single-request latency

**Throughput**:
```
Throughput ≈ DP × single_instance_throughput
```

**KV Cache**:
- Fully replicated per DP replica

**Key Rule**:
> DP improves throughput only, never latency.

---

### 4.2 Tensor Parallel (TP)

**Prefill Latency**:
- Reduced compute per GPU
- Additional communication overhead

Approximation:
```
T_prefill_TP ≈ T_compute / TP + T_comm_allreduce
```

**Decode Latency**:
- Often worse beyond small TP due to:
  - Per-token AllReduce
  - Synchronization overhead

**KV Cache**:
- KV cache is usually **replicated across TP ranks**

**Key Tradeoff**:
> TP reduces compute but increases communication and memory pressure.

---

### 4.3 Pipeline Parallel (PP)

**Prefill Latency**:
```
T_prefill_PP ≈ sum(stage_latency) + pipeline_fill
```

**Decode Latency**:
- Strongly affected by pipeline bubbles
- Poor scaling for small batch or single request

**Throughput**:
- Improves only with sufficiently large batch size

**KV Cache**:
- KV cache is **partitioned by layers** across stages

**Key Rule**:
> PP is memory-driven, not latency-driven.

---

### 4.4 Expert Parallel (EP / MoE)

**Latency**:
- Only active experts contribute to compute
- Router overhead is non-negligible

Approximation:
```
T_moe ≈ T_router + max(T_active_experts)
```

**Throughput**:
- Scales with number of experts only if token load is balanced

**KV Cache**:
- KV cache for MoE layers exists only on GPUs hosting experts

**Critical Constraint**:
> Load imbalance dominates MoE inference performance.

---

### 4.5 Sequence Parallel (SP)

**Latency**:
- Reduces attention compute for long context
- Introduces sequence-level communication

**Throughput**:
- Improves only for large sequence length

**KV Cache**:
- KV cache is partitioned by sequence dimension

**Key Rule**:
> SP is useful only for long-context inference.

---

## 5. Combined Strategy Performance Reasoning

The Agent MUST evaluate combined strategies using **critical-path analysis**, not multiplication.

### Example: PP + TP + EP

```
Latency ≈ max_stage_latency(
  TP_compute + TP_comm + EP_compute
)
```

Not:
```
Latency ≠ base_latency / (PP × TP × EP)
```

---

## 6. Decode vs Prefill Asymmetry (Critical)

The Agent MUST distinguish:

| Phase   | Dominant Factor |
|--------|-----------------|
| Prefill | Compute (FLOPs) |
| Decode  | Memory + Comm   |

Consequences:
- TP helps prefill more than decode
- PP hurts decode more than prefill
- KV cache dominates decode scalability

---

## 7. GPU Memory Budget Model

Total GPU memory usage:

```
Memory_total =
  Model_weights
+ KV_cache
+ Activation_buffers
+ Communication_buffers
```

If memory exceeds GPU capacity, the deployment plan is invalid.

---

## 8. Mandatory Evaluation Rules (Hard Constraints)

The Performance Evaluation Agent MUST:

- Explicitly estimate KV cache size
- Separate prefill and decode latency
- Account for communication overhead
- Reject FLOPs-only performance models
- Reject latency scaling assumptions based on parallel degree multiplication

---

## 9. Summary Heuristics (Safe Defaults)

- DP → throughput scaling only
- TP → prefill acceleration, decode risk
- PP → memory relief, latency penalty
- EP → sparse compute, load imbalance risk
- SP → long-context optimization

> **Inference performance is limited by the slowest stage and the largest memory consumer, not by total FLOPs.**

---

**End of Performance Knowledge File**