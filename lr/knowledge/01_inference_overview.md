# 01 Inference Overview

## Scope

This document defines the **high-level execution model of Large Language Model (LLM) inference**. It provides a unified mental model for how inference requests are processed, without prescribing any parallelization strategy.

**Out of scope**:
- Training-time behavior
- Optimizer states
- Gradient computation
- Fine-tuning or adaptation techniques

This knowledge applies strictly to **decoder-only Transformer models** during inference.

---

## High-Level Inference Flow

At a system level, LLM inference proceeds as a sequence of stages:

1. **Request intake**
2. **Batch formation**
3. **Prefill phase execution**
4. **Decode phase execution**
5. **Token emission and termination**

Each stage introduces distinct computational, memory, and scheduling characteristics that constrain downstream parallelism decisions.

---

## Requests, Sequences, and Tokens

### Request
A *request* corresponds to a single user prompt, consisting of an input token sequence and generation parameters (e.g., max tokens, stop conditions).

Key properties:
- Requests are independent at the semantic level
- Requests may arrive asynchronously
- Requests may have heterogeneous sequence lengths

---

### Sequence
A *sequence* represents the evolving token stream associated with one request.

- During inference, a sequence grows one token at a time
- Each sequence maintains its own KV Cache state
- Sequence length directly affects memory footprint and compute cost

---

### Token
A *token* is the atomic unit of inference computation.

- Prefill processes many tokens per sequence
- Decode processes exactly one new token per active sequence per step

Token-level dependencies impose strict ordering constraints during decoding.

---

## Batch Semantics

### Definition
A *batch* is a set of sequences processed together in a single inference step.

Batching is a **system-level scheduling construct**, not a model-level concept.

---

### Static vs Dynamic Batching

- **Static batching**: batch size fixed at launch time
- **Dynamic batching**: batch composition changes over time due to
  - Request arrival
  - Request completion
  - Early termination

Dynamic batching introduces variability in:
- Effective batch size per step
- Memory usage
- Compute utilization

---

### Padding and Ragged Batches

Because sequences may have different lengths:

- Padding may be introduced to align tensor shapes
- Alternatively, ragged or packed representations may be used

These choices affect both efficiency and parallelization feasibility.

---

## Prefill Phase

### Definition
The *prefill phase* processes the entire input prompt for each request to initialize model state.

Characteristics:
- Operates on full input sequences
- Initializes KV Cache
- High arithmetic intensity
- Parallelism scales with sequence length

Prefill computation is largely parallelizable across tokens and sequences.

---

## Decode Phase

### Definition
The *decode phase* generates new tokens autoregressively, one token per sequence per step.

Characteristics:
- Strict token-level dependency
- Latency-sensitive
- Reuses KV Cache from previous steps
- Parallelism primarily comes from batch size

Decode introduces inherent sequential constraints absent in prefill.

---

## Prefill vs Decode: Conceptual Contrast

| Dimension | Prefill | Decode |
|--------|--------|--------|
| Tokens per sequence | Many | One per step |
| Parallelism source | Sequence length, batch size | Batch size |
| Dependency | None across tokens | Strict autoregressive |
| Compute intensity | High | Lower per step |
| Latency sensitivity | Moderate | High |

This distinction is fundamental to all inference parallelization strategies.

---

## KV Cache Lifecycle

- KV Cache is allocated during prefill
- KV Cache is read and appended during decode
- KV Cache persists for the lifetime of a sequence

KV Cache placement and access patterns impose strong constraints on:
- Memory layout
- Device partitioning
- Communication requirements

---

## Termination Conditions

A sequence terminates when one of the following is met:
- End-of-sequence token generated
- Maximum token limit reached
- External cancellation

Termination causes:
- Batch size to shrink
- Resource reclamation opportunities
- Scheduling re-evaluation

---

## Implications for Parallelism (Non-Prescriptive)

This overview establishes several invariant facts:

- Prefill and decode phases have fundamentally different parallelism properties
- Decode introduces unavoidable sequential dependencies
- Batch size is dynamic over time
- KV Cache dominates memory behavior in long-running sequences

**No specific parallelism strategy is implied here.**
These facts serve as constraints that downstream reasoning must respect.

---

## Summary

Inference is a staged, stateful, and dynamically scheduled process.

Understanding:
- how requests become sequences,
- how sequences evolve token by token,
- and how prefill and decode differ

is a prerequisite for any correct reasoning about inference parallelism or DAG construction.

