# Common Failure Modes

## Purpose

This document enumerates **frequent failure modes** encountered when constructing or executing inference DAGs for large language models.

It serves three goals:
- Help agents proactively avoid invalid strategies
- Provide diagnostics for observed runtime failures
- Supply prior knowledge for auto-fix and fallback logic

---

## Failure Mode Taxonomy

Failure modes are grouped into four categories:

1. Semantic violations
2. Parallelism misuse
3. Scheduling and synchronization errors
4. System-level resource failures

---

## Semantic Violations

### Cross-Phase Dependency Leakage

**Description**:
- Edges directly connecting prefill nodes to decode nodes
- KV cache written in prefill but read without phase barrier

**Symptoms**:
- Incorrect outputs
- Non-deterministic behavior

**Root Cause**:
- Missing PrefillEnd / DecodeBegin control nodes

---

### KV Cache Ordering Violations

**Description**:
- KVRead occurs before corresponding KVWrite
- Concurrent writes to the same KV slot

**Symptoms**:
- Attention attending to uninitialized values
- Silent numerical corruption

**Root Cause**:
- Improper decode step serialization

---

## Parallelism Misuse

### Over-Parallelization

**Description**:
- Excessive TP/SP applied to small batch or decode workloads

**Symptoms**:
- Higher latency despite more devices
- Communication dominating compute

**Root Cause**:
- Ignoring latency-vs-throughput objective

---

### Illegal SP Usage in Decode

**Description**:
- Applying sequence parallelism to decode tokens

**Symptoms**:
- Incorrect attention scores
- Broken KV semantics

**Root Cause**:
- Treating decode like prefill

---

### EP–TP Incompatibility

**Description**:
- Experts sharded inconsistently across TP groups

**Symptoms**:
- Mismatched tensor shapes
- Runtime collective failures

**Root Cause**:
- Missing expert-to-device mapping constraints

---

## Scheduling and Synchronization Errors

### Pipeline Bubble Amplification

**Description**:
- Poor microbatching or stage imbalance

**Symptoms**:
- Low device utilization
- Throughput collapse

**Root Cause**:
- Uneven layer distribution across PP stages

---

### Missing Synchronization After Collectives

**Description**:
- Downstream compute begins before all-reduce completion

**Symptoms**:
- Incorrect results
- Race conditions

**Root Cause**:
- DAG edges missing after communication nodes

---

### Deadlock in Multi-Stage Pipelines

**Description**:
- Cyclic send/recv dependencies across stages

**Symptoms**:
- Execution hang
- No progress in runtime

**Root Cause**:
- Inconsistent send/recv ordering

---

## System-Level Resource Failures

### Out-of-Memory (OOM)

**Description**:
- HBM exhausted by parameters + KV cache + activations

**Symptoms**:
- Runtime OOM errors
- Forced eviction or crashes

**Root Cause**:
- Excessive batch size or decode length

---

### Communication Fabric Saturation

**Description**:
- Interconnect bandwidth fully consumed

**Symptoms**:
- Latency spikes
- Collective timeouts

**Root Cause**:
- Large TP or EP collectives on weak topology

---

### Load Imbalance Across Devices

**Description**:
- Uneven token or expert distribution

**Symptoms**:
- Straggler devices on critical path

**Root Cause**:
- Skewed routing or poor partitioning

---

## Diagnostic Mapping

| Symptom | Likely Failure Mode |
|------|------------------|
| Wrong output | KV ordering / phase leakage |
| Latency regression | Over-parallelization |
| Hang | Deadlock |
| OOM | Memory misestimation |

---

## Summary

Common failure modes represent **hard-earned system knowledge**.

Encoding them explicitly enables:
- Safer automatic DAG generation
- Faster debugging
- Intelligent fallback and repair strategies

This module is a prerequisite for **DAG validation and auto-fix logic**.