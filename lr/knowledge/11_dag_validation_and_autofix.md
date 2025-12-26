# DAG Validation and Auto-Fix

## Purpose

This document defines the **validation rules, legality checks, and automatic repair mechanisms** applied to inference execution DAGs.

Its goal is to ensure that any DAG passed to runtime is:
- Semantically correct
- Deadlock-free
- Compatible with hardware and phase constraints

This module operationalizes knowledge from:
- DAG primitives (08)
- Metrics and objectives (09)
- Common failure modes (10)

---

## Validation Pipeline Overview

DAG validation proceeds in ordered stages:

1. Structural validation
2. Semantic validation
3. Parallelism legality checks
4. Resource feasibility checks
5. Objective consistency checks

Failure at any stage triggers **auto-fix** or **strategy rejection**.

---

## 1. Structural Validation

### Acyclicity Check

**Rule**:
- DAG must be strictly acyclic

**Detection**:
- Topological sort failure

**Auto-Fix**:
- None (hard rejection)

---

### Node Connectivity

**Rule**:
- Every non-input node must have at least one incoming edge

**Detection**:
- Isolated subgraphs

**Auto-Fix**:
- Remove unreachable subgraphs

---

## 2. Semantic Validation

### Phase Boundary Enforcement

**Rule**:
- No edges allowed across Prefill → Decode without control nodes

**Detection**:
- Cross-phase edge scan

**Auto-Fix**:
- Insert PrefillEnd / DecodeBegin nodes

---

### KV Cache Correctness

**Rule**:
- Each KVRead must depend on the corresponding KVWrite

**Detection**:
- Step index analysis

**Auto-Fix**:
- Insert missing dependency edges

---

## 3. Parallelism Legality Checks

### Tensor Parallel Consistency

**Rule**:
- All TP ranks must participate in required collectives

**Detection**:
- Collective group mismatch

**Auto-Fix**:
- Reduce TP degree or collapse to single rank

---

### Sequence Parallel Restrictions

**Rule**:
- SP forbidden in decode phase

**Detection**:
- SP nodes inside decode subgraph

**Auto-Fix**:
- Remove SP and rebalance TP

---

### Expert Parallel Mapping

**Rule**:
- Experts must map to disjoint or explicitly shared devices

**Detection**:
- Overlapping expert-device assignments

**Auto-Fix**:
- Repartition experts or serialize execution

---

## 4. Resource Feasibility Checks

### Memory Fit Check

**Rule**:
- Peak memory per device ≤ available HBM

**Detection**:
- Static memory estimation

**Auto-Fix**:
- Reduce batch size
- Reduce KV cache length
- Lower parallel degree

---

### Communication Feasibility

**Rule**:
- Collective volume must fit topology bandwidth

**Detection**:
- Communication cost model

**Auto-Fix**:
- Switch to smaller TP groups
- Favor pipeline parallelism

---

## 5. Objective Consistency Checks

### Latency Objective Alignment

**Rule**:
- Latency-optimized DAGs must minimize critical path

**Detection**:
- Excessive synchronization on critical path

**Auto-Fix**:
- Move collectives off critical path
- Reduce microbatching depth

---

### Throughput Objective Alignment

**Rule**:
- Throughput-optimized DAGs must saturate devices

**Detection**:
- Large idle regions in pipeline

**Auto-Fix**:
- Increase microbatch count
- Rebalance pipeline stages

---

## Auto-Fix Strategy Hierarchy

Auto-fix actions follow a strict preference order:

1. Insert missing dependencies or barriers
2. Reduce parallelism degree
3. Change parallelism type (TP → PP → single-device)
4. Reduce workload parameters

If all fixes fail, the DAG is rejected.

---

## Validation Outcome States

- **Valid**: DAG accepted for execution
- **Fixed**: DAG modified and accepted
- **Rejected**: No legal fix available

All outcomes must be logged with diagnostics.

---

## Summary

This module converts static knowledge into **enforceable system rules**.

It enables:
- Correct-by-construction DAG generation
- Automatic recovery from common design errors
- Robust deployment in heterogeneous environments

With this layer, the Agent becomes **production-grade**.