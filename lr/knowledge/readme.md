# Inference Parallel Strategy DAG Knowledge Base

## Overview

This repository defines a **complete, engineering-grade knowledge system** for constructing, evaluating, validating, and executing **parallel inference DAGs** for decoder-only Large Language Models (LLMs).

Its primary consumer is an **autonomous Agent** that generates inference execution DAGs under real-world constraints.

**Scope**:
- Inference stage only (Prefill + Decode)
- Decoder-only Transformer architectures
- Production serving environments

Training-time semantics are explicitly excluded.

---

## Design Goals

This knowledge base is designed to ensure that generated DAGs are:

1. **Semantically correct** (KV cache, phase boundaries)
2. **Parallelism-aware** (TP / SP / PP / EP / DP)
3. **Hardware-constrained** (memory, interconnect, compute)
4. **Objective-driven** (latency vs throughput)
5. **Failure-resistant** (validation + auto-fix)

The end result is **correct-by-construction inference execution plans**.

---

## Knowledge Stack Architecture

The knowledge is organized as a strict dependency stack:

```
Model Semantics
      ↓
Parallel Primitives
      ↓
Module-Level Mapping
      ↓
Execution & Scheduling
      ↓
Hardware & System Constraints
      ↓
DAG Construction Primitives
      ↓
Metrics & Objectives
      ↓
Failure Modes
      ↓
Validation & Auto-Fix
```

Each layer assumes correctness of the layers below it.

---

## Directory Structure

```
01_model_semantics/
  ├─ attention_structure.md
  ├─ kv_cache_semantics.md

03_inference_phases/
  ├─ prefill_phase.md
  ├─ decode_phase.md

04_parallel_primitives/
  ├─ tensor_parallelism.md
  ├─ sequence_parallelism.md
  ├─ pipeline_parallelism.md
  ├─ expert_parallelism.md
  ├─ dp_in_inference.md

05_parallelism_by_module/
  ├─ attention_parallelism.md
  ├─ ffn_parallelism.md
  ├─ moe_parallelism.md
  ├─ embedding_and_lm_head.md

06_execution_and_scheduling/
  ├─ pipeline_scheduling.md
  ├─ microbatching.md
  ├─ overlap_compute_communication.md
  ├─ synchronization_points.md

07_hardware_and_system_constraints/
  ├─ memory_constraints.md
  ├─ interconnect_topology.md
  ├─ latency_vs_throughput.md
  ├─ accelerator_specific_notes.md

08_dag_construction_primitives.md
09_metrics_and_objectives.md
10_common_failure_modes.md
11_dag_validation_and_autofix.md
README.md
```

---

## How an Agent Should Use This Knowledge

A well-behaved DAG generation Agent should follow this pipeline:

1. **Parse model & phase semantics** (01, 03)
2. **Select candidate parallel primitives** (04)
3. **Map primitives to model modules** (05)
4. **Construct execution schedule** (06)
5. **Apply hardware constraints** (07)
6. **Assemble DAG using primitives** (08)
7. **Evaluate using metrics & objectives** (09)
8. **Reject illegal designs using failure knowledge** (10)
9. **Validate and auto-fix before output** (11)

At no point should the Agent skip validation.

---

## Intended Agent Capabilities

An Agent equipped with this knowledge can:

- Automatically synthesize inference parallel strategies
- Generate explicit execution DAGs
- Adapt strategies to different hardware topologies
- Optimize for latency or throughput
- Detect and repair invalid designs

---

## Non-Goals

This knowledge base does NOT attempt to:

- Define training-time parallelism
- Optimize numerical kernels
- Replace runtime schedulers

It focuses strictly on **strategy-level correctness and feasibility**.

---

## Recommended Extensions

Optional future additions:

- Online profiling feedback loops
- Cost models calibrated per accelerator
- Reinforcement learning–based strategy search

---

## Summary

This repository encodes **hard system knowledge** accumulated from real-world LLM inference deployments.

When used correctly, it enables:

- Deterministic inference execution planning
- High-performance parallel serving
- Reduced production risk

This README serves as the **entry point and contract** for any Agent consuming this knowledge base.