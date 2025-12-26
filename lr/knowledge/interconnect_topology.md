# Interconnect Topology

## Scope

This document defines the **interconnect topology constraints and considerations** for multi-device decoder-only Transformer inference.

It focuses on:
- Device interconnection structures (intra-node and inter-node)
- Communication characteristics relevant to TP, SP, PP, EP, and DP
- How topology influences parallel strategy feasibility and scheduling

**Out of scope**:
- Vendor-specific hardware tuning
- Low-level network stack implementation

---

## Interconnect Levels

### Intra-Node Interconnect

- High-bandwidth, low-latency links (e.g., NVLink-class, on-chip fabrics)
- Typically connects devices within the same server or accelerator box
- Preferred for latency-sensitive communication

### Inter-Node Interconnect

- Lower bandwidth and higher latency compared to intra-node links
- Used for scaling beyond a single node
- Communication cost dominates for fine-grained synchronization

---

## Topology Characteristics

Key properties affecting inference parallelism:
- **Bandwidth**: Determines feasibility of large tensor transfers
- **Latency**: Critical for decode phase and pipeline stage boundaries
- **Topology shape**: Ring, mesh, fully connected, hierarchical
- **Oversubscription**: Impacts effective throughput under concurrent communication

---

## Interaction with Parallel Primitives

### Tensor Parallelism (TP)

- Requires frequent all-reduce/all-gather operations
- Strongly prefers intra-node, high-bandwidth topology
- Cross-node TP significantly increases latency and often degrades decode performance

### Sequence Parallelism (SP)

- Prefill-only communication for token partition aggregation
- Can tolerate slightly higher latency than TP
- Still benefits from intra-node placement

### Pipeline Parallelism (PP)

- Stage-to-stage activation transfer
- Sensitive to latency at stage boundaries
- Cross-node PP is feasible if stages are coarse-grained

### Expert Parallelism (EP)

- Scatter/gather patterns based on routing
- Communication volume is sparse but irregular
- Topology-aware expert placement reduces cross-node traffic

### Data Parallelism (DP)

- Minimal communication during inference
- Output aggregation can tolerate higher latency
- Most topology-agnostic primitive

---

## Phase-Specific Considerations

### Prefill Phase

- Communication volume is higher due to multiple tokens
- Overlapping compute and communication is effective
- Topology influences achievable throughput but less sensitive to latency

### Decode Phase

- Single-token per step makes latency critical
- TP and PP communication must be minimized
- Cross-node communication often becomes bottleneck

---

## Topology-Aware Strategy Guidelines

- Place TP groups within the same node
- Assign PP stages to minimize cross-node boundaries
- Co-locate frequently communicating experts for EP
- Use DP to scale across nodes when possible
- Avoid cross-node SP and TP in decode

---

## Failure-Prone Assumptions

- Assuming uniform communication cost across topology levels
- Ignoring inter-node latency in decode scheduling
- Placing TP ranks across slow links
- Overloading a single interconnect with multiple primitives simultaneously

---

## Summary

Interconnect topology defines **the physical communication constraints** for inference parallelism:
- Intra-node links enable fine-grained parallelism (TP, SP)
- Inter-node links favor coarse-grained parallelism (PP, DP)
- Decode phase is especially sensitive to latency

This module ensures **parallel strategy DAGs respect real-world system topology constraints**.