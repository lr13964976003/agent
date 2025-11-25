## Phase-1 Key Points (Large-Scale Cross-Node Expert Parallelism for MoE)

1. Core Goal: maximise expert-level parallelism by deploying ≤ 1 expert per GPU across nodes.
2. Large EP regime: EP ≥ 16 defined as minimum for “large” setup.
3. Bottleneck Shift: from intra-GPU expert contention to network communication (mitigated by overlap & topology-aware placement).
4. Topology-aware expert placement minimises max tokens per link while respecting one-expert-per-GPU.
5. Dynamic gating + token batching + async routing ensure balanced expert load and hide latency.
6. Compatible with intra-expert TP and global DP; supports 100 B+ MoE models.