## Phase-2 Methodology Extract

### Expert Placement Strategy
- Single-expert-per-GPU rule: at most one expert instance per GPU; when E ≤ G assign uniquely, else replicate to maximise concurrency.
- Cross-node distribution algorithm: minimises maximum tokens per network link using bandwidth/latency matrix, per-node GPU memory, and predicted routing heat-map.

### Routing & Load Balancing
- Top-K (K=2) gating network selects experts per token.
- Token-level sharding: group by destination expert → fewer messages; async send to overlap with compute; per-batch load monitor adjusts gating logits to rebalance.

### Communication Overlap & Scheduling
- CUDA-stream / NCCL/MPI async all-to-all token transfers interleaved with expert compute.
- Layer-wise pipeline: tokens forwarded to next-layer experts as soon as partial batch arrives → reduces GPU idle time.

### Scalability Integration
- Large EP ≥ 16 makes network bandwidth limiting resource; amortised via batching & topology routing.
- Intra-expert tensor parallelism (TP) and global data parallelism (DP) supported when single-expert memory exceeds GPU limit.