## Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

### Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP"—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

### 1 Introduction
Mixture-of-Experts (MoE) architectures replace FFN layers with multiple experts and a gating network that activates only a subset of experts per token, achieving higher parameter counts without proportional compute cost. Traditional MoE parallelization colocates several experts per GPU to reduce inter-node communication, creating compute bottlenecks and limiting expert-level parallelism. We instead distribute experts across nodes with at most one expert per GPU, pushing EP ≥ 16 to maximize concurrent computation, shifting the bottleneck from contention to network communication, which we mitigate via topology-aware placement and overlapping strategies.

### 2 Background
MoE models use multiple "experts" in place of FFN layers, with a gating mechanism determining expert activation per token. Parallelism strategies include DP, TP, PP, and EP. Standard implementations use moderate EP, placing multiple experts per GPU to limit communication. With advanced interconnects (NVLink, InfiniBand, NVSwitch), communication cost becomes less dominant than compute concurrency gains. Large EP (≥16) distributes experts across many devices—ideally one per GPU—minimizing contention and maximizing parallel execution, with challenges in cross-node coordination and load balancing.

### 3 Methods
#### 3.1 Expert Placement Strategy
- Single-expert-per-GPU rule: at most one expert instance per GPU; when E ≤ G assign uniquely, else replicate to maximise concurrency.
- Cross-node distribution algorithm: minimises maximum tokens per network link using bandwidth/latency matrix, per-node GPU memory, and predicted routing heat-map.

#### 3.2 Routing & Load Balancing
- Top-K (K=2) gating network selects experts per token.
- Token-level sharding: group by destination expert → fewer messages; async send to overlap with compute; per-batch load monitor adjusts gating logits to rebalance.

#### 3.3 Communication Overlap & Scheduling
- CUDA-stream / NCCL/MPI async all-to-all token transfers interleaved with expert compute.
- Layer-wise pipeline: tokens forwarded to next-layer experts as soon as partial batch arrives → reduces GPU idle time.

#### 3.4 Scalability Integration
- Large EP ≥ 16 makes network bandwidth limiting resource; amortised via batching & topology routing.
- Intra-expert tensor parallelism (TP) and global data parallelism (DP) supported when single-expert memory exceeds GPU limit.

### 4 Experiments
#### 4.1 Setup (Inference-Only)
- Model: 16-layer MoE, 16 experts/layer, expert = MLP(BF16)
- Batch: 128 sequences × 10,000 tokens, token_dim = 4096
- MHA: 32 heads × 128 dim/head
- MLP hidden = 16,384
- HW: unlimited H100 GPUs, 400 TFLOPS/GPU, 64 GB VRAM, 1.8 TB/s @ 80 % utilisation
- Metrics: TPS, TPOT

#### 4.2 Deployments
Baseline (TP=8 PP=2): 16 GPUs, each GPU holds TP shard + 2 experts/layer → TPS = 120,000, TPOT = 8.3 ms.
Proposed (EP=16 1-expert/GPU): 256 GPUs (16 layers × 16 experts), each GPU hosts exactly one expert per layer, async all-to-all token routing → TPS = 450,000, TPOT = 2.2 ms.
Speed-up: 3.75× throughput, 3.8× latency reduction; near-linear scaling with 16 GPUs/layer.

### 5 Conclusion
Large-scale cross-node expert parallelism with one expert per GPU maximizes compute concurrency, balances load via topology-aware placement and dynamic gating, and overlaps communication with computation, yielding near-linear scaling and significant speed-ups for large-scale MoE inference.