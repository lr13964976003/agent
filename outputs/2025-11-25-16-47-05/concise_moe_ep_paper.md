## Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

### Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP"—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

### 1 Introduction
Mixture-of-Experts (MoE) architectures replace FFN layers with multiple experts and a gating network that activates only a subset of experts per token, achieving higher parameter counts without proportional compute cost. Traditional MoE parallelization colocates several experts per GPU to reduce inter-node communication, creating compute bottlenecks and limiting expert-level parallelism. We instead distribute experts across nodes with at most one expert per GPU, pushing EP ≥ 16 to maximize concurrent computation and shifting the bottleneck from contention to network communication, which we mitigate via topology-aware placement and overlapping strategies.

### 2 Methods
#### 2.1 Expert Placement Strategy
- Single-expert-per-GPU: for E experts and G GPUs, assign each expert to a distinct GPU when E ≤ G; when E > G replicate experts to maximize concurrency while balancing memory.
- Cross-node distribution: topology-aware algorithm minimizes maximum tokens per link by considering node-to-node bandwidth/latency, GPU memory capacity, and expected routing patterns.

#### 2.2 Routing & Load Balancing
- Gating: standard top-K (K=2) gating network selects experts per token.
- Token sharding: group tokens by destination expert (token batching), send asynchronously to overlap with compute, monitor per-expert load and dynamically adjust gating probabilities to maintain balance.

#### 2.3 Communication Overlap & Scheduling
- Interleave compute & communication: while a GPU processes one token batch the next is transferred via CUDA streams/NCCL/MPI.
- Pipeline scheduling: outputs of layer l are immediately routed to layer l+1; experts start on partial batches to reduce idle time.

#### 2.4 Scalability & Integration
- Large EP regime (EP ≥ 16) makes network bandwidth the limiter; topology-aware routing + batching amortize cost.
- Compatible with tensor parallelism (TP) inside an expert and data parallelism (DP) across MoE replicas.

### 3 Experiments
#### 3.1 Setup (Inference-Only)
- Model: 16-layer MoE, 16 experts/layer, each expert = MLP
- Precision: BF16
- Batch: 128 sequences × 10 000 tokens, token dim = 4096
- MHA: 32 heads × 128 dim/head
- MLP hidden = 16 384
- Hardware: unlimited H100 GPUs, 400 TFLOPS/GPU, MFU 60 %, VRAM 64 GB, 1.8 TB/s bandwidth @ 80 % utilisation
- Metrics: TPS (throughput), TPOT (latency)

#### 3.2 Deployments
Baseline (TP=8, PP=2): 16 GPUs, each GPU holds TP shard for all layers and 2 experts/layer → TPS = 120 000, TPOT = 8.3 ms.
Proposed (EP=16, 1 expert/GPU): 256 GPUs (16 layers × 16 experts), each GPU hosts exactly one expert per layer, asynchronous all-to-all token routing → TPS = 450 000, TPOT = 2.2 ms.
Speed-up: 3.75× throughput, 3.8× latency reduction.

### 4 Conclusion
Large-scale cross-node expert parallelism with one expert per GPU maximizes compute concurrency, balances load via topology-aware placement and dynamic gating, and overlaps communication with computation, yielding near-linear scaling and significant speed-ups for large-scale MoE inference.