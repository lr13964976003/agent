## Phase-3 Experiments Extract

### Experimental Setup (Inference-Only)
- Model: 16-layer MoE, 16 experts/layer, expert = MLP(BF16)
- Batch: 128 sequences × 10 000 tokens, token_dim = 4096
- MHA: 32 heads × 128 dim/head
- MLP hidden = 16 384
- HW: unlimited H100 GPUs, 400 TFLOPS/GPU, 64 GB VRAM, 1.8 TB/s @ 80 % utilisation
- Metrics: TPS, TPOT

### Baseline vs Proposed
| Method | GPUs | Placement | TPS | TPOT (ms) |
|---|---|---|---|---|
| Baseline (TP=8 PP=2) | 16 | TP shard + 2 experts/GPU | 120 000 | 8.3 |
| Proposed (EP=16 1-expert/GPU) | 256 | 1 expert/GPU/layer + async all-to-all | 450 000 | 2.2 |
Speed-up: 3.75× throughput, 3.8× latency reduction; near-linear scaling with 16 GPUs/layer.

### Discussion
Single-expert-per-GPU fully saturates compute units; async routing hides communication; system scales when EP ≥ 16.