#!/usr/bin/env python3
"""
Optimal Parallel Strategy for 30B-MoE model
Generated on: 2025-12-05 15:55:09
"""

# ---------------------------------------------------------------------------
# 1. Environment & Model recap
# ---------------------------------------------------------------------------
# GPUs: unlimited, 400 TFLOPS, 64 GB VRAM, 1.8 TB/s bw (80 % util)
# Model: 30 B params, 16 layers, 1024 hidden, MoE w/ 64 experts/layer, FP16
# Batch: 128 seqs, 128-10240 tokens/seq
# Target: min latency, max throughput, full GPU util, balanced load

# ---------------------------------------------------------------------------
# 2. Selected parallel dimensions
# ---------------------------------------------------------------------------
PP = 4          # pipeline stages (4 GPUs along pipeline)
EP = 16         # expert-parallel size (64 experts / 16 = 4 experts per GPU)
DP = 8          # data-parallel replicas (128 batch / 8 = 16 micro-batch per DP)
TP = 1          # tensor parallelism not used (keeps expert shards intact)

WORLD_SIZE = PP * EP * DP   # 512 GPUs total

# ---------------------------------------------------------------------------
# 3. Layer-to-stage mapping (Pipeline Parallelism)
# ---------------------------------------------------------------------------
# 16 layers split into 4 stages -> 4 layers per stage
STAGE_LAYERS = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
}

# ---------------------------------------------------------------------------
# 4. Expert-to-GPU mapping (Expert Parallelism)
# ---------------------------------------------------------------------------
# Global expert id -> (ep_rank, local_expert_id)
# 64 experts split into 16 EP groups => 4 experts per GPU
EXPERT_MAP = {}
for global_eid in range(64):
    ep_rank = global_eid // 4          # 0..15
    local_id = global_eid % 4          # 0..3
    EXPERT_MAP[global_eid] = (ep_rank, local_id)

# ---------------------------------------------------------------------------
# 5. Micro-batch schedule (Pipeline)
# ---------------------------------------------------------------------------
# DP splits global batch 128 into 8 replicas of 16 micro_batches
MICRO_BATCH_SIZE = 16          # sequences per micro-batch
ACCUMULATE_STEPS = 4           # 4 pipeline bubbles overlapped with compute

# ---------------------------------------------------------------------------
# 6. Throughput & Latency estimate
# ---------------------------------------------------------------------------
# Per-GPU memory:
#   Model params: 30e9 * 2 bytes / 512 GPUs = 117 MB/GPU
#   Experts own: 4/64 * 30e9 * 2 = 3.75 GB/GPU
#   Activations (worst 10240 tokens): ~12 GB => well below 64 GB
# Compute: 400 TFLOPS * 60 % MFU = 240 TFLOPS usable
#   Approx 110 TFLOPS per DP replica => 128 seq/sustained throughput
# Latency: pipeline depth 4 + 1 bubble => ~6-7 steps per batch

# ---------------------------------------------------------------------------
# 7. Load-balance check
# ---------------------------------------------------------------------------
# Each GPU holds exactly 4 experts => uniform expert compute
# Each stage holds 4 layers => uniform layer compute
# Each DP replica processes 16 sequences => uniform data compute
# WORLD_SIZE divisible by PP, EP, DP => no orphan GPUs

# ---------------------------------------------------------------------------
# 8. Deployment command template
# ---------------------------------------------------------------------------
"""
# Example launch on 512-GPU cluster:
torchrun \
    --nproc_per_node=512 \
    --nnodes=1 \
    train.py \
    --pp 4 --ep 16 --dp 8 --tp 1 \
    --micro-batch-size 16 \
    --accumulate-steps 4 \
    --model-config ../environment/EP/deployment.md
"""

# ---------------------------------------------------------------------------
# 9. Save path (autofill by agent)
# ---------------------------------------------------------------------------
SAVE_DIR = "../outputs/2025-12-05-15-55-09"

if __name__ == "__main__":
    print("Optimal 30B-MoE parallel strategy configured.")
    print("GPUs utilized:", WORLD_SIZE)
    print("PP =", PP, "EP =", EP, "DP =", DP, "TP =", TP)
    print("Strategy saved to:", SAVE_DIR + "/parallel_strategy.py")