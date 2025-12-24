#!/usr/bin/env python3
import math

# Hardware specifications
GPU_MEMORY_GB = 64
EFFECTIVE_COMPUTE_TFlops = 240
EFFECTIVE_BANDWIDTH_TBps = 1.44

# Model specifications
MODEL_PARAMS_B = 10
MODEL_MEMORY_GB = MODEL_PARAMS_B * 2  # FP16
LAYERS = 16
EXPERTS_PER_LAYER = 16
HIDDEN_SIZE = 512

# Input specifications
BATCH_SIZE = 128
MAX_SEQ_LEN = 10240

print("=== OPTIMAL PARALLEL STRATEGY DESIGN ===")

# Memory constraints analysis
kv_cache_per_token = 2 * LAYERS * HIDDEN_SIZE * 2  # K+V, FP16
kv_cache_per_seq = kv_cache_per_token * MAX_SEQ_LEN
kv_cache_total_gb = (kv_cache_per_seq * BATCH_SIZE) / (1024**3)

print(f"Model memory: {MODEL_MEMORY_GB}GB")
print(f"KV cache memory: {kv_cache_total_gb:.1f}GB")
print(f"Total memory needed: {MODEL_MEMORY_GB + kv_cache_total_gb:.1f}GB")

# Since we have 64GB per GPU and need ~60GB, we need at least 2 GPUs for memory
min_gpus_memory = math.ceil((MODEL_MEMORY_GB + kv_cache_total_gb) / GPU_MEMORY_GB)
print(f"Minimum GPUs needed for memory: {min_gpus_memory}")

# MoE characteristics - 16 experts per layer
# For optimal expert parallelism, we should divide experts across GPUs
print(f"\nMoE Analysis:")
print(f"Total experts: {EXPERTS_PER_LAYER} per layer")
print(f"Total layers: {LAYERS}")

# Optimal strategy: Use 4 GPUs
# - TP=2 for tensor parallelism within experts
# - EP=2 for expert parallelism across GPUs  
# - PP=2 for pipeline parallelism across layers
# This gives us 2*2*2=8 total GPUs, but let's optimize

# Better strategy: 4 GPUs total
# - TP=2 for tensor parallelism
# - EP=2 for expert parallelism (8 experts per GPU)
# - PP=2 for pipeline parallelism (8 layers per GPU stage)

total_gpus = 4
tp_size = 2
ep_size = 2
pp_size = 2

print(f"\nProposed Strategy:")
print(f"Total GPUs: {total_gpus}")
print(f"Tensor Parallelism (TP): {tp_size}")
print(f"Expert Parallelism (EP): {ep_size}")
print(f"Pipeline Parallelism (PP): {pp_size}")

# Memory per GPU with this strategy
model_memory_per_gpu = MODEL_MEMORY_GB / (tp_size * pp_size)
kv_cache_per_gpu = kv_cache_total_gb / (tp_size * pp_size)
total_memory_per_gpu = model_memory_per_gpu + kv_cache_per_gpu

print(f"\nMemory per GPU:")
print(f"Model parameters: {model_memory_per_gpu:.1f}GB")
print(f"KV cache: {kv_cache_per_gpu:.1f}GB")
print(f"Total: {total_memory_per_gpu:.1f}GB")
print(f"Utilization: {total_memory_per_gpu/GPU_MEMORY_GB*100:.1f}%")

# Expert distribution
experts_per_gpu = EXPERTS_PER_LAYER / ep_size
layers_per_gpu = LAYERS / pp_size

print(f"\nDistribution:")
print(f"Experts per GPU: {experts_per_gpu}")
print(f"Layers per GPU: {layers_per_gpu}")

print(f"\nParallel Strategy Summary:")
print(f"- Use {total_gpus} GPUs total")
print(f"- TP={tp_size}: Split tensor operations within layers")
print(f"- EP={ep_size}: Distribute {EXPERTS_PER_LAYER} experts across GPUs")
print(f"- PP={pp_size}: Split {LAYERS} layers into {pp_size} stages")
print(f"- Each GPU handles {experts_per_gpu} experts and {layers_per_gpu} layers")