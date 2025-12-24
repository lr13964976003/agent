#!/usr/bin/env python3
import math

# Hardware specifications
GPU_MEMORY_GB = 64
EFFECTIVE_COMPUTE_TFlops = 240  # 400 * 0.6
EFFECTIVE_BANDWIDTH_TBps = 1.44  # 1.8 * 0.8

# Model specifications
MODEL_PARAMS_B = 10
MODEL_MEMORY_GB = MODEL_PARAMS_B * 2  # FP16
LAYERS = 16
EXPERTS_PER_LAYER = 16
HIDDEN_SIZE = 512
MHA_HEADS = 16
HEAD_DIM = 32
MOE_HIDDEN = 1024

# Input specifications
BATCH_SIZE = 128
MAX_SEQ_LEN = 10240

# Performance requirements
REQUIRED_THROUGHPUT_PER_GPU = 100  # tokens/ms
MAX_TTFT = 10  # seconds

print("=== PARALLEL STRATEGY ANALYSIS ===")
print(f"Model size: {MODEL_MEMORY_GB}GB")
print(f"GPU memory: {GPU_MEMORY_GB}GB")
print(f"Memory utilization: {MODEL_MEMORY_GB/GPU_MEMORY_GB*100:.1f}%")
print()

# Memory analysis
kv_cache_per_token = 2 * LAYERS * HIDDEN_SIZE * 2  # 2 for K+V, 2 for FP16 bytes
kv_cache_per_seq = kv_cache_per_token * MAX_SEQ_LEN
kv_cache_total = kv_cache_per_seq * BATCH_SIZE
kv_cache_gb = kv_cache_total / (1024**3)

print(f"KV cache per token: {kv_cache_per_token} bytes")
print(f"KV cache per sequence (max): {kv_cache_per_seq / (1024**2):.1f} MB")
print(f"Total KV cache for batch: {kv_cache_gb:.1f} GB")
print()

# Calculate required parallelism
remaining_memory = GPU_MEMORY_GB - MODEL_MEMORY_GB - kv_cache_gb
print(f"Remaining memory after model and KV cache: {remaining_memory:.1f} GB")

# Performance analysis
print("\n=== PERFORMANCE ANALYSIS ===")

# For prefill phase (compute bound)
compute_per_token_prefill = 2 * MODEL_PARAMS_B * 2  # rough estimate
print(f"Compute per token (prefill): {compute_per_token_prefill} FLOPs")

# For decode phase (memory bound)
memory_per_token_decode = 2 * MODEL_PARAMS_B * 2  # memory access pattern
print(f"Memory access per token (decode): {memory_per_token_decode} bytes")

# Calculate required parallelism for throughput
required_compute_per_ms = REQUIRED_THROUGHPUT_PER_GPU * compute_per_token_prefill / 1000
print(f"Required compute for throughput: {required_compute_per_ms} GFLOPS/ms")
print(f"Available compute: {EFFECTIVE_COMPUTE_TFlops * 1000} GFLOPS/ms")

compute_parallelism_needed = required_compute_per_ms / (EFFECTIVE_COMPUTE_TFlops * 1000)
print(f"Compute parallelism needed: {compute_parallelism_needed:.2f}")

# Calculate required parallelism for memory bandwidth
required_memory_per_ms = REQUIRED_THROUGHPUT_PER_GPU * memory_per_token_decode / 1000
print(f"Required memory bandwidth for throughput: {required_memory_per_ms/1000:.2f} GB/ms")
print(f"Available memory bandwidth: {EFFECTIVE_BANDWIDTH_TBps * 1000} GB/ms")

memory_parallelism_needed = (required_memory_per_ms/1000) / (EFFECTIVE_BANDWIDTH_TBps * 1000)
print(f"Memory parallelism needed: {memory_parallelism_needed:.2f}")

print(f"\nRecommended total GPU count: {max(2, math.ceil(1/max(compute_parallelism_needed, memory_parallelism_needed)))}")