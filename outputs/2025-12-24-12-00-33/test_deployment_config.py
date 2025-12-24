#!/usr/bin/env python3
"""Test the exact configuration from the deployment plan"""

# Hardware specifications
GPU_MEMORY_GB = 64
TOTAL_PARAMS_B = 10  # 10B parameters
PRECISION_BYTES = 2  # FP16
BATCH_SIZE = 128
TOKEN_DIM = 512
MAX_SEQ_LEN = 10240
LAYERS = 16

# Test the deployment plan configuration: PP=4, SP=2, EP=4, TP=2
pp = 4
sp = 2
ep = 4
tp = 2
total_gpus = pp * sp * ep * tp

# Base model weights
total_model_weights_gb = TOTAL_PARAMS_B * 1e9 * PRECISION_BYTES / 1e9
print(f"Total model weights: {total_model_weights_gb:.1f} GB")

# CRITICAL FIX: Account for ALL parallelism dimensions
weight_parallelism = tp * ep * pp  # These affect model weights
model_memory_per_gpu = total_model_weights_gb / weight_parallelism

# Activation memory (rough estimate, scales with SP and batch)
avg_seq_len = (128 + 10240) / 2  # Use average for estimation
activation_memory_gb = (BATCH_SIZE * avg_seq_len * TOKEN_DIM * PRECISION_BYTES * 4) / (sp * 1e9)

# KV cache memory (scales with sequence and batch)
kv_cache_per_layer_gb = (BATCH_SIZE * MAX_SEQ_LEN * TOKEN_DIM * PRECISION_BYTES * 2) / (sp * 1e9)
total_kv_cache_gb = kv_cache_per_layer_gb * (LAYERS / pp)

total_memory_per_gpu = model_memory_per_gpu + activation_memory_gb + total_kv_cache_gb

print(f"\nDeployment Plan Configuration: PP={pp} SP={sp} EP={ep} TP={tp}")
print(f"Total GPUs: {total_gpus}")
print(f"Model weights per GPU: {model_memory_per_gpu:.2f} GB")
print(f"Activations per GPU: {activation_memory_gb:.2f} GB")
print(f"KV cache per GPU: {total_kv_cache_gb:.2f} GB")
print(f"Total memory per GPU: {total_memory_per_gpu:.2f} GB")
print(f"Memory utilization: {total_memory_per_gpu/GPU_MEMORY_GB*100:.1f}%")

# Check if configuration is feasible
if total_memory_per_gpu <= GPU_MEMORY_GB * 0.9:  # 90% threshold
    print(f"\n✓ CONFIGURATION FEASIBLE")
else:
    print(f"\n✗ CONFIGURATION NOT FEASIBLE")