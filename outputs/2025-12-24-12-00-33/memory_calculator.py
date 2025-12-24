#!/usr/bin/env python3
"""
Memory and Performance Calculator for LLM Parallel Strategy
Addresses critical issues from previous submission
"""

import math

# Hardware specifications
GPU_MEMORY_GB = 64
GPU_BANDWIDTH_TBPS = 1.8
GPU_COMPUTE_TFLOPS = 400
MFU = 0.6
BANDWIDTH_UTIL = 0.8

# Model specifications
TOTAL_PARAMS_B = 10  # 10B parameters
LAYERS = 16
EXPERTS_PER_LAYER = 16
PRECISION_BYTES = 2  # FP16
TOKEN_DIM = 512
MAX_SEQ_LEN = 10240
BATCH_SIZE = 128

# Performance requirements
TARGET_THROUGHPUT_TOKENS_PER_MS = 100
TARGET_TTFT_S = 10

def calculate_memory_requirements():
    """Calculate accurate memory requirements considering all parallelism dimensions"""
    
    # Base model weights
    total_model_weights_gb = TOTAL_PARAMS_B * 1e9 * PRECISION_BYTES / 1e9
    print(f"Total model weights: {total_model_weights_gb:.1f} GB")
    
    # Test different parallelism configurations
    configs = [
        {"pp": 2, "sp": 2, "ep": 4, "tp": 1},
        {"pp": 4, "sp": 1, "ep": 4, "tp": 2},
        {"pp": 2, "sp": 4, "ep": 2, "tp": 2},
        {"pp": 4, "sp": 2, "ep": 2, "tp": 2},
    ]
    
    print("\nMemory Analysis for Different Parallel Configurations:")
    print("=" * 70)
    
    for i, config in enumerate(configs, 1):
        pp, sp, ep, tp = config["pp"], config["sp"], config["ep"], config["tp"]
        total_gpus = pp * sp * ep * tp
        
        # CRITICAL FIX: Account for ALL parallelism dimensions
        # Each parallelism dimension reduces memory requirements differently:
        # - TP: splits tensors (model weights)
        # - EP: splits experts (model weights)
        # - PP: splits layers (model weights)
        # - SP: splits sequence (activations, not weights)
        
        weight_parallelism = tp * ep * pp  # These affect model weights
        model_memory_per_gpu = total_model_weights_gb / weight_parallelism
        
        # Activation memory (rough estimate, scales with SP and batch)
        # SP reduces per-GPU sequence length
        avg_seq_len = (128 + 10240) / 2  # Use average for estimation
        activation_memory_gb = (BATCH_SIZE * avg_seq_len * TOKEN_DIM * PRECISION_BYTES * 4) / (sp * 1e9)
        
        # KV cache memory (scales with sequence and batch)
        kv_cache_per_layer_gb = (BATCH_SIZE * MAX_SEQ_LEN * TOKEN_DIM * PRECISION_BYTES * 2) / (sp * 1e9)
        total_kv_cache_gb = kv_cache_per_layer_gb * (LAYERS / pp)
        
        total_memory_per_gpu = model_memory_per_gpu + activation_memory_gb + total_kv_cache_gb
        
        print(f"Config {i}: PP={pp} SP={sp} EP={ep} TP={tp}")
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Model weights per GPU: {model_memory_per_gpu:.2f} GB")
        print(f"  Activations per GPU: {activation_memory_gb:.2f} GB")
        print(f"  KV cache per GPU: {total_kv_cache_gb:.2f} GB")
        print(f"  Total memory per GPU: {total_memory_per_gpu:.2f} GB")
        print(f"  Memory utilization: {total_memory_per_gpu/GPU_MEMORY_GB*100:.1f}%")
        print()
        
        # Check if configuration is feasible
        if total_memory_per_gpu <= GPU_MEMORY_GB * 0.9:  # 90% threshold
            print(f"  ✓ CONFIGURATION FEASIBLE")
        else:
            print(f"  ✗ CONFIGURATION NOT FEASIBLE")
        print()

def calculate_performance_metrics():
    """Calculate realistic performance metrics"""
    
    effective_compute_tfops = GPU_COMPUTE_TFLOPS * MFU
    effective_bandwidth_tbops = GPU_BANDWIDTH_TBPS * BANDWIDTH_UTIL
    
    print("Performance Analysis:")
    print("=" * 50)
    print(f"Effective compute per GPU: {effective_compute_tfops:.0f} TFlops")
    print(f"Effective bandwidth per GPU: {effective_bandwidth_tbops:.1f} TB/s")
    
    # More realistic FLOPS calculation for inference
    # For a 10B model, typical FLOPS per token is much lower than 20B
    # Rough estimate: 2 * params for forward pass, but with optimizations
    model_flops_per_token = TOTAL_PARAMS_B * 1e9 * 1.5  # 1.5x accounts for attention, etc.
    
    print(f"\nModel FLOPs per token: {model_flops_per_token/1e9:.1f} B")
    
    # Theoretical compute-limited throughput
    compute_throughput_tokens_per_ms = effective_compute_tfops * 1e12 / model_flops_per_token / 1000
    
    print(f"Compute-limited throughput: {compute_throughput_tokens_per_ms:.1f} tokens/ms")
    
    # Memory bandwidth analysis (more realistic)
    # Memory access per token includes:
    # - Model weights (with caching)
    # - Activations
    # - KV cache access
    memory_access_per_token_gb = (TOTAL_PARAMS_B / 10) + 0.1  # Rough but more realistic
    
    bandwidth_throughput_tokens_per_ms = effective_bandwidth_tbops * 1e12 / (memory_access_per_token_gb * 1e9) / 1000
    
    print(f"Memory bandwidth per token: {memory_access_per_token_gb:.2f} GB")
    print(f"Bandwidth-limited throughput: {bandwidth_throughput_tokens_per_ms:.1f} tokens/ms")
    
    # Actual throughput is the minimum of compute and bandwidth limits
    actual_throughput = min(compute_throughput_tokens_per_ms, bandwidth_throughput_tokens_per_ms)
    
    print(f"\nActual expected throughput: {actual_throughput:.1f} tokens/ms")
    print(f"Target throughput: {TARGET_THROUGHPUT_TOKENS_PER_MS} tokens/ms")
    
    if actual_throughput >= TARGET_THROUGHPUT_TOKENS_PER_MS:
        print("✓ THROUGHPUT REQUIREMENT MET")
    else:
        print("✗ THROUGHPUT REQUIREMENT NOT MET")

def calculate_ttft():
    """Calculate Time to First Token for prefill phase"""
    
    print(f"\nTTFT Analysis:")
    print("=" * 30)
    
    # For prefill with PP=4, SP=2
    pp = 4
    sp = 2
    total_gpus_prefill = pp * sp * 4 * 2  # 4 PP, 2 SP, 4 EP, 2 TP = 64 GPUs
    
    # Longest sequence
    max_tokens = MAX_SEQ_LEN
    
    # Effective processing rate (tokens per second across all GPUs)
    # With parallelism, we can process multiple tokens in parallel
    parallel_tokens_per_second = total_gpus_prefill * 50  # Conservative estimate
    
    ttft_seconds = max_tokens / parallel_tokens_per_second
    
    print(f"Max sequence length: {max_tokens} tokens")
    print(f"Total GPUs for prefill: {total_gpus_prefill}")
    print(f"Parallel processing rate: {parallel_tokens_per_second} tokens/s")
    print(f"Estimated TTFT: {ttft_seconds:.1f} seconds")
    print(f"Target TTFT: {TARGET_TTFT_S} seconds")
    
    if ttft_seconds <= TARGET_TTFT_S:
        print("✓ TTFT REQUIREMENT MET")
    else:
        print("✗ TTFT REQUIREMENT NOT MET")

if __name__ == "__main__":
    calculate_memory_requirements()
    calculate_performance_metrics()
    calculate_ttft()