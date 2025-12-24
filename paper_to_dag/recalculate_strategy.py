#!/usr/bin/env python3

import math

# Hardware configuration
NUM_GPUS = 8
GPU_MEMORY_GB = 80
MAX_MEMORY_USAGE_GB = GPU_MEMORY_GB * 0.85  # 85% max usage

# Model configuration
MODEL_WEIGHTS_GB = 140
NUM_LAYERS = 80
HIDDEN_SIZE = 8192
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 8192

# Performance requirements
TARGET_THROUGHPUT_RPS = 8
MAX_DECODE_LATENCY_MS = 100
TARGET_GPU_UTILIZATION = 0.70

def calculate_memory_usage_realistic(pp_degree, tp_degree, sp_degree, batch_size, seq_len):
    """
    Calculate realistic memory usage for given parallel strategy
    """
    # Model weights distribution - TP splits the model weights
    weights_per_gpu = MODEL_WEIGHTS_GB / tp_degree
    
    # KV cache memory (per GPU) - scales with batch size and sequence length
    kv_cache_per_token_kb = 1.0
    # Account for KV cache optimization and sharing
    kv_cache_per_gpu = (batch_size * seq_len * kv_cache_per_token_kb / 1024) / (tp_degree * max(1, sp_degree//2))
    
    # Activation memory (per GPU) - significant for prefill, minimal for decode
    activation_per_token_kb = 0.5
    activation_memory = (batch_size * seq_len * activation_per_token_kb / 1024) / (tp_degree * sp_degree)
    
    # Pipeline buffer overhead (smaller for inference)
    pipeline_overhead = weights_per_gpu * 0.05 if pp_degree > 1 else 0
    
    # Memory for logits and embeddings
    embedding_memory = (batch_size * seq_len * HIDDEN_SIZE * 2) / (1024**3) / (tp_degree * sp_degree)
    
    total_memory = weights_per_gpu + kv_cache_per_gpu + activation_memory + pipeline_overhead + embedding_memory
    
    return total_memory

def calculate_throughput_estimate_realistic(pp_degree, tp_degree, sp_degree):
    """
    More realistic throughput estimation
    """
    # Base throughput for single GPU on Llama-70B
    base_rps = 2.5
    
    # PP improves throughput significantly for large models
    pp_factor = 1.0 + (pp_degree - 1) * 0.7
    
    # TP has communication overhead but enables larger effective batch size
    tp_factor = 1.0 + (tp_degree - 1) * 0.3
    
    # SP helps with memory efficiency, enabling larger batches
    sp_factor = 1.0 + (sp_degree - 1) * 0.2
    
    # Combined throughput
    estimated_rps = base_rps * pp_factor * tp_factor * sp_factor
    
    return estimated_rps

def calculate_latency_estimate_realistic(pp_degree, tp_degree, sp_degree):
    """
    More realistic latency estimation
    """
    # Base decode latency for Llama-70B on H100
    base_latency_ms = 85
    
    # PP adds pipeline stages but each stage is faster
    # For decode, pipeline bubbles are the main concern
    pp_latency_factor = 1.0 + (pp_degree - 1) * 0.12
    
    # TP reduces compute per device but adds AllReduce communication
    # H100 has fast NVLink (900 GB/s) so TP communication is relatively cheap
    tp_latency_factor = (1.0 / tp_degree) * 1.15  # 15% communication overhead
    
    # SP has minimal impact on decode latency (single token)
    sp_latency_factor = 1.0
    
    estimated_latency = base_latency_ms * pp_latency_factor * tp_latency_factor * sp_latency_factor
    
    return estimated_latency

def find_optimal_strategy_realistic():
    """
    Find optimal parallel strategy with realistic calculations
    """
    best_strategy = None
    best_score = -1
    
    # Valid parallel strategies that use exactly 8 GPUs
    valid_strategies = []
    
    for pp in [1, 2, 4, 8]:
        for tp in [1, 2, 4, 8]:
            if pp * tp <= 8:
                remaining_gpus = 8 // (pp * tp)
                for sp in [1, 2, 4, 8]:
                    if sp <= remaining_gpus and remaining_gpus % sp == 0:
                        if pp * tp * sp == 8:
                            valid_strategies.append((pp, tp, sp))
    
    print("Valid strategies (PP×TP×SP = 8 GPUs):")
    print("=" * 60)
    
    feasible_strategies = []
    
    for pp, tp, sp in valid_strategies:        # Try different batch sizes to find feasible configuration
        best_batch_size = 0
        best_memory = float('inf')
        
        for batch_size in [8, 16, 32, 64]:
            for seq_len in [1024, 2048, 4096]:
                memory_usage = calculate_memory_usage_realistic(pp, tp, sp, batch_size, seq_len)
                if memory_usage <= MAX_MEMORY_USAGE_GB and memory_usage < best_memory:
                    best_memory = memory_usage
                    best_batch_size = batch_size
                    best_seq_len = seq_len
        
        if best_batch_size > 0:
            # Calculate final metrics with optimal batch size
            throughput = calculate_throughput_estimate_realistic(pp, tp, sp)
            latency = calculate_latency_estimate_realistic(pp, tp, sp)
            
            # Check constraints
            memory_feasible = best_memory <= MAX_MEMORY_USAGE_GB
            throughput_feasible = throughput >= TARGET_THROUGHPUT_RPS
            latency_feasible = latency <= MAX_DECODE_LATENCY_MS
            
            # Calculate score (higher is better)
            if memory_feasible and throughput_feasible and latency_feasible:
                score = throughput / latency  # Throughput per latency
                feasible_strategies.append((pp, tp, sp, best_memory, throughput, latency, score))
            else:
                score = -1
            
            print(f"PP={pp}×TP={tp}×SP={sp} (batch={best_batch_size}, seq={best_seq_len}):")
            print(f"  Memory usage: {best_memory:.1f}GB ({'✓' if memory_feasible else '✗'})")
            print(f"  Throughput: {throughput:.1f} RPS ({'✓' if throughput_feasible else '✗'})")
            print(f"  Decode latency: {latency:.1f}ms ({'✓' if latency_feasible else '✗'})")
            print(f"  Score: {score:.3f}")
            print()
    
    # Find best feasible strategy
    if feasible_strategies:
        best_strategy = max(feasible_strategies, key=lambda x: x[6])
        return best_strategy
    else:
        return None

if __name__ == "__main__":
    optimal = find_optimal_strategy_realistic()
    if optimal:
        pp, tp, sp, memory, throughput, latency, score = optimal
        print(f"OPTIMAL STRATEGY: PP={pp}×TP={tp}×SP={sp}")
        print(f"Total GPUs used: {pp * tp * sp}")
        print(f"\nPerformance metrics:")
        print(f"  Memory utilization: {memory/80*100:.1f}%")
        print(f"  Projected throughput: {throughput:.1f} RPS")
        print(f"  Projected decode latency: {latency:.1f}ms")
        print(f"  Optimization score: {score:.3f}")
    else:
        print("No feasible strategy found!")
        # Let's find the closest match
        print("\nSearching for closest feasible strategy...")
        # Implementation would continue here...