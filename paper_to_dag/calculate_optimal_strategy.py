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

def calculate_memory_usage(pp_degree, tp_degree, sp_degree, batch_size, seq_len):
    """
    Calculate memory usage for given parallel strategy
    """
    # Model weights distribution
    weights_per_gpu = MODEL_WEIGHTS_GB / tp_degree  # TP splits weights
    
    # KV cache memory (per GPU)
    kv_cache_per_token_kb = 1.0
    kv_cache_per_gpu = (batch_size * seq_len * kv_cache_per_token_kb / 1024) / tp_degree
    
    # Activation memory (per GPU)
    activation_per_token_kb = 0.5
    activation_memory = (batch_size * seq_len * activation_per_token_kb / 1024) / (tp_degree * sp_degree)
    
    # Pipeline buffer overhead (10% of weights)
    pipeline_overhead = weights_per_gpu * 0.1 if pp_degree > 1 else 0
    
    total_memory = weights_per_gpu + kv_cache_per_gpu + activation_memory + pipeline_overhead
    
    return total_memory

def calculate_throughput_estimate(pp_degree, tp_degree, sp_degree):
    """
    Rough throughput estimation based on parallel strategy
    """
    # Base throughput factor (higher is better)
    base_factor = 1.0
    
    # PP improves throughput but with diminishing returns
    pp_factor = 1.0 + (pp_degree - 1) * 0.6
    
    # TP improves latency but can hurt throughput due to communication
    tp_factor = 1.0 / (1.0 + (tp_degree - 1) * 0.2)
    
    # SP helps with memory and can improve throughput for long sequences
    sp_factor = 1.0 + (sp_degree - 1) * 0.3
    
    # Combined factor
    combined_factor = base_factor * pp_factor * tp_factor * sp_factor
    
    # Convert to estimated RPS (rough approximation)
    estimated_rps = combined_factor * 6.0  # Base 6 RPS for single GPU
    
    return estimated_rps

def calculate_latency_estimate(pp_degree, tp_degree, sp_degree):
    """
    Rough latency estimation for decode phase
    """
    # Base latency
    base_latency_ms = 120
    
    # PP adds pipeline bubbles, especially for single token decode
    pp_latency_factor = 1.0 + (pp_degree - 1) * 0.15
    
    # TP reduces compute per device but adds communication overhead
    tp_latency_factor = 1.0 / tp_degree + (tp_degree - 1) * 0.05
    
    # SP has limited benefit for decode (single token)
    sp_latency_factor = 1.0
    
    estimated_latency = base_latency_ms * pp_latency_factor * tp_latency_factor * sp_latency_factor
    
    return estimated_latency

def find_optimal_strategy():
    """
    Find optimal parallel strategy within GPU constraints
    """
    best_strategy = None
    best_score = -1
    
    # Valid parallel strategies that use exactly 8 GPUs
    valid_strategies = []
    
    for pp in [1, 2, 4, 8]:  # PP must divide 8 evenly
        for tp in [1, 2, 4, 8]:  # TP must divide 8 evenly
            if pp * tp <= 8:  # Must fit in available GPUs
                remaining_gpus = 8 // (pp * tp)
                for sp in [1, 2, 4, 8]:  # SP must divide remaining capacity
                    if sp <= remaining_gpus and remaining_gpus % sp == 0:
                        # This strategy uses exactly 8 GPUs: pp * tp * sp = 8
                        if pp * tp * sp == 8:
                            valid_strategies.append((pp, tp, sp))
    
    print("Valid strategies (PP×TP×SP = 8 GPUs):")
    print("=" * 50)
    
    for pp, tp, sp in valid_strategies:
        # Calculate metrics
        memory_usage = calculate_memory_usage(pp, tp, sp, batch_size=32, seq_len=2048)
        throughput = calculate_throughput_estimate(pp, tp, sp)
        latency = calculate_latency_estimate(pp, tp, sp)
        
        # Check constraints
        memory_feasible = memory_usage <= MAX_MEMORY_USAGE_GB
        throughput_feasible = throughput >= TARGET_THROUGHPUT_RPS
        latency_feasible = latency <= MAX_DECODE_LATENCY_MS
        
        # Calculate score (higher is better)
        if memory_feasible and throughput_feasible and latency_feasible:
            score = throughput / latency  # Throughput per latency
        else:
            score = -1
        
        print(f"PP={pp}×TP={tp}×SP={sp}:")
        print(f"  Memory usage: {memory_usage:.1f}GB ({'✓' if memory_feasible else '✗'})")
        print(f"  Throughput: {throughput:.1f} RPS ({'✓' if throughput_feasible else '✗'})")
        print(f"  Decode latency: {latency:.1f}ms ({'✓' if latency_feasible else '✗'})")
        print(f"  Score: {score:.3f}")
        print()
        
        if score > best_score:
            best_score = score
            best_strategy = (pp, tp, sp)
    
    return best_strategy

if __name__ == "__main__":
    optimal = find_optimal_strategy()
    if optimal:
        pp, tp, sp = optimal
        print(f"Optimal strategy: PP={pp}×TP={tp}×SP={sp}")
        print(f"Total GPUs used: {pp * tp * sp}")
        
        # Final validation
        memory = calculate_memory_usage(pp, tp, sp, batch_size=32, seq_len=2048)
        throughput = calculate_throughput_estimate(pp, tp, sp)
        latency = calculate_latency_estimate(pp, tp, sp)
        
        print(f"\nFinal performance:")
        print(f"  Memory utilization: {memory/80*100:.1f}%")
        print(f"  Projected throughput: {throughput:.1f} RPS")
        print(f"  Projected decode latency: {latency:.1f}ms")
    else:
        print("No feasible strategy found!")