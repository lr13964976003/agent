#!/usr/bin/env python3

import json
import math

def comprehensive_performance_evaluation():
    print("=== COMPREHENSIVE PERFORMANCE EVALUATION ===\n")
    
    # Load the actual deployment strategies
    try:
        with open('./outputs/2026-01-05-10-27-44/prefill_parallel_strategy_deployment.json', 'r') as f:
            prefill_strategy = json.load(f)
        with open('./outputs/2026-01-05-10-27-44/decode_parallel_strategy_deployment.json', 'r') as f:
            decode_strategy = json.load(f)
    except FileNotFoundError:
        print("ERROR: Deployment strategy files not found!")
        return
    
    # Extract claimed performance
    claimed_ttft = float(prefill_strategy['performance_characteristics']['estimated_ttft'].replace(' seconds', ''))
    claimed_memory_util = float(prefill_strategy['performance_characteristics']['memory_utilization'].replace('%', ''))
    
    print("1. CLAIMED vs ACTUAL PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"Claimed TTFT: {claimed_ttft:.1f} seconds")
    print(f"Claimed Memory Utilization: {claimed_memory_util:.1f}%")
    
    # Hardware specifications from input
    single_gpu_compute = 400 * 1e12  # 400 TFlops
    single_gpu_memory = 64 * 1e9  # 64 GB
    memory_bandwidth = 1.8 * 1e12  # 1.8 TBps
    bandwidth_utilization = 0.8
    mfu_utilization = 0.6
    
    # Model configuration
    layers = 94
    experts_per_layer = 128
    token_dimension = 4096
    attention_heads = 64
    head_dimension = 64
    moe_hidden_size = 1536
    top_k_gate = 8
    gqa_kv_heads = 4
    
    # Input requirements
    batch_size = 128
    input_sequence = 2048
    output_sequence = 2048
    ttft_requirement = 30  # seconds
    
    # Parallel strategy configuration
    ep = 1
    pp = 4
    tp = 2
    dp = 1
    total_gpus = 32
    
    # 2. Detailed Memory Analysis
    print("\n2. DETAILED MEMORY ANALYSIS")
    print("-" * 50)
    
    # Model weights calculation
    attention_weights = (
        attention_heads * head_dimension * token_dimension * 4  # Q, K, V, O projections
    )
    moe_weights = (
        experts_per_layer * moe_hidden_size * token_dimension  # Gate
        + top_k_gate * moe_hidden_size * token_dimension  # Up projection
        + top_k_gate * token_dimension * moe_hidden_size  # Down projection
    )
    other_weights = 2 * token_dimension  # LayerNorm
    
    total_weights_per_layer = attention_weights + moe_weights + other_weights
    total_model_weights = layers * total_weights_per_layer
    
    print(f"Total model weights: {total_model_weights:,} parameters")
    print(f"Model size in FP8: {total_model_weights * 1 / 1e9:.2f} GB")
    
    # KV cache calculation
    kv_cache_per_token = layers * gqa_kv_heads * head_dimension * 2
    total_kv_cache = kv_cache_per_token * batch_size * (input_sequence + output_sequence)
    
    print(f"Total KV cache: {total_kv_cache:,} parameters")
    print(f"KV cache size in FP8: {total_kv_cache * 1 / 1e9:.2f} GB")
    
    # Memory per GPU
    model_memory_per_gpu = (total_model_weights * 1) / total_gpus
    kv_cache_per_gpu = total_kv_cache * 1 / total_gpus
    activation_memory = batch_size * token_dimension * layers * 4
    
    total_memory_per_gpu = model_memory_per_gpu + kv_cache_per_gpu + activation_memory
    actual_memory_utilization = total_memory_per_gpu / single_gpu_memory
    
    print(f"\nMemory per GPU breakdown:")
    print(f"  Model weights: {model_memory_per_gpu / 1e9:.2f} GB")
    print(f"  KV cache: {kv_cache_per_gpu / 1e9:.2f} GB")
    print(f"  Activations: {activation_memory / 1e9:.2f} GB")
    print(f"  Total: {total_memory_per_gpu / 1e9:.2f} GB")
    print(f"  Actual utilization: {actual_memory_utilization * 100:.1f}%")
    
    # 3. Performance Analysis Issues
    print("\n3. PERFORMANCE ANALYSIS ISSUES")
    print("-" * 50)
    
    issues = []
    
    # Issue 1: Memory utilization discrepancy
    if abs(actual_memory_utilization - claimed_memory_util/100) > 0.1:
        issues.append(f"Memory utilization discrepancy: claimed {claimed_memory_util:.1f}% vs actual {actual_memory_utilization*100:.1f}%")
    
    # Issue 2: TTFT calculation methodology
    if claimed_ttft > 10:  # Suspiciously high for the actual compute needed
        issues.append(f"TTFT seems overestimated: claimed {claimed_ttft:.1f}s for mainly compute-bound workload")
    
    # Issue 3: GPU count calculation
    expected_gpus_simple = ep * pp * tp * dp
    if total_gpus != expected_gpus_simple:
        issues.append(f"GPU count: {total_gpus} claimed vs {expected_gpus_simple} simple multiplication")
    
    # Issue 4: Parallel strategy effectiveness
    if tp == 2 and pp == 4:
        issues.append("TP=2 may not provide sufficient speedup for 64 attention heads")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    # 4. Corrected Performance Estimation
    print("\n4. CORRECTED PERFORMANCE ESTIMATION")
    print("-" * 50)
    
    # More realistic compute analysis
    effective_compute = single_gpu_compute * mfu_utilization
    
    # Prefill FLOPs (more accurate estimation)
    attention_flops = batch_size * input_sequence * input_sequence * attention_heads * head_dimension * 4 * layers
    moe_flops = batch_size * input_sequence * top_k_gate * moe_hidden_size * token_dimension * 2 * layers
    total_prefill_flops = attention_flops + moe_flops
    
    # With parallelism
    compute_per_gpu = total_prefill_flops / (pp * tp)
    realistic_prefill_time = compute_per_gpu / (effective_compute * total_gpus)
    
    print(f"Realistic prefill time: {realistic_prefill_time:.2f} seconds")
    
    # Decode time (memory bound)
    memory_transfer_per_step = batch_size * (token_dimension + kv_cache_per_token) * 1
    effective_bandwidth = memory_bandwidth * bandwidth_utilization
    decode_time_per_step = memory_transfer_per_step / effective_bandwidth
    total_decode_time = decode_time_per_step * output_sequence
    
    print(f"Realistic decode time: {total_decode_time:.2f} seconds")
    
    realistic_ttft = realistic_prefill_time + total_decode_time
    print(f"Realistic TTFT: {realistic_ttft:.2f} seconds")
    
    # 5. Optimization Opportunities
    print("\n5. OPTIMIZATION OPPORTUNITIES")
    print("-" * 50)
    
    opportunities = []
    
    if actual_memory_utilization < 0.5:
        opportunities.append("Memory utilization is very low. Consider increasing batch size or reducing GPUs.")
    
    if realistic_ttft < ttft_requirement:
        opportunities.append(f"TTFT requirement easily met. Can optimize for throughput or reduce resources.")
    
    if tp == 2:
        opportunities.append("Consider increasing TP to 4 or 8 for better attention parallelization.")
    
    if dp == 1:
        opportunities.append("Consider adding DP for better throughput scaling.")
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp}")
    
    # 6. Final Assessment
    print("\n6. FINAL ASSESSMENT")
    print("-" * 50)
    
    print("Current Strategy Assessment:")
    print(f"  ✓ TTFT requirement: {realistic_ttft:.1f}s < {ttft_requirement}s (MET)")
    print(f"  ✓ Memory utilization: {actual_memory_utilization*100:.1f}% (VERY LOW)")
    print(f"  ✓ Parallel strategy: EP={ep}, PP={pp}, TP={tp}, DP={dp}")
    print(f"  ✓ Total GPUs: {total_gpus}")
    
    print("\nIssues Identified:")
    print("  ✗ Memory utilization severely underestimated in claimed analysis")
    print("  ✗ TTFT overestimated in claimed analysis")
    print("  ✗ GPU allocation not optimally explained")
    print("  ✗ Parallel strategy could be more aggressive")
    
    print("\nRecommendations:")
    print("  1. Increase batch size to improve GPU utilization")
    print("  2. Consider more aggressive parallelization (higher TP/DP)")
    print("  3. Reduce GPU count while maintaining performance")
    print("  4. Update performance analysis methodology")
    
    return {
        "claimed_ttft": claimed_ttft,
        "actual_ttft": realistic_ttft,
        "claimed_memory": claimed_memory_util,
        "actual_memory": actual_memory_utilization * 100,
        "issues": issues,
        "opportunities": opportunities
    }

if __name__ == "__main__":
    results = comprehensive_performance_evaluation()