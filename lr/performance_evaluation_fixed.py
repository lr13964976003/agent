#!/usr/bin/env python3

import json
import math

def calculate_performance_metrics():
    # Hardware specifications from input
    single_gpu_compute = 400 * 1e12  # 400 TFlops
    single_gpu_memory = 64 * 1e9  # 64 GB
    memory_bandwidth = 1.8 * 1e12  # 1.8 TBps
    bandwidth_utilization = 0.8
    mfu_utilization = 0.6
    
    # Model configuration
    layers = 94
    experts_per_layer = 128
    precision = "FP8"  # 1 byte per parameter
    token_dimension = 4096
    attention_heads = 64
    head_dimension = 64
    moe_hidden_size = 1536
    top_k_gate = 8
    vocabulary_size = 151936
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
    
    print("=== PERFORMANCE EVALUATION REPORT ===\n")
    
    # 1. Model Weights Calculation
    print("1. MODEL WEIGHTS ANALYSIS")
    print("-" * 40)
    
    # Attention weights per layer
    attention_weights = (
        attention_heads * head_dimension * token_dimension  # Q projection
        + attention_heads * head_dimension * token_dimension  # K projection
        + attention_heads * head_dimension * token_dimension  # V projection
        + attention_heads * head_dimension * token_dimension  # O projection
    )
    
    # MoE weights per layer (assuming top-k experts active)
    moe_weights = (
        experts_per_layer * moe_hidden_size * token_dimension  # Gate
        + top_k_gate * moe_hidden_size * token_dimension  # Up projection
        + top_k_gate * token_dimension * moe_hidden_size  # Down projection
    )
    
    # Other layer weights (LayerNorm, embeddings, etc.)
    other_weights = 2 * token_dimension  # LayerNorm
    
    total_weights_per_layer = attention_weights + moe_weights + other_weights
    total_model_weights = layers * total_weights_per_layer
    
    print(f"Attention weights per layer: {attention_weights:,} parameters")
    print(f"MoE weights per layer: {moe_weights:,} parameters")
    print(f"Total weights per layer: {total_weights_per_layer:,} parameters")
    print(f"Total model weights: {total_model_weights:,} parameters")
    print(f"Model size in FP8: {{total_model_weights * 1 / 1e9:.2f}} GB")
    
    # 2. KV Cache Memory Calculation
    print("\n2. KV CACHE MEMORY ANALYSIS")
    print("-" * 40)
    
    kv_cache_per_token = (
        layers * gqa_kv_heads * head_dimension * 2  # K + V
    )
    total_kv_cache = kv_cache_per_token * batch_size * (input_sequence + output_sequence)
    
    print(f"KV cache per token: {kv_cache_per_token} parameters")
    print(f"Total KV cache for batch: {total_kv_cache:,} parameters")
    print(f"KV cache size in FP8: {total_kv_cache * 1 / 1e9:.2f} GB")
    
    # 3. Memory Utilization Analysis
    print("\n3. MEMORY UTILIZATION ANALYSIS")
    print("-" * 40)
    
    model_memory_per_gpu = (total_model_weights * 1) / total_gpus  # FP8
    kv_cache_per_gpu = total_kv_cache * 1 / total_gpus  # FP8
    activation_memory = batch_size * token_dimension * layers * 4  # Rough estimate
    
    total_memory_per_gpu = model_memory_per_gpu + kv_cache_per_gpu + activation_memory
    memory_utilization = total_memory_per_gpu / single_gpu_memory
    
    print(f"Model memory per GPU: {model_memory_per_gpu / 1e9:.2f} GB")
    print(f"KV cache per GPU: {kv_cache_per_gpu / 1e9:.2f} GB")
    print(f"Activation memory per GPU: {activation_memory / 1e9:.2f} GB")
    print(f"Total memory per GPU: {total_memory_per_gpu / 1e9:.2f} GB")
    print(f"Memory utilization: {memory_utilization * 100:.1f}%")
    
    # 4. Compute Analysis for Prefill
    print("\n4. PREFILL COMPUTE ANALYSIS")
    print("-" * 40)
    
    # Attention FLOPs (O(L²) complexity)
    attention_flops = (
        batch_size * input_sequence * input_sequence * attention_heads * head_dimension * 4
    ) * layers
    
    # MoE FLOPs
    moe_flops = (
        batch_size * input_sequence * top_k_gate * moe_hidden_size * token_dimension * 2
    ) * layers
    
    total_prefill_flops = attention_flops + moe_flops
    effective_compute = single_gpu_compute * mfu_utilization
    
    # With parallel strategies
    compute_per_gpu = total_prefill_flops / (pp * tp)
    prefill_time = compute_per_gpu / effective_compute
    
    print(f"Attention FLOPs: {attention_flops / 1e12:.2f} TFLOPs")
    print(f"MoE FLOPs: {moe_flops / 1e12:.2f} TFLOPs")
    print(f"Total prefill FLOPs: {total_prefill_flops / 1e12:.2f} TFLOPs")
    print(f"Effective compute per GPU: {effective_compute / 1e12:.2f} TFLOPs")
    print(f"Compute per GPU (with parallelism): {compute_per_gpu / 1e12:.2f} TFLOPs")
    print(f"Estimated prefill time: {prefill_time:.2f} seconds")
    
    # 5. Decode Analysis
    print("\n5. DECODE COMPUTE ANALYSIS")
    print("-" * 40)
    
    # Decode is memory-bound, not compute-bound
    # Estimate based on memory bandwidth
    memory_transfer_per_token = (
        batch_size * (token_dimension + kv_cache_per_token) * 1  # FP8
    )
    
    effective_bandwidth = memory_bandwidth * bandwidth_utilization
    decode_time_per_token = memory_transfer_per_token / effective_bandwidth
    total_decode_time = decode_time_per_token * output_sequence
    
    print(f"Memory transfer per token: {memory_transfer_per_token / 1e9:.2f} GB")
    print(f"Effective memory bandwidth: {effective_bandwidth / 1e12:.2f} TB/s")
    print(f"Decode time per token: {decode_time_per_token * 1000:.2f} ms")
    print(f"Total decode time: {total_decode_time:.2f} seconds")
    
    # 6. Total TTFT Analysis
    print("\n6. TOTAL TTFT ANALYSIS")
    print("-" * 40)
    
    total_ttft = prefill_time + total_decode_time
    meets_ttft = total_ttft <= ttft_requirement
    
    print(f"Prefill time: {prefill_time:.2f} seconds")
    print(f"Decode time: {total_decode_time:.2f} seconds")
    print(f"Total TTFT: {total_ttft:.2f} seconds")
    print(f"TTFT requirement: {ttft_requirement} seconds")
    print(f"Meets TTFT requirement: {meets_ttft}")
    
    # 7. Throughput Analysis
    print("\n7. THROUGHPUT ANALYSIS")
    print("-" * 40)
    
    total_tokens = batch_size * (input_sequence + output_sequence)
    throughput = total_tokens / total_ttft
    
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Total time: {total_ttft:.2f} seconds")
    print(f"Throughput: {throughput:.0f} tokens/second")
    
    # 8. Optimization Recommendations
    print("\n8. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if memory_utilization > 0.8:
        recommendations.append("Memory utilization is high. Consider increasing PP or reducing batch size.")
    
    if not meets_ttft:
        recommendations.append("TTFT requirement not met. Consider increasing TP or reducing sequence length.")
    
    if prefill_time > total_decode_time:
        recommendations.append("Prefill dominates latency. Focus on TP optimization.")
    else:
        recommendations.append("Decode dominates latency. Focus on memory bandwidth optimization.")
    
    if throughput < 1000:
        recommendations.append("Throughput is low. Consider increasing DP for better utilization.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # 9. Summary
    print("\n9. SUMMARY")
    print("-" * 40)
    print(f"Memory Utilization: {memory_utilization * 100:.1f}%")
    print(f"TTFT: {total_ttft:.2f} seconds (Requirement: {ttft_requirement}s)")
    print(f"Throughput: {throughput:.0f} tokens/second")
    print(f"Total GPUs: {total_gpus}")
    print(f"Parallel Strategy: EP={ep}, PP={pp}, TP={tp}, DP={dp}")
    
    return {
        "memory_utilization": memory_utilization,
        "ttft": total_ttft,
        "meets_ttft": meets_ttft,
        "throughput": throughput,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    results = calculate_performance_metrics()