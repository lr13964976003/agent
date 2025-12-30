#!/usr/bin/env python3

import math

# Model configuration
weights = 10_000_000_000  # 10B parameters
layers = 16
precision = 2  # FP16 = 2 bytes
token_dim = 512
num_heads = 16
head_dim = 32
mlp_hidden = 1024

# Input data
batch_size = 128
seq_length_min = 128
seq_length_max = 10240

# Hardware
gpu_memory = 64_000_000_000  # 64GB in bytes
single_gpu_tflops = 400  # TFlops
mfu_utilization = 0.6

# Memory calculations
param_memory = weights * precision  # 20GB
kv_cache_per_layer = num_heads * head_dim * 2 * seq_length_max * batch_size * precision
kv_cache_total = kv_cache_per_layer * layers  # 42.95 GB
activation_per_layer = batch_size * seq_length_max * token_dim * precision
activation_total = activation_per_layer * layers  # 21.47 GB

def calculate_memory_usage(TP, PP, DP, SP):
    """Calculate memory usage per GPU for given parallel configuration"""
    param_per_gpu = param_memory / (TP * PP)
    kv_per_gpu = kv_cache_total / (PP * SP)
    activation_per_gpu = activation_total / (PP * SP * TP)
    
    subtotal = param_per_gpu + kv_per_gpu + activation_per_gpu
    overhead = subtotal * 0.05
    total_per_gpu = subtotal + overhead
    
    return total_per_gpu

def calculate_performance(TP):
    """Calculate basic performance metrics"""
    effective_compute = single_gpu_tflops * mfu_utilization
    
    # Rough prefill FLOPs estimation
    attention_flops = batch_size * seq_length_max * seq_length_max * num_heads * head_dim
    ffn_flops = batch_size * seq_length_max * mlp_hidden * token_dim * layers
    total_prefill_flops = (attention_flops + ffn_flops) / 1e12
    
    # With TP, compute is roughly halved but add communication overhead
    prefill_time = (total_prefill_flops / TP) / effective_compute * 1.1
    
    # Decode throughput estimation
    token_flops = (num_heads * head_dim + mlp_hidden) * token_dim * layers / 1e12
    token_time = token_flops / effective_compute
    throughput_per_gpu = 1 / token_time
    
    return prefill_time, throughput_per_gpu

print("=== OPTIMIZATION ANALYSIS ===")
print("Testing different parallel configurations...")
print()

# Test different configurations
configs = [
    (2, 2, 1, 2),  # TP=2, PP=2, DP=1, SP=2 = 4 GPUs
    (2, 2, 2, 1),  # TP=2, PP=2, DP=2, SP=1 = 8 GPUs  
    (2, 1, 2, 2),  # TP=2, PP=1, DP=2, SP=2 = 4 GPUs
    (1, 2, 2, 2),  # TP=1, PP=2, DP=2, SP=2 = 4 GPUs
    (4, 2, 1, 1),  # TP=4, PP=2, DP=1, SP=1 = 8 GPUs
    (2, 4, 1, 1),  # TP=2, PP=4, DP=1, SP=1 = 8 GPUs
]

best_config = None
best_score = float('inf')

for TP, PP, DP, SP in configs:
    total_gpus = TP * PP * DP
    memory_per_gpu = calculate_memory_usage(TP, PP, DP, SP)
    prefill_time, throughput = calculate_performance(TP)
    
    # Check requirements
    memory_ok = memory_per_gpu < gpu_memory * 0.8  # Keep under 80%
    latency_ok = prefill_time < 10  # Under 10 seconds
    throughput_ok = throughput >= 100000  # 100k tokens/sec
    
    # Calculate efficiency score (lower is better)
    # Factor in GPU usage, memory efficiency, and performance margins
    memory_efficiency = memory_per_gpu / gpu_memory  # Higher utilization is better
    gpu_efficiency = 1 / total_gpus  # Fewer GPUs is better
    latency_margin = max(0.1, 10 - prefill_time)  # More margin is better
    throughput_margin = throughput / 100000  # Higher margin is better
    
    score = total_gpus * (1 - memory_efficiency) * (1 / latency_margin) * (1 / throughput_margin)
    
    print(f"TP={TP}, PP={PP}, DP={DP}, SP={SP} ({total_gpus} GPUs)")
    print(f"  Memory per GPU: {memory_per_gpu/1e9:.2f} GB ({memory_per_gpu/gpu_memory*100:.1f}%)")
    print(f"  Prefill time: {prefill_time:.3f}s")
    print(f"  Throughput: {throughput/1000:.1f}k tokens/s")
    print(f"  Requirements: Memory={memory_ok}, Latency={latency_ok}, Throughput={throughput_ok}")
    print(f"  Efficiency score: {score:.2f}")
    
    if memory_ok and latency_ok and throughput_ok:
        if score < best_score:
            best_score = score
            best_config = (TP, PP, DP, SP, total_gpus, memory_per_gpu, prefill_time, throughput)
    print()

if best_config:
    TP, PP, DP, SP, total_gpus, memory_per_gpu, prefill_time, throughput = best_config
    print("=== OPTIMAL CONFIGURATION ===")
    print(f"Best configuration: TP={TP}, PP={PP}, DP={DP}, SP={SP}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Memory per GPU: {memory_per_gpu/1e9:.2f} GB ({memory_per_gpu/gpu_memory*100:.1f}%)")
    print(f"Prefill time: {prefill_time:.3f}s")
    print(f"Throughput: {throughput/1000:.1f}k tokens/s")
    print()
    
    # Module division check
    print("=== MODULE DIVISION VERIFICATION ===")
    print(f"Layers per PP stage: {layers/PP} = {layers//PP}")
    print(f"Heads per TP group: {num_heads/TP} = {num_heads//TP}")
    if SP > 1:
        print(f"Sequence partition: {SP} ways")
    print(f"Memory utilization: {memory_per_gpu/gpu_memory*100:.1f}%")
else:
    print("No valid configuration found!")