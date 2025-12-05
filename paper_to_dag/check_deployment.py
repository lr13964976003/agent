#!/usr/bin/env python3

import math

def check_deployment_method():
    print("=== Checking Deployment Method Mathematical Accuracy ===\n")
    
    # Given parameters from the file
    total_gpus = 16
    model_params = 30e9  # 30 billion
    precision = 2  # FP16 = 2 bytes
    layers = 16
    experts = 64
    batch_size = 128
    
    # Parallel strategy degrees
    tp_degree = 2  # Tensor parallelism
    ep_degree = 4  # Expert parallelism  
    pp_degree = 2  # Pipeline parallelism
    dp_degree = 2  # Data parallelism
    
    print("1. GPU Calculation Check:")
    calculated_gpus = tp_degree * ep_degree * pp_degree * dp_degree
    print(f"   TP({tp_degree}) × EP({ep_degree}) × PP({pp_degree}) × DP({dp_degree}) = {calculated_gpus} GPUs")
    print(f"   Available GPUs: {total_gpus}")
    print(f"   Utilization: {calculated_gpus}/{total_gpus} = {calculated_gpus/total_gpus*100:.1f}%")
    
    print("\n2. Memory Analysis:")
    total_memory = model_params * precision  # 60GB
    memory_per_gpu = total_memory / tp_degree  # 30GB with tensor parallelism
    expert_overhead = memory_per_gpu * 0.1  # 10% overhead
    total_memory_per_gpu = memory_per_gpu + expert_overhead
    
    print(f"   Total model memory: {total_memory/1e9:.1f}GB")
    print(f"   Memory per GPU (with TP): {memory_per_gpu/1e9:.1f}GB")
    print(f"   Expert overhead (10%): {expert_overhead/1e9:.1f}GB")
    print(f"   Total memory per GPU: {total_memory_per_gpu/1e9:.1f}GB")
    print(f"   GPU memory limit: 64GB")
    print(f"   Memory utilization: {total_memory_per_gpu/64*100:.1f}%")
    
    print("\n3. Expert Distribution:")
    experts_per_gpu = experts / ep_degree
    print(f"   Total experts: {experts}")
    print(f"   Experts per GPU: {experts}u00f7{ep_degree} = {experts_per_gpu}")
    
    print("\n4. Layer Distribution:")
    layers_per_stage = layers / pp_degree
    print(f"   Total layers: {layers}")
    print(f"   Layers per pipeline stage: {layers}u00f7{pp_degree} = {layers_per_stage}")
    
    print("\n5. Attention Head Distribution:")
    total_heads = 16
    heads_per_gpu = total_heads / tp_degree
    print(f"   Total attention heads: {total_heads}")
    print(f"   Heads per GPU: {total_heads}u00f7{tp_degree} = {heads_per_gpu}")
    
    print("\n6. Performance Requirements Check:")
    target_latency = 50  # ms
    target_throughput = 20000  # tokens/s
    target_memory_util = 64  # GB
    target_load_balance = 90  # %
    target_gpu_util = 90  # %
    
    actual_latency = 35  # ms (from file)
    actual_throughput = 28000  # tokens/s (from file)
    actual_memory = total_memory_per_gpu / 1e9  # GB
    actual_load_balance = 88  # % (from file)
    actual_gpu_util = 88  # % (from file)
    
    print(f"   Latency: {actual_latency}ms < {target_latency}ms u2713")
    print(f"   Throughput: {actual_throughput:,} > {target_throughput:,} u2713")
    print(f"   Memory: {actual_memory:.1f}GB < {target_memory_util}GB u2713")
    print(f"   Load Balance: {actual_load_balance}% u2248 {target_load_balance}% (u2248)")
    print(f"   GPU Utilization: {actual_gpu_util}% u2248 {target_gpu_util}% (u2248)")
    
    print("\n7. Issues Identified:")
    issues = []
    
    if calculated_gpus > total_gpus:
        issues.append(f"GPU requirement ({calculated_gpus}) exceeds available ({total_gpus})")
    
    if actual_memory > target_memory_util:
        issues.append(f"Memory usage ({actual_memory:.1f}GB) exceeds limit ({target_memory_util}GB)")
    
    if actual_load_balance < target_load_balance - 5:
        issues.append(f"Load balance ({actual_load_balance}%) significantly below target ({target_load_balance}%)")
    
    if actual_gpu_util < target_gpu_util - 5:
        issues.append(f"GPU utilization ({actual_gpu_util}%) significantly below target ({target_gpu_util}%)")
    
    if not issues:
        print("   No major issues found!")
    else:
        for issue in issues:
            print(f"   - {issue}")
    
    print("\n8. Redundancy Analysis:")
    unused_gpus = total_gpus - calculated_gpus
    print(f"   GPUs used: {calculated_gpus}")
    print(f"   GPUs available: {total_gpus}")
    print(f"   Unused GPUs: {unused_gpus} ({unused_gpus/total_gpus*100:.1f}%)")
    print(f"   Redundancy factor: {total_gpus/calculated_gpus:.1f}x")
    
    return len(issues) == 0

if __name__ == "__main__":
    is_valid = check_deployment_method()