#!/usr/bin/env python3

import json
import math

def validate_parallel_strategy():
    """Validate the PP=2×TP=4×SP=1 parallel strategy"""
    
    print("=== PARALLEL STRATEGY VALIDATION ===")
    print("Strategy: PP=2 × TP=4 × SP=1")
    print()
    
    # Basic mathematical validation
    pp = 2
    tp = 4 
    sp = 1
    total_gpus = pp * tp * sp
    
    print(f"PP (Pipeline Parallelism): {pp}")
    print(f"TP (Tensor Parallelism): {tp}")
    print(f"SP (Sequence Parallelism): {sp}")
    print(f"Total GPUs Required: {total_gpus}")
    print()
    
    # Available resources
    available_gpus = 8
    gpu_memory_gb = 80
    max_memory_utilization = 0.85
    
    print(f"Available GPUs: {available_gpus}")
    print(f"GPU Memory: {gpu_memory_gb}GB")
    print(f"Max Memory Utilization: {max_memory_utilization*100}%")
    print()
    
    # Mathematical correctness check
    if total_gpus == available_gpus:
        print("✅ MATHEMATICAL CORRECTNESS: PASSED")
        print(f"   {pp} × {tp} × {sp} = {total_gpus} GPUs (exact match)")
    else:
        print("❌ MATHEMATICAL CORRECTNESS: FAILED")
        print(f"   Required: {total_gpus}, Available: {available_gpus}")
    print()
    
    # Model configuration
    total_layers = 80
    model_size_gb = 140  # FP16
    
    print(f"Model Layers: {total_layers}")
    print(f"Model Size: {model_size_gb}GB (FP16)")
    print()
    
    # Layer distribution
    layers_per_pp_stage = total_layers // pp
    print(f"Layers per PP stage: {layers_per_pp_stage}")
    if total_layers % pp == 0:
        print("✅ LAYER DISTRIBUTION: EVEN")
    else:
        print("⚠️  LAYER DISTRIBUTION: UNEVEN")
    print()
    
    # Memory analysis
    memory_per_gpu = model_size_gb / (pp * tp)  # SP doesn't affect model weights
    max_memory_per_gpu = gpu_memory_gb * max_memory_utilization
    
    print(f"Model weights per GPU: {memory_per_gpu:.1f}GB")
    print(f"Max memory per GPU: {max_memory_per_gpu:.1f}GB")    
    if memory_per_gpu <= max_memory_per_gpu:
        print("✅ MEMORY CONSTRAINT: SATISFIED")
        print(f"   Utilization: {memory_per_gpu/gpu_memory_gb*100:.1f}%")
    else:
        print("❌ MEMORY CONSTRAINT: VIOLATED")
        print(f"   Required: {memory_per_gpu:.1f}GB, Max: {max_memory_per_gpu:.1f}GB")
    print()
    
    # Performance targets from deployment plan
    target_throughput = 8.0  # RPS
    target_latency_p99 = 100.0  # ms
    projected_throughput = 8.1  # RPS
    projected_latency = 27.4  # ms
    
    print("=== PERFORMANCE VALIDATION ===")
    print(f"Target Throughput: {target_throughput} RPS")
    print(f"Projected Throughput: {projected_throughput} RPS")
    if projected_throughput >= target_throughput:
        print("✅ THROUGHPUT TARGET: MET")
    else:
        print("❌ THROUGHPUT TARGET: MISSED")
    print()
    
    print(f"Target Latency P99: {target_latency_p99}ms")
    print(f"Projected Latency P99: {projected_latency}ms")
    if projected_latency <= target_latency_p99:
        print("✅ LATENCY TARGET: MET")
    else:
        print("❌ LATENCY TARGET: MISSED")
    print()
    
    # Load balancing check
    print("=== LOAD BALANCING ===")
    gpus_per_pp_stage = tp * sp
    total_gpus_needed = pp * gpus_per_pp_stage
    print(f"GPUs per PP stage: {gpus_per_pp_stage}")
    print(f"Total GPUs needed: {total_gpus_needed}")
    
    if total_gpus_needed == available_gpus:
        print("✅ LOAD BALANCING: OPTIMAL")
        print("   Perfect GPU utilization with no waste")
    else:
        print("⚠️  LOAD BALANCING: SUBOPTIMAL")
    print()
    
    # Overall assessment
    print("=== OVERALL ASSESSMENT ===")
    issues = []
    
    if total_gpus != available_gpus:
        issues.append("Mathematical incorrectness")
    if memory_per_gpu > max_memory_per_gpu:
        issues.append("Memory constraint violation")
    if projected_throughput < target_throughput:
        issues.append("Throughput target missed")
    if projected_latency > target_latency_p99:
        issues.append("Latency target missed")
    if total_layers % pp != 0:
        issues.append("Uneven layer distribution")
    
    if not issues:
        print("✅ STRATEGY VALIDATION: PASSED")
        print("   Ready for production deployment")
    else:
        print("❌ STRATEGY VALIDATION: FAILED")
        for issue in issues:
            print(f"   - {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    validate_parallel_strategy()