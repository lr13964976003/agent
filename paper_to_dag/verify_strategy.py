#!/usr/bin/env python3

import math

def verify_parallel_strategy():
    # Model parameters
    total_params = 30e9  # 30B parameters
    num_layers = 16
    num_experts = 64
    expert_hidden_size = 2048
    token_dim = 1024
    num_heads = 16
    head_dim = 64
    
    # Hardware parameters
    num_gpus = 8
    vram_per_gpu = 64e9  # 64GB in bytes
    bandwidth = 1.8e12  # 1.8TBps in bytes/sec
    
    # Batch parameters
    batch_size = 128
    seq_length_min = 128
    seq_length_max = 10240
    
    print("=== Parallel Strategy Verification ===")
    print(f"Model: {total_params/1e9:.1f}B parameters, {num_layers} layers, {num_experts} experts")
    print(f"Hardware: {num_gpus} GPUs, {vram_per_gpu/1e9:.1f}GB VRAM each, {bandwidth/1e12:.1f}TBps bandwidth")
    print()
    
    # 1. Expert Parallelism Verification
    experts_per_gpu = num_experts // num_gpus
    print(f"1. Expert Parallelism:")
    print(f"   - Experts per GPU: {experts_per_gpu} ({num_experts} ÷ {num_gpus})")
    print(f"   - Load balanced: {'✓' if num_experts % num_gpus == 0 else '✗'}")
    print()
    
    # 2. Tensor Parallelism Verification
    tp_size = 2
    print(f"2. Tensor Parallelism:")
    print(f"   - TP size: {tp_size}")
    print(f"   - Attention split: {'✓' if num_heads % tp_size == 0 else '✗'} ({num_heads} heads)")
    print(f"   - Expert network split: {'✓' if expert_hidden_size % tp_size == 0 else '✗'} ({expert_hidden_size} hidden)")
    print()
    
    # 3. Pipeline Parallelism Verification
    pp_stages = 4
    layers_per_stage = num_layers // pp_stages
    print(f"3. Pipeline Parallelism:")
    print(f"   - Pipeline stages: {pp_stages}")
    print(f"   - Layers per stage: {layers_per_stage} ({num_layers} ÷ {pp_stages})")
    print(f"   - Balanced: {'✓' if num_layers % pp_stages == 0 else '✗'}")
    print()
    
    # 4. Total GPU Configuration
    total_gpus_needed = tp_size * pp_stages
    print(f"4. GPU Configuration:")
    print(f"   - Total GPUs needed: {total_gpus_needed} ({tp_size} TP × {pp_stages} PP)")
    print(f"   - Available GPUs: {num_gpus}")
    print(f"   - Match: {'✓' if total_gpus_needed == num_gpus else '✗'}")
    print()
    
    # 5. Memory Estimation
    # Rough estimation of memory usage
    params_per_gpu = total_params / num_gpus  # Simplified distribution
    memory_for_params = params_per_gpu * 4  # FP32: 4 bytes per parameter
    memory_for_activations = batch_size * seq_length_max * token_dim * 4 * 0.5  # Rough estimate with checkpointing
    
    total_memory_per_gpu = (memory_for_params + memory_for_activations) / 1e9  # Convert to GB
    memory_utilization = total_memory_per_gpu / (vram_per_gpu / 1e9) * 100
    
    print(f"5. Memory Analysis:")
    print(f"   - Estimated memory per GPU: {total_memory_per_gpu:.1f}GB")
    print(f"   - Available memory per GPU: {vram_per_gpu/1e9:.1f}GB")
    print(f"   - Memory utilization: {memory_utilization:.1f}%")
    print(f"   - Within limits: {'✓' if memory_utilization < 80 else '✗'}")
    print()
    
    # 6. Module Division Verification
    total_expert_modules = num_layers * num_experts
    modules_per_gpu = total_expert_modules // num_gpus
    print(f"6. Module Division:")
    print(f"   - Total expert modules: {total_expert_modules} ({num_layers} × {num_experts})")
    print(f"   - Modules per GPU: {modules_per_gpu}")
    print(f"   - Perfectly balanced: {'✓' if total_expert_modules % num_gpus == 0 else '✗'}")
    print()
    
    # Overall assessment
    issues = []
    if num_experts % num_gpus != 0:
        issues.append("Expert distribution not perfectly balanced")
    if total_gpus_needed != num_gpus:
        issues.append("GPU configuration mismatch")
    if memory_utilization > 80:
        issues.append("Memory utilization too high")
    if num_heads % tp_size != 0:
        issues.append("Attention heads not divisible by TP size")
    if expert_hidden_size % tp_size != 0:
        issues.append("Expert hidden size not divisible by TP size")
    if num_layers % pp_stages != 0:
        issues.append("Layers not perfectly divisible by PP stages")
    
    print("=== Overall Assessment ===")
    if not issues:
        print("✓ All checks passed! The parallel strategy appears sound.")
    else:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"   - {issue}")

if __name__ == "__main__":
    verify_parallel_strategy()