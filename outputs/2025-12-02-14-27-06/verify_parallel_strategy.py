#!/usr/bin/env python3

import json
import math

def verify_parallel_strategy():
    """Verify the parallel strategy compatibility and optimization potential"""
    
    # Load the deployment configuration
    config = {
        "strategy_name": "EP3_TP2_PP1_Enhanced",
        "hardware_assumption": "3 GPUs, 64 GB each, 400 TFLOPS per GPU",
        "model_parameters": {
            "layers": 24,
            "experts_per_layer": 63,
            "token_dim": 4096,
            "batch_size": 64,
            "seq_len": 1024,
            "ffn_hidden_size": 16384,
            "attention_heads": 32,
            "vocab_size": 50000
        },
        "parallel_configuration": {
            "expert_parallelism": 3,
            "tensor_parallelism": 2,
            "pipeline_parallelism": 1
        },
        "expert_distribution": {
            "gpu_0": 21,
            "gpu_1": 21,
            "gpu_2": 21
        },
        "tensor_parallel_distribution": {
            "gpu_0": ["attention_qkv_0", "mlp_fc1_0", "mlp_fc2_0"],
            "gpu_1": ["attention_qkv_1", "mlp_fc1_1", "mlp_fc2_1"],
            "gpu_2": ["attention_out_0", "mlp_out_1"]
        }
    }
    
    issues = []
    warnings = []
    
    # 1. Check hardware compatibility
    print("=== Hardware Compatibility Check ===")
    gpus = 3
    memory_per_gpu = 64  # GB
    flops_per_gpu = 400  # TFLOPS
    
    # 2. Check parallel configuration consistency
    print("\n=== Parallel Configuration Check ===")
    ep = config["parallel_configuration"]["expert_parallelism"]
    tp = config["parallel_configuration"]["tensor_parallelism"]
    pp = config["parallel_configuration"]["pipeline_parallelism"]
    
    print(f"EP={ep}, TP={tp}, PP={pp}")
    
    # Check if EP divides experts evenly
    experts_per_layer = config["model_parameters"]["experts_per_layer"]
    if experts_per_layer % ep != 0:
        issues.append(f"Experts per layer ({experts_per_layer}) not divisible by EP ({ep})")
    else:
        experts_per_gpu = experts_per_layer // ep
        print(f"Experts per GPU: {experts_per_gpu}")
    
    # Check TP consistency
    if tp != 2:
        warnings.append(f"TP={tp} is not standard (usually 2, 4, or 8)")
    
    # Check tensor parallel distribution
    print("\n=== Tensor Parallel Distribution Check ===")
    tp_dist = config["tensor_parallel_distribution"]
    
    # With TP=2, we should have 2-way splitting, but we have 3 GPUs
    if len(tp_dist) != 2 and tp == 2:
        issues.append(f"TP=2 but tensor_parallel_distribution has {len(tp_dist)} GPUs")
    
    # Check if the distribution makes sense
    for gpu, modules in tp_dist.items():
        print(f"{gpu}: {modules}")
    
    # Check expert distribution
    print("\n=== Expert Distribution Check ===")
    expert_dist = config["expert_distribution"]
    total_experts = sum(expert_dist.values())
    
    if total_experts != experts_per_layer:
        issues.append(f"Total experts in distribution ({total_experts}) != experts per layer ({experts_per_layer})")
    
    # Check if experts are evenly distributed
    expert_counts = list(expert_dist.values())
    if len(set(expert_counts)) > 1:
        warnings.append("Experts not evenly distributed across GPUs")
    
    print(f"Expert distribution: {expert_dist}")
    print(f"Total experts: {total_experts}")
    
    # 3. Check memory utilization
    print("\n=== Memory Utilization Check ===")
    token_dim = config["model_parameters"]["token_dim"]
    batch_size = config["model_parameters"]["batch_size"]
    seq_len = config["model_parameters"]["seq_len"]
    vocab_size = config["model_parameters"]["vocab_size"]
    layers = config["model_parameters"]["layers"]
    
    # Calculate activation memory
    activation_memory = batch_size * seq_len * token_dim * 4  # 4 bytes for FP32
    activation_memory_gb = activation_memory / (1024**3)
    
    # Calculate parameter memory
    # Rough estimate for transformer model
    param_memory_gb = (layers * token_dim * token_dim * 4 * 4) / (1024**3)  # 4 matrices per layer
    
    # Expert memory (MoE)
    expert_memory_gb = (experts_per_layer * token_dim * 16384 * 4) / (1024**3)  # FFN size
    
    total_memory_gb = activation_memory_gb + param_memory_gb + expert_memory_gb
    memory_per_gpu_estimated = total_memory_gb / gpus
    
    print(f"Estimated activation memory: {activation_memory_gb:.2f} GB")
    print(f"Estimated parameter memory: {param_memory_gb:.2f} GB")
    print(f"Estimated expert memory: {expert_memory_gb:.2f} GB")
    print(f"Estimated total memory: {total_memory_gb:.2f} GB")
    print(f"Estimated memory per GPU: {memory_per_gpu_estimated:.2f} GB")
    
    if memory_per_gpu_estimated > memory_per_gpu:
        issues.append(f"Estimated memory per GPU ({memory_per_gpu_estimated:.2f} GB) exceeds available memory ({memory_per_gpu} GB)")
    
    # 4. Check compute utilization
    print("\n=== Compute Utilization Check ===")
    
    # Rough FLOPs calculation for transformer
    seq_flops = batch_size * seq_len * token_dim * token_dim * 4 * layers
    expert_flops = batch_size * seq_len * token_dim * 16384 * 2 * experts_per_layer
    total_flops = seq_flops + expert_flops
    
    total_tf = total_flops / (10**12)
    tf_per_gpu = total_tf / gpus
    
    print(f"Estimated total TFLOPs: {total_tf:.2f}")
    print(f"Estimated TFLOPs per GPU: {tf_per_gpu:.2f}")
    print(f"Available TFLOPs per GPU: {flops_per_gpu}")
    
    if tf_per_gpu > flops_per_gpu:
        issues.append(f"Estimated compute per GPU ({tf_per_gpu:.2f} TFLOPs) exceeds available compute ({flops_per_gpu} TFLOPs)")
    
    # 5. Optimization potential
    print("\n=== Optimization Potential Check ===")
    
    utilization = tf_per_gpu / flops_per_gpu * 100
    print(f"Estimated compute utilization: {utilization:.1f}%")
    
    if utilization < 70:
        warnings.append("Low compute utilization - consider increasing batch size or model size")
    elif utilization > 90:
        warnings.append("High compute utilization - may cause performance issues")
    
    return issues, warnings

if __name__ == "__main__":
    issues, warnings = verify_parallel_strategy()
    
    print("\n=== SUMMARY ===")
    if issues:
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("No issues found - strategy appears compatible and optimized!")
    
    print(f"\nTotal issues: {len(issues)}")
    print(f"Total warnings: {len(warnings)}")