#!/usr/bin/env python3

import json
import math

def verify_deployment_strategy(file_path, strategy_name):
    """Verify GPU allocation and module division for a deployment strategy"""
    
    with open(file_path, 'r') as f:
        strategy = json.load(f)
    
    print(f"\n=== {strategy_name.upper()} VERIFICATION ===")
    
    # Extract parallel parameters
    pp = strategy['parallel_strategy']['pp']
    tp = strategy['parallel_strategy']['tp']
    ep = strategy['parallel_strategy']['ep']
    dp = strategy['parallel_strategy']['dp']
    sp = strategy['parallel_strategy']['sp']
    
    total_gpus = strategy['gpu_allocation']['total_gpus']
    
    print(f"Parallel Strategy: PP={pp}, TP={tp}, EP={ep}, DP={dp}, SP={sp}")
    print(f"Total GPUs: {total_gpus}")
    
    # According to knowledge file rules:
    # - EP ≈ GPU_total for MoE inference
    # - TP and EP are not multiplicative
    # - PP is outer structure
    # - GPU count is determined by outermost structural parallelism
    
    if strategy_name == "prefill":
        # For prefill: PP stages × TP groups = 4 × 8 = 32 GPUs
        calculated_gpus = pp * tp
        print(f"Calculated GPUs (PP × TP): {pp} × {tp} = {calculated_gpus}")
        
        # EP should equal total GPUs for MoE
        print(f"EP mapping: {ep} experts to {total_gpus} GPUs")        
    else:  # decode
        # For decode: PP stages × max(TP groups, DP groups) = 2 × max(4, 4) = 8
        # But we need to account for the actual GPU allocation
        calculated_gpus = pp * max(tp, dp) * (ep // ep)  # EP dominates
        print(f"Calculated GPUs based on structure: {pp} × max({tp}, {dp}) = {calculated_gpus}")
        print(f"But EP requires {ep} GPUs for experts")
        
    
    # Verify module division
    total_layers = 94
    layers_per_stage = strategy['model_partitioning']['layers_per_stage']
    
    print(f"\nModule Division:")
    print(f"Total layers: {total_layers}")
    print(f"Layers per PP stage: {layers_per_stage}")
    print(f"PP stages: {pp}")
    print(f"Total layers accounted for: {layers_per_stage} × {pp} = {layers_per_stage * pp}")
    
    # Check for balanced division
    if layers_per_stage * pp == total_layers:
        print("✓ Module division is balanced")
    else:
        print("✗ Module division is not balanced")
    
    # Performance requirements check
    ttft_target = strategy['performance_characteristics']['ttft_target']
    print(f"\nPerformance Target: TTFT ≤ {ttft_target}s")
    
    return total_gpus, calculated_gpus

def main():
    print("GPU Deployment Verification for Qwen3-235B Model")
    print("=" * 50)
    
    # Verify prefill strategy
    prefill_gpus, prefill_calc = verify_deployment_strategy(
        './outputs/2026-01-04-17-39-40/prefill_parallel_strategy.json', 
        'prefill'
    )
    
    # Verify decode strategy  
    decode_gpus, decode_calc = verify_deployment_strategy(
        './outputs/2026-01-04-17-39-40/decode_parallel_strategy.json', 
        'decode'
    )
    
    print(f"\n=== SUMMARY ===")
    print(f"Prefill GPUs: {prefill_gpus} (calculated: {prefill_calc})")
    print(f"Decode GPUs: {decode_gpus} (calculated: {decode_calc})")
    print(f"Total GPUs required: {max(prefill_gpus, decode_gpus)}")

if __name__ == "__main__":
    main()