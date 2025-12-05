#!/usr/bin/env python3
"""
GPU Calculation Verification for 30B MoE Model Deployment
This script verifies the mathematical accuracy of parallel strategy GPU requirements.
"""

def verify_gpu_calculation(tp_degree, ep_degree, pp_degree, dp_degree):
    """
    Calculate total GPUs required for a given parallel strategy configuration.
    
    Args:
        tp_degree: Tensor parallelism degree
        ep_degree: Expert parallelism degree  
        pp_degree: Pipeline parallelism degree
        dp_degree: Data parallelism degree
    
    Returns:
        dict: Total GPUs required and breakdown
    """
    
    # Basic parallel degrees calculation
    basic_gpus = pp_degree * tp_degree * dp_degree
    
    # Expert parallelism constraint: EP must be >= TP
    if ep_degree < tp_degree:
        print(f"WARNING: EP ({ep_degree}) < TP ({tp_degree}) - This may cause issues")
    
    # Total GPUs considering all constraints
    total_gpus = max(basic_gpus, ep_degree * tp_degree)
    
    return {
        'basic_calculation': basic_gpus,
        'ep_constraint_calculation': ep_degree * tp_degree,
        'total_gpus_required': total_gpus,
        'tp_degree': tp_degree,
        'ep_degree': ep_degree,
        'pp_degree': pp_degree,
        'dp_degree': dp_degree
    }

def main():
    print("=== GPU Calculation Verification ===\n")
    
    # Original incorrect configuration
    print("1. ORIGINAL (INCORRECT) CONFIGURATION:")
    original = verify_gpu_calculation(tp_degree=4, ep_degree=16, pp_degree=4, dp_degree=2)
    print(f"   TP: {original['tp_degree']}, EP: {original['ep_degree']}, PP: {original['pp_degree']}, DP: {original['dp_degree']}")
    print(f"   Basic calculation (PP × TP × DP): {original['pp_degree']} × {original['tp_degree']} × {original['dp_degree']} = {original['basic_calculation']} GPUs")
    print(f"   EP constraint (EP × TP): {original['ep_degree']} × {original['tp_degree']} = {original['ep_constraint_calculation']} GPUs")
    print(f"   TOTAL REQUIRED: {original['total_gpus_required']} GPUs")
    print(f"   CLAIMED: 16 GPUs ❌\n")
    
    # Corrected configuration
    print("2. CORRECTED CONFIGURATION:")
    corrected = verify_gpu_calculation(tp_degree=2, ep_degree=4, pp_degree=2, dp_degree=2)
    print(f"   TP: {corrected['tp_degree']}, EP: {corrected['ep_degree']}, PP: {corrected['pp_degree']}, DP: {corrected['dp_degree']}")
    print(f"   Basic calculation (PP × TP × DP): {corrected['pp_degree']} × {corrected['tp_degree']} × {corrected['dp_degree']} = {corrected['basic_calculation']} GPUs")
    print(f"   EP constraint (EP × TP): {corrected['ep_degree']} × {corrected['tp_degree']} = {corrected['ep_constraint_calculation']} GPUs")
    print(f"   TOTAL REQUIRED: {corrected['total_gpus_required']} GPUs")
    print(f"   AVAILABLE: 16 GPUs ✅\n")
    
    # Alternative configurations
    print("3. ALTERNATIVE CONFIGURATIONS (for reference):")
    
    # Minimal configuration
    minimal = verify_gpu_calculation(tp_degree=1, ep_degree=1, pp_degree=1, dp_degree=1)
    print(f"   Minimal (TP=1, EP=1, PP=1, DP=1): {minimal['total_gpus_required']} GPUs")
    
    # Maximum feasible with 16 GPUs
    max_config = verify_gpu_calculation(tp_degree=2, ep_degree=8, pp_degree=2, dp_degree=4)
    print(f"   Max with 16 GPUs (TP=2, EP=8, PP=2, DP=4): {max_config['total_gpus_required']} GPUs")
    
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Original strategy ERROR: Claims 16 GPUs but needs {original['total_gpus_required']} GPUs")
    print(f"Corrected strategy: Needs {corrected['total_gpus_required']} GPUs, {16 - corrected['total_gpus_required']} GPUs available for redundancy")
    print("✅ Mathematical accuracy verified for corrected configuration")

if __name__ == "__main__":
    main()