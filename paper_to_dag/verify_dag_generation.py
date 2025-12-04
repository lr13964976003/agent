#!/usr/bin/env python3

import json
import sys

def verify_dag_generation():
    """Verify if the deployment method file contains sufficient information for DAG generation"""
    
    # Read the deployment method file
    try:
        with open('../outputs/2025-12-04-11-03-13/enhanced_parallel_strategy.json', 'r') as f:
            deployment_data = json.load(f)
    except Exception as e:
        print(f"Error reading deployment method file: {e}")
        return False
    
    deployment_method = deployment_data.get('deployment_method', {})
    
    # Check required components for DAG generation
    checks = {
        'hardware_environment': False,
        'model_specifications': False,
        'parallel_strategy': False,
        'gpu_assignment_matrix': False,
        'communication_patterns': False,
        'memory_analysis': False,
        'compute_analysis': False
    }
    
    # Verify each component
    if 'hardware_environment' in deployment_method:
        hw_env = deployment_method['hardware_environment']
        checks['hardware_environment'] = all(key in hw_env for key in ['total_gpus', 'gpu_memory_gb', 'gpu_compute_tflops'])
    
    if 'model_specifications' in deployment_method:
        model_spec = deployment_method['model_specifications']
        checks['model_specifications'] = all(key in model_spec for key in ['layers', 'total_experts', 'token_dimension'])
    
    if 'parallel_strategy' in deployment_method:
        parallel_strat = deployment_method['parallel_strategy']
        checks['parallel_strategy'] = all(key in parallel_strat for key in ['expert_parallelism', 'tensor_parallelism', 'total_gpus_used'])
    
    if 'gpu_assignment_matrix' in deployment_method:
        gpu_matrix = deployment_method['gpu_assignment_matrix']
        checks['gpu_assignment_matrix'] = len(gpu_matrix) > 0 and 'expert_0' in gpu_matrix
    
    if 'advanced_optimizations' in deployment_method:
        adv_opt = deployment_method['advanced_optimizations']
        checks['communication_patterns'] = 'communication' in adv_opt
    
    if 'memory_analysis' in deployment_method:
        checks['memory_analysis'] = True
    
    if 'compute_analysis' in deployment_method:
        checks['compute_analysis'] = True
    
    # Overall assessment
    all_checks_passed = all(checks.values())
    
    print("=== DAG Generation Verification Results ===")
    print(f"Hardware Environment: {'✓' if checks['hardware_environment'] else '✗'}")
    print(f"Model Specifications: {'✓' if checks['model_specifications'] else '✗'}")
    print(f"Parallel Strategy: {'✓' if checks['parallel_strategy'] else '✗'}")
    print(f"GPU Assignment Matrix: {'✓' if checks['gpu_assignment_matrix'] else '✗'}")
    print(f"Communication Patterns: {'✓' if checks['communication_patterns'] else '✗'}")
    print(f"Memory Analysis: {'✓' if checks['memory_analysis'] else '✗'}")
    print(f"Compute Analysis: {'✓' if checks['compute_analysis'] else '✗'}")
    print(f"\nOverall Result: {'✓ SUFFICIENT for DAG generation' if all_checks_passed else '✗ INSUFFICIENT for DAG generation'}")
    
    # Additional verification for specific DAG requirements
    if all_checks_passed:
        print("\n=== Additional DAG Capabilities ===")
        
        # Check if we can create nodes and edges
        gpu_matrix = deployment_method['gpu_assignment_matrix']
        expert_count = len(gpu_matrix)
        print(f"Can create {expert_count} expert nodes ✓")
        
        # Check communication patterns
        comm_patterns = deployment_method['advanced_optimizations']['communication']
        print(f"Communication patterns available: {list(comm_patterns.keys())}")
        
        # Check for hierarchical structure
        print(f"Hierarchical communication: {'hierarchical_all2all' in comm_patterns}")
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = verify_dag_generation()
    sys.exit(0 if success else 1)