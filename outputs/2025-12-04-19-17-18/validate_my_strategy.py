#!/usr/bin/env python3

import json
import sys

def validate_parallel_strategy(strategy_file):
    """Validate the parallel strategy for compatibility and optimization"""
    
    # Load the strategy
    with open(strategy_file, 'r') as f:
        strategy = json.load(f)
    
    print("=== PARALLEL STRATEGY VALIDATION ===")
    print(f"Strategy: {strategy['strategy']}")
    print(f"GPU Count: {strategy['gpu_count']}")
    
    issues = []
    warnings = []
    
    # 1. Check GPU allocation consistency
    print("\n1. GPU Allocation Check:")
    all_gpus = set()
    for stage in strategy['stages']:
        stage_gpus = set(stage['gpu_ids'])
        all_gpus.update(stage_gpus)
        print(f"  {stage['name']}: GPUs {stage['gpu_ids']}")
    
    # Check if GPU IDs are within range
    max_gpu_id = max(all_gpus)
    if max_gpu_id >= strategy['gpu_count']:
        issues.append(f"GPU ID {max_gpu_id} exceeds available GPU count {strategy['gpu_count']}")
    
    # Check for GPU overlaps in tensor splits
    print("\n2. Tensor Split Device Check:")
    for stage in strategy['stages']:
        if 'tensor_splits' in stage:
            for tensor_name, split_config in stage['tensor_splits'].items():
                devices = split_config['devices']
                print(f"  {tensor_name}: devices {devices}")
                
                # Check if devices are within GPU count
                for device in devices:
                    if device >= strategy['gpu_count']:
                        issues.append(f"Tensor {tensor_name} uses GPU {device} which exceeds available {strategy['gpu_count']} GPUs")
    
    # 3. Check communication patterns
    print("\n3. Communication Pattern Check:")
    for comm_name, comm_config in strategy['communication'].items():
        from_devices = comm_config['from_devices']
        to_devices = comm_config['to_devices']
        print(f"  {comm_name}: {from_devices} -> {to_devices}")
        
        # Check device validity
        all_comm_devices = set(from_devices + to_devices)
        for device in all_comm_devices:
            if device >= strategy['gpu_count']:
                issues.append(f"Communication {comm_name} involves GPU {device} which exceeds available {strategy['gpu_count']} GPUs")
    
    # 4. Check load balancing
    print("\n4. Load Balancing Check:")
    compute_dist = strategy['load_balancing']['compute_distribution']
    for gpu_name, workload in compute_dist.items():
        print(f"  {gpu_name}: {workload}")
        
        # Parse GPU number
        if 'gpu_' in gpu_name:
            gpu_num = int(gpu_name.split('_')[1].split('-')[0])
            if gpu_num >= strategy['gpu_count']:
                issues.append(f"Load balancing includes {gpu_name} which exceeds available {strategy['gpu_count']} GPUs")
    
    # 5. Check memory requirements
    print("\n5. Memory Requirements Check:")
    memory_balance = strategy['load_balancing']['memory_balance']
    max_memory = memory_balance['max_memory_per_gpu']
    print(f"  Max memory per GPU: {max_memory}")
    
    # Parse memory (4096*1024*4 bytes = 16MB - seems very low)
    try:
        memory_bytes = eval(max_memory.replace('bytes', '').strip())
        if memory_bytes < 1024*1024*1024:  # Less than 1GB
            warnings.append(f"Max memory per GPU is very low: {memory_bytes} bytes ({memory_bytes/1024/1024:.2f} MB)")
    except:
        warnings.append(f"Could not parse memory configuration: {max_memory}")
    
    # 6. Check performance metrics
    print("\n6. Performance Metrics Check:")
    metrics = strategy['performance_metrics']
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
        
        # Check GPU utilization claim
        if 'gpu_utilization' in metric_name:
            if '98%' in str(metric_value):
                warnings.append("GPU utilization claim of 98% may be optimistic for this configuration")
    
    # 7. Check for optimization opportunities
    print("\n7. Optimization Analysis:")
    
    # Check if tensor parallel stages are properly connected
    if len(strategy['stages']) >= 2:
        # Check if there's a bottleneck between embedding and expert stages
        embed_devices = set(strategy['stages'][0]['gpu_ids'])
        expert_devices = set(strategy['stages'][1]['gpu_ids'])
        
        if len(embed_devices.intersection(expert_devices)) == 0:
            warnings.append("No direct GPU overlap between embedding and expert stages - may cause communication overhead")
    
    # Check data parallel configuration
    agg_stage = strategy['stages'][2]
    if agg_stage['parallel_mode'] == 'data_parallel' and len(agg_stage['gpu_ids']) > 1:
        print("  Data parallel configuration looks good")
    
    # Check module division
    total_modules = strategy['module_division']['total_modules']
    gpu_count = strategy['gpu_count']
    if total_modules != gpu_count:
        issues.append(f"Module division mismatch: {total_modules} modules for {gpu_count} GPUs")
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    if issues:
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No critical issues found.")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No warnings found.")
    
    return len(issues) == 0, issues, warnings

if __name__ == "__main__":
    strategy_file = "optimized_parallel_strategy.json"
    is_valid, issues, warnings = validate_parallel_strategy(strategy_file)
    
    if is_valid:
        print("\n✅ PARALLEL STRATEGY IS VALID")
        sys.exit(0)
    else:
        print("\n❌ PARALLEL STRATEGY HAS ISSUES")
        sys.exit(1)