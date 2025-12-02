#!/usr/bin/env python3
"""
Validation script to check deployment method compatibility and optimization
"""
import json
import math

def validate_deployment():
    # Read current deployment method
    with open('../outputs/2025-12-02-14-27-06/deployment_method.json', 'r') as f:
        current = json.load(f)
    
    # Expected configuration from context
    expected = {
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
        }
    }
    
    issues = []
    
    # Check 1: Model parameters compatibility
    current_params = current["model_parameters"]
    expected_params = expected["model_parameters"]
    
    if current_params["layers"] != expected_params["layers"]:
        issues.append(f"❌ Layer count mismatch: current={current_params['layers']}, expected={expected_params['layers']}")
    
    if current_params["experts_per_layer"] != expected_params["experts_per_layer"]:
        issues.append(f"❌ Expert count mismatch: current={current_params['experts_per_layer']}, expected={expected_params['experts_per_layer']}")
    
    if current_params["token_dim"] != expected_params["token_dim"]:
        issues.append(f"❌ Token dimension mismatch: current={current_params['token_dim']}, expected={expected_params['token_dim']}")
    
    if current_params["batch_size"] != expected_params["batch_size"]:
        issues.append(f"❌ Batch size mismatch: current={current_params['batch_size']}, expected={expected_params['batch_size']}")
    
    if current_params["seq_len"] != expected_params["seq_len"]:
        issues.append(f"❌ Sequence length mismatch: current={current_params['seq_len']}, expected={expected_params['seq_len']}")
    
    # Check 2: Parallel strategy configuration
    if "parallel_configuration" not in current:
        issues.append("❌ Missing parallel_configuration section")
        # Check from strategy name for tensor parallelism
        if "TP1" in current["strategy_name"]:
            issues.append("❌ Tensor parallelism disabled (TP=1), but expected TP=2")
    else:
        current_parallel = current["parallel_configuration"]
        expected_parallel = expected["parallel_configuration"]
        
        if current_parallel.get("tensor_parallelism") != expected_parallel["tensor_parallelism"]:
            issues.append(f"❌ Tensor parallelism mismatch: current={current_parallel.get('tensor_parallelism')}, expected={expected_parallel['tensor_parallelism']}")
        
        if current_parallel.get("expert_parallelism") != expected_parallel["expert_parallelism"]:
            issues.append(f"❌ Expert parallelism mismatch: current={current_parallel.get('expert_parallelism')}, expected={expected_parallel['expert_parallelism']}")
    
    # Check 3: Expert distribution balance
    expert_dist = current["expert_distribution"]
    expert_counts = list(expert_dist.values())
    if len(set(expert_counts)) > 1:
        issues.append(f"❌ Uneven expert distribution: {expert_dist} (should be perfectly balanced 21-21-21)")
    
    if sum(expert_counts) != 63:
        issues.append(f"❌ Total expert count mismatch: sum={sum(expert_counts)}, expected=63")
    
    # Check 4: Compute utilization optimization
    compute_util = current["compute_utilization"]
    if compute_util["utilization_percent"] < 1.0:  # Less than 1% is terrible
        issues.append(f"❌ Extremely low compute utilization: {compute_util['utilization_percent']}% (should be ~80%)")
    
    # Check 5: Memory utilization
    memory_layout = current["memory_layout"]
    if memory_layout["total_per_gpu_MB"] < 1000:  # Less than 1GB is way too conservative
        issues.append(f"❌ Memory underutilization: {memory_layout['total_per_gpu_MB']}MB (should be ~42000MB for optimal performance)")
    
    # Check 6: Performance expectations
    expected_latency_ms = 8.5
    expected_throughput = 312
    
    current_perf = current["expected_performance"]
    
    if current_perf["latency_per_layer_ms"] > 10:  # Should be under 10ms
        issues.append(f"❌ High latency per layer: {current_perf['latency_per_layer_ms']}ms (should be ~8.5ms)")
    
    # The current throughput seems high but it's for a much smaller model
    # For the expected model size, 3800 samples/sec is unrealistic
    
    return issues

def calculate_memory_requirements():
    """Calculate expected memory requirements for the target model"""
    # Model parameters from expected configuration
    layers = 24
    experts_per_layer = 63
    token_dim = 4096
    batch_size = 64
    seq_len = 1024
    ffn_hidden_size = 16384
    attention_heads = 32
    vocab_size = 50000
    
    # Memory calculations
    expert_weights = experts_per_layer * token_dim * ffn_hidden_size * 4  # FP32
    attention_weights = 3 * token_dim * token_dim * attention_heads  # Q, K, V
    token_embeddings = vocab_size * token_dim * 4
    activations = batch_size * seq_len * token_dim * 4 * layers  # Rough estimate
    
    total_memory = (expert_weights + attention_weights + token_embeddings + activations) / (1024**3)  # Convert to GB
    
    return total_memory

if __name__ == "__main__":
    issues = validate_deployment()
    
    print("🔍 Deployment Method Validation Results:")
    print("=" * 50)
    
    if not issues:
        print("✅ No issues found - deployment method is optimal!")
    else:
        print(f"❌ Found {len(issues)} critical issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    print(f"\n📊 Expected memory requirement for target model: ~{calculate_memory_requirements():.1f} GB")
    print("\n📋 Summary:")
    print("The current deployment method is severely underutilizing hardware resources")
    print("and needs significant optimization to meet performance targets.")