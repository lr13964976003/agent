#!/usr/bin/env python3
"""
Simple validation script to check deployment method issues
"""
import json

def main():
    # Read current deployment method
    with open('../outputs/2025-12-02-14-27-06/deployment_method.json', 'r') as f:
        current = json.load(f)
    
    issues = []
    
    print("🔍 Deployment Method Validation Results:")
    print("=" * 50)
    
    # Expected vs Current comparison
    expected_model = {
        "layers": 24,
        "experts_per_layer": 63,
        "token_dim": 4096,
        "batch_size": 64,
        "seq_len": 1024
    }
    
    current_model = current["model_parameters"]
    
    print("\n📊 Model Parameters Comparison:")
    for param, expected_val in expected_model.items():
        current_val = current_model.get(param, "N/A")
        if current_val != expected_val:
            issues.append(f"Model parameter '{param}': current={current_val}, expected={expected_val}")
            print(f"❌ {param}: {current_val} ≠ {expected_val}")
        else:
            print(f"✅ {param}: {current_val}")
    
    # Check parallel configuration
    print("\n🔧 Parallel Configuration:")
    strategy_name = current["strategy_name"]
    if "TP1" in strategy_name:
        issues.append("Tensor parallelism disabled (TP=1), expected TP=2")
        print(f"❌ Tensor Parallelism: TP=1 (disabled), expected TP=2")
    else:
        print(f"✅ Tensor Parallelism: Found in strategy name")
    
    # Check expert distribution
    print("\n📈 Expert Distribution:")
    expert_dist = current["expert_distribution"]
    expert_counts = list(expert_dist.values())
    if len(set(expert_counts)) > 1:
        issues.append(f"Uneven expert distribution: {expert_dist}")
        print(f"❌ Uneven distribution: {expert_dist}")
    else:
        print(f"✅ Even distribution: {expert_dist}")
    
    if sum(expert_counts) != 63:
        issues.append(f"Total experts = {sum(expert_counts)}, expected 63")
        print(f"❌ Total experts: {sum(expert_counts)} ≠ 63")
    else:
        print(f"✅ Total experts: {sum(expert_counts)}")
    
    # Check utilization
    print("\n⚡ Resource Utilization:")
    compute_util = current["compute_utilization"]["utilization_percent"]
    if compute_util < 1.0:
        issues.append(f"Extremely low compute utilization: {compute_util}%")
        print(f"❌ Compute utilization: {compute_util}% (should be ~80%)")
    else:
        print(f"✅ Compute utilization: {compute_util}%")
    
    memory_mb = current["memory_layout"]["total_per_gpu_MB"]
    if memory_mb < 1000:
        issues.append(f"Memory underutilization: {memory_mb}MB")
        print(f"❌ Memory usage: {memory_mb}MB (should be ~42000MB)")
    else:
        print(f"✅ Memory usage: {memory_mb}MB")
    
    print(f"\n📋 Summary:")
    if issues:
        print(f"❌ Found {len(issues)} critical issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\n🚨 The deployment method is INCORRECT and needs major modifications.")
    else:
        print("✅ No issues found - deployment method is optimal!")
    
    return len(issues) == 0

if __name__ == "__main__":
    is_correct = main()
    exit(0 if is_correct else 1)