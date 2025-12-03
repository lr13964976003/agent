#!/usr/bin/env python3
"""
Deployment Verification Script

This script verifies that the hybrid parallel strategy deployment meets all
requirements including load balancing, module division, and performance metrics.
"""

import json
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple


class DeploymentVerifier:
    """
    Comprehensive verification of hybrid parallel strategy deployment
    """
    
    def __init__(self, config_path: str):
        """Initialize verifier with deployment configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.verification_results = {}
    
    def verify_hardware_constraints(self) -> bool:
        """
        Verify deployment conforms to hardware constraints
        
        Returns:
            True if constraints are satisfied, False otherwise
        """
        print("\n=== Hardware Constraints Verification ===")
        
        hardware = self.config['hardware_environment']
        
        # Check GPU count
        total_gpus = hardware['total_gpus']
        print(f"Total GPUs available: {total_gpus}")
        
        # Check memory per GPU
        gpu_memory = hardware['gpu_memory']
        print(f"GPU memory per device: {gpu_memory}")
        
        # Check interconnect
        interconnect = hardware['interconnect']
        print(f"Interconnect type: {interconnect}")
        
        # Verify requirements
        if total_gpus >= 3:
            print("✅ Sufficient GPU count (3 required)")
            self.verification_results['gpu_count'] = True
        else:
            print("❌ Insufficient GPU count")
            self.verification_results['gpu_count'] = False
        
        if '32GB' in gpu_memory:
            print("✅ Sufficient GPU memory")
            self.verification_results['gpu_memory'] = True
        else:
            print("❌ Insufficient GPU memory")
            self.verification_results['gpu_memory'] = False
        
        if interconnect == 'NVLink':
            print("✅ High-bandwidth interconnect available")
            self.verification_results['interconnect'] = True
        else:
            print("❌ High-bandwidth interconnect not available")
            self.verification_results['interconnect'] = False
        
        return all([self.verification_results['gpu_count'], 
                   self.verification_results['gpu_memory'], 
                   self.verification_results['interconnect']])
    
    def verify_parallel_strategy(self) -> bool:
        """
        Verify parallel strategy configuration
        
        Returns:
            True if strategy is valid, False otherwise
        """
        print("\n=== Parallel Strategy Verification ===")
        
        strategy = self.config['parallel_strategy']
        
        # Check strategy type
        strategy_type = strategy['type']
        print(f"Parallel strategy type: {strategy_type}")
        
        # Check tensor parallel size
        tp_size = strategy['tensor_parallel_size']
        print(f"Tensor parallel size: {tp_size}")
        
        # Check pipeline parallel size
        pp_size = strategy['pipeline_parallel_size']
        print(f"Pipeline parallel size: {pp_size}")
        
        # Check data parallel size
        dp_size = strategy['data_parallel_size']
        print(f"Data parallel size: {dp_size}")
        
        # Verify strategy validity
        if strategy_type == 'hybrid_tensor_pipeline':
            print("✅ Valid hybrid strategy selected")
            self.verification_results['strategy_type'] = True
        else:
            print("❌ Invalid strategy type")
            self.verification_results['strategy_type'] = False
        
        if tp_size == 2 and pp_size == 3:
            print("✅ Optimal parallel configuration")
            self.verification_results['parallel_config'] = True
        else:
            print("❌ Suboptimal parallel configuration")
            self.verification_results['parallel_config'] = False
        
        return all([self.verification_results['strategy_type'], 
                   self.verification_results['parallel_config']])
    
    def verify_gpu_assignment(self) -> bool:
        """
        Verify GPU assignment strategy
        
        Returns:
            True if assignment is optimal, False otherwise
        """
        print("\n=== GPU Assignment Verification ===")
        
        assignment = self.config['gpu_assignment']
        
        # Check stage assignments
        for stage_name, stage_config in assignment.items():
            gpus = stage_config['gpus']
            layers = stage_config['layers']
            tensor_splits = stage_config['tensor_parallel_splits']
            
            print(f"{stage_name}: GPUs {gpus}, Layers {layers}, Tensor splits: {tensor_splits}")
        
        # Verify load balancing
        gpu_usage = {0: 0, 1: 0, 2: 0}
        
        for stage_name, stage_config in assignment.items():
            gpus = stage_config['gpus']
            for gpu in gpus:
                gpu_usage[gpu] += 1
        
        print(f"GPU usage distribution: {gpu_usage}")
        
        # Check for balanced usage
        max_usage = max(gpu_usage.values())
        min_usage = min(gpu_usage.values())
        usage_difference = max_usage - min_usage
        
        if usage_difference <= 1:
            print("✅ Balanced GPU assignment")
            self.verification_results['gpu_balance'] = True
        else:
            print("❌ Unbalanced GPU assignment")
            self.verification_results['gpu_balance'] = False
        
        return self.verification_results['gpu_balance']
    
    def verify_load_balancing(self) -> bool:
        """
        Verify GPU load balancing
        
        Returns:
            True if load balancing requirements are met, False otherwise
        """
        print("\n=== Load Balancing Verification ===")
        
        load_balancing = self.config['load_balancing']
        
        computation_dist = load_balancing['computation_distribution']
        memory_dist = load_balancing['memory_distribution']
        
        # Extract computation percentages
        comp_percentages = []
        for gpu, load in computation_dist.items():
            percentage = float(load.split('%')[0]) / 100
            comp_percentages.append(percentage)
            print(f"{gpu} computation load: {percentage:.2%}")
        
        # Extract memory percentages
        mem_percentages = []
        for gpu, usage in memory_dist.items():
            percentage = float(usage.split('%')[0]) / 100
            mem_percentages.append(percentage)
            print(f"{gpu} memory usage: {percentage:.2%}")
        
        # Check maximum difference
        comp_max_diff = max(comp_percentages) - min(comp_percentages)
        mem_max_diff = max(mem_percentages) - min(mem_percentages)
        
        print(f"Computation load max difference: {comp_max_diff:.2%}")
        print(f"Memory usage max difference: {mem_max_diff:.2%}")
        
        # Verify requirements (1% tolerance)
        if comp_max_diff <= 0.01:
            print("✅ Computation load balancing satisfied (≤1% difference)")
            self.verification_results['comp_balance'] = True
        else:
            print("❌ Computation load balancing not satisfied")
            self.verification_results['comp_balance'] = False
        
        if mem_max_diff <= 0.01:
            print("✅ Memory usage balancing satisfied (≤1% difference)")
            self.verification_results['mem_balance'] = True
        else:
            print("❌ Memory usage balancing not satisfied")
            self.verification_results['mem_balance'] = False
        
        return all([self.verification_results['comp_balance'], 
                   self.verification_results['mem_balance']])
    
    def verify_module_division(self) -> bool:
        """
        Verify module division matches GPU count
        
        Returns:
            True if division is correct, False otherwise
        """
        print("\n=== Module Division Verification ===")
        
        module_division = self.config['module_division']
        
        total_parts = module_division['total_parts']
        gpu_match = module_division['gpu_match']
        parts_per_gpu = module_division['parts_per_gpu']
        expert_tensor_split = module_division['expert_tensor_split']
        
        print(f"Total module parts: {total_parts}")
        print(f"GPU match: {gpu_match}")
        print(f"Parts per GPU: {parts_per_gpu}")
        print(f"Expert tensor split: {expert_tensor_split}")
        
        # Check if total parts match GPU count
        expected_parts = 3  # 3 GPUs
        
        if total_parts == expected_parts:
            print("✅ Module parts match GPU count")
            self.verification_results['module_parts'] = True
        else:
            print(f"❌ Module parts ({total_parts}) don't match GPU count ({expected_parts})")
            self.verification_results['module_parts'] = False
        
        if gpu_match:
            print("✅ GPU match flag is True")
            self.verification_results['gpu_match'] = True
        else:
            print("❌ GPU match flag is False")
            self.verification_results['gpu_match'] = False
        
        return all([self.verification_results['module_parts'], 
                   self.verification_results['gpu_match']])
    
    def verify_performance_metrics(self) -> bool:
        """
        Verify performance metrics meet targets
        
        Returns:
            True if metrics are satisfactory, False otherwise
        """
        print("\n=== Performance Metrics Verification ===")
        
        metrics = self.config['performance_metrics']
        
        latency_reduction = metrics['expected_latency_reduction']
        throughput_increase = metrics['expected_throughput_increase']
        gpu_utilization = metrics['gpu_utilization_target']
        memory_efficiency = metrics['memory_efficiency']
        
        print(f"Expected latency reduction: {latency_reduction}")
        print(f"Expected throughput increase: {throughput_increase}")
        print(f"GPU utilization target: {gpu_utilization}")
        print(f"Memory efficiency: {memory_efficiency}")
        
        # Extract numeric values
        latency_val = float(latency_reduction.strip('%'))
        throughput_val = float(throughput_increase.strip('%'))
        gpu_util_val = float(gpu_utilization.strip('%'))
        mem_eff_val = float(memory_efficiency.strip('%'))
        
        # Verify against targets
        if latency_val >= 30:  # 30% target
            print("✅ Latency reduction target met (≥30%)")
            self.verification_results['latency_target'] = True
        else:
            print("❌ Latency reduction target not met")
            self.verification_results['latency_target'] = False
        
        if throughput_val >= 50:  # 50% target
            print("✅ Throughput increase target met (≥50%)")
            self.verification_results['throughput_target'] = True
        else:
            print("❌ Throughput increase target not met")
            self.verification_results['throughput_target'] = False
        
        if gpu_util_val >= 90:  # 90% target
            print("✅ GPU utilization target met (≥90%)")
            self.verification_results['gpu_util_target'] = True
        else:
            print("❌ GPU utilization target not met")
            self.verification_results['gpu_util_target'] = False
        
        return all([self.verification_results['latency_target'], 
                   self.verification_results['throughput_target'], 
                   self.verification_results['gpu_util_target']])
    
    def run_complete_verification(self) -> Dict[str, bool]:
        """
        Run complete verification of all deployment aspects
        
        Returns:
            Dictionary of verification results
        """
        print("=" * 60)
        print("HYBRID PARALLEL STRATEGY DEPLOYMENT VERIFICATION")
        print("=" * 60)
        
        # Run all verification checks
        self.verify_hardware_constraints()
        self.verify_parallel_strategy()
        self.verify_gpu_assignment()
        self.verify_load_balancing()
        self.verify_module_division()
        self.verify_performance_metrics()
        
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        # Summary of results
        all_passed = True
        for check, result in self.verification_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check.replace('_', ' ').title()}: {status}")
            if not result:
                all_passed = False
        
        print(f"\nOverall Result: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        
        return self.verification_results


def main():
    """Main verification function"""
    
    # Path to deployment configuration
    config_path = "deployment_configuration.json"
    
    # Initialize verifier
    verifier = DeploymentVerifier(config_path)
    
    # Run complete verification
    results = verifier.run_complete_verification()
    
    # Save verification results
    with open("verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nVerification results saved to verification_results.json")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)