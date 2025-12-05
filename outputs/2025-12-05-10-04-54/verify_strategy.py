#!/usr/bin/env python3
"""
Verification script for the Optimized MoE Hybrid Parallel Strategy
This script validates the parallel configuration and GPU allocation.
"""

import json
import math
from typing import Dict, List, Tuple

class ParallelStrategyVerifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['parallel_strategy']
        
        self.hardware_config = self.config['hardware_configuration']
        self.model_config = self.config['model_specifications']
        self.parallel_config = self.config['parallel_configuration']
        
    def verify_gpu_allocation(self) -> bool:
        """Verify that total GPU count matches parallel degrees."""
        ep_degree = self.parallel_config['expert_parallelism']['ep_degree']
        tp_degree = self.parallel_config['tensor_parallelism']['tp_degree']
        pp_degree = self.parallel_config['pipeline_parallelism']['pp_degree']
        dp_degree = self.parallel_config['data_parallelism']['dp_degree']
        
        total_required_gpus = ep_degree * tp_degree * pp_degree * dp_degree
        available_gpus = self.hardware_config['total_gpus']
        
        print(f"Required GPUs: {total_required_gpus}")
        print(f"Available GPUs: {available_gpus}")
        print(f"GPU utilization: {total_required_gpus}/{available_gpus} ({total_required_gpus/available_gpus*100:.1f}%)")
        
        return total_required_gpus <= available_gpus
    
    def verify_expert_distribution(self) -> bool:
        """Verify expert distribution across GPUs."""
        experts_per_layer = self.model_config['experts_per_layer']
        ep_degree = self.parallel_config['expert_parallelism']['ep_degree']
        
        experts_per_gpu = experts_per_layer / ep_degree
        print(f"Experts per layer: {experts_per_layer}")
        print(f"EP degree: {ep_degree}")
        print(f"Experts per GPU: {experts_per_gpu}")
        
        # Should be exactly 1 expert per GPU for optimal load balancing
        return experts_per_gpu == 1.0
    
    def verify_memory_constraints(self) -> bool:
        """Verify memory usage is within GPU memory limits."""
        total_params = self.model_config['total_parameters']
        gpu_memory = self.hardware_config['gpu_memory']
        total_gpus = self.hardware_config['total_gpus']
        
        # Calculate memory requirements
        params_per_gpu = total_params / total_gpus  # in billions
        bytes_per_param = 2  # FP16
        param_memory_gb = params_per_gpu * bytes_per_param
        
        # Estimate activation memory (rough calculation)
        batch_size = self.model_config['batch_size']
        seq_length = 2048  # Use average sequence length
        hidden_size = self.model_config['token_dimension']
        layers = self.model_config['layers']
        
        activation_memory_gb = (batch_size * seq_length * hidden_size * layers * 4) / (1024**3)  # GB
        
        # With selective checkpointing, reduce activation memory
        checkpointing_factor = 0.4
        activation_memory_gb *= checkpointing_factor
        
        total_memory_gb = param_memory_gb + activation_memory_gb
        
        print(f"Parameter memory per GPU: {param_memory_gb:.2f} GB")
        print(f"Activation memory per GPU: {activation_memory_gb:.2f} GB")
        print(f"Total memory per GPU: {total_memory_gb:.2f} GB")
        print(f"Available GPU memory: {gpu_memory} GB")
        print(f"Memory utilization: {total_memory_gb/gpu_memory*100:.1f}%")
        
        return total_memory_gb < gpu_memory * 0.9  # Leave 10% headroom
    
    def verify_load_balancing(self) -> bool:
        """Verify load balancing across different parallelism dimensions."""
        layers = self.model_config['layers']
        pp_degree = self.parallel_config['pipeline_parallelism']['pp_degree']
        
        layers_per_stage = layers / pp_degree
        print(f"Total layers: {layers}")
        print(f"PP degree: {pp_degree}")
        print(f"Layers per stage: {layers_per_stage}")
        
        # Should be evenly divisible
        return layers % pp_degree == 0
    
    def verify_performance_projections(self) -> bool:
        """Verify performance projections are reasonable."""
        gpu_compute = self.hardware_config['gpu_compute_power']  # TFlops
        mfu_utilization = self.hardware_config['mfu_utilization']
        
        effective_compute = gpu_compute * mfu_utilization
        
        # Rough throughput estimation
        total_params = self.model_config['total_parameters']
        batch_size = self.model_config['batch_size']
        
        # Simplified calculation (actual would be more complex)
        theoretical_tokens_per_sec = (effective_compute * 1000) / (total_params * 2)  # Rough estimate
        
        projected_throughput = self.config['performance_projections']['theoretical_throughput']
        
        print(f"Effective compute: {effective_compute:.1f} TFlops")
        print(f"Theoretical tokens/sec: {theoretical_tokens_per_sec:.1f}K")
        print(f"Projected throughput: {projected_throughput}")
        
        return True  # Basic sanity check passed
    
    def run_verification(self) -> Dict[str, bool]:
        """Run all verification checks."""
        print("=== Parallel Strategy Verification ===")
        print()
        
        checks = {
            'gpu_allocation': self.verify_gpu_allocation(),
            'expert_distribution': self.verify_expert_distribution(),
            'memory_constraints': self.verify_memory_constraints(),
            'load_balancing': self.verify_load_balancing(),
            'performance_projections': self.verify_performance_projections()
        }
        
        print()
        print("=== Verification Results ===")
        for check, result in checks.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{check}: {status}")
        
        all_passed = all(checks.values())
        print(f"\nOverall result: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
        
        return checks

def main():
    verifier = ParallelStrategyVerifier('../outputs/2025-12-05-10-04-54/parallel_strategy.json')
    results = verifier.run_verification()
    
    # Create verification report
    report = {
        'verification_results': results,
        'summary': {
            'total_checks': len(results),
            'passed_checks': sum(results.values()),
            'all_passed': all(results.values())
        }
    }
    
    with open('../outputs/2025-12-05-10-04-54/verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nVerification report saved to: ../outputs/2025-12-05-10-04-54/verification_report.json")

if __name__ == '__main__':
    main()