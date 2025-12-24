#!/usr/bin/env python3
"""
Parallel Strategy Validation Script
Validates that the generated parallel strategy meets all requirements
"""

import json
import math
from typing import Dict, Any, Tuple

class ParallelStrategyValidator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.hardware = {
            'total_gpus': 8,
            'gpu_memory_gb': 80,
            'nvlink_bw_gbps': 900,
            'pcie_bw_gbps': 64
        }
        
        self.model = {
            'num_layers': 80,
            'hidden_size': 8192,
            'vocab_size': 128256,
            'model_weights_gb': 140
        }
        
        self.requirements = {
            'prefill_p50_ms': 500,
            'prefill_p99_ms': 1000,
            'decode_p50_ms': 50,
            'decode_p99_ms': 100,
            'target_rps': 8,
            'max_gpu_memory_usage': 0.85,
            'gpu_utilization_target': 0.70
        }
    
    def validate_module_division(self) -> Tuple[bool, str]:
        """Check if module division matches GPU count"""
        total_gpus = self.config['parallel_strategy']['total_gpus']
        tp_size = self.config['parallel_strategy']['tensor_parallel_size']
        pp_size = self.config['parallel_strategy']['pipeline_parallel_size']
        dp_size = self.config['parallel_strategy']['data_parallel_size']        
        calculated_gpus = tp_size * pp_size * dp_size
        
        if calculated_gpus == total_gpus:
            return True, f"Module division matches GPU count: {calculated_gpus} modules for {total_gpus} GPUs"
        else:
            return False, f"Module division mismatch: {calculated_gpus} modules vs {total_gpus} GPUs"
    
    def validate_memory_constraints(self) -> Tuple[bool, str]:
        """Check if memory usage is within limits"""
        model_weights_per_gpu = self.config['memory_analysis']['model_weights_per_gpu_gb']
        available_memory = self.config['memory_analysis']['available_memory_per_gpu_gb']
        max_usage_percent = self.requirements['max_gpu_memory_usage']
        
        expected_usage = model_weights_per_gpu / (self.hardware['gpu_memory_gb'] * max_usage_percent)
        
        if expected_usage <= 1.0:
            return True, f"Memory usage valid: {model_weights_per_gpu:.1f}GB used of {available_memory:.1f}GB available"
        else:
            return False, f"Memory usage exceeds limit: {expected_usage:.1%} usage"
    
    def validate_load_balancing(self) -> Tuple[bool, str]:
        """Check GPU load balancing"""
        pp_size = self.config['parallel_strategy']['pipeline_parallel_size']
        layers_per_stage = self.model['num_layers'] // pp_size
        
        if self.model['num_layers'] % pp_size == 0:
            return True, f"Perfect load balancing: {layers_per_stage} layers per stage"
        else:
            remainder = self.model['num_layers'] % pp_size
            return False, f"Uneven load distribution: remainder of {remainder} layers"
    
    def validate_layer_distribution(self) -> Tuple[bool, str]:
        """Check if layers are properly distributed across pipeline stages"""
        pp_size = self.config['parallel_strategy']['pipeline_parallel_size']
        layers_per_stage = self.config['model_config']['layers_per_stage']
        total_distributed_layers = layers_per_stage * pp_size
        
        if total_distributed_layers == self.model['num_layers']:
            return True, f"All {self.model['num_layers']} layers distributed across {pp_size} stages"
        else:
            return False, f"Layer distribution mismatch: {total_distributed_layers} vs {self.model['num_layers']}"
    
    def validate_performance_targets(self) -> Tuple[bool, str]:
        """Check if performance targets are achievable"""
        # Simplified performance estimation
        tp_size = self.config['parallel_strategy']['tensor_parallel_size']
        pp_size = self.config['parallel_strategy']['pipeline_parallel_size']
        
        # Estimated speedup from parallelism
        tp_speedup = tp_size * 0.8  # 80% efficiency
        pp_speedup = 1.0 / (1.0 + (pp_size - 1.0) * 0.1)  # Pipeline overhead
        
        estimated_prefill_speedup = tp_speedup * pp_speedup
        estimated_decode_speedup = tp_speedup * pp_speedup
        
        # Check if targets are theoretically achievable
        baseline_prefill_ms = 2000  # Estimated baseline
        baseline_decode_ms = 200    # Estimated baseline
        
        estimated_prefill_ms = baseline_prefill_ms / estimated_prefill_speedup
        estimated_decode_ms = baseline_decode_ms / estimated_decode_speedup
        
        prefill_ok = estimated_prefill_ms <= self.requirements['prefill_p50_ms']
        decode_ok = estimated_decode_ms <= self.requirements['decode_p50_ms']
        
        if prefill_ok and decode_ok:
            return True, f"Performance targets achievable: Prefill ~{estimated_prefill_ms:.0f}ms, Decode ~{estimated_decode_ms:.0f}ms"
        else:
            return False, f"Performance targets may not be met: Prefill ~{estimated_prefill_ms:.0f}ms, Decode ~{estimated_decode_ms:.0f}ms"
    
    def validate_throughput_targets(self) -> Tuple[bool, str]:
        """Check throughput requirements"""
        batch_size = self.config['performance_config']['max_batch_size']
        target_rps = self.requirements['target_rps']
        
        # Estimate throughput capability
        # Assuming we can process batch_size requests in parallel
        estimated_rps = batch_size * 0.25  # Conservative estimate
        
        if estimated_rps >= target_rps:
            return True, f"Throughput target achievable: {estimated_rps:.1f} RPS vs {target_rps} target"
        else:
            return False, f"Throughput target may not be met: {estimated_rps:.1f} RPS vs {target_rps} target"
    
    def run_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        results = {}
        
        # Module division check
        results['module_division'] = self.validate_module_division()
        
        # Memory constraints check
        results['memory_constraints'] = self.validate_memory_constraints()
        
        # Load balancing check
        results['load_balancing'] = self.validate_load_balancing()
        
        # Layer distribution check
        results['layer_distribution'] = self.validate_layer_distribution()
        
        # Performance targets check
        results['performance_targets'] = self.validate_performance_targets()
        
        # Throughput targets check
        results['throughput_targets'] = self.validate_throughput_targets()
        
        # Overall validation result
        all_passed = all(result[0] for result in results.values())
        results['overall'] = (all_passed, "All validations passed" if all_passed else "Some validations failed")
        
        return results
    
    def print_validation_report(self):
        """Print detailed validation report"""
        results = self.run_validation()
        
        print("=" * 60)
        print("PARALLEL STRATEGY VALIDATION REPORT")
        print("=" * 60)
        
        for check_name, (passed, message) in results.items():
            if check_name == 'overall':
                print("-" * 60)
                print(f"OVERALL RESULT: {'✅ PASS' if passed else '❌ FAIL'}")
                print(f"{message}")
            else:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{check_name.replace('_', ' ').upper()}: {status}")
                print(f"  {message}")
                print()
        
        print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python strategy_validation.py <config_path>")
        sys.exit(1)
    
    validator = ParallelStrategyValidator(sys.argv[1])
    validator.print_validation_report()
    
    # Exit with appropriate code
    results = validator.run_validation()
    overall_passed = results['overall'][0]
    sys.exit(0 if overall_passed else 1)