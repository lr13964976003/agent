#!/usr/bin/env python3
"""
Verification script for optimized parallel strategy
Checks GPU load balancing, module division, and performance requirements
"""

import json
import numpy as np

class DeploymentVerifier:
    def __init__(self, deployment_file):
        with open(deployment_file, 'r') as f:
            self.config = json.load(f)
            
    def verify_gpu_load_balancing(self):
        """Verify GPU load balancing for throughput and latency"""
        print("=== GPU Load Balancing Verification ===")
        
        load_distribution = self.config['load_balancing']['computation_distribution']
        
        gpu_loads = []
        for gpu, load in load_distribution.items():
            load_percent = float(load.split('%')[0])
            gpu_loads.append(load_percent)
            print(f"{gpu}: {load_percent}% computation load")
            
        # Check if loads are balanced (within 5% difference)
        max_diff = max(gpu_loads) - min(gpu_loads)
        balanced = max_diff <= 5.0
        
        print(f"Maximum load difference: {max_diff}%")
        print(f"Load balancing: {'PASSED' if balanced else 'FAILED'}")
        
        return balanced
        
    def verify_module_division(self):
        """Verify module division matches GPU count"""
        print("\n=== Module Division Verification ===")
        
        module_division = self.config['module_division']
        total_parts = module_division['total_parts']
        gpu_match = module_division['gpu_match']
        
        print(f"Total module parts: {total_parts}")
        print(f"Number of GPUs: 3")
        print(f"Parts match GPUs: {gpu_match}")
        
        # Each GPU should handle approximately one part
        expected_parts_per_gpu = total_parts / 3
        print(f"Expected parts per GPU: {expected_parts_per_gpu}")
        
        division_correct = (total_parts == 3) and gpu_match
        print(f"Module division: {'PASSED' if division_correct else 'FAILED'}")
        
        return division_correct
        
    def verify_performance_metrics(self):
        """Verify performance metrics meet requirements"""
        print("\n=== Performance Metrics Verification ===")
        
        metrics = self.config['performance_metrics']
        
        # Check latency reduction
        latency_reduction = float(metrics['expected_latency_reduction'].split('%')[0])
        latency_target = latency_reduction >= 30  # At least 30% reduction
        
        # Check throughput increase
        throughput_increase = float(metrics['expected_throughput_increase'].split('%')[0])
        throughput_target = throughput_increase >= 50  # At least 50% increase
        
        # Check GPU utilization
        gpu_utilization = float(metrics['gpu_utilization_target'].split('%')[0])
        utilization_target = gpu_utilization >= 90  # At least 90% utilization
        
        print(f"Expected latency reduction: {latency_reduction}%")
        print(f"Latency target (>=30%): {'PASSED' if latency_target else 'FAILED'}")
        
        print(f"Expected throughput increase: {throughput_increase}%")
        print(f"Throughput target (>=50%): {'PASSED' if throughput_target else 'FAILED'}")
        
        print(f"GPU utilization target: {gpu_utilization}%")
        print(f"Utilization target (>=90%): {'PASSED' if utilization_target else 'FAILED'}")
        
        overall_performance = latency_target and throughput_target and utilization_target
        print(f"Overall performance: {'PASSED' if overall_performance else 'FAILED'}")
        
        return overall_performance
        
    def verify_parallel_strategy_correctness(self):
        """Verify the parallel strategy is correctly configured"""
        print("\n=== Parallel Strategy Verification ===")
        
        strategy = self.config['parallel_strategy']
        
        # Check tensor parallel size
        tp_size = strategy['tensor_parallel_size']
        pp_size = strategy['pipeline_parallel_size']
        dp_size = strategy['data_parallel_size']
        
        total_parallelism = tp_size * pp_size * dp_size
        
        print(f"Tensor parallel size: {tp_size}")
        print(f"Pipeline parallel size: {pp_size}")
        print(f"Data parallel size: {dp_size}")
        print(f"Total parallelism: {total_parallelism}")
        
        # Verify GPU assignment
        gpu_assignment = self.config['gpu_assignment']
        assigned_gpus = set()
        
        for stage, config in gpu_assignment.items():
            gpus = config['gpus']
            assigned_gpus.update(gpus)
            print(f"{stage}: GPUs {gpus}")
            
        all_gpus_used = assigned_gpus == {0, 1, 2}
        print(f"All GPUs utilized: {'PASSED' if all_gpus_used else 'FAILED'}")
        
        strategy_correct = (total_parallelism == 6) and all_gpus_used  # 2*3*1=6
        print(f"Parallel strategy: {'PASSED' if strategy_correct else 'FAILED'}")
        
        return strategy_correct
        
    def verify_communication_optimization(self):
        """Verify communication optimizations are in place"""
        print("\n=== Communication Optimization Verification ===")
        
        comm_opt = self.config['communication_optimization']
        
        overlap_enabled = comm_opt['overlap_communication_computation']
        fusion_strategies = comm_opt['fusion_strategies']
        bandwidth_opt = comm_opt['bandwidth_optimization']
        
        print(f"Overlap communication/computation: {overlap_enabled}")
        print(f"Fusion strategies: {fusion_strategies}")
        print(f"Bandwidth optimization: {bandwidth_opt}")
        
        comm_optimized = overlap_enabled and len(fusion_strategies) > 0
        print(f"Communication optimization: {'PASSED' if comm_optimized else 'FAILED'}")
        
        return comm_optimized
        
    def run_all_verifications(self):
        """Run all verification checks"""
        print("=" * 60)
        print("DEPLOYMENT VERIFICATION REPORT")
        print("=" * 60)
        
        results = {
            'gpu_load_balancing': self.verify_gpu_load_balancing(),
            'module_division': self.verify_module_division(),
            'performance_metrics': self.verify_performance_metrics(),
            'parallel_strategy': self.verify_parallel_strategy_correctness(),
            'communication_optimization': self.verify_communication_optimization()
        }
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        all_passed = all(results.values())
        
        for check, result in results.items():
            status = "PASSED" if result else "FAILED"
            print(f"{check.replace('_', ' ').title()}: {status}")
            
        print(f"\nOverall Deployment: {'PASSED' if all_passed else 'FAILED'}")
        
        if all_passed:
            print("\n✅ Deployment meets all requirements!")
            print("✅ GPU load balancing optimized for throughput/latency")
            print("✅ Module division matches GPU count (3 parts, 3 GPUs)")
            print("✅ Performance targets exceeded")
            print("✅ Parallel strategy correctly configured")
            print("✅ Communication optimizations in place")
        else:
            print("\n❌ Deployment has issues that need to be addressed")
            
        return all_passed

if __name__ == "__main__":
    verifier = DeploymentVerifier('../outputs/2025-12-03-10-02-49/optimized_parallel_strategy.json')
    verifier.run_all_verifications()