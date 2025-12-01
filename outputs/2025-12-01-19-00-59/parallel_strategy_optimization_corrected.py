#!/usr/bin/env python3
"""
Corrected Parallel Strategy Optimization for Single GPU Deployment
Addresses the critical hardware incompatibility issue where previous strategy
required 32 GPUs but system only has 1 GPU available.
"""

import json
import math
from typing import Dict, List, Tuple

class SingleGPUOptimizer:
    """Optimizer for single GPU deployment with EP1_TP1 configuration"""
    
    def __init__(self):
        # Actual hardware constraints (1 Tesla T4 GPU)
        self.available_gpus = 1
        self.gpu_memory_gb = 15.1  # Tesla T4 has ~15GB memory
        self.gpu_compute_tfops = 8.1  # Tesla T4 compute capacity
        
        # Model parameters from deployment analysis
        self.layers = 16
        self.experts_per_layer = 64
        self.total_experts = 1024
        self.token_dim = 4096
        self.moe_hidden_dim = 16384
        self.batch_size = 128
        self.sequence_length = 1024
        self.num_attention_heads = 32
        
        # Performance targets
        self.target_latency_ms = 5000  # Reasonable for single GPU
        self.target_throughput = 1000  # Realistic for single GPU
        
    def calculate_memory_requirements(self) -> Dict[str, float]:
        """Calculate memory requirements for single GPU deployment"""
        
        # Expert parameters (all experts on single GPU)
        expert_params = self.experts_per_layer * self.token_dim * self.moe_hidden_dim * 2  # 2 for gate and expert weights
        expert_params_gb = expert_params * 4 / (1024**3)  # float32
        
        # Attention parameters
        attention_params = self.layers * self.num_attention_heads * self.token_dim * self.token_dim / self.num_attention_heads
        attention_params_gb = attention_params * 4 / (1024**3)
        
        # Activation memory (batch size * sequence length * token dimension * layers)
        activation_memory = self.batch_size * self.sequence_length * self.token_dim * self.layers
        activation_memory_gb = activation_memory * 4 / (1024**3)
        
        # Communication buffers (minimal for single GPU)
        comm_buffers_gb = 0.1
        
        # Overhead for optimizer states, etc.
        overhead_gb = 1.0
        
        total_memory_gb = expert_params_gb + attention_params_gb + activation_memory_gb + comm_buffers_gb + overhead_gb
        
        return {
            "expert_parameters_gb": expert_params_gb,
            "attention_parameters_gb": attention_params_gb,
            "activation_memory_gb": activation_memory_gb,
            "communication_buffers_gb": comm_buffers_gb,
            "overhead_gb": overhead_gb,
            "total_memory_gb": total_memory_gb,
            "memory_utilization_percent": (total_memory_gb / self.gpu_memory_gb) * 100
        }
    
    def calculate_compute_requirements(self) -> Dict[str, float]:
        """Calculate compute requirements for single GPU"""
        
        # FLOPs for expert computation
        expert_flops = self.batch_size * self.sequence_length * self.experts_per_layer * self.token_dim * self.moe_hidden_dim * 2
        
        # FLOPs for attention computation
        attention_flops = self.batch_size * self.sequence_length * self.sequence_length * self.token_dim * self.layers * 2
        
        # Total FLOPs per forward pass
        total_flops = expert_flops + attention_flops
        
        # Compute time on single GPU (in milliseconds)
        compute_time_ms = (total_flops / (self.gpu_compute_tfops * 1e12)) * 1000
        
        return {
            "expert_flops": expert_flops,
            "attention_flops": attention_flops,
            "total_flops": total_flops,
            "compute_time_ms": compute_time_ms
        }
    
    def optimize_parallel_strategy(self) -> Dict:
        """Generate optimal parallel strategy for single GPU"""
        
        memory_analysis = self.calculate_memory_requirements()
        compute_analysis = self.calculate_compute_requirements()
        
        # Single GPU strategy: EP1_TP1 (1-way Expert Parallelism, 1-way Tensor Parallelism)
        strategy = {
            "parallel_strategy": "EP1_TP1",
            "expert_parallelism_degree": 1,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 1,
            "total_gpus_used": 1,
            "module_division": 1,
            "matches_gpu_count": True,
            "load_balancing": "perfect (single GPU)"
        }
        
        # Performance projections
        latency_ms = compute_analysis["compute_time_ms"] + 50  # Add communication overhead
        throughput_tokens_per_sec = (self.batch_size * self.sequence_length) / (latency_ms / 1000)
        
        performance = {
            "latency_ms": latency_ms,
            "throughput_tokens_per_sec": throughput_tokens_per_sec,
            "memory_utilization_percent": memory_analysis["memory_utilization_percent"],
            "compute_utilization_percent": min(100, (compute_analysis["compute_time_ms"] / latency_ms) * 100)
        }
        
        # Expert distribution for single GPU
        expert_distribution = {
            "gpu_0": {
                "experts": list(range(self.experts_per_layer)),
                "expert_count": self.experts_per_layer,
                "memory_allocation_gb": memory_analysis["total_memory_gb"]
            }
        }
        
        return {
            "strategy": strategy,
            "performance": performance,
            "memory_analysis": memory_analysis,
            "compute_analysis": compute_analysis,
            "expert_distribution": expert_distribution,
            "validation": {
                "memory_within_limits": memory_analysis["total_memory_gb"] <= self.gpu_memory_gb,
                "utilization_reasonable": memory_analysis["memory_utilization_percent"] <= 90,
                "load_balancing_achieved": True
            }
        }
    
    def generate_deployment_plan(self) -> Dict:
        """Generate complete deployment plan for single GPU"""
        
        optimization_result = self.optimize_parallel_strategy()
        
        deployment_plan = {
            "deployment_configuration": {
                "hardware_specification": {
                    "gpu_model": "Tesla T4",
                    "gpu_count": 1,
                    "gpu_memory_gb": self.gpu_memory_gb,
                    "gpu_compute_tfops": self.gpu_compute_tfops
                },
                "model_configuration": {
                    "layers": self.layers,
                    "experts_per_layer": self.experts_per_layer,
                    "total_experts": self.total_experts,
                    "token_dimension": self.token_dim,
                    "moe_hidden_dimension": self.moe_hidden_dim,
                    "batch_size": self.batch_size,
                    "sequence_length": self.sequence_length
                },
                "parallel_strategy": optimization_result["strategy"],
                "performance_targets": {
                    "target_latency_ms": self.target_latency_ms,
                    "target_throughput_tokens_per_sec": self.target_throughput
                }
            },
            "resource_allocation": {
                "expert_distribution": optimization_result["expert_distribution"],
                "memory_allocation": optimization_result["memory_analysis"],
                "compute_allocation": optimization_result["compute_analysis"]
            },
            "performance_projection": optimization_result["performance"],
            "optimization_summary": {
                "module_division_parts": 1,
                "gpu_load_balancing": "perfect (single GPU)",
                "memory_efficiency": f"{optimization_result['memory_analysis']['memory_utilization_percent']:.1f}%",
                "compute_efficiency": f"{optimization_result['performance']['compute_utilization_percent']:.1f}%",
                "meets_constraints": optimization_result["validation"]
            }
        }
        
        return deployment_plan

def main():
    """Main optimization function"""
    
    print("Single GPU Parallel Strategy Optimization")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = SingleGPUOptimizer()
    
    # Generate deployment plan
    deployment_plan = optimizer.generate_deployment_plan()
    
    # Save deployment plan
    with open("../outputs/2025-12-01-19-00-59/optimal_deployment_plan_corrected.json", "w") as f:
        json.dump(deployment_plan, f, indent=2)
    
    # Print summary
    print(f"Strategy: {deployment_plan['deployment_configuration']['parallel_strategy']['parallel_strategy']}")
    print(f"GPUs Used: {deployment_plan['deployment_configuration']['parallel_strategy']['total_gpus_used']}")
    print(f"Module Division: {deployment_plan['deployment_configuration']['parallel_strategy']['module_division']} parts")
    print(f"Memory Utilization: {deployment_plan['performance_projection']['memory_utilization_percent']:.1f}%")
    print(f"Latency: {deployment_plan['performance_projection']['latency_ms']:.1f} ms")
    print(f"Throughput: {deployment_plan['performance_projection']['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"Load Balancing: {deployment_plan['deployment_configuration']['parallel_strategy']['load_balancing']}")
    
    print("\nValidation Results:")
    validation = deployment_plan['optimization_summary']['meets_constraints']
    for check, result in validation.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check}: {status}")
    
    return deployment_plan

if __name__ == "__main__":
    main()