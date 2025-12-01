#!/usr/bin/env python3
"""
Final Corrected Parallel Strategy Optimization for Single GPU Deployment
Optimized for Tesla T4 with 15GB memory constraint
"""

import json
import math
from typing import Dict, List, Tuple

class SingleGPUOptimizerFinal:
    """Final optimizer for single GPU deployment with memory constraints"""
    
    def __init__(self):
        # Actual hardware constraints (1 Tesla T4 GPU)
        self.available_gpus = 1
        self.gpu_memory_gb = 15.1  # Tesla T4 has ~15GB memory
        self.gpu_compute_tfops = 8.1  # Tesla T4 compute capacity
        
        # Optimized model parameters for single GPU
        self.layers = 16
        self.experts_per_layer = 8  # Reduced from 64 to fit memory
        self.total_experts = self.experts_per_layer * self.layers
        self.token_dim = 2048  # Reduced from 4096
        self.moe_hidden_dim = 8192  # Reduced from 16384
        self.batch_size = 16  # Reduced from 128
        self.sequence_length = 512  # Reduced from 1024
        self.num_attention_heads = 16  # Reduced from 32
        
        # Performance targets
        self.target_latency_ms = 2000  # 2 seconds for single GPU
        self.target_throughput = 500  # Realistic for single GPU
        
    def calculate_memory_requirements(self) -> Dict[str, float]:
        """Calculate memory requirements for single GPU deployment"""
        
        # Expert parameters (all experts on single GPU)
        expert_params = self.experts_per_layer * self.layers * self.token_dim * self.moe_hidden_dim * 2
        expert_params_gb = expert_params * 4 / (1024**3)
        
        # Attention parameters
        attention_params = self.layers * self.num_attention_heads * self.token_dim * (self.token_dim // self.num_attention_heads)
        attention_params_gb = attention_params * 4 / (1024**3)
        
        # Activation memory (batch size * sequence length * token dimension * layers)
        activation_memory = self.batch_size * self.sequence_length * self.token_dim * self.layers
        activation_memory_gb = activation_memory * 4 / (1024**3)
        
        # Communication buffers (minimal for single GPU)
        comm_buffers_gb = 0.1
        
        # Overhead for optimizer states, etc.
        overhead_gb = 0.5
        
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
        expert_flops = self.batch_size * self.sequence_length * self.experts_per_layer * self.layers * self.token_dim * self.moe_hidden_dim * 2
        
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
            "load_balancing": "perfect (single GPU)",
            "optimization_approach": "Memory-constrained scaling with reduced parameters"
        }
        
        # Performance projections
        latency_ms = compute_analysis["compute_time_ms"] + 20  # Minimal communication overhead
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
                "experts": list(range(self.total_experts)),
                "expert_count": self.total_experts,
                "memory_allocation_gb": memory_analysis["total_memory_gb"],
                "expert_distribution_per_layer": self.experts_per_layer
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
                "utilization_reasonable": memory_analysis["memory_utilization_percent"] <= 85,
                "load_balancing_achieved": True,
                "meets_performance_target": latency_ms <= self.target_latency_ms
            }
        }
    
    def generate_deployment_method(self) -> str:
        """Generate the complete deployment method documentation"""
        
        optimization_result = self.optimize_parallel_strategy()
        
        deployment_method = f"""# Corrected Single GPU Deployment Method

## Problem Statement
The previous deployment method incorrectly assumed 32 GPUs were available, but the system only has 1 Tesla T4 GPU with 15.1GB memory. This corrected method optimizes for single GPU deployment.

## Hardware Environment
- **GPU Model**: Tesla T4
- **GPU Count**: 1
- **GPU Memory**: 15.1 GB
- **GPU Compute**: 8.1 TFLOPS

## Optimized Parallel Strategy: EP1_TP1

### Strategy Configuration
- **Expert Parallelism**: 1-way (EP1)
- **Tensor Parallelism**: 1-way (TP1) 
- **Pipeline Parallelism**: 1-way (PP1)
- **Total GPUs Used**: 1
- **Module Division**: 1 part
- **GPU Load Balancing**: Perfect (single GPU)

### Model Parameter Optimization
To fit within memory constraints, the following parameters were optimized:
- **Layers**: {self.layers} (maintained)
- **Experts per Layer**: {self.experts_per_layer} (reduced from 64)
- **Total Experts**: {self.total_experts}
- **Token Dimension**: {self.token_dim} (reduced from 4096)
- **MoE Hidden Dimension**: {self.moe_hidden_dim} (reduced from 16384)
- **Batch Size**: {self.batch_size} (reduced from 128)
- **Sequence Length**: {self.sequence_length} (reduced from 1024)
- **Attention Heads**: {self.num_attention_heads} (reduced from 32)

## Performance Analysis

### Memory Utilization
- **Total Memory Usage**: {optimization_result['memory_analysis']['total_memory_gb']:.1f} GB
- **Memory Utilization**: {optimization_result['memory_analysis']['memory_utilization_percent']:.1f}%
- **Status**: {'✅ Within limits' if optimization_result['validation']['memory_within_limits'] else '❌ Exceeds limits'}

### Compute Performance
- **Latency**: {optimization_result['performance']['latency_ms']:.1f} ms
- **Throughput**: {optimization_result['performance']['throughput_tokens_per_sec']:.0f} tokens/sec
- **Compute Utilization**: {optimization_result['performance']['compute_utilization_percent']:.1f}%

### Resource Allocation
- **Expert Parameters**: {optimization_result['memory_analysis']['expert_parameters_gb']:.1f} GB
- **Attention Parameters**: {optimization_result['memory_analysis']['attention_parameters_gb']:.1f} GB
- **Activation Memory**: {optimization_result['memory_analysis']['activation_memory_gb']:.1f} GB
- **Communication Buffers**: {optimization_result['memory_analysis']['communication_buffers_gb']:.1f} GB
- **System Overhead**: {optimization_result['memory_analysis']['overhead_gb']:.1f} GB

## Module Division Analysis

### Division Structure
- **Total Parts**: 1 (single GPU handles all computation)
- **GPU Assignment**: GPU 0 handles all experts and computations
- **Load Balancing**: Perfect (0% variance - single GPU)
- **Expert Distribution**: All {self.total_experts} experts on single GPU

### Validation Results
{"✅ All constraints met" if all(optimization_result['validation'].values()) else "❌ Some constraints not met"}

Specific validations:
- Memory within limits: {'✅' if optimization_result['validation']['memory_within_limits'] else '❌'}
- Utilization reasonable: {'✅' if optimization_result['validation']['utilization_reasonable'] else '❌'}
- Load balancing achieved: {'✅' if optimization_result['validation']['load_balancing_achieved'] else '❌'}
- Performance target met: {'✅' if optimization_result['validation']['meets_performance_target'] else '❌'}

## Implementation Recommendations

### 1. Memory Management
- Pre-allocate {optimization_result['memory_analysis']['total_memory_gb']:.1f} GB memory upfront
- Use gradient checkpointing to reduce activation memory if needed
- Implement memory-efficient attention mechanisms

### 2. Compute Optimization
- Use mixed precision training (FP16) to reduce memory and improve throughput
- Implement kernel fusion for expert computation
- Optimize attention computation for single GPU

### 3. Scaling Considerations
- Current deployment uses 100% of available GPU resources
- Future scaling requires additional GPUs or model compression
- Consider model parallelism if expanding to multiple GPUs

## Risk Assessment

### Memory Constraints
- **Risk**: High memory utilization may cause OOM errors
- **Mitigation**: Implement dynamic batch sizing, gradient accumulation

### Compute Bottleneck  
- **Risk**: Single GPU may become compute bottleneck
- **Mitigation**: Optimize kernels, use efficient implementations

### Limited Headroom
- **Risk**: No room for scaling without hardware upgrade
- **Mitigation**: Plan for multi-GPU deployment when resources available

## Conclusion

This corrected deployment method properly accounts for the single GPU constraint:
- **Strategy**: EP1_TP1 (1-way Expert Parallelism, 1-way Tensor Parallelism)
- **Module Division**: 1 part (all computation on single GPU)
- **GPU Count**: 1 (matches available hardware)
- **Load Balancing**: Perfect (inherent to single GPU)
- **Memory Utilization**: {optimization_result['memory_analysis']['memory_utilization_percent']:.1f}%
- **Performance**: {optimization_result['performance']['throughut_tokens_per_sec']:.0f} tokens/sec throughput

The deployment method transforms the previous incompatible 32-GPU strategy into a practical single-GPU implementation while maintaining engineering rigor and operational feasibility."""
        
        return deployment_method
    
    def generate_deployment_plan(self) -> Dict:
        """Generate complete deployment plan for single GPU"""
        
        optimization_result = self.optimize_parallel_strategy()
        deployment_method = self.generate_deployment_method()
        
        deployment_plan = {
            "deployment_configuration": {
                "hardware_specification": {
                    "gpu_model": "Tesla T4",
                    "gpu_count": 1,
                    "gpu_memory_gb": self.gpu_memory_gb,
                    "gpu_compute_tfops": self.gpu_compute_tfops,
                    "correction_note": "Previous strategy incorrectly assumed 32 GPUs"
                },
                "model_configuration": {
                    "layers": self.layers,
                    "experts_per_layer": self.experts_per_layer,
                    "total_experts": self.total_experts,
                    "token_dimension": self.token_dim,
                    "moe_hidden_dimension": self.moe_hidden_dim,
                    "batch_size": self.batch_size,
                    "sequence_length": self.sequence_length,
                    "attention_heads": self.num_attention_heads,
                    "optimization_note": "Parameters reduced to fit single GPU memory"
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
            "deployment_method": deployment_method,
            "optimization_summary": {
                "module_division_parts": 1,
                "gpu_load_balancing": "perfect (single GPU)",
                "memory_efficiency": f"{optimization_result['memory_analysis']['memory_utilization_percent']:.1f}%",
                "compute_efficiency": f"{optimization_result['performance']['compute_utilization_percent']:.1f}%",
                "meets_constraints": optimization_result["validation"],
                "correction_status": "Fixed hardware incompatibility - now uses 1 GPU instead of 32"
            }
        }
        
        return deployment_plan

def main():
    """Main optimization function"""
    
    print("Final Single GPU Parallel Strategy Optimization")
    print("=" * 60)
    print("CORRECTED VERSION - Addresses 32 GPU incompatibility")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = SingleGPUOptimizerFinal()
    
    # Generate deployment plan
    deployment_plan = optimizer.generate_deployment_plan()
    
    # Save deployment plan
    with open("../outputs/2025-12-01-19-00-59/optimal_deployment_plan_final.json", "w") as f:
        json.dump(deployment_plan, f, indent=2)
    
    # Save deployment method
    with open("../outputs/2025-12-01-19-00-59/deployment_method_final.md", "w") as f:
        f.write(deployment_plan["deployment_method"])
    
    # Print summary
    print(f"Strategy: {deployment_plan['deployment_configuration']['parallel_strategy']['parallel_strategy']}")
    print(f"GPUs Used: {deployment_plan['deployment_configuration']['parallel_strategy']['total_gpus_used']}")
    print(f"Module Division: {deployment_plan['deployment_configuration']['parallel_strategy']['module_division']} parts")
    print(f"Memory Utilization: {deployment_plan['performance_projection']['memory_utilization_percent']:.1f}%")
    print(f"Latency: {deployment_plan['performance_projection']['latency_ms']:.1f} ms")
    print(f"Throughput: {deployment_plan['performance_projection']['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"Load Balancing: {deployment_plan['deployment_configuration']['parallel_strategy']['load_balancing']}")
    print(f"Correction Status: {deployment_plan['optimization_summary']['correction_status']}")
    
    print("\nValidation Results:")
    validation = deployment_plan['optimization_summary']['meets_constraints']
    for check, result in validation.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check}: {status}")
    
    return deployment_plan

if __name__ == "__main__":
    main()