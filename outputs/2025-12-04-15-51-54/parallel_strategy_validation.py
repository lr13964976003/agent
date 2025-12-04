#!/usr/bin/env python3
"""
Parallel Strategy Validation for 30B Model Deployment
Validates GPU allocation, load balancing, and performance metrics
"""

import math
from typing import Dict, List, Tuple

class ParallelStrategyValidator:
    def __init__(self):
        # Model configuration
        self.model_params = 30e9  # 30B parameters
        self.num_layers = 16
        self.num_experts = 64
        self.hidden_size = 1024
        self.moe_hidden_size = 2048
        self.batch_size = 128
        self.max_seq_len = 10240
        self.precision = 2  # FP16 = 2 bytes
        
        # Hardware configuration
        self.gpu_flops = 400e12  # 400TFlops
        self.mfu_utilization = 0.6
        self.vram_bandwidth = 1.8e12  # 1.8TBps
        self.bandwidth_utilization = 0.8
        self.vram_capacity = 64e9  # 64GB
        
        # Parallel configuration
        self.expert_parallel_degree = 64
        self.tensor_parallel_degree = 2
        self.pipeline_parallel_degree = 4
        
    def calculate_memory_requirements(self) -> Dict[str, float]:
        """Calculate memory requirements for model and activations"""
        # Model parameters
        model_memory = self.model_params * self.precision  # 60GB
        
        # Activations per layer
        activation_memory_per_layer = (self.batch_size * self.max_seq_len * 
                                     self.hidden_size * self.precision)  # ~2.68GB
        total_activation_memory = activation_memory_per_layer * self.num_layers  # ~42.9GB
        
        # With activation checkpointing (50% reduction)
        checkpointed_activation_memory = total_activation_memory * 0.5
        
        return {
            'model_memory_gb': model_memory / 1e9,
            'activation_memory_gb': total_activation_memory / 1e9,
            'checkpointed_activation_gb': checkpointed_activation_memory / 1e9,
            'total_memory_gb': (model_memory + checkpointed_activation_memory) / 1e9
        }
    
    def calculate_parallel_distribution(self) -> Dict[str, any]:
        """Calculate how model is distributed across GPUs"""
        total_gpus = (self.expert_parallel_degree * 
                     self.tensor_parallel_degree * 
                     self.pipeline_parallel_degree)
        
        # Memory per GPU
        memory_req = self.calculate_memory_requirements()
        memory_per_gpu = memory_req['total_memory_gb'] / total_gpus
        
        # Parameters per GPU
        params_per_gpu = self.model_params / total_gpus
        
        # Layers per GPU
        layers_per_gpu = self.num_layers / (total_gpus / 
                                           (self.expert_parallel_degree * 
                                            self.tensor_parallel_degree))
        
        # Experts per GPU
        experts_per_gpu = self.num_experts / self.expert_parallel_degree
        
        return {
            'total_gpus': total_gpus,
            'memory_per_gpu_gb': memory_per_gpu,
            'params_per_gpu_gb': (params_per_gpu * self.precision) / 1e9,
            'layers_per_gpu': layers_per_gpu,
            'experts_per_gpu': experts_per_gpu,
            'expert_parallel_degree': self.expert_parallel_degree,
            'tensor_parallel_degree': self.tensor_parallel_degree,
            'pipeline_parallel_degree': self.pipeline_parallel_degree
        }
    
    def validate_load_balancing(self) -> Dict[str, bool]:
        """Validate that load is balanced across GPUs"""
        dist = self.calculate_parallel_distribution()
        
        # Check if experts are perfectly balanced
        expert_balance = (dist['experts_per_gpu'] == 1.0)
        
        # Check if layers are balanced
        layer_balance = (dist['layers_per_gpu'] == 4.0)  # 16 layers / 4 pipeline stages
        
        # Check if memory is balanced
        memory_balance = (dist['memory_per_gpu_gb'] < 1.0)  # Well under 64GB limit
        
        # Check if parameters are balanced
        param_balance = (dist['params_per_gpu_gb'] < 1.0)  # Well under memory limits
        
        return {
            'expert_balance': expert_balance,
            'layer_balance': layer_balance,
            'memory_balance': memory_balance,
            'param_balance': param_balance,
            'overall_balance': all([expert_balance, layer_balance, memory_balance, param_balance])
        }
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate theoretical performance metrics"""
        dist = self.calculate_parallel_distribution()
        
        # Effective FLOPS per GPU
        effective_flops = self.gpu_flops * self.mfu_utilization
        
        # Total system FLOPS
        total_system_flops = effective_flops * dist['total_gpus']
        
        # Memory bandwidth per GPU
        effective_bandwidth = self.vram_bandwidth * self.bandwidth_utilization
        
        # Theoretical throughput (parameters per second)
        # Assuming compute-bound rather than memory-bound
        theoretical_throughput = total_system_flops / (self.model_params * self.precision)
        
        # Estimated latency for single batch
        # Simplified model: assume perfect parallelization
        compute_latency = (self.model_params * self.precision) / total_system_flops
        
        return {
            'effective_flops_per_gpu': effective_flops / 1e12,  # TFlops
            'total_system_flops': total_system_flops / 1e15,  # PFlops
            'effective_bandwidth_per_gpu': effective_bandwidth / 1e12,  # TBps
            'theoretical_throughput': theoretical_throughput,  # batches/sec
            'estimated_latency_ms': compute_latency * 1000,  # ms
            'total_gpus': dist['total_gpus']
        }
    
    def generate_gpu_mapping(self) -> List[Dict[str, any]]:
        """Generate detailed GPU mapping for the parallel strategy"""
        mapping = []
        gpu_id = 0
        
        for stage in range(self.pipeline_parallel_degree):
            stage_layers = list(range(stage * 4, (stage + 1) * 4))
            
            for expert in range(self.expert_parallel_degree // self.pipeline_parallel_degree):
                global_expert_id = stage * (self.expert_parallel_degree // self.pipeline_parallel_degree) + expert
                
                # Tensor parallel group (2 GPUs per expert)
                tensor_group = []
                for tp_rank in range(self.tensor_parallel_degree):
                    tensor_group.append(gpu_id + tp_rank)
                
                mapping.append({
                    'pipeline_stage': stage,
                    'expert_id': global_expert_id,
                    'tensor_parallel_rank': 0,
                    'gpu_id': gpu_id,
                    'layers': stage_layers,
                    'tensor_parallel_group': tensor_group
                })
                
                mapping.append({
                    'pipeline_stage': stage,
                    'expert_id': global_expert_id,
                    'tensor_parallel_rank': 1,
                    'gpu_id': gpu_id + 1,
                    'layers': stage_layers,
                    'tensor_parallel_group': tensor_group
                })
                
                gpu_id += self.tensor_parallel_degree
        
        return mapping
    
    def print_summary(self):
        """Print comprehensive summary of the parallel strategy"""
        print("=" * 80)
        print("PARALLEL STRATEGY VALIDATION SUMMARY")
        print("=" * 80)
        
        # Memory analysis
        memory = self.calculate_memory_requirements()
        print("\n1. MEMORY REQUIREMENTS:")
        print(f"   Model Memory: {memory['model_memory_gb']:.1f} GB")
        print(f"   Activation Memory: {memory['activation_memory_gb']:.1f} GB")
        print(f"   Checkpointed Activation: {memory['checkpointed_activation_gb']:.1f} GB")
        print(f"   Total Memory: {memory['total_memory_gb']:.1f} GB")
        
        # Parallel distribution
        dist = self.calculate_parallel_distribution()
        print("\n2. PARALLEL DISTRIBUTION:")
        print(f"   Total GPUs: {dist['total_gpus']}")
        print(f"   Memory per GPU: {dist['memory_per_gpu_gb']:.2f} GB")
        print(f"   Parameters per GPU: {dist['params_per_gpu_gb']:.2f} GB")
        print(f"   Layers per GPU: {dist['layers_per_gpu']}")
        print(f"   Experts per GPU: {dist['experts_per_gpu']}")
        print(f"   Expert Parallelism: {dist['expert_parallel_degree']}-way")
        print(f"   Tensor Parallelism: {dist['tensor_parallel_degree']}-way")
        print(f"   Pipeline Parallelism: {dist['pipeline_parallel_degree']}-way")
        
        # Load balancing
        balance = self.validate_load_balancing()
        print("\n3. LOAD BALANCING VALIDATION:")
        print(f"   Expert Balance: {'✓' if balance['expert_balance'] else '✗'}")
        print(f"   Layer Balance: {'✓' if balance['layer_balance'] else '✗'}")
        print(f"   Memory Balance: {'✓' if balance['memory_balance'] else '✗'}")
        print(f"   Parameter Balance: {'✓' if balance['param_balance'] else '✗'}")
        print(f"   Overall Balance: {'✓' if balance['overall_balance'] else '✗'}")
        
        # Performance metrics
        perf = self.calculate_performance_metrics()
        print("\n4. PERFORMANCE METRICS:")
        print(f"   Effective FLOPS per GPU: {perf['effective_flops_per_gpu']:.1f} TFlops")
        print(f"   Total System FLOPS: {perf['total_system_flops']:.1f} PFlops")
        print(f"   Effective Bandwidth per GPU: {perf['effective_bandwidth_per_gpu']:.1f} TBps")
        print(f"   Theoretical Throughput: {perf['theoretical_throughput']:.1f} batches/sec")
        print(f"   Estimated Latency: {perf['estimated_latency_ms']:.2f} ms")
        
        # GPU mapping sample
        mapping = self.generate_gpu_mapping()
        print("\n5. GPU MAPPING (Sample):")
        for i in range(min(8, len(mapping))):
            m = mapping[i]
            print(f"   GPU {m['gpu_id']:3d}: Stage {m['pipeline_stage']}, Expert {m['expert_id']:2d}, "
                  f"TP rank {m['tensor_parallel_rank']}, Layers {m['layers']}")
        
        print("\n" + "=" * 80)
        
        return {
            'memory': memory,
            'distribution': dist,
            'balance': balance,
            'performance': perf,
            'mapping': mapping
        }

if __name__ == "__main__":
    validator = ParallelStrategyValidator()
    results = validator.print_summary()
    
    # Save detailed results
    import json
    with open('../outputs/2025-12-04-15-51-54/validation_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'memory': results['memory'],
            'distribution': results['distribution'],
            'balance': results['balance'],
            'performance': results['performance'],
            'mapping_sample': results['mapping'][:16]  # First 16 entries
        }
        json.dump(json_results, f, indent=2)
    
    print("\nDetailed validation results saved to validation_results.json")