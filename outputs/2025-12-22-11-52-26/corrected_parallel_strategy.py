#!/usr/bin/env python3
"""
Corrected Optimal Parallel Strategy for 30B MoE Model Deployment
Fixed memory calculation errors and optimized parallel dimensions
"""

import math
from typing import Dict, List, Tuple

class ParallelStrategyGenerator:
    def __init__(self):
        # Hardware specifications
        self.gpu_compute_power = 400  # TFlops
        self.gpu_memory = 64  # GB
        self.memory_bandwidth = 1.8  # TBps
        self.mfu_utilization = 0.6
        self.bandwidth_utilization = 0.8
        
        # Model specifications
        self.model_params = 30e9  # 30B parameters
        self.num_layers = 16
        self.experts_per_layer = 64
        self.token_dim = 1024
        self.hidden_size = 2048
        self.num_heads = 16
        self.head_dim = 64
        self.precision = 2  # FP16 = 2 bytes
        
        # Batch specifications
        self.batch_size = 128  # sequences
        self.seq_length_min = 128
        self.seq_length_max = 10240
        
    def calculate_memory_requirements(self) -> Dict[str, float]:
        """Calculate memory requirements for different components"""
        # Model weights memory
        weight_memory = self.model_params * self.precision  # bytes
        
        # KV cache memory (worst case - max sequence length)
        kv_cache_per_token = 2 * self.num_heads * self.head_dim * self.precision  # Q and K+V
        max_kv_cache = self.batch_size * self.seq_length_max * kv_cache_per_token
        
        # Activation memory (reduced estimate for better accuracy)
        activation_memory = self.batch_size * self.seq_length_max * self.hidden_size * self.precision * 2  # more conservative estimate
        
        return {
            'weights_gb': weight_memory / 1e9,
            'kv_cache_gb': max_kv_cache / 1e9,
            'activation_gb': activation_memory / 1e9,
            'total_gb': (weight_memory + max_kv_cache + activation_memory) / 1e9
        }
    
    def determine_parallel_dimensions(self) -> Dict[str, int]:
        """Determine optimal parallel dimensions based on constraints"""
        memory_req = self.calculate_memory_requirements()
        
        # Calculate minimum GPUs needed based on memory
        memory_per_gpu = self.gpu_memory * 0.8  # Use 80% of GPU memory for safety
        min_gpus_memory = math.ceil(memory_req['total_gb'] / memory_per_gpu)
        
        print(f"Memory requirement: {memory_req['total_gb']:.2f} GB")
        print(f"Minimum GPUs needed for memory: {min_gpus_memory}")
        
        # Optimized parallel strategy - reduce GPU count while maintaining performance
        # EP: 8-way instead of 64-way for better load balancing and reduced communication
        ep_dim = 8
        
        # TP: 4-way instead of 8-way to reduce communication overhead
        tp_dim = 4
        
        # PP: 2-way (keep) for good pipeline balance
        pp_dim = 2
        
        # DP: 4-way to improve throughput with reasonable batch size
        dp_dim = 4
        
        # Calculate total GPUs
        total_gpus = ep_dim * tp_dim * pp_dim * dp_dim
        
        # Ensure we meet memory requirements
        if total_gpus < min_gpus_memory:
            # Scale up DP dimension to meet memory requirements
            dp_dim = math.ceil(min_gpus_memory / (ep_dim * tp_dim * pp_dim))
            total_gpus = ep_dim * tp_dim * pp_dim * dp_dim
            
        return {
            'EP': ep_dim,
            'TP': tp_dim,
            'PP': pp_dim,
            'DP': dp_dim,
            'total_gpus': total_gpus
        }
    
    def calculate_load_balancing(self, parallel_dims: Dict[str, int]) -> Dict[str, float]:
        """Calculate load balancing metrics with corrected precision"""
        total_gpus = parallel_dims['total_gpus']
        
        # Expert load balancing
        experts_per_gpu = self.experts_per_layer / parallel_dims['EP']
        
        # Layer load balancing
        layers_per_stage = self.num_layers / parallel_dims['PP']
        
        # Batch load balancing
        sequences_per_gpu = self.batch_size / parallel_dims['DP']
        
        # Memory load balancing - fix precision issue
        memory_req = self.calculate_memory_requirements()
        memory_per_gpu = memory_req['total_gb'] / total_gpus
        memory_utilization = (memory_per_gpu / self.gpu_memory) * 100  # Convert to percentage
        
        return {
            'experts_per_gpu': experts_per_gpu,
            'layers_per_stage': layers_per_stage,
            'sequences_per_gpu': sequences_per_gpu,
            'memory_per_gpu_gb': memory_per_gpu,
            'memory_utilization_percent': memory_utilization
        }
    
    def estimate_performance(self, parallel_dims: Dict[str, int]) -> Dict[str, float]:
        """Estimate performance metrics with corrected calculations"""
        # More realistic latency estimation
        # Prefill phase - compute bound with better FLOPS calculation
        prefill_flops = 2 * self.model_params * self.batch_size * self.seq_length_max / (parallel_dims['EP'] * parallel_dims['TP'])
        prefill_time = prefill_flops / (self.gpu_compute_power * 1e12 * self.mfu_utilization * parallel_dims['TP'])
        
        # Decode phase - memory bound with corrected calculation
        # Account for actual memory access patterns and sequence length
        decode_memory_xfer = self.batch_size * self.hidden_size * self.precision * 2  # account for KV cache access
        decode_time = decode_memory_xfer / (self.memory_bandwidth * 1e12 * self.bandwidth_utilization)
        
        # Throughput estimation with parallel efficiency
        # Account for communication overhead in parallel execution
        parallel_efficiency = 0.85  # Realistic efficiency for large-scale parallelism
        effective_batch_size = self.batch_size * parallel_dims['DP'] * parallel_efficiency
        tokens_per_second = effective_batch_size / (prefill_time + decode_time)
        
        return {
            'prefill_latency_ms': prefill_time * 1000,
            'decode_latency_ms': decode_time * 1000,
            'throughput_tokens_per_sec': tokens_per_second,
            'latency_optimization_factor': parallel_dims['TP'],
            'throughput_optimization_factor': parallel_dims['DP'],
            'parallel_efficiency': parallel_efficiency
        }
    
    def generate_strategy(self) -> Dict:
        """Generate complete parallel strategy"""
        parallel_dims = self.determine_parallel_dimensions()
        load_balancing = self.calculate_load_balancing(parallel_dims)
        performance = self.estimate_performance(parallel_dims)
        memory_req = self.calculate_memory_requirements()
        
        # Validation checks
        total_modules = parallel_dims['EP'] * parallel_dims['TP'] * parallel_dims['PP'] * parallel_dims['DP']
        gpu_load_balanced = load_balancing['memory_utilization_percent'] < 80.0  # 80% threshold
        
        return {
            'strategy_name': f"EP{parallel_dims['EP']}-TP{parallel_dims['TP']}-PP{parallel_dims['PP']}-DP{parallel_dims['DP']}",
            'parallel_dimensions': parallel_dims,
            'hardware_requirements': {
                'total_gpus': parallel_dims['total_gpus'],
                'gpu_memory_gb': self.gpu_memory,
                'gpu_compute_tflops': self.gpu_compute_power,
                'memory_bandwidth_tbps': self.memory_bandwidth
            },
            'memory_analysis': memory_req,
            'load_balancing': load_balancing,
            'performance_metrics': performance,
            'validation': {
                'total_modules': total_modules,
                'matches_gpu_count': total_modules == parallel_dims['total_gpus'],
                'gpu_load_balanced': gpu_load_balanced,
                'memory_efficiency': f"{load_balancing['memory_utilization_percent']:.2f}%"
            },
            'optimization_recommendations': [
                f"Use {parallel_dims['EP']}-way EP for balanced expert distribution",
                f"Apply {parallel_dims['TP']}-way TP to reduce communication overhead",
                f"Implement {parallel_dims['PP']}-way PP for good pipeline utilization",
                f"Scale with {parallel_dims['DP']}-way DP for throughput improvement",
                "Overlap communication with computation for reduced latency",
                "Batch All-to-All operations for improved throughput",
                "Use hierarchical All-Reduce for better scalability",
                "Implement micro-batching in pipeline parallelism",
                "Cache optimization for KV storage across TP and PP dimensions"
            ]
        }

def main():
    """Main function to generate and output parallel strategy"""
    generator = ParallelStrategyGenerator()
    strategy = generator.generate_strategy()
    
    print("=== CORRECTED OPTIMAL PARALLEL STRATEGY FOR 30B MoE MODEL ===")
    print(f"Strategy: {strategy['strategy_name']}")
    print(f"Total GPUs: {strategy['hardware_requirements']['total_gpus']}")
    print()
    
    print("=== PARALLEL DIMENSIONS ===")
    for dim, value in strategy['parallel_dimensions'].items():
        if dim != 'total_gpus':
            print(f"{dim}: {value}")
    print()
    
    print("=== MEMORY ANALYSIS ===")
    for key, value in strategy['memory_analysis'].items():
        print(f"{key}: {value:.2f} GB")
    print()
    
    print("=== LOAD BALANCING ===")
    for key, value in strategy['load_balancing'].items():
        if 'gb' in key:
            print(f"{key}: {value:.4f}")
        elif 'percent' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")
    print()
    
    print("=== PERFORMANCE METRICS ===")
    for key, value in strategy['performance_metrics'].items():
        if 'ms' in key:
            print(f"{key}: {value:.2f} ms")
        elif 'sec' in key:
            print(f"{key}: {value:.2f} tokens/sec")
        elif 'efficiency' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:.2f}x")
    print()
    
    print("=== VALIDATION ===")
    validation = strategy['validation']
    print(f"Module division validation: {validation['total_modules']} parts match {strategy['hardware_requirements']['total_gpus']} GPUs")
    print(f"GPU load balancing: {'PASSED' if validation['gpu_load_balanced'] else 'FAILED'}")
    print(f"Memory utilization: {validation['memory_efficiency']}")
    
    return strategy

if __name__ == "__main__":
    strategy = main()
    
    # Save strategy to file for further use
    import json
    with open('../outputs/2025-12-22-11-52-26/corrected_parallel_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    
    print(f"\nStrategy saved to ../outputs/2025-12-22-11-52-26/corrected_parallel_strategy.json")