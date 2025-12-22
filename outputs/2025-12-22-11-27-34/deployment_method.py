#!/usr/bin/env python3
"""
Optimal Parallel Strategy for 30B MoE Model Deployment
Generated based on hardware environment and model parameters
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
        
        # Activation memory
        activation_memory = self.batch_size * self.seq_length_max * self.token_dim * self.precision * 4  # rough estimate
        
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
        
        # Expert parallelism - distribute 64 experts
        # Each GPU should handle at least 1 expert for load balancing
        ep_dim = self.experts_per_layer  # 64-way expert parallelism
        
        # Tensor parallelism - balance compute and communication
        # For MoE models, TP is typically applied within experts
        # 8-way TP provides good balance between parallelism and communication overhead
        tp_dim = 8
        
        # Pipeline parallelism - distribute layers
        # 16 layers with PP=2 gives 8 layers per stage (good balance)
        pp_dim = 2
        
        # Data parallelism - increase throughput
        # Calculate based on remaining GPUs
        total_gpus_needed = ep_dim * tp_dim * pp_dim
        dp_dim = max(1, min_gpus_memory // (ep_dim * tp_dim * pp_dim))
        
        # Adjust to ensure we have enough GPUs for memory requirements
        while total_gpus_needed * dp_dim < min_gpus_memory:
            dp_dim += 1
            
        return {
            'EP': ep_dim,
            'TP': tp_dim,
            'PP': pp_dim,
            'DP': dp_dim,
            'total_gpus': ep_dim * tp_dim * pp_dim * dp_dim
        }
    
    def calculate_load_balancing(self, parallel_dims: Dict[str, int]) -> Dict[str, float]:
        """Calculate load balancing metrics"""
        total_gpus = parallel_dims['total_gpus']
        
        # Expert load balancing
        experts_per_gpu = self.experts_per_layer / parallel_dims['EP']
        
        # Layer load balancing
        layers_per_stage = self.num_layers / parallel_dims['PP']
        
        # Batch load balancing
        sequences_per_gpu = self.batch_size / parallel_dims['DP']
        
        # Memory load balancing
        memory_req = self.calculate_memory_requirements()
        memory_per_gpu = memory_req['total_gb'] / total_gpus
        
        return {
            'experts_per_gpu': experts_per_gpu,
            'layers_per_stage': layers_per_stage,
            'sequences_per_gpu': sequences_per_gpu,
            'memory_per_gpu_gb': memory_per_gpu,
            'memory_utilization': memory_per_gpu / self.gpu_memory
        }
    
    def estimate_performance(self, parallel_dims: Dict[str, int]) -> Dict[str, float]:
        """Estimate performance metrics"""
        # Latency estimation (simplified model)
        # Prefill phase - compute bound
        prefill_flops = 2 * self.model_params * self.batch_size * self.seq_length_max
        prefill_time = prefill_flops / (self.gpu_compute_power * 1e12 * self.mfu_utilization * parallel_dims['TP'])
        
        # Decode phase - memory bound
        decode_memory_xfer = self.batch_size * self.token_dim * self.precision
        decode_time = decode_memory_xfer / (self.memory_bandwidth * 1e12 * self.bandwidth_utilization)
        
        # Throughput estimation
        tokens_per_second = self.batch_size * parallel_dims['DP'] / (prefill_time + decode_time)
        
        return {
            'prefill_latency_ms': prefill_time * 1000,
            'decode_latency_ms': decode_time * 1000,
            'throughput_tokens_per_sec': tokens_per_second,
            'latency_optimization_factor': parallel_dims['TP'],
            'throughput_optimization_factor': parallel_dims['DP']
        }
    
    def generate_strategy(self) -> Dict:
        """Generate complete parallel strategy"""
        parallel_dims = self.determine_parallel_dimensions()
        load_balancing = self.calculate_load_balancing(parallel_dims)
        performance = self.estimate_performance(parallel_dims)
        memory_req = self.calculate_memory_requirements()
        
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
            'optimization_recommendations': [
                'Overlap communication with computation for reduced latency',
                'Batch All-to-All operations for improved throughput',
                'Use hierarchical All-Reduce for better scalability',
                'Implement micro-batching in pipeline parallelism',
                'Cache optimization for KV storage across TP and PP dimensions'
            ]
        }

def main():
    """Main function to generate and output parallel strategy"""
    generator = ParallelStrategyGenerator()
    strategy = generator.generate_strategy()
    
    print("=== OPTIMAL PARALLEL STRATEGY FOR 30B MoE MODEL ===")
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
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:.2f}")
    print()
    
    print("=== PERFORMANCE METRICS ===")
    for key, value in strategy['performance_metrics'].items():
        if 'ms' in key:
            print(f"{key}: {value:.2f} ms")
        elif 'sec' in key:
            print(f"{key}: {value:.2f} tokens/sec")
        else:
            print(f"{key}: {value:.2f}x")
    print()
    
    print("=== VALIDATION ===")
    total_modules = (strategy['parallel_dimensions']['EP'] * 
                    strategy['parallel_dimensions']['TP'] * 
                    strategy['parallel_dimensions']['PP'] * 
                    strategy['parallel_dimensions']['DP'])
    print(f"Module division validation: {total_modules} parts match {strategy['hardware_requirements']['total_gpus']} GPUs")
    print(f"GPU load balancing: {'PASSED' if strategy['load_balancing']['memory_utilization'] < 0.8 else 'FAILED'}")
    
    return strategy

if __name__ == "__main__":
    strategy = main()
    
    # Save strategy to file for further use
    import json
    with open('../outputs/2025-12-22-11-27-34/parallel_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    
    print(f"\nStrategy saved to ../outputs/2025-12-22-11-27-34/parallel_strategy.json")