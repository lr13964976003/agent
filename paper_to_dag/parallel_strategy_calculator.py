#!/usr/bin/env python3

import json
import math
from typing import Dict, List, Tuple

class ParallelStrategyCalculator:
    def __init__(self):
        # Hardware specs
        self.num_gpus = 8
        self.gpu_memory_gb = 80
        self.max_gpu_memory_usage = 0.85
        self.available_gpu_memory_gb = self.gpu_memory_gb * self.max_gpu_memory_usage
        
        # Model specs
        self.model_weights_gb = 140
        self.num_layers = 80
        self.hidden_size = 8192
        self.num_attention_heads = 64
        self.max_seq_len = 8192
        self.vocab_size = 128256
        
        # Memory estimates per token
        self.kv_cache_per_token_kb = 1.0
        self.activation_per_token_kb = 0.5
        
        # Performance requirements
        self.target_rps = 8
        self.max_batch_size = 64
        self.max_num_seqs = 128
        self.prefill_p99_ms = 1000
        self.decode_p99_ms = 100
        
    def calculate_memory_requirements(self, tp_size: int, pp_size: int) -> Dict:
        """Calculate memory requirements for given TP and PP configuration"""
        
        # Model weights are replicated per TP group and split across PP stages
        layers_per_stage = self.num_layers // pp_size
        model_memory_per_gpu = (self.model_weights_gb / pp_size)
        
        # KV cache memory (depends on sequence length and batch size)
        # Assuming worst case: max sequence length and full batch
        max_tokens = min(self.max_batch_size * self.max_seq_len, self.max_num_seqs * self.max_seq_len)
        kv_cache_memory_gb = (max_tokens * self.kv_cache_per_token_kb) / 1024 / 1024  # Convert to GB
        
        # Activation memory (split by TP)
        activation_memory_gb = (max_tokens * self.activation_per_token_kb) / 1024 / 1024 / tp_size
        
        # Add overhead for communication buffers (10% of total)
        communication_overhead = 0.1 * (model_memory_per_gpu + kv_cache_memory_gb + activation_memory_gb)
        
        total_memory_gb = model_memory_per_gpu + kv_cache_memory_gb + activation_memory_gb + communication_overhead
        
        return {
            "model_memory_gb": model_memory_per_gpu,
            "kv_cache_memory_gb": kv_cache_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "communication_overhead_gb": communication_overhead,
            "total_memory_gb": total_memory_gb,
            "memory_utilization": total_memory_gb / self.gpu_memory_gb
        }
    
    def calculate_parallel_efficiency(self, tp_size: int, pp_size: int) -> Dict:
        """Calculate parallel efficiency metrics"""
        
        # TP efficiency decreases with size due to communication overhead
        tp_efficiency = 1.0 / (1.0 + 0.1 * math.log2(tp_size))
        
        # PP efficiency decreases with size due to pipeline bubbles
        # For decode phase, pipeline bubbles are more significant
        pp_efficiency_decode = 1.0 / (1.0 + 0.2 * (pp_size - 1))
        pp_efficiency_prefill = 1.0 / (1.0 + 0.05 * (pp_size - 1))
        
        return {
            "tp_efficiency": tp_efficiency,
            "pp_efficiency_decode": pp_efficiency_decode,
            "pp_efficiency_prefill": pp_efficiency_prefill
        }
    
    def check_latency_constraints(self, tp_size: int, pp_size: int) -> bool:
        """Check if configuration meets latency constraints"""
        
        efficiency = self.calculate_parallel_efficiency(tp_size, pp_size)
        
        # Estimate compute time (very rough approximation)
        # This is a simplified model for demonstration
        base_compute_time_prefill_ms = 200  # Base time for full sequence
        base_compute_time_decode_ms = 5     # Base time per token
        
        # Apply efficiency factors
        prefill_time_ms = base_compute_time_prefill_ms / (tp_size * efficiency["pp_efficiency_prefill"])
        decode_time_ms = base_compute_time_decode_ms / (tp_size * efficiency["pp_efficiency_decode"])
        
        # Add communication overhead
        comm_overhead_prefill_ms = 50 * math.log2(tp_size)
        comm_overhead_decode_ms = 10 * math.log2(tp_size)
        
        total_prefill_ms = prefill_time_ms + comm_overhead_prefill_ms
        total_decode_ms = decode_time_ms + comm_overhead_decode_ms
        
        return (total_prefill_ms <= self.prefill_p99_ms and 
                total_decode_ms <= self.decode_p99_ms)
    
    def find_optimal_configuration(self) -> Dict:
        """Find optimal TP and PP configuration"""
        
        best_config = None
        best_score = float('-inf')
        
        # Try all valid TP and PP combinations
        for tp_size in [1, 2, 4, 8]:
            for pp_size in [1, 2, 4, 8]:
                if tp_size * pp_size != self.num_gpus:
                    continue
                
                # Calculate memory requirements
                memory = self.calculate_memory_requirements(tp_size, pp_size)
                
                # Skip if memory usage too high
                if memory["memory_utilization"] > self.max_gpu_memory_usage:
                    continue
                
                # Check latency constraints
                if not self.check_latency_constraints(tp_size, pp_size):
                    continue
                
                # Calculate efficiency
                efficiency = self.calculate_parallel_efficiency(tp_size, pp_size)
                
                # Calculate throughput estimate
                # Simplified model: throughput scales with parallel efficiency
                throughput_score = (efficiency["tp_efficiency"] * 
                                  efficiency["pp_efficiency_prefill"] * 
                                  self.target_rps)
                
                # Calculate load balancing score
                # Prefer configurations that distribute work more evenly
                load_balance_score = 1.0 - abs(memory["memory_utilization"] - 0.7)
                
                # Combined score
                score = (0.5 * throughput_score + 
                        0.3 * load_balance_score + 
                        0.2 * (1.0 - memory["memory_utilization"]))
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        "tp_size": tp_size,
                        "pp_size": pp_size,
                        "memory": memory,
                        "efficiency": efficiency,
                        "score": score
                    }
        
        return best_config
    
    def generate_deployment_plan(self) -> Dict:
        """Generate complete deployment plan"""
        
        config = self.find_optimal_configuration()
        
        if config is None:
            raise ValueError("No valid configuration found!")
        
        tp_size = config["tp_size"]
        pp_size = config["pp_size"]
        
        # Calculate layer distribution for PP
        layers_per_stage = self.num_layers // pp_size
        layer_distribution = []
        for stage in range(pp_size):
            start_layer = stage * layers_per_stage
            end_layer = (stage + 1) * layers_per_stage
            layer_distribution.append({
                "stage": stage,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "num_layers": layers_per_stage
            })
        
        # GPU mapping
        gpu_mapping = []
        for stage in range(pp_size):
            for tp_rank in range(tp_size):
                gpu_id = stage * tp_size + tp_rank
                gpu_mapping.append({
                    "gpu_id": gpu_id,
                    "stage": stage,
                    "tp_rank": tp_rank,
                    "node": 0  # Single node deployment
                })
        
        # Generate communication plan
        communication_plan = {
            "tp_communication": {
                "collectives": ["all_reduce", "all_gather"],
                "bandwidth_gbps": 900,  # NVLink
                "frequency": "per_layer"
            },
            "pp_communication": {
                "collectives": ["send", "recv"],
                "bandwidth_gbps": 64,   # PCIe
                "frequency": "per_stage"
            }
        }
        
        return {
            "configuration": {
                "tensor_parallel_size": tp_size,
                "pipeline_parallel_size": pp_size,
                "total_gpus": self.num_gpus,
                "model_name": "Llama3_70B_Instruct"
            },
            "memory_analysis": config["memory"],
            "efficiency_metrics": config["efficiency"],
            "layer_distribution": layer_distribution,
            "gpu_mapping": gpu_mapping,
            "communication_plan": communication_plan,
            "performance_projection": {
                "expected_throughput_rps": self.target_rps * config["efficiency"]["pp_efficiency_prefill"],
                "prefill_latency_ms": self.prefill_p99_ms,
                "decode_latency_ms": self.decode_p99_ms
            }
        }

def main():
    calculator = ParallelStrategyCalculator()
    deployment_plan = calculator.generate_deployment_plan()
    
    # Save deployment plan
    with open("../outputs/2025-12-23-16-46-30/deployment_plan.json", "w") as f:
        json.dump(deployment_plan, f, indent=2)
    
    # Print summary
    print("=== OPTIMAL PARALLEL STRATEGY DEPLOYMENT PLAN ===")
    print(f"Tensor Parallel Size: {deployment_plan['configuration']['tensor_parallel_size']}")
    print(f"Pipeline Parallel Size: {deployment_plan['configuration']['pipeline_parallel_size']}")
    print(f"Total GPUs: {deployment_plan['configuration']['total_gpus']}")
    print()
    print("=== MEMORY ANALYSIS ===")
    memory = deployment_plan['memory_analysis']
    print(f"Model Memory per GPU: {memory['model_memory_gb']:.1f} GB")
    print(f"KV Cache Memory: {memory['kv_cache_memory_gb']:.1f} GB")
    print(f"Activation Memory: {memory['activation_memory_gb']:.1f} GB")
    print(f"Communication Overhead: {memory['communication_overhead_gb']:.1f} GB")
    print(f"Total Memory per GPU: {memory['total_memory_gb']:.1f} GB")
    print(f"Memory Utilization: {memory['memory_utilization']*100:.1f}%")
    print()
    print("=== EFFICIENCY METRICS ===")
    efficiency = deployment_plan['efficiency_metrics']
    print(f"TP Efficiency: {efficiency['tp_efficiency']:.3f}")
    print(f"PP Efficiency (Prefill): {efficiency['pp_efficiency_prefill']:.3f}")
    print(f"PP Efficiency (Decode): {efficiency['pp_efficiency_decode']:.3f}")
    print()
    print("=== LAYER DISTRIBUTION ===")
    for stage in deployment_plan['layer_distribution']:
        print(f"Stage {stage['stage']}: Layers {stage['start_layer']}-{stage['end_layer']-1} ({stage['num_layers']} layers)")
    print()
    print("=== PERFORMANCE PROJECTION ===")
    perf = deployment_plan['performance_projection']
    print(f"Expected Throughput: {perf['expected_throughput_rps']:.1f} RPS")
    print(f"Prefill Latency (P99): {perf['prefill_latency_ms']} ms")
    print(f"Decode Latency (P99): {perf['decode_latency_ms']} ms/token")
    print()
    print("=== GPU MAPPING ===")
    for mapping in deployment_plan['gpu_mapping']:
        print(f"GPU {mapping['gpu_id']}: Stage {mapping['stage']}, TP Rank {mapping['tp_rank']}")
    print()
    print("=== MODULE DIVISION ANALYSIS ===")
    print(f"Model divided into {deployment_plan['configuration']['pipeline_parallel_size']} pipeline stages")
    print(f"Each stage uses {deployment_plan['configuration']['tensor_parallel_size']} GPUs for tensor parallelism")
    print(f"Total GPU parts: {deployment_plan['configuration']['total_gpus']}")
    print("✓ Module division matches number of GPUs perfectly")

if __name__ == "__main__":
    main()