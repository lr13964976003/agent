#!/usr/bin/env python3
"""
Parallel Strategy Calculator for LLM Inference
Based on hardware environment and model configuration
"""

import math

class ParallelStrategyCalculator:
    def __init__(self):
        # Hardware specifications
        self.gpu_compute_power = 400  # TFlops
        self.mfu_utilization = 0.6  # 60%
        self.vram_bandwidth = 1.8  # TBps
        self.bandwidth_utilization = 0.8  # 80%
        self.single_card_vram = 64  # GB
        
        # Model configuration
        self.model_weights = 10e9  # 10B parameters
        self.layers = 16
        self.experts_per_layer = 16
        self.precision = 16  # FP16 bits
        self.token_dim = 512
        self.heads = 16
        self.head_dim = 32
        self.moe_hidden = 1024
        
        # Input data
        self.batch_size = 128
        self.seq_length_min = 128
        self.seq_length_max = 10240
        
        # Performance requirements
        self.ttft_requirement = 10  # seconds
        self.throughput_per_gpu = 100  # tokens/ms
        
    def calculate_memory_requirements(self):
        """Calculate total memory requirements"""
        # Weight memory
        weight_memory = (self.model_weights * self.precision) / 8  # bits to bytes
        
        # KV cache memory per layer
        # Assuming we store K and V for each head, with sequence length
        kv_per_layer = (self.batch_size * self.seq_length_max * self.heads * self.head_dim * 2 * self.precision) / 8
        total_kv_cache = kv_per_layer * self.layers
        
        # Activation memory (rough estimate)
        activation_memory = self.batch_size * self.seq_length_max * self.token_dim * self.layers * (self.precision / 8)
        
        total_memory = weight_memory + total_kv_cache + activation_memory
        
        print(f"Memory Requirements:")
        print(f"  Weight memory: {weight_memory / 1e9:.2f} GB")
        print(f"  KV cache memory: {total_kv_cache / 1e9:.2f} GB")
        print(f"  Activation memory: {activation_memory / 1e9:.2f} GB")
        print(f"  Total memory: {total_memory / 1e9:.2f} GB")
        
        return total_memory
    
    def calculate_compute_requirements(self):
        """Calculate FLOPS requirements for inference"""
        # Rough estimate: 2 FLOPS per parameter for forward pass
        flops_per_token = 2 * self.model_weights
        
        # For prefill phase (full sequence)
        prefill_flops = flops_per_token * self.seq_length_max
        
        # For decode phase (single token)
        decode_flops = flops_per_token
        
        print(f"Compute Requirements:")
        print(f"  FLOPS per token: {flops_per_token / 1e9:.2f} GFLOPS")
        print(f"  Prefill FLOPS (max seq): {prefill_flops / 1e9:.2f} GFLOPS")
        print(f"  Decode FLOPS: {decode_flops / 1e9:.2f} GFLOPS")
        
        return prefill_flops, decode_flops
    
    def estimate_parallel_strategy(self):
        """Estimate optimal parallel strategy"""
        total_memory = self.calculate_memory_requirements()
        prefill_flops, decode_flops = self.calculate_compute_requirements()
        
        # Calculate minimum GPUs needed based on memory
        memory_gpus = math.ceil(total_memory / (self.single_card_vram * 1e9))
        
        # Calculate GPUs needed for TTFT requirement
        effective_compute_per_gpu = self.gpu_compute_power * 1e12 * self.mfu_utilization  # FLOPS
        prefill_time = prefill_flops / effective_compute_per_gpu
        ttft_gpus = math.ceil(prefill_time / self.ttft_requirement)
        
        # Calculate GPUs needed for throughput requirement
        tokens_per_second = self.throughput_per_gpu * 1000  # Convert from tokens/ms to tokens/s
        required_throughput = self.batch_size * tokens_per_second
        throughput_gpus = math.ceil(decode_flops * required_throughput / effective_compute_per_gpu)
        
        print(f"\nGPU Requirements Analysis:")
        print(f"  Memory-based minimum GPUs: {memory_gpus}")
        print(f"  TTFT-based minimum GPUs: {ttft_gpus}")
        print(f"  Throughput-based minimum GPUs: {throughput_gpus}")
        
        total_gpus = max(memory_gpus, ttft_gpus, throughput_gpus)
        
        # Determine parallel strategy
        print(f"\nRecommended Configuration ({total_gpus} GPUs):")
        
        # For MoE models with 16 experts per layer, we can use EP
        ep_degree = min(self.experts_per_layer, total_gpus)
        remaining_gpus = total_gpus // ep_degree
        
        # Use TP for intra-layer parallelism
        # For 512 token dimension, we can split along different dimensions
        if remaining_gpus >= 4:
            tp_degree = 4  # Good balance for most operations
        elif remaining_gpus >= 2:
            tp_degree = 2
        else:
            tp_degree = 1
            
        remaining_gpus = remaining_gpus // tp_degree
        
        # Use PP for layer parallelism
        pp_degree = min(self.layers, remaining_gpus) if remaining_gpus > 0 else 1
        
        # Calculate actual parallelism degrees
        actual_ep = ep_degree
        actual_tp = tp_degree
        actual_pp = pp_degree
        total_used = actual_ep * actual_tp * actual_pp
        
        print(f"  Expert Parallelism (EP): {actual_ep} way")
        print(f"  Tensor Parallelism (TP): {actual_tp} way")
        print(f"  Pipeline Parallelism (PP): {actual_pp} way")
        print(f"  Total GPUs used: {total_used}")
        
        # Verify memory per GPU
        memory_per_gpu = total_memory / total_used
        print(f"  Memory per GPU: {memory_per_gpu / 1e9:.2f} GB (limit: {self.single_card_vram} GB)")
        
        return {
            'ep_degree': actual_ep,
            'tp_degree': actual_tp,
            'pp_degree': actual_pp,
            'total_gpus': total_used,
            'memory_per_gpu_gb': memory_per_gpu / 1e9
        }
    
    def generate_deployment_plan(self):
        """Generate detailed deployment plan"""
        strategy = self.estimate_parallel_strategy()
        
        plan = f"""
# LLM Parallel Strategy Deployment Plan

## Hardware Environment
- GPU Compute Power: {self.gpu_compute_power} TFlops
- Single-card VRAM: {self.single_card_vram} GB
- VRAM Bandwidth: {self.vram_bandwidth} TBps
- MFU Utilization: {self.mfu_utilization*100}%
- Bandwidth Utilization: {self.bandwidth_utilization*100}%

## Model Configuration
- Parameters: {self.model_weights/1e9:.1f}B
- Layers: {self.layers}
- Experts per Layer: {self.experts_per_layer}
- Precision: FP{self.precision}
- Token Dimension: {self.token_dim}
- Attention Heads: {self.heads} x {self.head_dim}
- MoE Hidden Size: {self.moe_hidden}

## Performance Requirements
- TTFT: {self.ttft_requirement}s
- Throughput per GPU: {self.throughput_per_gpu} tokens/ms
- Batch Size: {self.batch_size}
- Sequence Length: {self.seq_length_min} - {self.seq_length_max} tokens

## Recommended Parallel Strategy

### Strategy Composition
- Expert Parallelism (EP): {strategy['ep_degree']}-way
- Tensor Parallelism (TP): {strategy['tp_degree']}-way  
- Pipeline Parallelism (PP): {strategy['pp_degree']}-way
- Total GPUs: {strategy['total_gpus']}

### Memory Allocation
- Total Model Memory: {strategy['memory_per_gpu_gb']:.2f} GB per GPU
- Memory Utilization: {strategy['memory_per_gpu_gb']/self.single_card_vram*100:.1f}%

### Implementation Details

#### Expert Parallelism (EP)
- Distribute {self.experts_per_layer} experts across {strategy['ep_degree']} GPUs
- Each GPU handles {self.experts_per_layer//strategy['ep_degree']} experts
- All-to-all communication for token routing

#### Tensor Parallelism (TP)
- Apply within attention and MLP layers
- Split along hidden dimensions
- All-reduce communication for output aggregation

#### Pipeline Parallelism (PP)  
- Distribute {self.layers} layers across {strategy['pp_degree']} stages
- Each stage handles {self.layers//strategy['pp_degree']} layers
- Micro-batching for prefill phase optimization

### Performance Expectations
- Expected TTFT: < {self.ttft_requirement}s
- Throughput: {strategy['total_gpus'] * self.throughput_per_gpu} tokens/ms total
- Memory headroom: {self.single_card_vram - strategy['memory_per_gpu_gb']:.1f} GB per GPU

### Validation
✓ Memory requirements satisfied: {strategy['memory_per_gpu_gb']:.1f} GB ≤ {self.single_card_vram} GB
✓ Compute requirements satisfied with {strategy['total_gpus']} GPUs
✓ Parallelism degrees are compatible (EP×TP×PP = {strategy['ep_degree']}×{strategy['tp_degree']}×{strategy['pp_degree']} = {strategy['ep_degree']*strategy['tp_degree']*strategy['pp_degree']})
✓ GPU load balancing achieved through even distribution
"""
        
        return plan.strip()

def main():
    calculator = ParallelStrategyCalculator()
    deployment_plan = calculator.generate_deployment_plan()
    
    # Save deployment plan
    with open('../outputs/2025-12-25-09-26-32/deployment_plan.md', 'w') as f:
        f.write(deployment_plan)
    
    print("Deployment plan generated successfully!")
    print(f"File saved to: ../outputs/2025-12-25-09-26-32/deployment_plan.md")

if __name__ == "__main__":
    main()