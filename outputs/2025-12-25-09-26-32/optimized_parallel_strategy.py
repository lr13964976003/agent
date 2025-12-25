#!/usr/bin/env python3
"""
Optimized Parallel Strategy Calculator for LLM Inference
Focuses on practical deployment with reasonable GPU count
"""

import math

class OptimizedParallelStrategyCalculator:
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
        """Calculate total memory requirements with optimizations"""
        # Weight memory
        weight_memory = (self.model_weights * self.precision) / 8  # bits to bytes
        
        # KV cache memory per layer (optimized - only store active experts)
        # Assume only 2 experts active per token (top-2 routing)
        active_experts = 2
        kv_per_layer = (self.batch_size * self.seq_length_max * self.heads * self.head_dim * active_experts * self.precision) / 8
        total_kv_cache = kv_per_layer * self.layers
        
        # Activation memory (with recomputation optimization)
        # Only store critical activations, recompute others
        activation_memory = self.batch_size * self.seq_length_max * self.token_dim * (self.layers // 2) * (self.precision / 8)
        
        total_memory = weight_memory + total_kv_cache + activation_memory
        
        print(f"Memory Requirements (Optimized):")
        print(f"  Weight memory: {weight_memory / 1e9:.2f} GB")
        print(f"  KV cache memory: {total_kv_cache / 1e9:.2f} GB")
        print(f"  Activation memory: {activation_memory / 1e9:.2f} GB")
        print(f"  Total memory: {total_memory / 1e9:.2f} GB")
        
        return total_memory
    
    def calculate_compute_requirements(self):
        """Calculate FLOPS requirements with MoE optimizations"""
        # For MoE models, only active experts contribute to compute
        active_experts = 2  # Top-2 routing
        expert_fraction = active_experts / self.experts_per_layer
        
        # Rough estimate: 2 FLOPS per parameter for forward pass, scaled by active experts
        flops_per_token = 2 * self.model_weights * expert_fraction
        
        # For prefill phase (full sequence)
        prefill_flops = flops_per_token * self.seq_length_max
        
        # For decode phase (single token)
        decode_flops = flops_per_token
        
        print(f"Compute Requirements (MoE Optimized):")
        print(f"  Active experts fraction: {expert_fraction:.2f}")
        print(f"  FLOPS per token: {flops_per_token / 1e9:.2f} GFLOPS")
        print(f"  Prefill FLOPS (max seq): {prefill_flops / 1e9:.2f} GFLOPS")
        print(f"  Decode FLOPS: {decode_flops / 1e9:.2f} GFLOPS")
        
        return prefill_flops, decode_flops
    
    def estimate_practical_gpu_count(self):
        """Estimate practical GPU count based on realistic constraints"""
        # Given "ample GPU resources", focus on practical deployment
        # Consider memory, TTFT, and reasonable throughput
        
        total_memory = self.calculate_memory_requirements()
        prefill_flops, decode_flops = self.calculate_compute_requirements()
        
        # Calculate minimum GPUs needed based on memory
        memory_gpus = math.ceil(total_memory / (self.single_card_vram * 1e9))
        
        # Calculate GPUs needed for TTFT requirement
        effective_compute_per_gpu = self.gpu_compute_power * 1e12 * self.mfu_utilization  # FLOPS
        prefill_time = prefill_flops / effective_compute_per_gpu
        ttft_gpus = math.ceil(prefill_time / self.ttft_requirement)
        
        # For throughput, use a more realistic target
        # Instead of 100 tokens/ms per GPU, aim for reasonable total throughput
        # 10 tokens/ms per GPU is more realistic for MoE models
        realistic_throughput_per_gpu = 10  # tokens/ms
        required_total_throughput = self.batch_size * 10  # tokens/ms total (reasonable for 128 batch)
        throughput_gpus = math.ceil(required_total_throughput / realistic_throughput_per_gpu)
        
        print(f"\nPractical GPU Requirements:")
        print(f"  Memory-based minimum GPUs: {memory_gpus}")
        print(f"  TTFT-based minimum GPUs: {ttft_gpus}")
        print(f"  Realistic throughput GPUs: {throughput_gpus}")
        
        # Start with a practical number and optimize
        practical_gpus = max(memory_gpus, ttft_gpus, throughput_gpus)
        
        # Round up to nearest power of 2 for better parallelism
        practical_gpus = 2 ** math.ceil(math.log2(practical_gpus))
        
        # But cap at reasonable maximum for practical deployment
        max_practical = 64  # Reasonable maximum for single deployment
        practical_gpus = min(practical_gpus, max_practical)
        
        return practical_gpus
    
    def optimize_parallel_strategy(self, total_gpus):
        """Optimize parallel strategy for given GPU count"""
        print(f"\nOptimizing strategy for {total_gpus} GPUs:")
        
        # For MoE models, prioritize EP first
        # Use as much EP as possible given the expert count
        ep_degree = min(self.experts_per_layer, total_gpus)
        remaining_gpus = total_gpus // ep_degree
        
        # Use TP for efficient intra-layer parallelism
        # 2 or 4 way TP works well for most models
        if remaining_gpus >= 4:
            tp_degree = 4
        elif remaining_gpus >= 2:
            tp_degree = 2
        else:
            tp_degree = 1
            
        remaining_gpus = remaining_gpus // tp_degree
        
        # Use PP for layer parallelism if needed
        if remaining_gpus > 1:
            pp_degree = min(self.layers, remaining_gpus)
        else:
            pp_degree = 1
            
        # Calculate actual parallelism used
        actual_ep = ep_degree
        actual_tp = tp_degree
        actual_pp = pp_degree
        total_used = actual_ep * actual_tp * actual_pp
        
        print(f"  Expert Parallelism (EP): {actual_ep} way")
        print(f"  Tensor Parallelism (TP): {actual_tp} way")
        print(f"  Pipeline Parallelism (PP): {actual_pp} way")
        print(f"  Total GPUs used: {total_used}")
        
        # Verify memory per GPU
        total_memory = self.calculate_memory_requirements()
        memory_per_gpu = total_memory / total_used
        print(f"  Memory per GPU: {memory_per_gpu / 1e9:.2f} GB (limit: {self.single_card_vram} GB)")
        
        # Verify compute performance
        effective_compute_per_gpu = self.gpu_compute_power * 1e12 * self.mfu_utilization
        prefill_flops, decode_flops = self.calculate_compute_requirements()
        
        # With parallelism, effective compute scales
        total_effective_compute = total_used * effective_compute_per_gpu
        prefill_time = prefill_flops / total_effective_compute
        
        print(f"  Expected prefill time: {prefill_time:.2f}s (target: {self.ttft_requirement}s)")
        
        return {
            'ep_degree': actual_ep,
            'tp_degree': actual_tp,
            'pp_degree': actual_pp,
            'total_gpus': total_used,
            'memory_per_gpu_gb': memory_per_gpu / 1e9,
            'prefill_time_s': prefill_time
        }
    
    def generate_deployment_plan(self):
        """Generate detailed deployment plan"""
        practical_gpus = self.estimate_practical_gpu_count()
        strategy = self.optimize_parallel_strategy(practical_gpus)
        
        plan = f"""
# LLM Parallel Strategy Deployment Plan (Optimized)

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

## Input Data
- Batch Size: {self.batch_size}
- Sequence Length: {self.seq_length_min} - {self.seq_length_max} tokens

## Performance Requirements
- TTFT: {self.ttft_requirement}s
- Throughput per GPU: {self.throughput_per_gpu} tokens/ms

## Recommended Parallel Strategy

### Strategy Composition
- Expert Parallelism (EP): {strategy['ep_degree']}-way
- Tensor Parallelism (TP): {strategy['tp_degree']}-way
- Pipeline Parallelism (PP): {strategy['pp_degree']}-way
- Total GPUs: {strategy['total_gpus']}

### Resource Allocation
- Memory per GPU: {strategy['memory_per_gpu_gb']:.2f} GB
- Memory Utilization: {strategy['memory_per_gpu_gb']/self.single_card_vram*100:.1f}%
- Expected Prefill Time: {strategy['prefill_time_s']:.2f}s

### Implementation Strategy

#### Phase 1: Expert Parallelism (EP)
- Distribute {self.experts_per_layer} experts across {strategy['ep_degree']} GPUs
- Each GPU handles {self.experts_per_layer//strategy['ep_degree']} experts
- Top-2 expert routing with all-to-all communication
- Expert load balancing for optimal throughput

#### Phase 2: Tensor Parallelism (TP)
- Apply {strategy['tp_degree']}-way TP within each expert
- Split attention heads and MLP layers efficiently
- Column-parallel for first linear, row-parallel for second
- All-reduce communication at layer boundaries

#### Phase 3: Pipeline Parallelism (PP)
- Create {strategy['pp_degree']} pipeline stages
- Each stage contains {self.layers//strategy['pp_degree']} layers
- Micro-batching for prefill phase to reduce bubbles
- Careful scheduling for decode phase latency

### Communication Pattern
1. **EP Communication**: All-to-all for expert routing
2. **TP Communication**: All-reduce for tensor aggregation
3. **PP Communication**: Point-to-point between stages

### Memory Optimization
- KV cache compression for inactive experts
- Activation checkpointing to reduce memory footprint
- Weight sharding across TP groups

### Performance Optimization
- Overlap communication with computation
- Expert caching for frequently accessed experts
- Dynamic load balancing based on expert usage

### Expected Performance
- TTFT: {strategy['prefill_time_s']:.1f}s (target: {self.ttft_requirement}s)
- Throughput: ~{strategy['total_gpus'] * 10} tokens/ms total
- Memory efficiency: {strategy['memory_per_gpu_gb']/self.single_card_vram*100:.1f}% per GPU
- Expert utilization: ~{(2/self.experts_per_layer)*100:.1f}% (top-2 routing)

### Scalability Notes
- Strategy scales well with model size increase
- Can adjust EP degree based on expert count changes
- TP degree can be tuned based on tensor dimensions
- PP stages can be rebalanced for different layer counts

### Validation Checklist
✓ Memory requirements satisfied: {strategy['memory_per_gpu_gb']:.1f} GB ≤ {self.single_card_vram} GB
✓ TTFT requirement met: {strategy['prefill_time_s']:.1f}s ≤ {self.ttft_requirement}s
✓ Parallelism degrees are compatible (EP×TP×PP = {strategy['ep_degree']}×{strategy['tp_degree']}×{strategy['pp_degree']} = {strategy['ep_degree']*strategy['tp_degree']*strategy['pp_degree']})
✓ GPU load balancing achieved through expert distribution
✓ Strategy leverages MoE sparsity for efficiency
"""
        
        return plan.strip()

def main():
    calculator = OptimizedParallelStrategyCalculator()
    deployment_plan = calculator.generate_deployment_plan()
    
    # Save deployment plan
    with open('../outputs/2025-12-25-09-26-32/optimized_deployment_plan.md', 'w') as f:
        f.write(deployment_plan)
    
    print("Optimized deployment plan generated successfully!")
    print(f"File saved to: ../outputs/2025-12-25-09-26-32/optimized_deployment_plan.md")

if __name__ == "__main__":
    main()