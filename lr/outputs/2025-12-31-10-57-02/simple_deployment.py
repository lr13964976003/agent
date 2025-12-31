#!/usr/bin/env python3
"""
Qwen3-235B MoE Model Parallel Strategy Deployment Plan
Optimized for maximum throughput while meeting TTFT requirements
"""

# Model configuration from deployment.md
weights = 235_000_000_000  # 235B parameters
layers = 94
experts_per_layer = 128
top_k_gate = 8
precision = 1  # FP8 = 1 byte
token_dim = 4096
num_heads = 64
head_dim = 64
mlp_hidden = 1536
vocab_size = 151936
gqa_kv_heads = 4

# Input data
batch_size = 128
seq_length_max = 10240
seq_in = 2048

# Hardware configuration
gpu_memory = 64_000_000_000  # 64GB in bytes
single_gpu_tflops = 400  # TFlops
mfu_utilization = 0.6

# Performance requirements
target_ttft = 30  # seconds

def calculate_memory_requirements(TP, PP, EP, SP):
    """Calculate memory usage per GPU for MoE model"""
    
    # Parameter memory breakdown
    attention_params = 4 * num_heads * head_dim * token_dim
    expert_params = experts_per_layer * (token_dim * mlp_hidden + mlp_hidden * token_dim + token_dim)
    layer_params = attention_params + expert_params + 2 * token_dim
    total_params = layers * layer_params + vocab_size * token_dim
    
    # Memory per GPU with parallelism
    experts_per_gpu = experts_per_layer / EP
    params_per_gpu = total_params / EP
    
    # KV cache memory
    kv_cache_per_token = gqa_kv_heads * head_dim * 2 * precision
    kv_cache_total = batch_size * seq_length_max * layers * kv_cache_per_token
    kv_cache_per_gpu = kv_cache_total / (PP * SP)
    
    # Activation memory
    activation_per_layer = batch_size * seq_length_max * token_dim * precision
    activation_total = activation_per_layer * layers
    activation_per_gpu = activation_total / (TP * PP * SP)
    
    # Total memory per GPU
    total_memory = (params_per_gpu * precision + kv_cache_per_gpu + activation_per_gpu)
    overhead = total_memory * 0.1
    final_memory = total_memory + overhead
    
    return {
        'total_memory': final_memory,
        'experts_per_gpu': experts_per_gpu
    }

def calculate_prefill_time(TP, PP, EP, SP):
    """Calculate prefill time for the configuration"""
    
    effective_tflops = single_gpu_tflops * mfu_utilization
    
    # Attention FLOPs
    attention_flops = batch_size * seq_in * seq_in * num_heads * head_dim * layers / 1e12
    
    # MoE FLOPs
    active_experts = top_k_gate
    moe_flops = batch_size * seq_in * active_experts * mlp_hidden * token_dim * layers / 1e12
    
    # Total prefill FLOPs
    total_prefill_flops = attention_flops + moe_flops
    
    # With parallelism
    parallel_factor = max(TP, EP)
    communication_overhead = 1.15
    
    prefill_time = (total_prefill_flops / parallel_factor / effective_tflops * communication_overhead)
    
    return prefill_time

def calculate_throughput(TP, EP):
    """Calculate decode throughput"""
    
    effective_tflops = single_gpu_tflops * mfu_utilization
    
    # Per-token FLOPs
    attention_token_flops = num_heads * head_dim * layers / 1e12
    moe_token_flops = top_k_gate * mlp_hidden * token_dim * layers / 1e12
    total_token_flops = attention_token_flops + moe_token_flops
    
    # Token time with parallelism
    parallel_factor = max(TP, EP)
    token_time = total_token_flops / parallel_factor / effective_tflops * 1.1
    
    throughput_per_gpu = 1 / token_time
    total_throughput = throughput_per_gpu * EP
    
    return total_throughput

def main():
    print("=== QWEN3-235B MOE DEPLOYMENT OPTIMIZATION ===")
    print(f"Model: Qwen3-235B")
    print(f"Parameters: {weights/1e9:.1f}B")
    print(f"Layers: {layers}")
    print(f"Experts per layer: {experts_per_layer}")
    print(f"Target TTFT: {target_ttft}s")
    print(f"GPU memory: {gpu_memory/1e9:.1f}GB")
    print()
    
    best_config = None
    best_score = float('inf')
    valid_configs = []
    
    # Test configurations focusing on EP (MoE dominant)
    ep_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for EP in ep_candidates:
        if experts_per_layer % EP != 0:
            continue
            
        for PP in [1, 2, 4, 8]:
            if layers % PP != 0:
                continue
                
            for TP in [1, 2, 4, 8]:
                for SP in [1, 2, 4]:
                    
                    # Calculate memory and performance
                    memory_info = calculate_memory_requirements(TP, PP, EP, SP)
                    memory_per_gpu = memory_info['total_memory']
                    prefill_time = calculate_prefill_time(TP, PP, EP, SP)
                    throughput = calculate_throughput(TP, EP)
                    
                    # Check constraints
                    memory_ok = memory_per_gpu < gpu_memory * 0.85
                    ttft_ok = prefill_time < target_ttft
                    
                    if memory_ok and ttft_ok:
                        # Calculate efficiency score
                        total_gpus = EP
                        memory_efficiency = memory_per_gpu / gpu_memory
                        latency_margin = target_ttft - prefill_time
                        
                        # Score: fewer GPUs, higher memory efficiency, more latency margin
                        score = total_gpus * (1 - memory_efficiency) * (1 / max(latency_margin, 0.1))
                        
                        config_info = {
                            'TP': TP, 'PP': PP, 'EP': EP, 'SP': SP,
                            'total_gpus': total_gpus,
                            'memory_per_gpu': memory_per_gpu,
                            'prefill_time': prefill_time,
                            'throughput': throughput,
                            'score': score,
                            'experts_per_gpu': memory_info['experts_per_gpu']
                        }
                        
                        valid_configs.append(config_info)
                        
                        if score < best_score:
                            best_score = score
                            best_config = config_info
    
    # Display all valid configurations
    print("=== VALID CONFIGURATIONS ===")
    for config in sorted(valid_configs, key=lambda x: x['score']):
        print(f"TP={config['TP']}, PP={config['PP']}, EP={config['EP']}, SP={config['SP']} ({config['total_gpus']} GPUs)")
        print(f"  Memory per GPU: {config['memory_per_gpu']/1e9:.2f}GB ({config['memory_per_gpu']/gpu_memory*100:.1f}%)")
        print(f"  Prefill time: {config['prefill_time']:.2f}s")
        print(f"  Throughput: {config['throughput']/1e6:.1f}M tokens/s")
        print(f"  Experts per GPU: {config['experts_per_gpu']}")
        print(f"  Efficiency score: {config['score']:.3f}")
        print()
    
    if best_config:
        print("=== OPTIMAL CONFIGURATION ===")
        print(f"Best configuration: TP={best_config['TP']}, PP={best_config['PP']}, EP={best_config['EP']}, SP={best_config['SP']}")
        print(f"Total GPUs: {best_config['total_gpus']}")
        print(f"Memory per GPU: {best_config['memory_per_gpu']/1e9:.2f}GB ({best_config['memory_per_gpu']/gpu_memory*100:.1f}%)")
        print(f"Prefill time: {best_config['prefill_time']:.2f}s (target: {target_ttft}s)")
        print(f"Throughput: {best_config['throughput']/1e6:.1f}M tokens/s")
        print(f"Experts per GPU: {best_config['experts_per_gpu']}")
        print()
        
        # Module division verification
        print("=== MODULE DIVISION VERIFICATION ===")
        layers_per_stage = layers / best_config['PP']
        heads_per_tp = num_heads / best_config['TP']
        experts_per_gpu = experts_per_layer / best_config['EP']
        
        print(f"Layers per PP stage: {layers_per_stage}")
        print(f"Attention heads per TP group: {heads_per_tp}")
        print(f"Experts per GPU: {experts_per_gpu}")
        print(f"Sequence partition: {best_config['SP']} ways")
        print(f"Memory utilization: {best_config['memory_per_gpu']/gpu_memory*100:.1f}%")
        print()
        
        # Save deployment plan
        deployment_plan = f"""# Qwen3-235B MoE Parallel Strategy Deployment Plan
# Generated: 2025-12-31-10-57-02

## Model Configuration
- Model: Qwen3-235B
- Parameters: 235B
- Layers: 94
- Experts per layer: 128
- Top-K gate: 8
- Precision: FP8

## Optimal Parallel Strategy
- Tensor Parallel (TP): {best_config['TP']}
- Pipeline Parallel (PP): {best_config['PP']}
- Expert Parallel (EP): {best_config['EP']}
- Sequence Parallel (SP): {best_config['SP']}

## Resource Allocation
- Total GPUs: {best_config['total_gpus']}
- Memory per GPU: {best_config['memory_per_gpu']/1e9:.2f}GB
- Memory utilization: {best_config['memory_per_gpu']/gpu_memory*100:.1f}%

## Performance Metrics
- Prefill time: {best_config['prefill_time']:.2f}s
- Target TTFT: {target_ttft}s
- Throughput: {best_config['throughput']/1e6:.1f}M tokens/s

## Module Division
- Layers per stage: {layers_per_stage}
- Experts per GPU: {experts_per_gpu}
- Attention heads per TP: {heads_per_tp}
- Sequence partition: {best_config['SP']} ways

## Deployment Notes
- EP dominates GPU allocation (MoE inference rule)
- Each GPU hosts {experts_per_gpu} experts
- Balanced load across all GPUs
- Meets TTFT requirement with margin
- Maximizes throughput while minimizing GPU usage
"""
        
        with open('./outputs/2025-12-31-10-57-02/deployment_plan.md', 'w') as f:
            f.write(deployment_plan)
        
        print("Deployment plan saved to: ./outputs/2025-12-31-10-57-02/deployment_plan.md")
        
    else:
        print("No valid configuration found!")

if __name__ == "__main__":
    main()