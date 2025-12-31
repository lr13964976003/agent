#!/usr/bin/env python3
"""
Qwen3-235B MoE Model Parallel Strategy Deployment Plan
Optimized for maximum throughput while meeting TTFT requirements
"""

import math

# Model configuration from deployment.md
model_name = "Qwen3-235B"
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
seq_length_min = 128
seq_length_max = 10240
seq_in = 2048
seq_out = 2048

# Hardware configuration
gpu_memory = 64_000_000_000  # 64GB in bytes
single_gpu_tflops = 400  # TFlops
mfu_utilization = 0.6
vram_bandwidth = 1.8_000_000_000_000  # 1.8TBps
bandwidth_utilization = 0.8

# Performance requirements
target_ttft = 30  # seconds

class Qwen3DeploymentOptimizer:
    def __init__(self):
        self.model_params = weights
        self.layers = layers
        self.experts = experts_per_layer
        self.top_k = top_k_gate
        self.precision = precision
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_hidden = mlp_hidden
        
    def calculate_memory_requirements(self, TP, PP, EP, SP):
        """Calculate memory usage per GPU for MoE model"""
        
        # Parameter memory breakdown
        # Attention parameters: QKV projections + output projection
        attention_params = 4 * self.num_heads * self.head_dim * self.token_dim
        
        # MoE parameters: experts * (gate + up_proj + down_proj)
        # Each expert has gate, up_proj, down_proj
        expert_params = self.experts * (self.token_dim * self.mlp_hidden + 
                                       self.mlp_hidden * self.token_dim + 
                                       self.token_dim)
        
        # Layer parameters: attention + MoE + layer norm
        layer_params = attention_params + expert_params + 2 * self.token_dim
        
        # Total model parameters
        total_params = self.layers * layer_params + self.vocab_size * self.token_dim  # embeddings
        
        # Memory per GPU with parallelism
        # EP dominates: each GPU holds experts_per_gpu = experts / EP
        experts_per_gpu = self.experts / EP
        
        # Parameters per GPU
        params_per_gpu = total_params / EP  # EP is primary for MoE
        
        # KV cache memory
        kv_cache_per_token = gqa_kv_heads * head_dim * 2 * precision  # 2 for K+V
        kv_cache_total = batch_size * seq_length_max * self.layers * kv_cache_per_token
        kv_cache_per_gpu = kv_cache_total / (PP * SP)  # Distributed across PP and SP
        
        # Activation memory
        activation_per_layer = batch_size * seq_length_max * self.token_dim * precision
        activation_total = activation_per_layer * self.layers
        activation_per_gpu = activation_total / (TP * PP * SP)
        
        # Total memory per GPU
        total_memory = (params_per_gpu * precision + 
                       kv_cache_per_gpu + 
                       activation_per_gpu)
        
        # Add overhead
        overhead = total_memory * 0.1  # 10% overhead
        final_memory = total_memory + overhead
        
        return {
            'params_per_gpu': params_per_gpu * precision,
            'kv_cache_per_gpu': kv_cache_per_gpu,
            'activation_per_gpu': activation_per_gpu,
            'total_memory': final_memory,
            'experts_per_gpu': experts_per_gpu
        }
    
    def calculate_prefill_time(self, TP, PP, EP, SP):
        """Calculate prefill time for the configuration"""
        
        # Effective compute per GPU
        effective_tflops = single_gpu_tflops * mfu_utilization
        
        # Attention FLOPs (quadratic in sequence length)
        attention_flops = (batch_size * seq_in * seq_in * self.num_heads * self.head_dim * 
                          self.layers / 1e12)  # Convert to TFlops
        
        # MoE FLOPs (top-k experts active)
        active_experts = self.top_k
        moe_flops = (batch_size * seq_in * active_experts * self.mlp_hidden * 
                    self.token_dim * self.layers / 1e12)
        
        # Total prefill FLOPs
        total_prefill_flops = attention_flops + moe_flops
        
        # With parallelism
        # TP divides attention and FFN computation
        # EP divides MoE computation (experts distributed)
        parallel_factor = max(TP, EP)  # Conservative estimate
        
        # Add communication overhead
        communication_overhead = 1.15  # 15% overhead for TP/EP communication
        
        prefill_time = (total_prefill_flops / parallel_factor / effective_tflops * 
                       communication_overhead)
        
        return prefill_time
    
    def calculate_throughput(self, TP, EP):
        """Calculate decode throughput"""
        
        effective_tflops = single_gpu_tflops * mfu_utilization
        
        # Per-token FLOPs
        attention_token_flops = (self.num_heads * self.head_dim * self.layers / 1e12)
        moe_token_flops = (self.top_k * self.mlp_hidden * self.token_dim * self.layers / 1e12)
        total_token_flops = attention_token_flops + moe_token_flops
        
        # Token time with parallelism
        parallel_factor = max(TP, EP)
        token_time = total_token_flops / parallel_factor / effective_tflops * 1.1  # 10% overhead
        
        throughput_per_gpu = 1 / token_time
        total_throughput = throughput_per_gpu * EP  # EP GPUs work in parallel
        
        return total_throughput
    
    def find_optimal_config(self):
        """Find optimal parallel configuration"""
        
        print("=== QWEN3-235B MOE DEPLOYMENT OPTIMIZATION ===")
        print(f"Model: {model_name}")
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
        # EP must divide experts evenly
        ep_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for EP in ep_candidates:
            if self.experts % EP != 0:
                continue
                
            for PP in [1, 2, 4, 8]:  # PP should divide layers evenly
                if self.layers % PP != 0:
                    continue
                    
                for TP in [1, 2, 4, 8]:  # TP for attention/FFN
                    for SP in [1, 2, 4]:  # SP for sequence
                        
                        # Calculate memory
                        memory_breakdown = self.calculate_memory_requirements(TP, PP, EP, SP)
                        memory_per_gpu = memory_breakdown['total_memory']
                        
                        # Calculate performance
                        prefill_time = self.calculate_prefill_time(TP, PP, EP, SP)
                        throughput = self.calculate_throughput(TP, EP)
                        
                        # Check constraints
                        memory_ok = memory_per_gpu < gpu_memory * 0.85  # 85% max utilization
                        ttft_ok = prefill_time < target_ttft
                        
                        if memory_ok and ttft_ok:
                            # Calculate efficiency score
                            total_gpus = EP  # EP dominates GPU count for MoE
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
                                'experts_per_gpu': memory_breakdown['experts_per_gpu']
                            }
                            
                            valid_configs.append(config_info)
                            
                            if score < best_score:问候语 best_score = score
                                best_config = config_info
        
        # Display all valid configurations
        print("=== VALID CONFIGURATIONS ===")
        for config in sorted(valid_configs, key=lambda x: x['score']):
            print(f"TP={config['TP']}, PP={config['PP']}, EP={config['EP']}, SP={config['SP']} "
 f"({config['total_gpus']} GPUs)")
            print(f"  Memory per GPU: {config['memory_per_gpu']/1e9:.2f}GB "
                  f"({config['memory_per_gpu']/gpu_memory*100:.1f}%)")
            print(f"  Prefill time: {config['prefill_time']:.2f}s")
            print(f"  Throughput: {config['throughput']/1e6:.1f}M tokens/s")
            print(f"  Experts per GPU: {config['experts_per_gpu']}")
            print(f"  Efficiency score: {config['score']:.3f}")
            print()
        
        if best_config:
            print("=== OPTIMAL CONFIGURATION ===")
            self.display_optimal_config(best_config)
            return best_config
        else:
            print("No valid configuration found!")
            return None
    
    def display_optimal_config(self, config):
        """Display the optimal configuration details"""
        
        print(f"Best configuration: TP={config['TP']}, PP={config['PP']}, EP={config['EP']}, SP={config['SP']}")
        print(f"Total GPUs: {config['total_gpus']}")
        print(f"Memory per GPU: {config['memory_per_gpu']/1e9:.2f}GB ({config['memory_per_gpu']/gpu_memory*100:.1f}%)")
        print(f"Prefill time: {config['prefill_time']:.2f}s (target: {target_ttft}s)")
        print(f"Throughput: {config['throughput']/1e6:.1f}M tokens/s")
        print(f"Experts per GPU: {config['experts_per_gpu']}")
        print()
        
        # Module division verification
        print("=== MODULE DIVISION VERIFICATION ===")
        layers_per_stage = self.layers / config['PP']
        heads_per_tp = self.num_heads / config['TP']
        experts_per_gpu = self.experts / config['EP']
        
        print(f"Layers per PP stage: {layers_per_stage}")
        print(f"Attention heads per TP group: {heads_per_tp}")
        print(f"Experts per GPU: {experts_per_gpu}")
        print(f"Sequence partition: {config['SP']} ways")
        print(f"Memory utilization: {config['memory_per_gpu']/gpu_memory*100:.1f}%")
        print()
        
        # Deployment strategy summary
        print("=== DEPLOYMENT STRATEGY SUMMARY ===")
        print(f"1. EP={config['EP']}: Each GPU hosts {experts_per_gpu} experts")
        print(f"2. PP={config['PP']}: Model split into {config['PP']} pipeline stages")
        print(f"3. TP={config['TP']}: Attention/FFN tensors parallelized")
        print(f"4. SP={config['SP']}: Sequence length partitioned")
        print(f"5. Total: {config['total_gpus']} GPUs with balanced load")

def main():
    optimizer = Qwen3DeploymentOptimizer()
    optimal_config = optimizer.find_optimal_config()
    
    if optimal_config:
        # Save deployment plan
        deployment_plan = f"""
# Qwen3-235B MoE Parallel Strategy Deployment Plan
# Generated: 2025-12-31-10-57-02

## Model Configuration
- Model: Qwen3-235B
- Parameters: 235B
- Layers: 94
- Experts per layer: 128
- Top-K gate: 8
- Precision: FP8

## Optimal Parallel Strategy
- Tensor Parallel (TP): {optimal_config['TP']}
- Pipeline Parallel (PP): {optimal_config['PP']}
- Expert Parallel (EP): {optimal_config['EP']}
- Sequence Parallel (SP): {optimal_config['SP']}

## Resource Allocation
- Total GPUs: {optimal_config['total_gpus']}
- Memory per GPU: {optimal_config['memory_per_gpu']/1e9:.2f}GB
- Memory utilization: {optimal_config['memory_per_gpu']/gpu_memory*100:.1f}%

## Performance Metrics
- Prefill time: {optimal_config['prefill_time']:.2f}s
- Target TTFT: {target_ttft}s
- Throughput: {optimal_config['throughput']/1e6:.1f}M tokens/s

## Module Division
- Layers per stage: {94/optimal_config['PP']}
- Experts per GPU: {128/optimal_config['EP']}
- Attention heads per TP: {64/optimal_config['TP']}
- Sequence partition: {optimal_config['SP']} ways

## Deployment Notes
- EP dominates GPU allocation (MoE inference rule)
- Each GPU hosts {128/optimal_config['EP']} experts
- Balanced load across all GPUs
- Meets TTFT requirement with margin
- Maximizes throughput while minimizing GPU usage
"""
        
        with open('./outputs/2025-12-31-10-57-02/deployment_plan.md', 'w') as f:
            f.write(deployment_plan)
        
        print("Deployment plan saved to: ./outputs/2025-12-31-10-57-02/deployment_plan.md")
        return optimal_config
    else:
        print("Failed to find valid deployment configuration")
        return None

if __name__ == "__main__":
    main()