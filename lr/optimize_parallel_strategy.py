#!/usr/bin/env python3
"""
Optimize parallel strategy to meet TTFT requirement while minimizing GPU usage
"""

import math

# Model parameters
MODEL_WEIGHTS = 235e9  # 235B parameters
LAYERS = 94
EXPERTS_PER_LAYER = 128
TOKEN_DIM = 4096
HIDDEN_SIZE = 1536
VOCAB_SIZE = 151936
HEADS = 64
HEAD_DIM = 64
GQA_KV_HEADS = 4
PRECISION = 1  # FP8 = 1 byte per parameter

# Hardware parameters
SINGLE_GPU_FLOPS = 400e12  # 400TFlops
SINGLE_GPU_VRAM = 64e9  # 64GB
BANDWIDTH = 1.8e12  # 1.8TBps
MFU = 0.6  # 60% utilization
BANDWIDTH_UTIL = 0.8  # 80% bandwidth utilization

# Input requirements
BATCH_SIZE = 128
SEQ_LEN_RANGE = [128, 10240]
SEQ_IN = 2048
SEQ_OUT = 2048
MAX_TTFT = 30  # 30 seconds

def calculate_memory_requirements():
    """Calculate memory requirements for different components"""
    
    # Embedding weights
    embedding_weights = VOCAB_SIZE * TOKEN_DIM * PRECISION
    
    # Layer weights (per layer)
    # Attention weights
    qkv_weights = 3 * TOKEN_DIM * TOKEN_DIM * PRECISION  # Q, K, V
    o_weights = TOKEN_DIM * TOKEN_DIM * PRECISION  # Output projection
    
    # MoE weights (per expert)
    gate_weights = TOKEN_DIM * EXPERTS_PER_LAYER * PRECISION
    ffn1_weights = TOKEN_DIM * HIDDEN_SIZE * PRECISION
    ffn2_weights = HIDDEN_SIZE * TOKEN_DIM * PRECISION
    
    # Total per layer
    attention_per_layer = qkv_weights + o_weights
    moe_per_expert = ffn1_weights + ffn2_weights
    moe_per_layer = gate_weights + (moe_per_expert * EXPERTS_PER_LAYER)
    
    layer_total = attention_per_layer + moe_per_layer
    model_total = embedding_weights + (layer_total * LAYERS)
    
    return {
        'embedding': embedding_weights,
        'attention_per_layer': attention_per_layer,
        'moe_per_layer': moe_per_layer,
        'layer_total': layer_total,
        'model_total': model_total,
        'per_layer': layer_total
    }

def calculate_compute_requirements():
    """Calculate compute requirements for inference"""
    # Rough estimate: ~2 FLOPs per parameter for forward pass
    flops_per_token = MODEL_WEIGHTS * 2
    return flops_per_token

def find_min_gpus_for_ttft():
    """Find minimum GPUs needed to meet TTFT requirement"""
    
    memory_req = calculate_memory_requirements()
    compute_req = calculate_compute_requirements()
    
    print("=== Memory Requirements ===")
    print(f"Total model memory: {memory_req['model_total'] / 1e9:.1f} GB")
    print(f"Per layer memory: {memory_req['layer_total'] / 1e9:.1f} GB")
    
    # Calculate minimum GPUs needed for compute
    tokens_to_generate = SEQ_IN + SEQ_OUT
    total_compute = compute_req * tokens_to_generate * BATCH_SIZE
    
    # We need: total_compute / (total_gpus * single_gpu_flops * MFU) <= MAX_TTFT
    min_gpus_compute = total_compute / (MAX_TTFT * SINGLE_GPU_FLOPS * MFU)
    
    print(f"\n=== Compute Requirements ===")
    print(f"Total compute needed: {total_compute / 1e12:.1f} TFLOPs")
    print(f"Min GPUs for compute: {math.ceil(min_gpus_compute)}")
    
    # Now let's find a practical strategy
    # Start with minimum and increase until we meet constraints
    
    for total_gpus in range(int(min_gpus_compute), 100):  # reasonable upper bound
        
        # Try different combinations of EP, PP, TP
        # According to knowledge: EP ≈ GPU_total, but we can be more efficient
        
        best_strategy = None
        best_time = float('inf')
        
        for ep in range(1, min(EXPERTS_PER_LAYER, total_gpus) + 1):
            for pp in range(1, min(LAYERS, total_gpus) + 1):
                for tp in range(1, min(HEADS, total_gpus) + 1):
                    
                    # Check if this combination fits within total_gpus
                    # According to knowledge: Total GPUs = structural mapping, not simple multiplication
                    # For MoE: EP is dominant, PP is structural, TP is operator-level
                    
                    estimated_gpus = max(ep, pp) * tp  # Conservative estimate
                    
                    if estimated_gpus <= total_gpus:                        
                        # Calculate performance
                        available_compute = total_gpus * SINGLE_GPU_FLOPS * MFU
                        estimated_time = total_compute / available_compute
                        
                        if estimated_time <= MAX_TTFT and estimated_time < best_time:
                            best_time = estimated_time
                            best_strategy = {
                                'ep': ep,
                                'pp': pp,
                                'tp': tp,
                                'total_gpus': total_gpus,
                                'estimated_time': estimated_time
                            }
        
        if best_strategy:
            print(f"\n=== Optimal Strategy Found ===")
            print(f"Expert Parallel (EP): {best_strategy['ep']}")
            print(f"Pipeline Parallel (PP): {best_strategy['pp']}")
            print(f"Tensor Parallel (TP): {best_strategy['tp']}")
            print(f"Total GPUs: {best_strategy['total_gpus']}")
            print(f"Estimated TTFT: {best_strategy['estimated_time']:.1f} seconds")
            
            # Verify memory constraints
            layer_memory = memory_req['layer_total']
            layers_per_gpu = math.ceil(LAYERS / best_strategy['pp'])
            memory_per_gpu = layers_per_gpu * layer_memory
            
            print(f"Memory per GPU: {memory_per_gpu / 1e9:.1f} GB")
            print(f"GPU capacity: {SINGLE_GPU_VRAM / 1e9:.1f} GB")
            print(f"Memory OK: {'YES' if memory_per_gpu <= SINGLE_GPU_VRAM * 0.8 else 'NO'}")
            
            return best_strategy
    
    return None

if __name__ == "__main__":
    strategy = find_min_gpus_for_ttft()
    
    if strategy:
        print(f"\n=== Final Recommendation ===")
        print(f"Expert Parallel (EP): {strategy['ep']}")
        print(f"Pipeline Parallel (PP): {strategy['pp']}")
        print(f"Tensor Parallel (TP): {strategy['tp']}")
        print(f"Total GPUs: {strategy['total_gpus']}")
        print(f"Estimated TTFT: {strategy['estimated_time']:.1f} seconds")
    else:
        print("No valid strategy found!")