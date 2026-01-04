#!/usr/bin/env python3
"""
Calculate optimal parallel strategy for Qwen3-235B model deployment
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
    
    # Attention compute (per token)
    # QKV projection: 3 * seq_len * token_dim^2
    # Attention computation: seq_len^2 * token_dim
    # Output projection: seq_len * token_dim^2
    
    # MoE compute (per token, per expert)
    # Gate: token_dim * num_experts
    # Expert FFN: 2 * token_dim * hidden_size
    
    # For simplicity, let's estimate based on model size and typical inference patterns
    # Rough estimate: ~2 FLOPs per parameter for forward pass
    
    flops_per_token = MODEL_WEIGHTS * 2  # Rough estimate
    
    return flops_per_token

def find_optimal_strategy():
    """Find optimal parallel strategy"""
    
    memory_req = calculate_memory_requirements()
    compute_req = calculate_compute_requirements()
    
    print("=== Memory Requirements ===")
    print(f"Total model memory: {memory_req['model_total'] / 1e9:.1f} GB")
    print(f"Per layer memory: {memory_req['layer_total'] / 1e9:.1f} GB")
    print(f"Embedding memory: {memory_req['embedding'] / 1e9:.1f} GB")
    
    print("\n=== Compute Requirements ===")
    print(f"Compute per token: {compute_req / 1e9:.1f} GFLOPs")
    
    # Based on the knowledge file, for MoE inference:
    # EP ≈ GPU_total (one GPU per expert)
    # But we have 128 experts per layer, which would require too many GPUs
    # We need to find a balance
    
    print("\n=== Parallel Strategy Analysis ===")
    
    # Strategy 1: Expert Parallel dominant
    # We need to fit experts within GPU memory
    expert_memory = (HIDDEN_SIZE * TOKEN_DIM * 2 * PRECISION)  # Rough expert size
    print(f"Memory per expert: {expert_memory / 1e9:.1f} GB")
    
    # How many experts can fit in one GPU?
    experts_per_gpu = int(SINGLE_GPU_VRAM * 0.8 / expert_memory)  # 80% memory utilization
    print(f"Experts per GPU (memory limited): {experts_per_gpu}")
    
    # Total GPUs needed for experts
    gpus_for_experts = math.ceil(EXPERTS_PER_LAYER / experts_per_gpu)
    print(f"GPUs needed for expert parallelism: {gpus_for_experts}")
    
    # Strategy 2: Pipeline Parallel for layers
    # How many layers per GPU?
    layer_memory = memory_req['layer_total']
    layers_per_gpu = int(SINGLE_GPU_VRAM * 0.6 / layer_memory)  # 60% memory for layers
    print(f"Layers per GPU (memory limited): {layers_per_gpu}")
    
    pp_degree = math.ceil(LAYERS / layers_per_gpu)
    print(f"Pipeline parallel degree: {pp_degree}")
    
    # Strategy 3: Tensor Parallel for attention
    # For attention, we can parallelize across heads
    heads_per_gpu = 16  # Reasonable for good load balancing
    tp_degree = math.ceil(HEADS / heads_per_gpu)
    print(f"Tensor parallel degree: {tp_degree}")
    
    # Total GPUs (according to knowledge file rules)
    # EP is dominant, PP is structural, TP is operator-level
    # Total GPUs = max(EP, PP) * TP (if TP and EP operate in same phase)
    
    total_gpus = max(gpus_for_experts, pp_degree) * tp_degree
    print(f"Total GPUs estimated: {total_gpus}")
    
    # Check TTFT constraint
    # Rough calculation: time = compute / (total_gpus * single_gpu_flops * MFU)
    tokens_to_generate = SEQ_IN + SEQ_OUT
    total_compute = compute_req * tokens_to_generate * BATCH_SIZE
    available_compute = total_gpus * SINGLE_GPU_FLOPS * MFU
    estimated_time = total_compute / available_compute
    
    print(f"\n=== Performance Check ===")
    print(f"Estimated time for first token: {estimated_time:.1f} seconds")
    print(f"TTFT requirement: {MAX_TTFT} seconds")
    print(f"Meets TTFT: {'YES' if estimated_time <= MAX_TTFT else 'NO'}")
    
    return {
        'ep_degree': gpus_for_experts,
        'pp_degree': pp_degree,
        'tp_degree': tp_degree,
        'total_gpus': total_gpus,
        'meets_ttft': estimated_time <= MAX_TTFT,
        'estimated_time': estimated_time
    }

if __name__ == "__main__":
    strategy = find_optimal_strategy()
    
    print(f"\n=== Recommended Strategy ===")
    print(f"Expert Parallel (EP): {strategy['ep_degree']}")
    print(f"Pipeline Parallel (PP): {strategy['pp_degree']}")
    print(f"Tensor Parallel (TP): {strategy['tp_degree']}")
    print(f"Total GPUs: {strategy['total_gpus']}")
    print(f"Meets TTFT: {strategy['meets_ttft']}")