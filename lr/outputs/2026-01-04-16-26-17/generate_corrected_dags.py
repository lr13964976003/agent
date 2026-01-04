#!/usr/bin/env python3

import os

def generate_prefill_dag():
    """Generate corrected prefill phase DAG with detailed attention decomposition and all layers"""
    
    dot_content = """// Prefill Phase DAG - Qwen3-235B (Corrected)
digraph {
    rankdir=TB size="30,40"
    node [shape=rectangle style=filled]
    
    // Input node
    input [label="Input Embedding\\nGPU: 0-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgray shape=ellipse]
    
    // Stage 0: Layers 0-23 (GPUs 0-8)
    subgraph cluster_stage0 {
        fillcolor=lightblue label="Stage 0: Layers 0-23\\nGPUs 0-8" style=rounded
        
        // Layer 0
        layer0_qkv_proj [label="Layer 0 QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer0_attn_comp [label="Layer 0 Attention Computation\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer0_out_proj [label="Layer 0 Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
        layer0_gate [label="Layer 0 Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer0_moe [label="Layer 0 MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
        """
    
    # Generate intermediate layers 1-22
    for i in range(1, 23):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer{i}_attn_comp [label="Layer {i} Attention Computation\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
        """
    
    dot_content += """
        // Layer 23
        layer23_qkv_proj [label="Layer 23 QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer23_attn_comp [label="Layer 23 Attention Computation\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightblue]
        layer23_out_proj [label="Layer 23 Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
        layer23_gate [label="Layer 23 Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer23_moe [label="Layer 23 MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightblue]
    }
    
    // Communication between stages
    comm_0_1 [label="Pipeline Communication\\nGPUs: 0-8 → 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 0
    input -> layer0_qkv_proj
    layer0_qkv_proj -> layer0_attn_comp
    layer0_attn_comp -> layer0_out_proj
    layer0_out_proj -> layer0_gate
    layer0_gate -> layer0_moe [style=dashed]
    """
    
    # Add connections for intermediate layers
    for i in range(1, 23):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer22_moe -> layer23_qkv_proj
    layer23_qkv_proj -> layer23_attn_comp
    layer23_attn_comp -> layer23_out_proj
    layer23_out_proj -> layer23_gate
    layer23_gate -> layer23_moe [style=dashed]
    layer23_moe -> comm_0_1
    
    // Stage 1: Layers 24-47 (GPUs 9-17)
    subgraph cluster_stage1 {
        fillcolor=lightgreen label="Stage 1: Layers 24-47\\nGPUs 9-17" style=rounded
        
        // Layer 24
        layer24_qkv_proj [label="Layer 24 QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer24_attn_comp [label="Layer 24 Attention Computation\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer24_out_proj [label="Layer 24 Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
        layer24_gate [label="Layer 24 Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer24_moe [label="Layer 24 MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
        """
    
    # Generate intermediate layers 25-46
    for i in range(25, 47):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer{i}_attn_comp [label="Layer {i} Attention Computation\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
        """
    
    dot_content += """
        // Layer 47
        layer47_qkv_proj [label="Layer 47 QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer47_attn_comp [label="Layer 47 Attention Computation\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightgreen]
        layer47_out_proj [label="Layer 47 Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
        layer47_gate [label="Layer 47 Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer47_moe [label="Layer 47 MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightgreen]
    }
    
    // Communication between stages
    comm_1_2 [label="Pipeline Communication\\nGPUs: 9-17 → 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 1
    comm_0_1 -> layer24_qkv_proj
    layer24_qkv_proj -> layer24_attn_comp
    layer24_attn_comp -> layer24_out_proj
    layer24_out_proj -> layer24_gate
    layer24_gate -> layer24_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 25-46
    for i in range(25, 47):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer46_moe -> layer47_qkv_proj
    layer47_qkv_proj -> layer47_attn_comp
    layer47_attn_comp -> layer47_out_proj
    layer47_out_proj -> layer47_gate
    layer47_gate -> layer47_moe [style=dashed]
    layer47_moe -> comm_1_2
    
    // Stage 2: Layers 48-71 (GPUs 18-26)
    subgraph cluster_stage2 {
        fillcolor=lightyellow label="Stage 2: Layers 48-71\\nGPUs 18-26" style=rounded
        
        // Layer 48
        layer48_qkv_proj [label="Layer 48 QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer48_attn_comp [label="Layer 48 Attention Computation\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer48_out_proj [label="Layer 48 Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
        layer48_gate [label="Layer 48 Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer48_moe [label="Layer 48 MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
        """
    
    # Generate intermediate layers 49-70
    for i in range(49, 71):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer{i}_attn_comp [label="Layer {i} Attention Computation\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
        """
    
    dot_content += """
        // Layer 71
        layer71_qkv_proj [label="Layer 71 QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer71_attn_comp [label="Layer 71 Attention Computation\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightyellow]
        layer71_out_proj [label="Layer 71 Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
        layer71_gate [label="Layer 71 Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer71_moe [label="Layer 71 MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightyellow]
    }
    
    // Communication between stages
    comm_2_3 [label="Pipeline Communication\\nGPUs: 18-26 → 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 2
    comm_1_2 -> layer48_qkv_proj
    layer48_qkv_proj -> layer48_attn_comp
    layer48_attn_comp -> layer48_out_proj
    layer48_out_proj -> layer48_gate
    layer48_gate -> layer48_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 49-70
    for i in range(49, 71):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer70_moe -> layer71_qkv_proj
    layer71_qkv_proj -> layer71_attn_comp
    layer71_attn_comp -> layer71_out_proj
    layer71_out_proj -> layer71_gate
    layer71_gate -> layer71_moe [style=dashed]
    layer71_moe -> comm_2_3
    
    // Stage 3: Layers 72-93 (GPUs 27-34)
    subgraph cluster_stage3 {
        fillcolor=lightcoral label="Stage 3: Layers 72-93\\nGPUs 27-34" style=rounded
        
        // Layer 72
        layer72_qkv_proj [label="Layer 72 QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer72_attn_comp [label="Layer 72 Attention Computation\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer72_out_proj [label="Layer 72 Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
        layer72_gate [label="Layer 72 Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer72_moe [label="Layer 72 MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
        """
    
    # Generate intermediate layers 73-92
    for i in range(73, 93):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer{i}_attn_comp [label="Layer {i} Attention Computation\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
        """
    
    dot_content += """
        // Layer 93
        layer93_qkv_proj [label="Layer 93 QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer93_attn_comp [label="Layer 93 Attention Computation\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]" fillcolor=lightcoral]
        layer93_out_proj [label="Layer 93 Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
        layer93_gate [label="Layer 93 Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, top_k=8]" fillcolor=orange shape=parallelogram]
        layer93_moe [label="Layer 93 MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, d_model=4096]" fillcolor=lightcoral]
    }
    
    // Output node
    output [label="Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=2048, d_model=4096]\\nOutput: [batch_size=128, seq_len=2048, vocab_size=151936]" fillcolor=lightgray shape=ellipse]
    
    // Connections for Stage 3
    comm_2_3 -> layer72_qkv_proj
    layer72_qkv_proj -> layer72_attn_comp
    layer72_attn_comp -> layer72_out_proj
    layer72_out_proj -> layer72_gate
    layer72_gate -> layer72_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 73-92
    for i in range(73, 93):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer92_moe -> layer93_qkv_proj
    layer93_qkv_proj -> layer93_attn_comp
    layer93_attn_comp -> layer93_out_proj
    layer93_out_proj -> layer93_gate
    layer93_gate -> layer93_moe [style=dashed]
    layer93_moe -> output
}
    """
    
    return dot_content

def generate_decode_dag():
    """Generate corrected decode phase DAG with detailed attention decomposition and all layers"""
    
    dot_content = """// Decode Phase DAG - Qwen3-235B (Corrected)
digraph {
    rankdir=TB size="30,40"
    node [shape=rectangle style=filled]
    
    // Input node
    input [label="Input Token Embedding\\nGPU: 0-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgray shape=ellipse]
    
    // KV Cache node
    kv_cache [label="KV Cache Storage\\nGPU: 0-34\\nCache: [batch_size=128, seq_len=2048, heads=4, d_k=64]\\nHit Rate: 95%+" fillcolor=lightpink shape=parallelogram]
    
    // Stage 0: Layers 0-23 (GPUs 0-8)
    subgraph cluster_stage0 {
        fillcolor=lightblue label="Stage 0: Layers 0-23\\nGPUs 0-8" style=rounded
        
        // Layer 0
        layer0_qkv_proj [label="Layer 0 QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer0_attn_comp [label="Layer 0 Attention Computation + KV Cache\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer0_out_proj [label="Layer 0 Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
        layer0_gate [label="Layer 0 Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer0_moe [label="Layer 0 MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
        """
    
    # Generate intermediate layers 1-22
    for i in range(1, 23):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer{i}_attn_comp [label="Layer {i} Attention Computation + KV Cache\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
        """
    
    dot_content += """
        // Layer 23
        layer23_qkv_proj [label="Layer 23 QKV Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer23_attn_comp [label="Layer 23 Attention Computation + KV Cache\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightblue]
        layer23_out_proj [label="Layer 23 Output Projection\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
        layer23_gate [label="Layer 23 Gate Selection (Top-8)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer23_moe [label="Layer 23 MoE (128 experts)\\nGPU: 0-8\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightblue]
    }
    
    // KV Cache access nodes
    kv_access_0 [label="KV Cache Access\\nGPU: 0-8\\nCache: [batch_size=128, seq_len=2048, heads=4, d_k=64]" fillcolor=lightcyan shape=parallelogram]
    
    // Communication between stages
    comm_0_1 [label="Pipeline Communication (Single Token)\\nGPUs: 0-8 → 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 0
    input -> kv_cache
    kv_cache -> kv_access_0
    kv_access_0 -> layer0_qkv_proj
    layer0_qkv_proj -> layer0_attn_comp
    layer0_attn_comp -> layer0_out_proj
    layer0_out_proj -> layer0_gate
    layer0_gate -> layer0_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 1-22
    for i in range(1, 23):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer22_moe -> layer23_qkv_proj
    layer23_qkv_proj -> layer23_attn_comp
    layer23_attn_comp -> layer23_out_proj
    layer23_out_proj -> layer23_gate
    layer23_gate -> layer23_moe [style=dashed]
    layer23_moe -> comm_0_1
    
    // Stage 1: Layers 24-47 (GPUs 9-17)
    subgraph cluster_stage1 {
        fillcolor=lightgreen label="Stage 1: Layers 24-47\\nGPUs 9-17" style=rounded
        
        // Layer 24
        layer24_qkv_proj [label="Layer 24 QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer24_attn_comp [label="Layer 24 Attention Computation + KV Cache\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer24_out_proj [label="Layer 24 Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
        layer24_gate [label="Layer 24 Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer24_moe [label="Layer 24 MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
        """
    
    # Generate intermediate layers 25-46
    for i in range(25, 47):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer{i}_attn_comp [label="Layer {i} Attention Computation + KV Cache\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
        """
    
    dot_content += """
        // Layer 47
        layer47_qkv_proj [label="Layer 47 QKV Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer47_attn_comp [label="Layer 47 Attention Computation + KV Cache\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightgreen]
        layer47_out_proj [label="Layer 47 Output Projection\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
        layer47_gate [label="Layer 47 Gate Selection (Top-8)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer47_moe [label="Layer 47 MoE (128 experts)\\nGPU: 9-17\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightgreen]
    }
    
    // KV Cache access nodes
    kv_access_1 [label="KV Cache Access\\nGPU: 9-17\\nCache: [batch_size=128, seq_len=2048, heads=4, d_k=64]" fillcolor=lightcyan shape=parallelogram]
    
    // Communication between stages
    comm_1_2 [label="Pipeline Communication (Single Token)\\nGPUs: 9-17 → 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 1
    comm_0_1 -> kv_access_1
    kv_access_1 -> layer24_qkv_proj
    layer24_qkv_proj -> layer24_attn_comp
    layer24_attn_comp -> layer24_out_proj
    layer24_out_proj -> layer24_gate
    layer24_gate -> layer24_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 25-46
    for i in range(25, 47):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer46_moe -> layer47_qkv_proj
    layer47_qkv_proj -> layer47_attn_comp
    layer47_attn_comp -> layer47_out_proj
    layer47_out_proj -> layer47_gate
    layer47_gate -> layer47_moe [style=dashed]
    layer47_moe -> comm_1_2
    
    // Stage 2: Layers 48-71 (GPUs 18-26)
    subgraph cluster_stage2 {
        fillcolor=lightyellow label="Stage 2: Layers 48-71\\nGPUs 18-26" style=rounded
        
        // Layer 48
        layer48_qkv_proj [label="Layer 48 QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer48_attn_comp [label="Layer 48 Attention Computation + KV Cache\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer48_out_proj [label="Layer 48 Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
        layer48_gate [label="Layer 48 Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer48_moe [label="Layer 48 MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
        """
    
    # Generate intermediate layers 49-70
    for i in range(49, 71):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer{i}_attn_comp [label="Layer {i} Attention Computation + KV Cache\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
        """
    
    dot_content += """
        // Layer 71
        layer71_qkv_proj [label="Layer 71 QKV Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer71_attn_comp [label="Layer 71 Attention Computation + KV Cache\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightyellow]
        layer71_out_proj [label="Layer 71 Output Projection\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
        layer71_gate [label="Layer 71 Gate Selection (Top-8)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer71_moe [label="Layer 71 MoE (128 experts)\\nGPU: 18-26\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightyellow]
    }
    
    // KV Cache access nodes
    kv_access_2 [label="KV Cache Access\\nGPU: 18-26\\nCache: [batch_size=128, seq_len=2048, heads=4, d_k=64]" fillcolor=lightcyan shape=parallelogram]
    
    // Communication between stages
    comm_2_3 [label="Pipeline Communication (Single Token)\\nGPUs: 18-26 → 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=pink shape=ellipse style=dashed]
    
    // Connections for Stage 2
    comm_1_2 -> kv_access_2
    kv_access_2 -> layer48_qkv_proj
    layer48_qkv_proj -> layer48_attn_comp
    layer48_attn_comp -> layer48_out_proj
    layer48_out_proj -> layer48_gate
    layer48_gate -> layer48_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 49-70
    for i in range(49, 71):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer70_moe -> layer71_qkv_proj
    layer71_qkv_proj -> layer71_attn_comp
    layer71_attn_comp -> layer71_out_proj
    layer71_out_proj -> layer71_gate
    layer71_gate -> layer71_moe [style=dashed]
    layer71_moe -> comm_2_3
    
    // Stage 3: Layers 72-93 (GPUs 27-34)
    subgraph cluster_stage3 {
        fillcolor=lightcoral label="Stage 3: Layers 72-93\\nGPUs 27-34" style=rounded
        
        // Layer 72
        layer72_qkv_proj [label="Layer 72 QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer72_attn_comp [label="Layer 72 Attention Computation + KV Cache\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer72_out_proj [label="Layer 72 Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
        layer72_gate [label="Layer 72 Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer72_moe [label="Layer 72 MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
        """
    
    # Generate intermediate layers 73-92
    for i in range(73, 93):
        dot_content += f"""
        // Layer {i}
        layer{i}_qkv_proj [label="Layer {i} QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer{i}_attn_comp [label="Layer {i} Attention Computation + KV Cache\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer{i}_out_proj [label="Layer {i} Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
        layer{i}_gate [label="Layer {i} Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer{i}_moe [label="Layer {i} MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
        """
    
    dot_content += """
        // Layer 93
        layer93_qkv_proj [label="Layer 93 QKV Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer93_attn_comp [label="Layer 93 Attention Computation + KV Cache\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, heads=64, d_k=64]" fillcolor=lightcoral]
        layer93_out_proj [label="Layer 93 Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
        layer93_gate [label="Layer 93 Gate Selection (Top-8)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, top_k=8]" fillcolor=orange shape=parallelogram]
        layer93_moe [label="Layer 93 MoE (128 experts)\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, d_model=4096]" fillcolor=lightcoral]
    }
    
    // KV Cache access nodes
    kv_access_3 [label="KV Cache Access\\nGPU: 27-34\\nCache: [batch_size=128, seq_len=2048, heads=4, d_k=64]" fillcolor=lightcyan shape=parallelogram]
    
    // Output node
    output [label="Output Projection\\nGPU: 27-34\\nInput: [batch_size=128, seq_len=1, d_model=4096]\\nOutput: [batch_size=128, seq_len=1, vocab_size=151936]" fillcolor=lightgray shape=ellipse]
    
    // Connections for Stage 3
    comm_2_3 -> kv_access_3
    kv_access_3 -> layer72_qkv_proj
    layer72_qkv_proj -> layer72_attn_comp
    layer72_attn_comp -> layer72_out_proj
    layer72_out_proj -> layer72_gate
    layer72_gate -> layer72_moe [style=dashed]
    """
    
    # Add connections for intermediate layers 73-92
    for i in range(73, 93):
        dot_content += f"""
    layer{i-1}_moe -> layer{i}_qkv_proj
    layer{i}_qkv_proj -> layer{i}_attn_comp
    layer{i}_attn_comp -> layer{i}_out_proj
    layer{i}_out_proj -> layer{i}_gate
    layer{i}_gate -> layer{i}_moe [style=dashed]
        """
    
    dot_content += """
    layer92_moe -> layer93_qkv_proj
    layer93_qkv_proj -> layer93_attn_comp
    layer93_attn_comp -> layer93_out_proj
    layer93_out_proj -> layer93_gate
    layer93_gate -> layer93_moe [style=dashed]
    layer93_moe -> output
}
    """
    
    return dot_content

def main():
    # Generate prefill DAG
    prefill_dag = generate_prefill_dag()
    
    # Generate decode DAG  
    decode_dag = generate_decode_dag()
    
    # Write prefill DAG
    with open('./outputs/2026-01-04-16-26-17/prefill_dag_corrected.dot', 'w') as f:
        f.write(prefill_dag)
    
    # Write decode DAG
    with open('./outputs/2026-01-04-16-26-17/decode_dag_corrected.dot', 'w') as f:
        f.write(decode_dag)
    
    print("Generated corrected DAG files:")
    print("- ./outputs/2026-01-04-16-26-17/prefill_dag_corrected.dot")
    print("- ./outputs/2026-01-04-16-26-17/decode_dag_corrected.dot")
    
    # Generate SVG files using graphviz
    try:
        os.system('dot -Tsvg ./outputs/2026-01-04-16-26-17/prefill_dag_corrected.dot -o ./outputs/2026-01-04-16-26-17/prefill_dag_corrected.svg')
        os.system('dot -Tsvg ./outputs/2026-01-04-16-26-17/decode_dag_corrected.dot -o ./outputs/2026-01-04-16-26-17/decode_dag_corrected.svg')
        print("Generated SVG files:")
        print("- ./outputs/2026-01-04-16-26-17/prefill_dag_corrected.svg")
        print("- ./outputs/2026-01-04-16-26-17/decode_dag_corrected.svg")
    except Exception as e:
        print(f"Warning: Could not generate SVG files: {e}")
        print("DOT files are available for manual conversion")

if __name__ == "__main__":
    main()