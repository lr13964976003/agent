#!/usr/bin/env python3
"""
Generate corrected DAGs for FA Pool paper with proper connectivity
"""

import os

def generate_baseline_dag():
    """Generate baseline transformer DAG with complete connectivity"""
    dot_content = '''digraph baseline_transformer_final {
    rankdir=TB;
    compound=true;
    splines=ortho;
    node [shape=rectangle, style=filled, fontname="monospace"];
    
    graph [label="Baseline 4-Layer Dense Transformer (TP=8, PP=2)\nTotal GPUs: 16\nFully Connected Architecture", fontsize=20];
    
    // Input
    input [shape=ellipse, label="Input\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: Host", fillcolor="#E8F4FD"];
    
    // Embedding and Input Processing
    embed_split [shape=parallelogram, label="Split Embedding (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=?, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    embed_0 [label="Embedding GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_0", fillcolor="#90EE90"];
    embed_1 [label="Embedding GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_1", fillcolor="#90EE90"];
    embed_2 [label="Embedding GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_2", fillcolor="#90EE90"];
    embed_3 [label="Embedding GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_3", fillcolor="#90EE90"];
    embed_4 [label="Embedding GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_4", fillcolor="#90EE90"];
    embed_5 [label="Embedding GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_5", fillcolor="#90EE90"];
    embed_6 [label="Embedding GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_6", fillcolor="#90EE90"];
    embed_7 [label="Embedding GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_7", fillcolor="#90EE90"];
    
    embed_gather [shape=parallelogram, label="All-Gather Embedding\nInput: 8×[batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    pos_enc [label="Positional Encoding\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#87CEEB"];
    
    // Layer 0 Components (Stage 0 - GPUs 0-7)
    layernorm_0 [label="LayerNorm 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // Attention components for layer 0
    q_proj_0 [label="Query Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_0 [label="Key Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_0 [label="Value Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_0 [label="Flash Attention 0\nInput: Q,K,V [batch_size=1024, seq_len=?, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_0 [label="Concat Heads\nInput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_output_0 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_0 [label="Residual Add 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 0
    ffn_linear1_0 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_0 [label="GELU Activation\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_0 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_0 [label="FFN Residual Add 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Pipeline communication to stage 1
    pipeline_comm_0 [shape=ellipse, label="Pipeline Communication\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7 → 8-15", fillcolor="#FFD700"];
    
    // Layer 1 Components (Stage 1 - GPUs 8-15)
    layernorm_1 [label="LayerNorm 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#DDA0DD"];
    
    q_proj_1 [label="Query Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 8-15", fillcolor="#FFD700"];
    k_proj_1 [label="Key Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 8-15", fillcolor="#FFD700"];
    v_proj_1 [label="Value Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 8-15", fillcolor="#FFD700"];
    
    flash_attn_1 [label="Flash Attention 1\nInput: Q,K,V [batch_size=1024, seq_len=?, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 8-15", fillcolor="#FFD700"];
    
    attn_concat_1 [label="Concat Heads\nInput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#FFD700"];
    
    attn_output_1 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#FFD700"];
    
    residual_1 [label="Residual Add 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#DDA0DD"];
    
    // FFN components for layer 1
    ffn_linear1_1 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_gelu_1 [label="GELU Activation\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_linear2_1 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_residual_1 [label="FFN Residual Add 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#98FB98"];
    
    // Pipeline communication back to stage 0
    pipeline_comm_1 [shape=ellipse, label="Pipeline Communication\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15 → 0-7", fillcolor="#FFD700"];
    
    // Layer 2 Components (Stage 0 - GPUs 0-7)
    layernorm_2 [label="LayerNorm 2\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_2 [label="Query Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_2 [label="Key Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_2 [label="Value Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_2 [label="Flash Attention 2\nInput: Q,K,V [batch_size=1024, seq_len=?, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_2 [label="Concat Heads\nInput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_output_2 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_2 [label="Residual Add 2\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 2
    ffn_linear1_2 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_2 [label="GELU Activation\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_2 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_2 [label="FFN Residual Add 2\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Pipeline communication to stage 1
    pipeline_comm_2 [shape=ellipse, label="Pipeline Communication\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7 → 8-15", fillcolor="#FFD700"];
    
    // Layer 3 Components (Stage 1 - GPUs 8-15)
    layernorm_3 [label="LayerNorm 3\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#DDA0DD"];
    
    q_proj_3 [label="Query Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 8-15", fillcolor="#FFD700"];
    k_proj_3 [label="Key Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_k=128]\nGPU: 8-15", fillcolor="#FFD700"];
    v_proj_3 [label="Value Projection\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 8-15", fillcolor="#FFD700"];
    
    flash_attn_3 [label="Flash Attention 3\nInput: Q,K,V [batch_size=1024, seq_len=?, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nGPU: 8-15", fillcolor="#FFD700"];
    
    attn_concat_3 [label="Concat Heads\nInput: [batch_size=1024, seq_len=?, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#FFD700"];
    
    attn_output_3 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#FFD700"];
    
    residual_3 [label="Residual Add 3\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#DDA0DD"];
    
    // FFN components for layer 3
    ffn_linear1_3 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_gelu_3 [label="GELU Activation\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_ff=16384]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_linear2_3 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=?, d_ff=16384]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#98FB98"];
    ffn_residual_3 [label="FFN Residual Add 3\nInput: [batch_size=1024, seq_len=?, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#98FB98"];
    
    pipeline_comm_final [shape=ellipse, label="Pipeline Communication Final\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15 → 0-7", fillcolor="#FFD700"];
    
    // Output
    output_split [shape=parallelogram, label="Split Output\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=?, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    output_0 [label="Linear Output GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_0", fillcolor="#FFB6C1"];
    output_1 [label="Linear Output GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_1", fillcolor="#FFB6C1"];
    output_2 [label="Linear Output GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_2", fillcolor="#FFB6C1"];
    output_3 [label="Linear Output GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_3", fillcolor="#FFB6C1"];
    output_4 [label="Linear Output GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_4", fillcolor="#FFB6C1"];
    output_5 [label="Linear Output GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_5", fillcolor="#FFB6C1"];
    output_6 [label="Linear Output GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_6", fillcolor="#FFB6C1"];
    output_7 [label="Linear Output GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_7", fillcolor="#FFB6C1"];
    
    output_concat [shape=parallelogram, label="All-Gather Output\nInput: 8×[batch_size=1024, seq_len=?, vocab=4000]\nOutput: [batch_size=1024, seq_len=?, vocab=32000]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    final_output [shape=ellipse, label="Final Output\nInput: [batch_size=1024, seq_len=?, vocab=32000]\nOutput: [batch_size=1024, seq_len=?, vocab=32000]\nGPU: 0-7", fillcolor="#E8F4FD"];
    
    // Complete directed edges ensuring no disconnected nodes
    input -> embed_split -> {embed_0 embed_1 embed_2 embed_3 embed_4 embed_5 embed_6 embed_7} -> embed_gather -> pos_enc;
    
    pos_enc -> layernorm_0 -> q_proj_0;
    pos_enc -> layernorm_0 -> k_proj_0;
    pos_enc -> layernorm_0 -> v_proj_0;
    
    q_proj_0 -> flash_attn_0;
    k_proj_0 -> flash_attn_0;
    v_proj_0 -> flash_attn_0;
    
    flash_attn_0 -> attn_concat_0 -> attn_output_0 -> residual_0;
    pos_enc -> residual_0 [style=dashed, label="Residual"];
    
    residual_0 -> ffn_linear1_0 -> ffn_gelu_0 -> ffn_linear2_0 -> ffn_residual_0;
    residual_0 -> ffn_residual_0 [style=dashed, label="Residual"];
    
    ffn_residual_0 -> pipeline_comm_0 -> layernorm_1;
    
    layernorm_1 -> q_proj_1;
    layernorm_1 -> k_proj_1;
    layernorm_1 -> v_proj_1;
    
    q_proj_1 -> flash_attn_1;
    k_proj_1 -> flash_attn_1;
    v_proj_1 -> flash_attn_1;
    
    flash_attn_1 -> attn_concat_1 -> attn_output_1 -> residual_1;
    pipeline_comm_0 -> residual_1 [style=dashed, label="Residual"];
    
    residual_1 -> ffn_linear1_1 -> ffn_gelu_1 -> ffn_linear2_1 -> ffn_residual_1;
    residual_1 -> ffn_residual_1 [style=dashed, label="Residual"];
    
    ffn_residual_1 -> pipeline_comm_1 -> layernorm_2;
    
    layernorm_2 -> q_proj_2;
    layernorm_2 -> k_proj_2;
    layernorm_2 -> v_proj_2;
    
    q_proj_2 -> flash_attn_2;
    k_proj_2 -> flash_attn_2;
    v_proj_2 -> flash_attn_2;
    
    flash_attn_2 -> attn_concat_2 -> attn_output_2 -> residual_2;
    pipeline_comm_1 -> residual_2 [style=dashed, label="Residual"];
    
    residual_2 -> ffn_linear1_2 -> ffn_gelu_2 -> ffn_linear2_2 -> ffn_residual_2;
    residual_2 -> ffn_residual_2 [style=dashed, label="Residual"];
    
    ffn_residual_2 -> pipeline_comm_2 -> layernorm_3;
    
    layernorm_3 -> q_proj_3;
    layernorm_3 -> k_proj_3;
    layernorm_3 -> v_proj_3;
    
    q_proj_3 -> flash_attn_3;
    k_proj_3 -> flash_attn_3;
    v_proj_3 -> flash_attn_3;
    
    flash_attn_3 -> attn_concat_3 -> attn_output_3 -> residual_3;
    pipeline_comm_2 -> residual_3 [style=dashed, label="Residual"];
    
    residual_3 -> ffn_linear1_3 -> ffn_gelu_3 -> ffn_linear2_3 -> ffn_residual_3;
    residual_3 -> ffn_residual_3 [style=dashed, label="Residual"];
    
    ffn_residual_3 -> pipeline_comm_final -> output_split -> {output_0 output_1 output_2 output_3 output_4 output_5 output_6 output_7} -> output_concat -> final_output;
}'''
    return dot_content

def generate_fa_pool_short_sequence_dag():
    """Generate FA Pool short sequence DAG with complete connectivity"""
    dot_content = '''digraph fa_pool_short_sequence_final {
    rankdir=TB;
    compound=true;
    splines=ortho;
    node [shape=rectangle, style=filled, fontname="monospace"];
    
    graph [label="FA Pool - Short Sequence (≤4096 tokens)\nBase Layer: 8 GPUs (0-7)\nFully Connected Architecture", fontsize=20];
    
    // Input and sequence check
    input [shape=ellipse, label="Input\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: Host", fillcolor="#E8F4FD"];
    
    seq_check [shape=parallelogram, label="Sequence Length Check\nThreshold: 4096 tokens\nResult: Use Base Only\nGPU: Host", fillcolor="#FFE4B5"];
    
    // Embedding and Input Processing
    embed_split [shape=parallelogram, label="Split Embedding (TP=8)\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    embed_0 [label="Embedding GPU0\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_0", fillcolor="#90EE90"];
    embed_1 [label="Embedding GPU1\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_1", fillcolor="#90EE90"];
    embed_2 [label="Embedding GPU2\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_2", fillcolor="#90EE90"];
    embed_3 [label="Embedding GPU3\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_3", fillcolor="#90EE90"];
    embed_4 [label="Embedding GPU4\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_4", fillcolor="#90EE90"];
    embed_5 [label="Embedding GPU5\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_5", fillcolor="#90EE90"];
    embed_6 [label="Embedding GPU6\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_6", fillcolor="#90EE90"];
    embed_7 [label="Embedding GPU7\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: gpu_7", fillcolor="#90EE90"];
    
    embed_gather [shape=parallelogram, label="All-Gather Embedding\nInput: 8×[batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    pos_enc [label="Positional Encoding\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#87CEEB"];
    
    // Layer 0 Components
    layernorm_0 [label="LayerNorm 0\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_0 [label="Query Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_0 [label="Key Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_0 [label="Value Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_base_0 [label="Flash Attention 0\nInput: Q,K,V [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_0 [label="Concat Heads\nInput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    attn_output_0 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_0 [label="Residual Add 0\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 0
    ffn_linear1_0 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_0 [label="GELU Activation\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_0 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_0 [label="FFN Residual Add 0\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Layer 1 Components
    layernorm_1 [label="LayerNorm 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_1 [label="Query Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_1 [label="Key Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_1 [label="Value Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_base_1 [label="Flash Attention 1\nInput: Q,K,V [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_1 [label="Concat Heads\nInput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    attn_output_1 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_1 [label="Residual Add 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 1
    ffn_linear1_1 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_1 [label="GELU Activation\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_1 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_1 [label="FFN Residual Add 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Layer 2 Components
    layernorm_2 [label="LayerNorm 2\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_2 [label="Query Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_2 [label="Key Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_2 [label="Value Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_base_2 [label="Flash Attention 2\nInput: Q,K,V [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_2 [label="Concat Heads\nInput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    attn_output_2 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_2 [label="Residual Add 2\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 2
    ffn_linear1_2 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_2 [label="GELU Activation\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_2 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_2 [label="FFN Residual Add 2\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Layer 3 Components
    layernorm_3 [label="LayerNorm 3\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_3 [label="Query Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_3 [label="Key Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_3 [label="Value Projection\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    flash_attn_base_3 [label="Flash Attention 3\nInput: Q,K,V [batch_size=1024, seq_len=≤4096, heads=32, d_k=128]\nOutput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    attn_concat_3 [label="Concat Heads\nInput: [batch_size=1024, seq_len=≤4096, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    attn_output_3 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_3 [label="Residual Add 3\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 3
    ffn_linear1_3 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_3 [label="GELU Activation\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_3 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len=≤4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_3 [label="FFN Residual Add 3\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Output
    output_split [shape=parallelogram, label="Split Output\nInput: [batch_size=1024, seq_len=≤4096, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=≤4096, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    output_0 [label="Linear Output GPU0\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_0", fillcolor="#FFB6C1"];
    output_1 [label="Linear Output GPU1\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_1", fillcolor="#FFB6C1"];
    output_2 [label="Linear Output GPU2\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_2", fillcolor="#FFB6C1"];
    output_3 [label="Linear Output GPU3\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_3", fillcolor="#FFB6C1"];
    output_4 [label="Linear Output GPU4\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_4", fillcolor="#FFB6C1"];
    output_5 [label="Linear Output GPU5\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_5", fillcolor="#FFB6C1"];
    output_6 [label="Linear Output GPU6\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_6", fillcolor="#FFB6C1"];
    output_7 [label="Linear Output GPU7\nInput: [batch_size=1024, seq_len=≤4096, d_model=512]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=4000]\nGPU: gpu_7", fillcolor="#FFB6C1"];
    
    output_concat [shape=parallelogram, label="All-Gather Output\nInput: 8×[batch_size=1024, seq_len=≤4096, vocab=4000]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=32000]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    final_output [shape=ellipse, label="Final Output\nInput: [batch_size=1024, seq_len=≤4096, vocab=32000]\nOutput: [batch_size=1024, seq_len=≤4096, vocab=32000]\nGPU: 0-7", fillcolor="#E8F4FD"];
    
    // Complete directed edges ensuring no disconnected nodes
    input -> seq_check -> embed_split -> {embed_0 embed_1 embed_2 embed_3 embed_4 embed_5 embed_6 embed_7} -> embed_gather -> pos_enc;
    
    pos_enc -> layernorm_0 -> q_proj_0;
    pos_enc -> layernorm_0 -> k_proj_0;
    pos_enc -> layernorm_0 -> v_proj_0;
    
    q_proj_0 -> flash_attn_base_0;
    k_proj_0 -> flash_attn_base_0;
    v_proj_0 -> flash_attn_base_0;
    
    flash_attn_base_0 -> attn_concat_0 -> attn_output_0 -> residual_0;
    pos_enc -> residual_0 [style=dashed, label="Residual"];
    
    residual_0 -> ffn_linear1_0 -> ffn_gelu_0 -> ffn_linear2_0 -> ffn_residual_0;
    residual_0 -> ffn_residual_0 [style=dashed, label="Residual"];
    
    ffn_residual_0 -> layernorm_1 -> q_proj_1;
    ffn_residual_0 -> layernorm_1 -> k_proj_1;
    ffn_residual_0 -> layernorm_1 -> v_proj_1;
    
    q_proj_1 -> flash_attn_base_1;
    k_proj_1 -> flash_attn_base_1;
    v_proj_1 -> flash_attn_base_1;
    
    flash_attn_base_1 -> attn_concat_1 -> attn_output_1 -> residual_1;
    ffn_residual_0 -> residual_1 [style=dashed, label="Residual"];
    
    residual_1 -> ffn_linear1_1 -> ffn_gelu_1 -> ffn_linear2_1 -> ffn_residual_1;
    residual_1 -> ffn_residual_1 [style=dashed, label="Residual"];
    
    ffn_residual_1 -> layernorm_2 -> q_proj_2;
    ffn_residual_1 -> layernorm_2 -> k_proj_2;
    ffn_residual_1 -> layernorm_2 -> v_proj_2;
    
    q_proj_2 -> flash_attn_base_2;
    k_proj_2 -> flash_attn_base_2;
    v_proj_2 -> flash_attn_base_2;
    
    flash_attn_base_2 -> attn_concat_2 -> attn_output_2 -> residual_2;
    ffn_residual_1 -> residual_2 [style=dashed, label="Residual"];
    
    residual_2 -> ffn_linear1_2 -> ffn_gelu_2 -> ffn_linear2_2 -> ffn_residual_2;
    residual_2 -> ffn_residual_2 [style=dashed, label="Residual"];
    
    ffn_residual_2 -> layernorm_3 -> q_proj_3;
    ffn_residual_2 -> layernorm_3 -> k_proj_3;
    ffn_residual_2 -> layernorm_3 -> v_proj_3;
    
    q_proj_3 -> flash_attn_base_3;
    k_proj_3 -> flash_attn_base_3;
    v_proj_3 -> flash_attn_base_3;
    
    flash_attn_base_3 -> attn_concat_3 -> attn_output_3 -> residual_3;
    ffn_residual_2 -> residual_3 [style=dashed, label="Residual"];
    
    residual_3 -> ffn_linear1_3 -> ffn_gelu_3 -> ffn_linear2_3 -> ffn_residual_3;
    residual_3 -> ffn_residual_3 [style=dashed, label="Residual"];
    
    ffn_residual_3 -> output_split -> {output_0 output_1 output_2 output_3 output_4 output_5 output_6 output_7} -> output_concat -> final_output;
}'''
    return dot_content

def generate_fa_pool_long_sequence_dag():
    """Generate FA Pool long sequence DAG with complete connectivity"""
    dot_content = '''digraph fa_pool_long_sequence_final {
    rankdir=TB;
    compound=true;
    splines=ortho;
    node [shape=rectangle, style=filled, fontname="monospace"];
    
    graph [label="FA Pool - Long Sequence (>4096 tokens)\nBase Layer: 8 GPUs (0-7)\nAttention Pool: 32 GPUs (8-39)\nFully Connected Architecture", fontsize=20];
    
    // Input and resource management
    input [shape=ellipse, label="Input\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: Host", fillcolor="#E8F4FD"];
    
    resource_manager [shape=parallelogram, label="Resource Manager\nActivate 32 GPUs for Attention Pool\nGPU: Host", fillcolor="#FFE4B5"];
    
    seq_check [shape=parallelogram, label="Sequence Length Check\nResult: Use Attention Pool\nGPU: Host", fillcolor="#FFE4B5"];
    
    // Embedding and Input Processing
    embed_split [shape=parallelogram, label="Split Embedding (TP=8)\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len>4096, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    embed_0 [label="Embedding GPU0\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_0", fillcolor="#90EE90"];
    embed_1 [label="Embedding GPU1\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_1", fillcolor="#90EE90"];
    embed_2 [label="Embedding GPU2\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_2", fillcolor="#90EE90"];
    embed_3 [label="Embedding GPU3\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_3", fillcolor="#90EE90"];
    embed_4 [label="Embedding GPU4\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_4", fillcolor="#90EE90"];
    embed_5 [label="Embedding GPU5\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_5", fillcolor="#90EE90"];
    embed_6 [label="Embedding GPU6\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_6", fillcolor="#90EE90"];
    embed_7 [label="Embedding GPU7\nInput: [batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=512]\nGPU: gpu_7", fillcolor="#90EE90"];
    
    embed_gather [shape=parallelogram, label="All-Gather Embedding\nInput: 8×[batch_size=1024, seq_len>4096, d_model=512]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFE4B5"];
    
    pos_enc [label="Positional Encoding\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#87CEEB"];
    
    // Attention pool processing
    pool_split [shape=parallelogram, label="Split Sequence Blocks\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: 32×[batch_size=1024, seq_len=ceil(n/32), d_model=4096]\nGPU: 8-39", fillcolor="#FFE4B5"];
    
    // Attention pool GPUs (simplified representation)
    attention_pool_0 [label="Attention Pool Block 0\nInput: [batch_size=1024, seq_len=ceil(n/32), d_model=4096]\nOutput: [batch_size=1024, seq_len=ceil(n/32), heads=32, d_v=128]\nGPU: gpu_8-15", fillcolor="#FFD700"];
    attention_pool_1 [label="Attention Pool Block 1\nInput: [batch_size=1024, seq_len=ceil(n/32), d_model=4096]\nOutput: [batch_size=1024, seq_len=ceil(n/32), heads=32, d_v=128]\nGPU: gpu_16-23", fillcolor="#FFD700"];
    attention_pool_2 [label="Attention Pool Block 2\nInput: [batch_size=1024, seq_len=ceil(n/32), d_model=4096]\nOutput: [batch_size=1024, seq_len=ceil(n/32), heads=32, d_v=128]\nGPU: gpu_24-31", fillcolor="#FFD700"];
    attention_pool_3 [label="Attention Pool Block 3\nInput: [batch_size=1024, seq_len=ceil(n/32), d_model=4096]\nOutput: [batch_size=1024, seq_len=ceil(n/32), heads=32, d_v=128]\nGPU: gpu_32-39", fillcolor="#FFD700"];
    
    gather_attention [shape=parallelogram, label="Gather Pool Results\nInput: 32×[batch_size=1024, seq_len=ceil(n/32), heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_v=128]\nGPU: All GPUs", fillcolor="#FFE4B5"];
    
    // Layer 0 Components
    layernorm_0 [label="LayerNorm 0\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // Layer 0 attention handled by attention pool
    q_proj_0 [label="Query Projection\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_0 [label="Key Projection\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    v_proj_0 [label="Value Projection\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_v=128]\nGPU: 0-7", fillcolor="#FFD700"];
    
    // Attention pool processing
    flash_attn_pool_0 [label="Flash Attention Pool 0\nInput: Q,K,V across all blocks\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_v=128]\nGPU: 8-39", fillcolor="#FFD700"];
    
    attn_concat_0 [label="Concat Heads\nInput: [batch_size=1024, seq_len>4096, heads=32, d_v=128]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    attn_output_0 [label="Attention Output Linear\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#FFD700"];
    
    residual_0 [label="Residual Add 0\nInput: [batch_size=1024, seq_len>4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    // FFN components for layer 0
    ffn_linear1_0 [label="FFN Linear 1\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_gelu_0 [label="GELU Activation\nInput: [batch_size=1024, seq_len>4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len>4096, d_ff=16384]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_linear2_0 [label="FFN Linear 2\nInput: [batch_size=1024, seq_len>4096, d_ff=16384]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    ffn_residual_0 [label="FFN Residual Add 0\nInput: [batch_size=1024, seq_len>4096, d_model=4096]×2\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#98FB98"];
    
    // Similar structure for layers 1-3
    layernorm_1 [label="LayerNorm 1\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    
    q_proj_1 [label="Query Projection\nInput: [batch_size=1024, seq_len>4096, d_model=4096]\nOutput: [batch_size=1024, seq_len>4096, heads=32, d_k=128]\nGPU: 0-7", fillcolor="#FFD700"];
    k_proj_1 [label="Key Projection\nInput: [batch_size=1024, seq_len>4096, d