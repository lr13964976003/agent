#!/usr/bin/env python3
"""
Generate baseline DAG for 4-layer Dense Transformer with TP=8, PP=2
Baseline uses 16 GPUs total: 8-way tensor parallelism + 2-way pipeline parallelism
"""

import os

def generate_baseline_dag():
    dot_content = '''digraph baseline_transformer_dag {
    rankdir=TB;
    compound=true;
    splines=ortho;
    node [shape=rectangle, style=filled, fontname="monospace"];
    
    // Global attributes
    graph [label="Baseline 4-Layer Dense Transformer (TP=8, PP=2)\nTotal GPUs: 16", fontsize=20];
    
    // Input node
    input [shape=ellipse, label="Input\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: Host", fillcolor="#E8F4FD"];
    
    // Embedding layer - distributed across first 8 GPUs
    subgraph cluster_embedding {
        label="Embedding Layer (TP=8)";
        style=dashed;
        fillcolor="#FFF2E6";
        
        embed_split [shape=parallelogram, label="Split (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: All", fillcolor="#FFE4B5"];
        
        embed_0 [label="Embedding GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_0", fillcolor="#90EE90"];
        embed_1 [label="Embedding GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_1", fillcolor="#90EE90"];
        embed_2 [label="Embedding GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_2", fillcolor="#90EE90"];
        embed_3 [label="Embedding GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_3", fillcolor="#90EE90"];
        embed_4 [label="Embedding GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_4", fillcolor="#90EE90"];
        embed_5 [label="Embedding GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_5", fillcolor="#90EE90"];
        embed_6 [label="Embedding GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_6", fillcolor="#90EE90"];
        embed_7 [label="Embedding GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_7", fillcolor="#90EE90"];
        
        embed_gather [shape=parallelogram, label="All-Gather (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: All", fillcolor="#FFE4B5"];
    }
    
    // Positional Encoding
    pos_enc [label="Positional Encoding\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: All", fillcolor="#87CEEB"];
    
    // Layer 0 - Pipeline Stage 0
    subgraph cluster_layer0 {
        label="Layer 0 - Pipeline Stage 0 (GPUs 0-7)";
        style=rounded;
        fillcolor="#F0F8FF";
        
        // Layer Norm 0
        layernorm_0 [label="LayerNorm 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
        
        // Multi-Head Attention - distributed
        subgraph cluster_attention_0 {
            label="Multi-Head Attention (TP=8)";
            style=dashed;
            fillcolor="#FFE4E1";
            
            // Q, K, V projections
            qkv_split_0 [shape=parallelogram, label="Split QKV (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=?, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
            
            q_proj_0 [label="Q Proj GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_0", fillcolor="#FFB6C1"];
            k_proj_0 [label="K Proj GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_0", fillcolor="#FFB6C1"];
            v_proj_0 [label="V Proj GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_0", fillcolor="#FFB6C1"];
            
            q_proj_1 [label="Q Proj GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_1", fillcolor="#FFB6C1"];
            k_proj_1 [label="K Proj GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_1", fillcolor="#FFB6C1"];
            v_proj_1 [label="V Proj GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_1", fillcolor="#FFB6C1"];
            
            // Similar for GPUs 2-7
            q_proj_2 [label="Q Proj GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_2", fillcolor="#FFB6C1"];
            k_proj_2 [label="K Proj GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_2", fillcolor="#FFB6C1"];
            v_proj_2 [label="V Proj GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_2", fillcolor="#FFB6C1"];
            
            q_proj_3 [label="Q Proj GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_3", fillcolor="#FFB6C1"];
            k_proj_3 [label="K Proj GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_3", fillcolor="#FFB6C1"];
            v_proj_3 [label="V Proj GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_3", fillcolor="#FFB6C1"];
            
            q_proj_4 [label="Q Proj GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_4", fillcolor="#FFB6C1"];
            k_proj_4 [label="K Proj GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_4", fillcolor="#FFB6C1"];
            v_proj_4 [label="V Proj GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_4", fillcolor="#FFB6C1"];
            
            q_proj_5 [label="Q Proj GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_5", fillcolor="#FFB6C1"];
            k_proj_5 [label="K Proj GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_5", fillcolor="#FFB6C1"];
            v_proj_5 [label="V Proj GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_5", fillcolor="#FFB6C1"];
            
            q_proj_6 [label="Q Proj GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_6", fillcolor="#FFB6C1"];
            k_proj_6 [label="K Proj GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_6", fillcolor="#FFB6C1"];
            v_proj_6 [label="V Proj GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_6", fillcolor="#FFB6C1"];
            
            q_proj_7 [label="Q Proj GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_7", fillcolor="#FFB6C1"];
            k_proj_7 [label="K Proj GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_7", fillcolor="#FFB6C1"];
            v_proj_7 [label="V Proj GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_7", fillcolor="#FFB6C1"];
            
            // Attention computation
            attn_0 [label="Attention GPU0\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_0", fillcolor="#FFD700"];
            attn_1 [label="Attention GPU1\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_1", fillcolor="#FFD700"];
            attn_2 [label="Attention GPU2\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_2", fillcolor="#FFD700"];
            attn_3 [label="Attention GPU3\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_3", fillcolor="#FFD700"];
            attn_4 [label="Attention GPU4\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_4", fillcolor="#FFD700"];
            attn_5 [label="Attention GPU5\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_5", fillcolor="#FFD700"];
            attn_6 [label="Attention GPU6\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_6", fillcolor="#FFD700"];
            attn_7 [label="Attention GPU7\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: gpu_7", fillcolor="#FFD700"];
            
            // Output projection
            out_proj_0 [label="Output Proj GPU0\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_0", fillcolor="#FFB6C1"];
            out_proj_1 [label="Output Proj GPU1\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_1", fillcolor="#FFB6C1"];
            out_proj_2 [label="Output Proj GPU2\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_2", fillcolor="#FFB6C1"];
            out_proj_3 [label="Output Proj GPU3\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_3", fillcolor="#FFB6C1"];
            out_proj_4 [label="Output Proj GPU4\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_4", fillcolor="#FFB6C1"];
            out_proj_5 [label="Output Proj GPU5\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_5", fillcolor="#FFB6C1"];
            out_proj_6 [label="Output Proj GPU6\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_6", fillcolor="#FFB6C1"];
            out_proj_7 [label="Output Proj GPU7\nInput: [batch_size=1024, seq_len=?, heads=4, d_k=128]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_7", fillcolor="#FFB6C1"];
            
            concat_0 [shape=parallelogram, label="Concat (TP=8)\nInput: 8×[batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: All", fillcolor="#FFE4B5"];
        }
        
        // Residual connection
        residual_0 [label="Residual Add 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
        
        // FFN
        subgraph cluster_ffn_0 {
            label="FFN Layer (TP=8)";
            style=dashed;
            fillcolor="#E6E6FA";
            
            // First linear (column parallel)
            ffn_split_0 [shape=parallelogram, label="Split FFN Input (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=?, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
            
            ffn_linear1_0 [label="FFN Linear1 GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_0", fillcolor="#98FB98"];
            ffn_linear1_1 [label="FFN Linear1 GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_1", fillcolor="#98FB98"];
            ffn_linear1_2 [label="FFN Linear1 GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_2", fillcolor="#98FB98"];
            ffn_linear1_3 [label="FFN Linear1 GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_3", fillcolor="#98FB98"];
            ffn_linear1_4 [label="FFN Linear1 GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_4", fillcolor="#98FB98"];
            ffn_linear1_5 [label="FFN Linear1 GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_5", fillcolor="#98FB98"];
            ffn_linear1_6 [label="FFN Linear1 GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_6", fillcolor="#98FB98"];
            ffn_linear1_7 [label="FFN Linear1 GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_7", fillcolor="#98FB98"];
            
            // GELU activation
            ffn_gelu_0 [label="GELU GPU0\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_0", fillcolor="#87CEEB"];
            ffn_gelu_1 [label="GELU GPU1\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_1", fillcolor="#87CEEB"];
            ffn_gelu_2 [label="GELU GPU2\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_2", fillcolor="#87CEEB"];
            ffn_gelu_3 [label="GELU GPU3\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_3", fillcolor="#87CEEB"];
            ffn_gelu_4 [label="GELU GPU4\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_4", fillcolor="#87CEEB"];
            ffn_gelu_5 [label="GELU GPU5\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_5", fillcolor="#87CEEB"];
            ffn_gelu_6 [label="GELU GPU6\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_6", fillcolor="#87CEEB"];
            ffn_gelu_7 [label="GELU GPU7\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nGPU: gpu_7", fillcolor="#87CEEB"];
            
            // Second linear (row parallel)
            ffn_linear2_0 [label="FFN Linear2 GPU0\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_0", fillcolor="#98FB98"];
            ffn_linear2_1 [label="FFN Linear2 GPU1\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_1", fillcolor="#98FB98"];
            ffn_linear2_2 [label="FFN Linear2 GPU2\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_2", fillcolor="#98FB98"];
            ffn_linear2_3 [label="FFN Linear2 GPU3\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_3", fillcolor="#98FB98"];
            ffn_linear2_4 [label="FFN Linear2 GPU4\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_4", fillcolor="#98FB98"];
            ffn_linear2_5 [label="FFN Linear2 GPU5\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_5", fillcolor="#98FB98"];
            ffn_linear2_6 [label="FFN Linear2 GPU6\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_6", fillcolor="#98FB98"];
            ffn_linear2_7 [label="FFN Linear2 GPU7\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\nOutput: [batch_size=1024, seq_len=?, d_model=512]\nGPU: gpu_7", fillcolor="#98FB98"];
            
            ffn_reduce_0 [shape=parallelogram, label="All-Reduce Sum (TP=8)\nInput: 8×[batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: All", fillcolor="#FFE4B5"];
        }
        
        residual_0_ffn [label="Residual Add 0\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7", fillcolor="#DDA0DD"];
    }
    
    // Communication to next pipeline stage
    pipeline_comm_0 [shape=ellipse, label="Pipeline Communication\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7 → 8-15", fillcolor="#FFD700"];
    
    // Layer 1 - Pipeline Stage 1
    subgraph cluster_layer1 {
        label="Layer 1 - Pipeline Stage 1 (GPUs 8-15)";
        style=rounded;
        fillcolor="#F0F8FF";
        
        layernorm_1 [label="LayerNorm 1\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\nGPU: 8-15", fillcolor="#DDA0DD"];
        
        // Similar attention structure for layer 1 with GPUs 8-15
        // (abbreviated for brevity - full structure follows same pattern)
        qkv_split_1 [shape=parallelogram, label="Split QKV (TP=8)\nGPU: 8-15", fillcolor="#FFE4B5"];
        
        // Attention and FFN for layer 1
        attn_1_result [label="Multi-Head Attention 1 (TP=8)\nGPU: 8-15", fillcolor="#FFD700"];
        residual_1 [label="Residual Add 1\nGPU: 8-15", fillcolor="#DDA0DD"];
        ffn_1_result [label="FFN 1 (TP=8)\nGPU: 8-15", fillcolor="#98FB98"];
        residual_1_ffn [label="Residual Add 1\nGPU: 8-15", fillcolor="#DDA0DD"];
    }
    
    // Pipeline Communication to Stage 0
    pipeline_comm_1 [shape=ellipse, label="Pipeline Communication\nGPU: 8-15 → 0-7", fillcolor="#FFD700"];
    
    // Layer 2 - Pipeline Stage 0
    subgraph cluster_layer2 {
        label="Layer 2 - Pipeline Stage 0 (GPUs 0-7)";
        style=rounded;
        fillcolor="#F0F8FF";
        
        layernorm_2 [label="LayerNorm 2\nGPU: 0-7", fillcolor="#DDA0DD"];
        attn_2_result [label="Multi-Head Attention 2 (TP=8)\nGPU: 0-7", fillcolor="#FFD700"];
        residual_2 [label="Residual Add 2\nGPU: 0-7", fillcolor="#DDA0DD"];
        ffn_2_result [label="FFN 2 (TP=8)\nGPU: 0-7", fillcolor="#98FB98"];
        residual_2_ffn [label="Residual Add 2\nGPU: 0-7", fillcolor="#DDA0DD"];
    }
    
    pipeline_comm_2 [shape=ellipse, label="Pipeline Communication\nGPU: 0-7 → 8-15", fillcolor="#FFD700"];
    
    // Layer 3 - Pipeline Stage 1
    subgraph cluster_layer3 {
        label="Layer 3 - Pipeline Stage 1 (GPUs 8-15)";
        style=rounded;
        fillcolor="#F0F8FF";
        
        layernorm_3 [label="LayerNorm 3\nGPU: 8-15", fillcolor="#DDA0DD"];
        attn_3_result [label="Multi-Head Attention 3 (TP=8)\nGPU: 8-15", fillcolor="#FFD700"];
        residual_3 [label="Residual Add 3\nGPU: 8-15", fillcolor="#DDA0DD"];
        ffn_3_result [label="FFN 3 (TP=8)\nGPU: 8-15", fillcolor="#98FB98"];
        residual_3_ffn [label="Residual Add 3\nGPU: 8-15", fillcolor="#DDA0DD"];
    }
    
    // Final communication back to Stage 0 for output
    pipeline_comm_final [shape=ellipse, label="Pipeline Communication\nGPU: 8-15 → 0-7", fillcolor="#FFD700"];
    
    // Output layer
    subgraph cluster_output {
        label="Output Layer (TP=8)";
        style=dashed;
        fillcolor="#FFE4E1";
        
        output_split [shape=parallelogram, label="Split Output (TP=8)\nInput: [batch_size=1024, seq_len=?, d_model=4096]\nOutput: 8×[batch_size=1024, seq_len=?, d_model=512]\nGPU: 0-7", fillcolor="#FFE4B5"];
        
        output_0 [label="Linear GPU0\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_0", fillcolor="#FFB6C1"];
        output_1 [label="Linear GPU1\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_1", fillcolor="#FFB6C1"];
        output_2 [label="Linear GPU2\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_2", fillcolor="#FFB6C1"];
        output_3 [label="Linear GPU3\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_3", fillcolor="#FFB6C1"];
        output_4 [label="Linear GPU4\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_4", fillcolor="#FFB6C1"];
        output_5 [label="Linear GPU5\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_5", fillcolor="#FFB6C1"];
        output_6 [label="Linear GPU6\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_6", fillcolor="#FFB6C1"];
        output_7 [label="Linear GPU7\nInput: [batch_size=1024, seq_len=?, d_model=512]\nOutput: [batch_size=1024, seq_len=?, vocab=4000]\nGPU: gpu_7", fillcolor="#FFB6C1"];
        
        output_concat [shape=parallelogram, label="Concat Output (TP=8)\nInput: 8×[batch_size=1024, seq_len=?, vocab=4000]\nOutput: [batch_size=1024, seq_len=?, vocab=32000]\nGPU: All", fillcolor="#FFE4B5"];
        
        final_output [shape=ellipse, label="Final Output\nInput: [batch_size=1024, seq_len=?, vocab=32000]\nOutput: [batch_size=1024, seq_len=?, vocab=32000]\nGPU: All", fillcolor="#E8F4FD"];
    }
    
    // Connections for complete flow
    input -> embed_split;
    embed_split -> {embed_0 embed_1 embed_2 embed_3 embed_4 embed_5 embed_6 embed_7};
    {embed_0 embed_1 embed_2 embed_3 embed_4 embed_5 embed_6 embed_7} -> embed_gather -> pos_enc;
    pos_enc -> layernorm_0 -> qkv_split_0;
    
    // Attention flow
    qkv_split_0 -> {q_proj_0 k_proj_0 v_proj_0 q_proj_1 k_proj_1 v_proj_1 q_proj_2 k_proj_2 v_proj_2 q_proj_3 k_proj_3 v_proj_3 q_proj_4 k_proj_4 v_proj_4 q_proj_5 k_proj_5 v_proj_5 q_proj_6 k_proj_6 v_proj_6 q_proj_7 k_proj_7 v_proj_7};
    
    q_proj_0 -> k_proj_0 -> v_proj_0 -> attn_0 -> out_proj_0;
    q_proj_1 -> k_proj_1 -> v_proj_1 -> attn_1 -> out_proj_1;
    q_proj_2 -> k_proj_2 -> v_proj_2 -> attn_2 -> out_proj_2;
    q_proj_3 -> k_proj_3 -> v_proj_3 -> attn_3 -> out_proj_3;
    q_proj_4 -> k_proj_4 -> v_proj_4 -> attn_4 -> out_proj_4;
    q_proj_5 -> k_proj_5 -> v_proj_5 -> attn_5 -> out_proj_5;
    q_proj_6 -> k_proj_6 -> v_proj_6 -> attn_6 -> out_proj_6;
    q_proj_7 -> k_proj_7 -> v_proj_7 -> attn_7 -> out_proj_7;
    
    {out_proj_0 out_proj_1 out_proj_2 out_proj_3 out_proj_4 out_proj_5 out_proj_6 out_proj_7} -> concat_0 -> residual_0;
    residual_0 -> layernorm_0 [style=dashed, label="Residual"];
    residual_0 -> ffn_split_0;
    
    // FFN flow
    ffn_split_0 -> {ffn_linear1_0 ffn_linear1_1 ffn_linear1_2 ffn_linear1_3 ffn_linear1_4 ffn_linear1_5 ffn_linear1_6 ffn_linear1_7};
    ffn_linear1_0 -> ffn_gelu_0 -> ffn_linear2_0;
    ffn_linear1_1 -> ffn_gelu_1 -> ffn_linear2_1;
    ffn_linear1_2 -> ffn_gelu_2 -> ffn_linear2_2;
    ffn_linear1_3 -> ffn_gelu_3 -> ffn_linear2_3;
    ffn_linear1_4 -> ffn_gelu_4 -> ffn_linear2_4;
    ffn_linear1_5 -> ffn_gelu_5 -> ffn_linear2_5;
    ffn_linear1_6 -> ffn_gelu_6 -> ffn_linear2_6;
    ffn_linear1_7 -> ffn_gelu_7 -> ffn_linear2_7;
    
    {ffn_linear2_0 ffn_linear2_1 ffn_linear2_2 ffn_linear2_3 ffn_linear2_4 ffn_linear2_5 ffn_linear2_6 ffn_linear2_7} -> ffn_reduce_0 -> residual_0_ffn;
    residual_0_ffn -> residual_0 [style=dashed, label="Residual"];
    
    // Pipeline communications (simplified)
    residual_0_ffn -> pipeline_comm_0 -> layernorm_1 -> attn_1_result -> residual_1 -> ffn_1_result -> residual_1_ffn -> pipeline_comm_1 -> layernorm_2 -> attn_2_result -> residual_2 -> ffn_2_result -> residual_2_ffn -> pipeline_comm_2 -> layernorm_3 -> attn_3_result -> residual_3 -> ffn_3_result -> residual_3_ffn -> pipeline_comm_final -> output_split;
    
    // Output flow
    output_split -> {output_0 output_1 output_2 output_3 output_4 output_5 output_6 output_7} -> output_concat -> final_output;
}
'''
    
    # Write DOT file
    with open('../outputs/2025-10-16-18-18-15/baseline_transformer_dag.dot', 'w') as f:
        f.write(dot_content)
    
    # Generate SVG using Graphviz
    os.system('dot -Tsvg ../outputs/2025-10-16-18-18-15/baseline_transformer_dag.dot -o ../outputs/2025-10-16-18-18-15/baseline_transformer_dag.svg')
    
    return "baseline_transformer_dag.dot"

if __name__ == "__main__":
    generate_baseline_dag()