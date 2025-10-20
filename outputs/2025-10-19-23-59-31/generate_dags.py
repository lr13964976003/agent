#!/usr/bin/env python3
"""
Complete DAG generator for MA Separation and Baseline configurations
"""

import os

def generate_ma_separation_dag():
    """Generate complete MA Separation DAG"""
    content = """digraph ma_separation_complete {
    graph [comment="MA Separation Complete Model DAG - 16 GPUs", rankdir=TB, size="25,35"]
    node [fontname="Arial", fontsize=10, shape=ellipse, style=filled]
    
    // Input and Output nodes
    input [fillcolor="#e6f3ff", label="Input\nInput: [batch_size=1024, seq_len=2048]\nOutput: [batch_size=1024, seq_len=2048]\nGPU: 0", shape=ellipse]
    output [fillcolor="#e6f3ff", label="Output\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, vocab_size=50265]\nGPU: 0", shape=ellipse]
    
    // Embedding layer
    embedding [fillcolor="#000080", label="Embedding\nInput: [batch_size=1024, seq_len=2048]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    
    // Layer 0 components
    ln1_layer_0 [fillcolor="#000080", label="LayerNorm 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    
    // QKV projections for Layer 0 across 12 attention GPUs
    qkv_0_0 [fillcolor="#0014ff", label="QKV Proj 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 0", shape=rectangle]
    qkv_0_1 [fillcolor="#001eff", label="QKV Proj 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 1", shape=rectangle]
    qkv_0_2 [fillcolor="#0028ff", label="QKV Proj 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 2", shape=rectangle]
    qkv_0_3 [fillcolor="#0032ff", label="QKV Proj 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 3", shape=rectangle]
    qkv_0_4 [fillcolor="#003cff", label="QKV Proj 4\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 4", shape=rectangle]
    qkv_0_5 [fillcolor="#0046ff", label="QKV Proj 5\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 5", shape=rectangle]
    qkv_0_6 [fillcolor="#0050ff", label="QKV Proj 6\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 6", shape=rectangle]
    qkv_0_7 [fillcolor="#005aff", label="QKV Proj 7\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=3, d_k=128]\nGPU: 7", shape=rectangle]
    qkv_0_8 [fillcolor="#0064ff", label="QKV Proj 8\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=2, d_k=128]\nGPU: 8", shape=rectangle]
    qkv_0_9 [fillcolor="#006eff", label="QKV Proj 9\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=2, d_k=128]\nGPU: 9", shape=rectangle]
    qkv_0_10 [fillcolor="#0078ff", label="QKV Proj 10\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=2, d_k=128]\nGPU: 10", shape=rectangle]
    qkv_0_11 [fillcolor="#0082ff", label="QKV Proj 11\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: Q:[batch_size=1024, seq_len=2048, heads=2, d_k=128]\nGPU: 11", shape=rectangle]

    // Attention computation and aggregation
    attn_agg_0 [fillcolor="#ffffcc", label="All-Reduce Attention\nInput: [12× partial attention outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    out_proj_0 [fillcolor="#000080", label="Output Projection 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    residual1_0 [fillcolor="#ffffcc", label="Residual Add 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]

    // MoE components
    ln2_0 [fillcolor="#000080", label="LayerNorm 0 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    gate_0 [fillcolor="#00ff00", label="Gating Network 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
    
    // Expert nodes across 4 MoE GPUs
    expert_0_12 [fillcolor="#ff8800", label="Expert 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 12", shape=rectangle]
    expert_1_12 [fillcolor="#ff8800", label="Expert 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 12", shape=rectangle]
    expert_2_12 [fillcolor="#ff8800", label="Expert 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 12", shape=rectangle]
    expert_3_12 [fillcolor="#ff8800", label="Expert 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 12", shape=rectangle]
    expert_4_13 [fillcolor="#ff8800", label="Expert 4\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 13", shape=rectangle]
    expert_5_13 [fillcolor="#ff8800", label="Expert 5\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 13", shape=rectangle]
    expert_6_13 [fillcolor="#ff8800", label="Expert 6\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 13", shape=rectangle]
    expert_7_13 [fillcolor="#ff8800", label="Expert 7\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 13", shape=rectangle]
    expert_8_14 [fillcolor="#ff8800", label="Expert 8\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 14", shape=rectangle]
    expert_9_14 [fillcolor="#ff8800", label="Expert 9\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 14", shape=rectangle]
    expert_10_14 [fillcolor="#ff8800", label="Expert 10\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 14", shape=rectangle]
    expert_11_14 [fillcolor="#ff8800", label="Expert 11\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 14", shape=rectangle]
    expert_12_15 [fillcolor="#ff8800", label="Expert 12\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 15", shape=rectangle]
    expert_13_15 [fillcolor="#ff8800", label="Expert 13\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 15", shape=rectangle]
    expert_14_15 [fillcolor="#ff8800", label="Expert 14\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 15", shape=rectangle]
    expert_15_15 [fillcolor="#ff8800", label="Expert 15\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 15", shape=rectangle]

    expert_agg_0 [fillcolor="#ffffcc", label="Expert Aggregation 0\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    residual2_0 [fillcolor="#ffffcc", label="Residual Add 0 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]

    // Layer 1 components
    ln1_layer_1 [fillcolor="#000080", label="LayerNorm 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    attn_agg_1 [fillcolor="#ffffcc", label="All-Reduce Attention 1\nInput: [12× partial attention outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    out_proj_1 [fillcolor="#000080", label="Output Projection 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    residual1_1 [fillcolor="#ffffcc", label="Residual Add 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    ln2_1 [fillcolor="#000080", label="LayerNorm 1 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    gate_1 [fillcolor="#00ff00", label="Gating Network 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
    expert_agg_1 [fillcolor="#ffffcc", label="Expert Aggregation 1\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    residual2_1 [fillcolor="#ffffcc", label="Residual Add 1 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]

    // Layer 2 components
    ln1_layer_2 [fillcolor="#000080", label="LayerNorm 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    attn_agg_2 [fillcolor="#ffffcc", label="All-Reduce Attention 2\nInput: [12× partial attention outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    out_proj_2 [fillcolor="#000080", label="Output Projection 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    residual1_2 [fillcolor="#ffffcc", label="Residual Add 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    ln2_2 [fillcolor="#000080", label="LayerNorm 2 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    gate_2 [fillcolor="#00ff00", label="Gating Network 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
    expert_agg_2 [fillcolor="#ffffcc", label="Expert Aggregation 2\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    residual2_2 [fillcolor="#ffffcc", label="Residual Add 2 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]

    // Layer 3 components
    ln1_layer_3 [fillcolor="#000080", label="LayerNorm 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    attn_agg_3 [fillcolor="#ffffcc", label="All-Reduce Attention 3\nInput: [12× partial attention outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    out_proj_3 [fillcolor="#000080", label="Output Projection 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    residual1_3 [fillcolor="#ffffcc", label="Residual Add 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    ln2_3 [fillcolor="#000080", label="LayerNorm 3 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
    gate_3 [fillcolor="#00ff00", label="Gating Network 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
    expert_agg_3 [fillcolor="#ffffcc", label="Expert Aggregation 3\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    residual2_3 [fillcolor="#ffffcc", label="Residual Add 3 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]

    // Edges - Complete flow
    input -> embedding
    
    // Layer 0 flow
    embedding -> ln1_layer_0
    ln1_layer_0 -> qkv_0_0
    ln1_layer_0 -> qkv_0_1
    ln1_layer_0 -> qkv_0_2
    ln1_layer_0 -> qkv_0_3
    ln1_layer_0 -> qkv_0_4
    ln1_layer_0 -> qkv_0_5
    ln1_layer_0 -> qkv_0_6
    ln1_layer_0 -> qkv_0_7
    ln1_layer_0 -> qkv_0_8
    ln1_layer_0 -> qkv_0_9
    ln1_layer_0 -> qkv_0_10
    ln1_layer_0 -> qkv_0_11
    
    qkv_0_0 -> attn_agg_0
    qkv_0_1 -> attn_agg_0
    qkv_0_2 -> attn_agg_0
    qkv_0_3 -> attn_agg_0
    qkv_0_4 -> attn_agg_0
    qkv_0_5 -> attn_agg_0
    qkv_0_6 -> attn_agg_0
    qkv_0_7 -> attn_agg_0
    qkv_0_8 -> attn_agg_0
    qkv_0_9 -> attn_agg_0
    qkv_0_10 -> attn_agg_0
    qkv_0_11 -> attn_agg_0
    
    attn_agg_0 -> out_proj_0
    out_proj_0 -> residual1_0
    embedding -> residual1_0
    residual1_0 -> ln2_0
    
    // Layer 0 MoE flow
    ln2_0 -> gate_0
    ln2_0 -> expert_0_12
    ln2_0 -> expert_1_12
    ln2_0 -> expert_2_12
    ln2_0 -> expert_3_12
    ln2_0 -> expert_4_13
    ln2_0 -> expert_5_13
    ln2_0 -> expert_6_13
    ln2_0 -> expert_7_13
    ln2_0 -> expert_8_14
    ln2_0 -> expert_9_14
    ln2_0 -> expert_10_14
    ln2_0 -> expert_11_14
    ln2_0 -> expert_12_15
    ln2_0 -> expert_13_15
    ln2_0 -> expert_14_15
    ln2_0 -> expert_15_15
    
    gate_0 -> expert_0_12 [style=dashed]
    gate_0 -> expert_1_12 [style=dashed]
    gate_0 -> expert_2_12 [style=dashed]
    gate_0 -> expert_3_12 [style=dashed]
    gate_0 -> expert_4_13 [style=dashed]
    gate_0 -> expert_5_13 [style=dashed]
    gate_0 -> expert_6_13 [style=dashed]
    gate_0 -> expert_7_13 [style=dashed]
    gate_0 -> expert_8_14 [style=dashed]
    gate_0 -> expert_9_14 [style=dashed]
    gate_0 -> expert_10_14 [style=dashed]
    gate_0 -> expert_11_14 [style=dashed]
    gate_0 -> expert_12_15 [style=dashed]
    gate_0 -> expert_13_15 [style=dashed]
    gate_0 -> expert_14_15 [style=dashed]
    gate_0 -> expert_15_15 [style=dashed]
    
    expert_0_12 -> expert_agg_0
    expert_1_12 -> expert_agg_0
    expert_2_12 -> expert_agg_0
    expert_3_12 -> expert_agg_0
    expert_4_13 -> expert_agg_0
    expert_5_13 -> expert_agg_0
    expert_6_13 -> expert_agg_0
    expert_7_13 -> expert_agg_0
    expert_8_14 -> expert_agg_0
    expert_9_14 -> expert_agg_0
    expert_10_14 -> expert_agg_0
    expert_11_14 -> expert_agg_0
    expert_12_15 -> expert_agg_0
    expert_13_15 -> expert_agg_0
    expert_14_15 -> expert_agg_0
    expert_15_15 -> expert_agg_0
    
    expert_agg_0 -> residual2_0
    residual1_0 -> residual2_0
    residual2_0 -> ln1_layer_1
    
    // Layer 1 flow (similar to layer 0)
    ln1_layer_1 -> attn_agg_1
    attn_agg_1 -> out_proj_1
    out_proj_1 -> residual1_1
    residual2_0 -> residual1_1
    residual1_1 -> ln2_1
    
    ln2_1 -> gate_1
    ln2_1 -> expert_0_12
    ln2_1 -> expert_1_12
    ln2_1 -> expert_2_12
    ln2_1 -> expert_3_12
    ln2_1 -> expert_4_13
    ln2_1 -> expert_5_13
    ln2_1 -> expert_6_13
    ln2_1 -> expert_7_13
    ln2_1 -> expert_8_14
    ln2_1 -> expert_9_14
    ln2_1 -> expert_10_14
    ln2_1 -> expert_11_14
    ln2_1 -> expert_12_15
    ln2_1 -> expert_13_15
    ln2_1 -> expert_14_15
    ln2_1 -> expert_15_15
    
    gate_1 -> expert_0_12 [style=dashed]
    gate_1 -> expert_1_12 [style=dashed]
    gate_1 -> expert_2_12 [style=dashed]
    gate_1 -> expert_3_12 [style=dashed]
    gate_1 -> expert_4_13 [style=dashed]
    gate_1 -> expert_5_13 [style=dashed]
    gate_1 -> expert_6_13 [style=dashed]
    gate_1 -> expert_7_13 [style=dashed]
    gate_1 -> expert_8_14 [style=dashed]
    gate_1 -> expert_9_14 [style=dashed]
    gate_1 -> expert_10_14 [style=dashed]
    gate_1 -> expert_11_14 [style=dashed]
    gate_1 -> expert_12_15 [style=dashed]
    gate_1 -> expert_13_15 [style=dashed]
    gate_1 -> expert_14_15 [style=dashed]
    gate_1 -> expert_15_15 [style=dashed]
    
    expert_agg_1 -> residual2_1
    residual1_1 -> residual2_1
    residual2_1 -> ln1_layer_2
    
    // Layer 2 flow
    ln1_layer_2 -> attn_agg_2
    attn_agg_2 -> out_proj_2
    out_proj_2 -> residual1_2
    residual2_1 -> residual1_2
    residual1_2 -> ln2_2
    
    ln2_2 -> gate_2
    ln2_2 -> expert_0_12
    ln2_2 -> expert_1_12
    ln2_2 -> expert_2_12
    ln2_2 -> expert_3_12
    ln2_2 -> expert_4_13
    ln2_2 -> expert_5_13
    ln2_2 -> expert_6_13
    ln2_2 -> expert_7_13
    ln2_2 -> expert_8_14
    ln2_2 -> expert_9_14
    ln2_2 -> expert_10_14
    ln2_2 -> expert_11_14
    ln2_2 -> expert_12_15
    ln2_2 -> expert_13_15
    ln2_2 -> expert_14_15
    ln2_2 -> expert_15_15
    
    gate_2 -> expert_0_12 [style=dashed]
    gate_2 -> expert_1_12 [style=dashed]
    gate_2 -> expert_2_12 [style=dashed]
    gate_2 -> expert_3_12 [style=dashed]
    gate_2 -> expert_4_13 [style=dashed]
    gate_2 -> expert_5_13 [style=dashed]
    gate_2 -> expert_6_13 [style=dashed]
    gate_2 -> expert_7_13 [style=dashed]
    gate_2 -> expert_8_14 [style=dashed]
    gate_2 -> expert_9_14 [style=dashed]
    gate_2 -> expert_10_14 [style=dashed]
    gate_2 -> expert_11_14 [style=dashed]
    gate_2 -> expert_12_15 [style=dashed]
    gate_2 -> expert_13_15 [style=dashed]
    gate_2 -> expert_14_15 [style=dashed]
    gate_2 -> expert_15_15 [style=dashed]
    
    expert_agg_2 -> residual2_2
    residual1_2 -> residual2_2
    residual2_2 -> ln1_layer_3
    
    // Layer 3 flow
    ln1_layer_3 -> attn_agg_3
    attn_agg_3 -> out_proj_3
    out_proj_3 -> residual1_3
    residual2_2 -> residual1_3
    residual1_3 -> ln2_3
    
    ln2_3 -> gate_3
    ln2_3 -> expert_0_12
    ln2_3 -> expert_1_12
    ln2_3 -> expert_2_12
    ln2_3 -> expert_3_12
    ln2_3 -> expert_4_13
    ln2_3 -> expert_5_13
    ln2_3 -> expert_6_13
    ln2_3 -> expert_7_13
    ln2_3 -> expert_8_14
    ln2_3 -> expert_9_14
    ln2_3 -> expert_10_14
    ln2_3 -> expert_11_14
    ln2_3 -> expert_12_15
    ln2_3 -> expert_13_15
    ln2_3 -> expert_14_15
    ln2_3 -> expert_15_15
    
    gate_3 -> expert_0_12 [style=dashed]
    gate_3 -> expert_1_12 [style=dashed]
    gate_3 -> expert_2_12 [style=dashed]
    gate_3 -> expert_3_12 [style=dashed]
    gate_3 -> expert_4_13 [style=dashed]
    gate_3 -> expert_5_13 [style=dashed]
    gate_3 -> expert_6_13 [style=dashed]
    gate_3 -> expert_7_13 [style=dashed]
    gate_3 -> expert_8_14 [style=dashed]
    gate_3 -> expert_9_14 [style=dashed]
    gate_3 -> expert_10_14 [style=dashed]
    gate_3 -> expert_11_14 [style=dashed]
    gate_3 -> expert_12_15 [style=dashed]
    gate_3 -> expert_13_15 [style=dashed]
    gate_3 -> expert_14_15 [style=dashed]
    gate_3 -> expert_15_15 [style=dashed]
    
    expert_agg_3 -> residual2_3
    residual1_3 -> residual2_3
    residual2_3 -> output
}
"""
    return content

def generate_baseline_dag():
    """Generate complete baseline Hybrid TP+PP DAG"""
    content = """digraph baseline_hybrid_tp_pp {
    graph [comment="Baseline Hybrid TP=8, PP=2 Complete Model DAG - 16 GPUs", rankdir=TB, size="25,35"]
    node [fontname="Arial", fontsize=10, shape=ellipse, style=filled]
    
    // Input and Output nodes
    input [fillcolor="#e6f3ff", label="Input\nInput: [batch_size=1024, seq_len=2048]\nOutput: [batch_size=1024, seq_len=2048]\nGPU: 0", shape=ellipse]
    output [fillcolor="#e6f3ff", label="Output\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, vocab_size=50265]\nGPU: 0", shape=ellipse]
    
    // Pipeline Stage 0 (GPUs 0-7) - Layers 0 and 1
    subgraph cluster_stage0 {
        label="Pipeline Stage 0\nGPUs 0-7 (8-way TP)"
        style=dashed
        
        // Embedding and Layer 0
        embedding [fillcolor="#000080", label="Embedding\nInput: [batch_size=1024, seq_len=2048]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        
        // Layer 0 - Attention and MoE across 8 GPUs
        ln1_0 [fillcolor="#000080", label="LayerNorm 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        
        // Attention across 8 GPUs (TP=8)
        qkv_0_0 [fillcolor="#0014ff", label="QKV Proj 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 0", shape=rectangle]
        qkv_0_1 [fillcolor="#001eff", label="QKV Proj 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 1", shape=rectangle]
        qkv_0_2 [fillcolor="#0028ff", label="QKV Proj 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 2", shape=rectangle]
        qkv_0_3 [fillcolor="#0032ff", label="QKV Proj 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 3", shape=rectangle]
        qkv_0_4 [fillcolor="#003cff", label="QKV Proj 4\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 4", shape=rectangle]
        qkv_0_5 [fillcolor="#0046ff", label="QKV Proj 5\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 5", shape=rectangle]
        qkv_0_6 [fillcolor="#0050ff", label="QKV Proj 6\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 6", shape=rectangle]
        qkv_0_7 [fillcolor="#005aff", label="QKV Proj 7\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 7", shape=rectangle]
        
        attention_0 [fillcolor="#ff0000", label="Multi-Head Attention 0\nInput: [8× partial heads]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0-7", shape=rectangle]
        out_proj_0 [fillcolor="#000080", label="Output Projection 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        residual1_0 [fillcolor="#ffffcc", label="Residual Add 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
        
        // MoE Layer 0 across 8 GPUs
        ln2_0 [fillcolor="#000080", label="LayerNorm 0 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        gate_0 [fillcolor="#00ff00", label="Gating Network 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
        
        // 16 experts across 8 GPUs (2 experts per GPU)
        exp_0_0 [fillcolor="#ff8800", label="Expert 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        exp_1_0 [fillcolor="#ff8800", label="Expert 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        exp_2_1 [fillcolor="#ff8800", label="Expert 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 1", shape=rectangle]
        exp_3_1 [fillcolor="#ff8800", label="Expert 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 1", shape=rectangle]
        exp_4_2 [fillcolor="#ff8800", label="Expert 4\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 2", shape=rectangle]
        exp_5_2 [fillcolor="#ff8800", label="Expert 5\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 2", shape=rectangle]
        exp_6_3 [fillcolor="#ff8800", label="Expert 6\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 3", shape=rectangle]
        exp_7_3 [fillcolor="#ff8800", label="Expert 7\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 3", shape=rectangle]
        exp_8_4 [fillcolor="#ff8800", label="Expert 8\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 4", shape=rectangle]
        exp_9_4 [fillcolor="#ff8800", label="Expert 9\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 4", shape=rectangle]
        exp_10_5 [fillcolor="#ff8800", label="Expert 10\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 5", shape=rectangle]
        exp_11_5 [fillcolor="#ff8800", label="Expert 11\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 5", shape=rectangle]
        exp_12_6 [fillcolor="#ff8800", label="Expert 12\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 6", shape=rectangle]
        exp_13_6 [fillcolor="#ff8800", label="Expert 13\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 6", shape=rectangle]
        exp_14_7 [fillcolor="#ff8800", label="Expert 14\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 7", shape=rectangle]
        exp_15_7 [fillcolor="#ff8800", label="Expert 15\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 7", shape=rectangle]
        
        expert_agg_0 [fillcolor="#ffffcc", label="Expert Aggregation 0\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
        residual2_0 [fillcolor="#ffffcc", label="Residual Add 0\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
        
        // Layer 1 in Stage 0
        ln1_1 [fillcolor="#000080", label="LayerNorm 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        attention_1 [fillcolor="#ff0000", label="Multi-Head Attention 1\nInput: [8× partial heads]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0-7", shape=rectangle]
        out_proj_1 [fillcolor="#000080", label="Output Projection 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        residual1_1 [fillcolor="#ffffcc", label="Residual Add 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
        
        // MoE Layer 1
        ln2_1 [fillcolor="#000080", label="LayerNorm 1 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=rectangle]
        gate_1 [fillcolor="#00ff00", label="Gating Network 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 0", shape=parallelogram, style=dashed]
        expert_agg_1 [fillcolor="#ffffcc", label="Expert Aggregation 1\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
        residual2_1 [fillcolor="#ffffcc", label="Residual Add 1\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0", shape=parallelogram]
    }
    
    // Pipeline Stage 1 (GPUs 8-15) - Layers 2 and 3
    subgraph cluster_stage1 {
        label="Pipeline Stage 1\nGPUs 8-15 (8-way TP)"
        style=dashed
        
        // Layer 2
        ln1_2 [fillcolor="#000080", label="LayerNorm 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        qkv_2_8 [fillcolor="#0090ff", label="QKV Proj 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 8", shape=rectangle]
        qkv_2_9 [fillcolor="#009aff", label="QKV Proj 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 9", shape=rectangle]
        qkv_2_10 [fillcolor="#00a4ff", label="QKV Proj 4\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 10", shape=rectangle]
        qkv_2_11 [fillcolor="#00aeff", label="QKV Proj 5\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 11", shape=rectangle]
        qkv_2_12 [fillcolor="#00b8ff", label="QKV Proj 6\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 12", shape=rectangle]
        qkv_2_13 [fillcolor="#00c2ff", label="QKV Proj 7\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 13", shape=rectangle]
        qkv_2_14 [fillcolor="#00ccff", label="QKV Proj 8\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 14", shape=rectangle]
        qkv_2_15 [fillcolor="#00d6ff", label="QKV Proj 9\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, heads=4, d_k=128]\nGPU: 15", shape=rectangle]
        
        attention_2 [fillcolor="#ff0000", label="Multi-Head Attention 2\nInput: [8× partial heads]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8-15", shape=rectangle]
        out_proj_2 [fillcolor="#000080", label="Output Projection 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        residual1_2 [fillcolor="#ffffcc", label="Residual Add 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
        
        // MoE Layer 2
        ln2_2 [fillcolor="#000080", label="LayerNorm 2 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        gate_2 [fillcolor="#00ff00", label="Gating Network 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 8", shape=parallelogram, style=dashed]
        expert_agg_2 [fillcolor="#ffffcc", label="Expert Aggregation 2\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
        residual2_2 [fillcolor="#ffffcc", label="Residual Add 2\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
        
        // Layer 3
        ln1_3 [fillcolor="#000080", label="LayerNorm 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        attention_3 [fillcolor="#ff0000", label="Multi-Head Attention 3\nInput: [8× partial heads]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8-15", shape=rectangle]
        out_proj_3 [fillcolor="#000080", label="Output Projection 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        residual1_3 [fillcolor="#ffffcc", label="Residual Add 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
        
        // MoE Layer 3
        ln2_3 [fillcolor="#000080", label="LayerNorm 3 (MoE)\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=rectangle]
        gate_3 [fillcolor="#00ff00", label="Gating Network 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\nGPU: 8", shape=parallelogram, style=dashed]
        expert_agg_3 [fillcolor="#ffffcc", label="Expert Aggregation 3\nInput: [16× expert outputs]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
        residual2_3 [fillcolor="#ffffcc", label="Residual Add 3\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 8", shape=parallelogram]
    }
    
    // Pipeline communication
    pipeline_comm [fillcolor="#ff00ff", label="Pipeline Communication\nInput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nOutput: [batch_size=1024, seq_len=2048, hidden_size=4096]\nGPU: 0->8", shape=ellipse, style=dashed]
    
    // Edges - Complete flow
    input -> embedding
    
    // Stage 0 flow
    embedding -> ln1_0
    ln1_0 -> qkv_0_0
    ln1_0 -> qkv_0_1
    ln1_0 -> qkv_0_2
    ln1_0 -> qkv_0_3
    ln1_0 -> qkv_0_4
    ln1_0 -> qkv_0_5
    ln1_0 -> qkv_0_6
    ln1_0 -> qkv_0_7
    
    qkv_0_0 -> attention_0
    qkv_0_1 -> attention_0
    qkv_0_2 -> attention_0
    qkv_0_3 -> attention_0
    qkv_0_4 -> attention_0
    qkv_0_5 -> attention_0
    qkv_0_6 -> attention_0
    qkv_0_7 -> attention_0
    
    attention_0 -> out_proj_0
    out_proj_0 -> residual1_0
    embedding -> residual1_0
    residual1_0 -> ln2_0
    
    // Layer 0 MoE in Stage 0
    ln2_0 -> gate_0
    ln2_0 -> exp_0_0
    ln2_0 -> exp_1_0
    ln2_0 -> exp_2_1
    ln2_0 -> exp_3_1
    ln2_0 -> exp_4_2
    ln2_0 -> exp_5_2
    ln2_0 -> exp_6_3
    ln2_0 -> exp_7_3
    ln2_0 -> exp_8_4
    ln2_0 -> exp_9_4
    ln2_0 -> exp_10_5
    ln2_0 -> exp_11_5
    ln2_0 -> exp_12_6
    ln2_0 -> exp_13_6
    ln2_0 -> exp_14_7
    ln2_0 -> exp_15_7
    
    gate_0 -> exp_0_0 [style=dashed]
    gate_0 -> exp_1_0 [style=dashed]
    gate_0 -> exp_2_1 [style=dashed]
    gate_0 -> exp_3_1 [style=dashed]
    gate_0 -> exp_4_2 [style=dashed]
    gate_0 -> exp_5_2 [style=dashed]
    gate_0 -> exp_6_3 [style=dashed]
    gate_0 -> exp_7_3 [style=dashed]
    gate_0 -> exp_8_4 [style=dashed]
    gate_0 -> exp_9_4 [style=dashed]
    gate_0 -> exp_10_5 [style=dashed]
    gate_0 -> exp_11_5 [style=dashed]
    gate_0 -> exp_12_6 [style=dashed]
    gate_0 -> exp_13_6 [style=dashed]
    gate_0 -> exp_14_7 [style=dashed]
    gate_0 -> exp_15_7 [style=dashed]
    
    exp_0_0 -> expert_agg_0
    exp_1_0 -> expert_agg_0
    exp_2_1 -> expert_agg_0
    exp_3_1 -> expert_agg_0
    exp_4_2 -> expert_agg_0
    exp_5_2 -> expert_agg_0
    exp_6_3 -> expert_agg_0
    exp_7_3 -> expert_agg_0
    exp_8_4 -> expert_agg_0
    exp_9_4 -> expert_agg_0
    exp_10_5 -> expert_agg_0
    exp_11_5 -> expert_agg_0
    exp_12_6 -> expert_agg_0
    exp_13_6 -> expert_agg_0
    exp_14_7 -> expert_agg_0
    exp_15_7 -> expert_agg_0
    
    expert_agg_0 -> residual2_0
    residual1_0 -> residual2_0
    residual2_0 -> ln1_1
    
    // Layer 1 in Stage 0
    ln1_1 -> qkv_0_0
    ln1_1 -> qkv_0_1
    ln1_1 -> qkv_0_2
    ln1_1 -> qkv_0_3
    ln1_1 -> qkv_0_4
    ln1_1 -> qkv_0_5
    ln1_1 -> qkv_0_6
    ln1_1 -> qkv_0_7
    
    qkv_0_0 -> attention_1
    qkv_0_1 -> attention_1
    qkv_0_2 -> attention_1
    qkv_0_3 -> attention_1
    qkv_0_4 -> attention_1
    qkv_0_5 -> attention_1
    qkv_0_6 -> attention_1
    qkv_0_7 -> attention_1
    
    attention_1 -> out_proj_1
    out_proj_1 -> residual1_1
    residual2_0 -> residual1_1
    residual1_1 -> ln2_1
    
    ln2_1 -> gate_1
    ln2_1 -> exp_0_0
    ln2_1 -> exp_1_0
    ln2_1 -> exp_2_1
    ln2_1 -> exp_3_1
    ln2_1 -> exp_4_2
    ln2_1 -> exp_5_2
    ln2_1 -> exp_6_3
    ln2_1 -> exp_7_3
    ln2_1 -> exp_8_4
    ln2_1 -> exp_9_4
    ln2_1 -> exp_10_5
    ln2_1 -> exp_11_5
    ln2_1 -> exp_12_6
    ln2_1 -> exp_13_6
    ln2_1 -> exp_14_7
    ln2_1 -> exp_15_7
    
    gate_1 -> exp_0_0 [style=dashed]
    gate_1 -> exp_1_0 [style=dashed]
    gate_1 -> exp_2_1 [style=dashed]
    gate_1 -> exp_3_1 [style=dashed]
    gate_1 -> exp_4_2 [style=dashed]
    gate_1 -> exp_5_2 [style=dashed]
    gate_1 -> exp_6_3 [style=dashed]
    gate_1 -> exp_7_3 [style=dashed]
    gate_1 -> exp_8_4 [style=dashed]
    gate_1 -> exp_9_4 [style=dashed]
    gate_1 -> exp_10_5 [style=dashed]
    gate_1 -> exp_11_5 [style=dashed]
    gate_1 -> exp_12_6 [style=dashed]
    gate_1 -> exp_13_6 [style=dashed]
    gate_1 -> exp_14_7 [style=dashed]
    gate_1 -> exp_15_7 [style=dashed]
    
    expert_agg_1 -> residual2_1
    residual1_1 -> residual2_1
    residual2_1 -> pipeline_comm
    
    // Stage 1 flow
    pipeline_comm -> ln1_2
    ln1_2 -> qkv_2_8
    ln1_2 -> qkv_2_9
    ln1_2 -> qkv_2_10
    ln1_2 -> qkv_2_11
    ln1_2 -> qkv_2_12
    ln1_2 -> qkv_2_13
    ln1_2 -> qkv_2_14
    ln1_2 -> qkv_2_15
    
    qkv_2_8 -> attention_2
    qkv_2_9 -> attention_2
    qkv_2_10 -> attention_2
    qkv_2_11 -> attention_2
    qkv_2_12 -> attention_2
    qkv_2_13 -> attention_2
    qkv_2_14 -> attention_2
    qkv_2_15 -> attention_2
    
    attention_2 -> out_proj_2
    out_proj_2 -> residual1_2
    pipeline_comm -> residual1_2
    residual1_2 -> ln2_2
    
    ln2_2 -> gate_2
    ln2_2 -> exp_0_0
    ln2_2 -> exp_1_0
    ln2_2 -> exp_2_1
    ln2_2 -> exp_3_1
    ln2_2 -> exp_4_2
    ln2_2 -> exp_5_2
    ln2_2 -> exp_6_3
    ln2_2 -> exp_7_3
    ln2_2 -> exp_8_4
    ln2_2 -> exp_9_4
    ln2_2 -> exp_10_5
    ln2_2 -> exp_11_5
    ln2_2 -> exp_12_6
    ln2_2 -> exp_13_6
    ln2_2 -> exp_14_7
    ln2_2 -> exp_15_7
    
    gate_2 -> exp_0_0 [style=dashed]
    gate_2 -> exp_1_0 [style=dashed]
    gate_2 -> exp_2_1 [style=dashed]
    gate_2 -> exp_3_1 [style=dashed]
    gate_2 -> exp_4_2 [style=dashed]
    gate_2 -> exp_5_2 [style=dashed]
    gate_2 -> exp_6_3 [style=dashed]
    gate_2 -> exp_7_3 [style=dashed]
    gate_2 -> exp_8_4 [style=dashed]
    gate_2 -> exp_9_4 [style=dashed]
    gate_2 -> exp_10_5 [style=dashed]
    gate_2 -> exp_11_5 [style=dashed]
    gate_2 -> exp_12_6 [style=dashed]
    gate_2 -> exp_13_6 [style=dashed]
    gate_2 -> exp_14_7 [style=dashed]
    gate_2 -> exp_15_7 [style=dashed]
    
    expert_agg_2 -> residual2_2
    residual1_2 -> residual2_2
    residual2_2 -> ln1_3
    
    ln1_3 -> qkv_2_8
    ln1_3 -> qkv_2_9
    ln1_3 -> qkv_2_10
    ln1_3 -> qkv_2_11
    ln1_3 -> qkv_2_12
    ln1_3 -> qkv_2_13
    ln1_3 -> qkv_2_14
    ln1_3 -> qkv_2_15
    
    qkv_2_8 -> attention_3
    qkv_2_9 -> attention_3
    qkv_2_10 -> attention_3
    qkv_2_11 -> attention_3
    qkv_2_12 -> attention_3
    qkv_2_13 -> attention_3
    qkv_2_14 -> attention_3
    qkv_2_15 -> attention_3
    
    attention_3 -> out_proj_3
    out_proj_3 -> residual1_3
    residual2_2 -> residual1_3
    residual1_3 -> ln2_3
    
    ln2_3 -> gate_3
    ln2_3 -> exp_0_0
    ln2_3 -> exp_1_0
    ln2_3 -> exp_2_1
    ln2_3 -> exp_3_1
    ln2