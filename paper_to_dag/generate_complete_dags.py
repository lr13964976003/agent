#!/usr/bin/env python3

import os
import subprocess

def generate_corrected_baseline_dag():
    """Generate corrected baseline DAG with TP8-EP4-PP2 strategy"""
    
    dot_content = '''digraph Corrected_Baseline_TP8_EP4_PP2 {
    rankdir=TB
    splines=ortho
    graph [fontname=Arial, fontsize=12]
    
    // Node styles
    node [shape=rectangle, style=filled, fillcolor=lightblue, fontname=Arial, fontsize=10]
    
    // Input/Output
    input [label="Input\\n[batch_size=128, seq_len=1024, d_model=1024]\\nGPU: All", fillcolor=lightgreen, shape=ellipse]
    output [label="Output\\n[batch_size=128, seq_len=1024, d_model=1024]\\nGPU: All", fillcolor=lightgreen, shape=ellipse]
    
    // Communication nodes
    comm_tp_allreduce [label="TP All-Reduce\\nGPU: 0-7, 8-15, 16-23, 24-31, 32-39, 40-47, 48-55, 56-63", fillcolor=lightgray, shape=ellipse]
    comm_ep_all2all [label="EP All-to-All\\nGPU: Per TP group", fillcolor=lightgray, shape=ellipse]
    comm_pp_stage0_1 [label="PP Stage 0→1\\nGPU: 31→32", fillcolor=orange, shape=diamond]
    
    // Stage 0: Layers 0-7 (GPUs 0-31)
    // TP Group 0: GPUs 0-7, TP Group 1: GPUs 8-15, TP Group 2: GPUs 16-23, TP Group 3: GPUs 24-31
    
    // Layer 0 - Stage 0
    layer_0_input_split [label="Input Split\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=128]\\nGPU: Per TP group", fillcolor=yellow, shape=parallelogram]
    
    // QKV Linear with TP=8 (column parallel)
    layer_0_qkv_split [label="QKV Split\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=128]\\nGPU: 0-7 (TP=8)", fillcolor=yellow, shape=parallelogram]
    layer_0_qkv_gpu0 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 0", fillcolor=lightcoral]
    layer_0_qkv_gpu1 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 1", fillcolor=lightcoral]
    layer_0_qkv_gpu2 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 2", fillcolor=lightcoral]
    layer_0_qkv_gpu3 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 3", fillcolor=lightcoral]
    layer_0_qkv_gpu4 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 4", fillcolor=lightcoral]
    layer_0_qkv_gpu5 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 5", fillcolor=lightcoral]
    layer_0_qkv_gpu6 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 6", fillcolor=lightcoral]
    layer_0_qkv_gpu7 [label="QKV Linear\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 7", fillcolor=lightcoral]
    
    // Multi-head attention computation
    layer_0_attn_gpu0 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 0", fillcolor=lightcoral]
    layer_0_attn_gpu1 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 1", fillcolor=lightcoral]
    layer_0_attn_gpu2 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 2", fillcolor=lightcoral]
    layer_0_attn_gpu3 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 3", fillcolor=lightcoral]
    layer_0_attn_gpu4 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 4", fillcolor=lightcoral]
    layer_0_attn_gpu5 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 5", fillcolor=lightcoral]
    layer_0_attn_gpu6 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 6", fillcolor=lightcoral]
    layer_0_attn_gpu7 [label="Attention Compute\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 7", fillcolor=lightcoral]
    
    // Attention output projection with TP=8 (row parallel)
    layer_0_proj_gpu0 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 0", fillcolor=lightcoral]
    layer_0_proj_gpu1 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 1", fillcolor=lightcoral]
    layer_0_proj_gpu2 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 2", fillcolor=lightcoral]
    layer_0_proj_gpu3 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 3", fillcolor=lightcoral]
    layer_0_proj_gpu4 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 4", fillcolor=lightcoral]
    layer_0_proj_gpu5 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 5", fillcolor=lightcoral]
    layer_0_proj_gpu6 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 6", fillcolor=lightcoral]
    layer_0_proj_gpu7 [label="Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 7", fillcolor=lightcoral]
    
    // Attention all-reduce and residual connection
    layer_0_attn_allreduce [label="Attention All-Reduce\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_add_norm1 [label="Add+Residual+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightblue]
    
    // MoE Layer with EP=4 (64 experts total, 16 per GPU within EP group)
    layer_0_gate [label="MoE Gate\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,topk=2]\\nGPU: 0-7", fillcolor=yellow, shape=parallelogram]
    
    // Expert distribution across EP groups (4 groups of 8 GPUs each)
    // EP Group 0: GPUs 0-7, EP Group 1: GPUs 8-15, EP Group 2: GPUs 16-23, EP Group 3: GPUs 24-31
    
    layer_0_ep0_scatter [label="EP Scatter\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_ep1_scatter [label="EP Scatter\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 8-15", fillcolor=lightgray, shape=ellipse]
    layer_0_ep2_scatter [label="EP Scatter\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 16-23", fillcolor=lightgray, shape=ellipse]
    layer_0_ep3_scatter [label="EP Scatter\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 24-31", fillcolor=lightgray, shape=ellipse]
    
    // Experts (16 per EP group, 64 total)
    // EP Group 0 (GPUs 0-7): 16 experts, 2 per GPU
    layer_0_exp0_gpu0 [label="Expert 0\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0", fillcolor=lightgreen]
    layer_0_exp1_gpu0 [label="Expert 1\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0", fillcolor=lightgreen]
    layer_0_exp2_gpu1 [label="Expert 2\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 1", fillcolor=lightgreen]
    layer_0_exp3_gpu1 [label="Expert 3\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 1", fillcolor=lightgreen]
    layer_0_exp4_gpu2 [label="Expert 4\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 2", fillcolor=lightgreen]
    layer_0_exp5_gpu2 [label="Expert 5\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 2", fillcolor=lightgreen]
    layer_0_exp6_gpu3 [label="Expert 6\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 3", fillcolor=lightgreen]
    layer_0_exp7_gpu3 [label="Expert 7\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 3", fillcolor=lightgreen]
    layer_0_exp8_gpu4 [label="Expert 8\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 4", fillcolor=lightgreen]
    layer_0_exp9_gpu4 [label="Expert 9\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 4", fillcolor=lightgreen]
    layer_0_exp10_gpu5 [label="Expert 10\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 5", fillcolor=lightgreen]
    layer_0_exp11_gpu5 [label="Expert 11\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 5", fillcolor=lightgreen]
    layer_0_exp12_gpu6 [label="Expert 12\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 6", fillcolor=lightgreen]
    layer_0_exp13_gpu6 [label="Expert 13\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 6", fillcolor=lightgreen]
    layer_0_exp14_gpu7 [label="Expert 14\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 7", fillcolor=lightgreen]
    layer_0_exp15_gpu7 [label="Expert 15\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 7", fillcolor=lightgreen]
    
    layer_0_ep0_gather [label="EP Gather\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_ep1_gather [label="EP Gather\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 8-15", fillcolor=lightgray, shape=ellipse]
    layer_0_ep2_gather [label="EP Gather\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 16-23", fillcolor=lightgray, shape=ellipse]
    layer_0_ep3_gather [label="EP Gather\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 24-31", fillcolor=lightgray, shape=ellipse]
    
    layer_0_moe_agg [label="MoE Aggregation\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-31", fillcolor=yellow, shape=parallelogram]
    layer_0_add_norm2 [label="Add+Residual+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-31", fillcolor=lightblue]
    
    // Connections for Layer 0
    input -> layer_0_input_split
    
    // QKV computation
    layer_0_input_split -> layer_0_qkv_split
    layer_0_qkv_split -> layer_0_qkv_gpu0
    layer_0_qkv_split -> layer_0_qkv_gpu1
    layer_0_qkv_split -> layer_0_qkv_gpu2
    layer_0_qkv_split -> layer_0_qkv_gpu3
    layer_0_qkv_split -> layer_0_qkv_gpu4
    layer_0_qkv_split -> layer_0_qkv_gpu5
    layer_0_qkv_split -> layer_0_qkv_gpu6
    layer_0_qkv_split -> layer_0_qkv_gpu7
    
    layer_0_qkv_gpu0 -> layer_0_attn_gpu0
    layer_0_qkv_gpu1 -> layer_0_attn_gpu1
    layer_0_qkv_gpu2 -> layer_0_attn_gpu2
    layer_0_qkv_gpu3 -> layer_0_attn_gpu3
    layer_0_qkv_gpu4 -> layer_0_attn_gpu4
    layer_0_qkv_gpu5 -> layer_0_attn_gpu5
    layer_0_qkv_gpu6 -> layer_0_attn_gpu6
    layer_0_qkv_gpu7 -> layer_0_attn_gpu7
    
    layer_0_attn_gpu0 -> layer_0_proj_gpu0
    layer_0_attn_gpu1 -> layer_0_proj_gpu1
    layer_0_attn_gpu2 -> layer_0_proj_gpu2
    layer_0_attn_gpu3 -> layer_0_proj_gpu3
    layer_0_attn_gpu4 -> layer_0_proj_gpu4
    layer_0_attn_gpu5 -> layer_0_proj_gpu5
    layer_0_attn_gpu6 -> layer_0_proj_gpu6
    layer_0_attn_gpu7 -> layer_0_proj_gpu7
    
    layer_0_proj_gpu0 -> layer_0_attn_allreduce
    layer_0_proj_gpu1 -> layer_0_attn_allreduce
    layer_0_proj_gpu2 -> layer_0_attn_allreduce
    layer_0_proj_gpu3 -> layer_0_attn_allreduce
    layer_0_proj_gpu4 -> layer_0_attn_allreduce
    layer_0_proj_gpu5 -> layer_0_attn_allreduce
    layer_0_proj_gpu6 -> layer_0_attn_allreduce
    layer_0_proj_gpu7 -> layer_0_attn_allreduce
    
    layer_0_attn_allreduce -> layer_0_add_norm1
    input -> layer_0_add_norm1 [style=dashed, label="residual"]
    
    // MoE computation
    layer_0_add_norm1 -> layer_0_gate
    layer_0_add_norm1 -> layer_0_ep0_scatter
    layer_0_gate -> layer_0_ep0_scatter [style=dashed, label="routing"]
    
    // Expert routing and computation
    layer_0_ep0_scatter -> layer_0_exp0_gpu0
    layer_0_ep0_scatter -> layer_0_exp1_gpu0
    layer_0_ep0_scatter -> layer_0_exp2_gpu1
    layer_0_ep0_scatter -> layer_0_exp3_gpu1
    layer_0_ep0_scatter -> layer_0_exp4_gpu2
    layer_0_ep0_scatter -> layer_0_exp5_gpu2
    layer_0_ep0_scatter -> layer_0_exp6_gpu3
    layer_0_ep0_scatter -> layer_0_exp7_gpu3
    layer_0_ep0_scatter -> layer_0_exp8_gpu4
    layer_0_ep0_scatter -> layer_0_exp9_gpu4
    layer_0_ep0_scatter -> layer_0_exp10_gpu5
    layer_0_ep0_scatter -> layer_0_exp11_gpu5
    layer_0_ep0_scatter -> layer_0_exp12_gpu6
    layer_0_ep0_scatter -> layer_0_exp13_gpu6
    layer_0_ep0_scatter -> layer_0_exp14_gpu7
    layer_0_ep0_scatter -> layer_0_exp15_gpu7
    
    layer_0_exp0_gpu0 -> layer_0_ep0_gather
    layer_0_exp1_gpu0 -> layer_0_ep0_gather
    layer_0_exp2_gpu1 -> layer_0_ep0_gather
    layer_0_exp3_gpu1 -> layer_0_ep0_gather
    layer_0_exp4_gpu2 -> layer_0_ep0_gather
    layer_0_exp5_gpu2 -> layer_0_ep0_gather
    layer_0_exp6_gpu3 -> layer_0_ep0_gather
    layer_0_exp7_gpu3 -> layer_0_ep0_gather
    layer_0_exp8_gpu4 -> layer_0_ep0_gather
    layer_0_exp9_gpu4 -> layer_0_ep0_gather
    layer_0_exp10_gpu5 -> layer_0_ep0_gather
    layer_0_exp11_gpu5 -> layer_0_ep0_gather
    layer_0_exp12_gpu6 -> layer_0_ep0_gather
    layer_0_exp13_gpu6 -> layer_0_ep0_gather
    layer_0_exp14_gpu7 -> layer_0_ep0_gather
    layer_0_exp15_gpu7 -> layer_0_ep0_gather
    
    layer_0_ep0_gather -> layer_0_moe_agg
    layer_0_ep1_gather -> layer_0_moe_agg
    layer_0_ep2_gather -> layer_0_moe_agg
    layer_0_ep3_gather -> layer_0_moe_agg
    
    layer_0_moe_agg -> layer_0_add_norm2
    layer_0_add_norm1 -> layer_0_add_norm2 [style=dashed, label="residual"]
    
    // Simplified representation for other layers (showing key differences)
    // Layer 7 (last layer in stage 0)
    layer_7_add_norm2 [label="Layer 7 Add+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-31", fillcolor=lightblue]
    
    // Stage 1: Layers 8-15 (GPUs 32-63)
    layer_8_input_split [label="Stage 1 Input Split\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=128]\\nGPU: 32-39,40-47,48-55,56-63", fillcolor=yellow, shape=parallelogram]
    layer_15_add_norm2 [label="Layer 15 Add+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 32-63", fillcolor=lightblue]
    
    // Connections
    layer_0_add_norm2 -> layer_7_add_norm2 [style=dotted, label="layers 1-6"]
    layer_7_add_norm2 -> comm_pp_stage0_1
    comm_pp_stage0_1 -> layer_8_input_split
    layer_8_input_split -> layer_15_add_norm2 [style=dotted, label="layers 8-14"]
    layer_15_add_norm2 -> output
}
'''
    
    return dot_content

def generate_corrected_proposed_dag():
    """Generate corrected proposed DAG with optimized expert distribution"""
    
    dot_content = '''digraph Corrected_Proposed_TP8_EP4_PP2 {
    rankdir=TB
    splines=ortho
    graph [fontname=Arial, fontsize=12]
    
    // Node styles
    node [shape=rectangle, style=filled, fillcolor=lightblue, fontname=Arial, fontsize=10]
    
    // Input/Output
    input [label="Input\\n[batch_size=128, seq_len=1024, d_model=1024]\\nGPU: All", fillcolor=lightgreen, shape=ellipse]
    output [label="Output\\n[batch_size=128, seq_len=1024, d_model=1024]\\nGPU: All", fillcolor=lightgreen, shape=ellipse]
    
    // Communication nodes
    comm_tp_allreduce [label="TP All-Reduce\\nOptimized ring algorithm\\nGPU: Per TP group", fillcolor=lightgray, shape=ellipse]
    comm_ep_hierarchical [label="EP Hierarchical All-to-All\\nNVLink + InfiniBand\\nGPU: Per EP group", fillcolor=lightgray, shape=ellipse]
    comm_pp_async [label="PP Async Communication\\nDouble buffered\\nGPU: 31→32", fillcolor=orange, shape=diamond]
    
    // Optimized Stage 0: Layers 0-7 with load balancing
    
    // Layer 0 - Optimized attention with better tensor splits
    layer_0_input_opt [label="Optimized Input Split\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=128]\\nGPU: Per TP group", fillcolor=yellow, shape=parallelogram]
    
    // Optimized QKV with fused operations
    layer_0_qkv_fused_gpu0 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 0", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu1 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 1", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu2 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 2", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu3 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 3", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu4 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 4", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu5 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 5", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu6 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 6", fillcolor=lightgreen]
    layer_0_qkv_fused_gpu7 [label="Fused QKV\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 7", fillcolor=lightgreen]
    
    // Optimized attention with flash attention
    layer_0_flash_attn_gpu0 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 0", fillcolor=lightgreen]
    layer_0_flash_attn_gpu1 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 1", fillcolor=lightgreen]
    layer_0_flash_attn_gpu2 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 2", fillcolor=lightgreen]
    layer_0_flash_attn_gpu3 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 3", fillcolor=lightgreen]
    layer_0_flash_attn_gpu4 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 4", fillcolor=lightgreen]
    layer_0_flash_attn_gpu5 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 5", fillcolor=lightgreen]
    layer_0_flash_attn_gpu6 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 6", fillcolor=lightgreen]
    layer_0_flash_attn_gpu7 [label="Flash Attention\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,heads=2,d_k=64]\\nGPU: 7", fillcolor=lightgreen]
    
    // Optimized projection
    layer_0_proj_opt_gpu0 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 0", fillcolor=lightgreen]
    layer_0_proj_opt_gpu1 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 1", fillcolor=lightgreen]
    layer_0_proj_opt_gpu2 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 2", fillcolor=lightgreen]
    layer_0_proj_opt_gpu3 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 3", fillcolor=lightgreen]
    layer_0_proj_opt_gpu4 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 4", fillcolor=lightgreen]
    layer_0_proj_opt_gpu5 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 5", fillcolor=lightgreen]
    layer_0_proj_opt_gpu6 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 6", fillcolor=lightgreen]
    layer_0_proj_opt_gpu7 [label="Optimized Projection\\n[batch=128,seq=1024,heads=2,d_k=64]→[batch=128,seq=1024,d_model=128]\\nGPU: 7", fillcolor=lightgreen]
    
    // Optimized all-reduce with ring algorithm
    layer_0_attn_allreduce_opt [label="Optimized All-Reduce\\nRing algorithm, 2ms latency\\n[batch=128,seq=1024,d_model=128]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_add_norm1_opt [label="Fused Add+Residual+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgreen]
    
    // Optimized MoE with load balancing
    layer_0_gate_opt [label="Load-Balanced Gate\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,topk=2]\\nGPU: 0-7", fillcolor=yellow, shape=parallelogram]
    
    // Optimized expert distribution with hierarchical communication
    layer_0_ep0_scatter_opt [label="Hierarchical EP Scatter\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_ep1_scatter_opt [label="Hierarchical EP Scatter\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 8-15", fillcolor=lightgray, shape=ellipse]
    layer_0_ep2_scatter_opt [label="Hierarchical EP Scatter\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 16-23", fillcolor=lightgray, shape=ellipse]
    layer_0_ep3_scatter_opt [label="Hierarchical EP Scatter\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 24-31", fillcolor=lightgray, shape=ellipse]
    
    // Optimized experts with expert parallelism
    layer_0_exp0_opt_gpu0 [label="Expert 0 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0", fillcolor=lightcoral]
    layer_0_exp1_opt_gpu0 [label="Expert 1 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0", fillcolor=lightcoral]
    layer_0_exp2_opt_gpu1 [label="Expert 2 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 1", fillcolor=lightcoral]
    layer_0_exp3_opt_gpu1 [label="Expert 3 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 1", fillcolor=lightcoral]
    layer_0_exp4_opt_gpu2 [label="Expert 4 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 2", fillcolor=lightcoral]
    layer_0_exp5_opt_gpu2 [label="Expert 5 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 2", fillcolor=lightcoral]
    layer_0_exp6_opt_gpu3 [label="Expert 6 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 3", fillcolor=lightcoral]
    layer_0_exp7_opt_gpu3 [label="Expert 7 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 3", fillcolor=lightcoral]
    layer_0_exp8_opt_gpu4 [label="Expert 8 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 4", fillcolor=lightcoral]
    layer_0_exp9_opt_gpu4 [label="Expert 9 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 4", fillcolor=lightcoral]
    layer_0_exp10_opt_gpu5 [label="Expert 10 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 5", fillcolor=lightcoral]
    layer_0_exp11_opt_gpu5 [label="Expert 11 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 5", fillcolor=lightcoral]
    layer_0_exp12_opt_gpu6 [label="Expert 12 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 6", fillcolor=lightcoral]
    layer_0_exp13_opt_gpu6 [label="Expert 13 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 6", fillcolor=lightcoral]
    layer_0_exp14_opt_gpu7 [label="Expert 14 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 7", fillcolor=lightcoral]
    layer_0_exp15_opt_gpu7 [label="Expert 15 (Optimized)\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 7", fillcolor=lightcoral]
    
    layer_0_ep0_gather_opt [label="Hierarchical EP Gather\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-7", fillcolor=lightgray, shape=ellipse]
    layer_0_ep1_gather_opt [label="Hierarchical EP Gather\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 8-15", fillcolor=lightgray, shape=ellipse]
    layer_0_ep2_gather_opt [label="Hierarchical EP Gather\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 16-23", fillcolor=lightgray, shape=ellipse]
    layer_0_ep3_gather_opt [label="Hierarchical EP Gather\\nNVLink优先\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 24-31", fillcolor=lightgray, shape=ellipse]
    
    layer_0_moe_agg_opt [label="Optimized MoE Aggregation\\nLoad balanced\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-31", fillcolor=yellow, shape=parallelogram]
    layer_0_add_norm2_opt [label="Fused Add+Residual+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 0-31", fillcolor=lightgreen]
    
    // Optimized Stage 1 representation
    layer_15_add_norm2_opt [label="Optimized Layer 15\\nAdd+Residual+Norm\\n[batch=128,seq=1024,d_model=1024]→[batch=128,seq=1024,d_model=1024]\\nGPU: 32-63", fillcolor=lightgreen]
    
    // Connections for optimized version
    input -> layer_0_input_opt
    
    // Optimized attention path
    layer_0_input_opt -> layer_0_qkv_fused_gpu0
    layer_0_input_opt -> layer_0_qkv_fused_gpu1
    layer_0_input_opt -> layer_0_qkv_fused_gpu2
    layer_0_input_opt -> layer_0_qkv_fused_gpu3
    layer_0_input_opt -> layer_0_qkv_fused_gpu4
    layer_0_input_opt -> layer_0_qkv_fused_gpu5
    layer_0_input_opt -> layer_0_qkv_fused_gpu6
    layer_0_input_opt -> layer_0_qkv_fused_gpu7
    
    layer_0_qkv_fused_gpu0 -> layer_0_flash_attn_gpu0
    layer_0_qkv_fused_gpu1 -> layer_0_flash_attn_gpu1
    layer_0_qkv_fused_gpu2 -> layer_0_flash_attn_gpu2
    layer_0_qkv_fused_gpu3 -> layer_0_flash_attn_gpu3
    layer_0_qkv_fused_gpu4 -> layer_0_flash_attn_gpu4
    layer_0_qkv_fused_gpu5 -> layer_0_flash_attn_gpu5
    layer_0_qkv_fused_gpu6 -> layer_0_flash_attn_gpu6
    layer_0_qkv_fused_gpu7 -> layer_0_flash_attn_gpu7
    
    layer_0_flash_attn_gpu0 -> layer_0_proj_opt_gpu0
    layer_0_flash_attn_gpu1 -> layer_0_proj_opt_gpu1
    layer_0_flash_attn_gpu2 -> layer_0_proj_opt_gpu2
    layer_0_flash_attn_gpu3 -> layer_0_proj_opt_gpu3
    layer_0_flash_attn_gpu4 -> layer_0_proj_opt_gpu4
    layer_0_flash_attn_gpu5 -> layer_0_proj_opt_gpu5
    layer_0_flash_attn_gpu6 -> layer_0_proj_opt_gpu6
    layer_0_flash_attn_gpu7 -> layer_0_proj_opt_gpu7
    
    layer_0_proj_opt_gpu0 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu1 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu2 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu3 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu4 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu5 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu6 -> layer_0_attn_allreduce_opt
    layer_0_proj_opt_gpu7 -> layer_0_attn_allreduce_opt
    
    layer_0_attn_allreduce_opt -> layer_0_add_norm1_opt
    input -> layer_0_add_norm1_opt [style=dashed, label="residual"]
    
    // Optimized MoE path
    layer_0_add_norm1_opt -> layer_0_gate_opt
    layer_0_add_norm1_opt -> layer_0_ep0_scatter_opt
    layer_0_gate_opt -> layer_0_ep0_scatter_opt [style=dashed, label="routing"]
    
    layer_0_ep0_scatter_opt -> layer_0_exp0_opt_gpu0
    layer_0_ep0_scatter_opt -> layer_0_exp1_opt_gpu0
    layer_0_ep0_scatter_opt -> layer_0_exp2_opt_gpu1
    layer_0_ep0_scatter_opt -> layer_0_exp3_opt_gpu1
    layer_0_ep0_scatter_opt -> layer_0_exp4_opt_gpu2
    layer_0_ep0_scatter_opt -> layer_0_exp5_opt_gpu2
    layer_0_ep0_scatter_opt -> layer_0_exp6_opt_gpu3
    layer_0_ep0_scatter_opt -> layer_0_exp7_opt_gpu3
    layer_0_ep0_scatter_opt -> layer_0_exp8_opt_gpu4
    layer_0_ep0_scatter_opt -> layer_0_exp9_opt_gpu4
    layer_0_ep0_scatter_opt -> layer_0_exp10_opt_gpu5
    layer_0_ep0_scatter_opt -> layer_0_exp11_opt_gpu5
    layer_0_ep0_scatter_opt -> layer_0_exp12_opt_gpu6
    layer_0_ep0_scatter_opt -> layer_0_exp13_opt_gpu6
    layer_0_ep0_scatter_opt -> layer_0_exp14_opt_gpu7
    layer_0_ep0_scatter_opt -> layer_0_exp15_opt_gpu7
    
    layer_0_exp0_opt_gpu0 -> layer_0_ep0_gather_opt
    layer_0_exp1_opt_gpu0 -> layer_0_ep0_gather_opt
    layer_0_exp2_opt_gpu1 -> layer_0_ep0_gather_opt
    layer_0_exp3_opt_gpu1 -> layer_0_ep0_gather_opt
    layer_0_exp4_opt_gpu2 -> layer_0_ep0_gather_opt
    layer_0_exp5_opt_gpu2 -> layer_0_ep0_gather_opt
    layer_0_exp6_opt_gpu3 -> layer_0_ep0_gather_opt
    layer_0_exp7_opt_gpu3 -> layer_0_ep0_gather_opt
    layer_0_exp8_opt_gpu4 -> layer_0_ep0_gather_opt
    layer_0_exp9_opt_gpu4 -> layer_0_ep0_gather_opt
    layer_0_exp10_opt_gpu5 -> layer_0_ep0_gather_opt
    layer_0_exp11_opt_gpu5 -> layer_0_ep0_gather_opt
    layer_0_exp12_opt_gpu6 -> layer_0_ep0_gather_opt
    layer_0_exp13_opt_gpu6 -> layer_0_ep0_gather_opt
    layer_0_exp14_opt_gpu7 -> layer_0_ep0_gather_opt
    layer_0_exp15_opt_gpu7 -> layer_0_ep0_gather_opt
    
    layer_0_ep0_gather_opt -> layer_0_moe_agg_opt
    layer_0_ep1_gather_opt -> layer_0_moe_agg_opt
    layer_0_ep2_gather_opt -> layer_0_moe_agg_opt
    layer_0_ep3_gather_opt -> layer_0_moe_agg_opt
    
    layer_0_moe_agg_opt -> layer_0_add_norm2_opt
    layer_0_add_norm1_opt -> layer_0_add_norm2_opt [style=dashed, label="residual"]
    
    // Simplified path to output (representing all layers)
    layer_0_add_norm2_opt -> layer_15_add_norm2_opt [style=dotted, label="layers 1-14 optimized"]
    layer_15_add_norm2_opt -> output
}
'''
    
    return dot_content

def main():
    # Create output directory
    output_dir = "../outputs/2025-12-01-12-05-00"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate corrected baseline DAG
    baseline_dot = generate_corrected_baseline_dag()
    baseline_path = os.path.join(output_dir, "corrected_baseline_tp8_ep4_pp2.dot")
    
    with open(baseline_path, 'w') as f:
        f.write(baseline_dot)
    
    # Generate corrected proposed DAG
    proposed_dot = generate_corrected_proposed_dag()
    proposed_path = os.path.join(output_dir, "corrected_proposed_tp8_ep4_pp2.dot")
    
    with open(proposed_path, 'w') as f:
        f.write(proposed_dot)
    
    # Generate SVG images using dot command
    try:
        # Generate baseline SVG
        baseline_svg = os.path.join(output_dir, "corrected_baseline_tp8_ep4_pp2.svg")
        subprocess.run(['dot', '-Tsvg', baseline_path, '-o', baseline_svg], check=True)
        
        # Generate proposed SVG
        proposed_svg = os.path.join(output_dir, "corrected_proposed_tp8_ep4_pp2.svg")
        subprocess.run(['dot', '-Tsvg', proposed_path, '-o', proposed_svg], check=True)
        
        print(f"Successfully generated DAGs:")
        print(f"- Baseline: {baseline_path}")
        print(f"- Baseline SVG: {baseline_svg}")
        print(f"- Proposed: {proposed_path}")
        print(f"- Proposed SVG: {proposed_svg}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG images: {e}")
        print("DOT files were created successfully, but SVG generation failed.")
    
    # Return paths for JSON output
    return {
        "baseline_dot": baseline_path,
        "baseline_svg": baseline_svg if 'baseline_svg' in locals() else None,
        "proposed_dot": proposed_path,
        "proposed_svg": proposed_svg if 'proposed_svg' in locals() else None
    }

if __name__ == "__main__":
    result = main()
    print("\nGenerated files:")
    for key, path in result.items():
        if path:
            print(f"{key}: {path}")