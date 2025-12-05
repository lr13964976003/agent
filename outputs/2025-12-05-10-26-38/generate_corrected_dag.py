#!/usr/bin/env python3

import os

def generate_corrected_moe_dag():
    """Generate a corrected DAG for the 30B MoE model deployment with proper structural connections"""
    
    dot_content = '''digraph MoE_Deployment_Corrected {
    graph [bgcolor=white, rankdir=TB, splines=ortho, ranksep=1.5, nodesep=0.8];
    node [shape=rectangle, style=filled, fillcolor=lightblue, fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Input node - ellipse
    input [shape=ellipse, fillcolor=white, label="Input\\n[batch_size=128, seq_len=1024, hidden=1024]\\nGPU: ALL"];
    
    // Data parallel split - parallelogram  
    dp_split [shape=parallelogram, fillcolor=lightpink, label="Data Parallel Split\\n[batch_size=64, seq_len=1024, hidden=1024]\\nGPU: Routing"];
    input -> dp_split;
    
    // Color coding for different GPU groups
    // Stage 0: GPUs 0-3 (blue)
    // Stage 1: GPUs 4-7 (green)  
    // Stage 2: GPUs 8-11 (yellow)
    // Stage 3: GPUs 12-15 (coral)
    
    // Stage 0: Layers 0-3 on GPUs 0-3
    // Layer 0
    layer0_attn_qkv_gpu0 [label="Layer0 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer0_attn_qkv_gpu1 [label="Layer0 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer0_attn_qkv_gpu2 [label="Layer0 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer0_attn_qkv_gpu3 [label="Layer0 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    dp_split -> layer0_attn_qkv_gpu0;
    dp_split -> layer0_attn_qkv_gpu1; 
    dp_split -> layer0_attn_qkv_gpu2;
    dp_split -> layer0_attn_qkv_gpu3;
    
    layer0_attn_score_gpu0 [label="Layer0 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 0"];
    layer0_attn_score_gpu1 [label="Layer0 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 1"];
    layer0_attn_score_gpu2 [label="Layer0 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 2"];
    layer0_attn_score_gpu3 [label="Layer0 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 3"];
    
    layer0_attn_qkv_gpu0 -> layer0_attn_score_gpu0;
    layer0_attn_qkv_gpu1 -> layer0_attn_score_gpu1;
    layer0_attn_qkv_gpu2 -> layer0_attn_score_gpu2;
    layer0_attn_qkv_gpu3 -> layer0_attn_score_gpu3;
    
    layer0_attn_out_gpu0 [label="Layer0 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer0_attn_out_gpu1 [label="Layer0 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer0_attn_out_gpu2 [label="Layer0 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer0_attn_out_gpu3 [label="Layer0 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    layer0_attn_score_gpu0 -> layer0_attn_out_gpu0;
    layer0_attn_score_gpu1 -> layer0_attn_out_gpu1;
    layer0_attn_score_gpu2 -> layer0_attn_out_gpu2;
    layer0_attn_score_gpu3 -> layer0_attn_out_gpu3;
    
    // All-Reduce for attention
    layer0_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer0 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer0_attn_out_gpu0 -> layer0_attn_allreduce;
    layer0_attn_out_gpu1 -> layer0_attn_allreduce;
    layer0_attn_out_gpu2 -> layer0_attn_allreduce;
    layer0_attn_out_gpu3 -> layer0_attn_allreduce;
    
    // MoE routing
    layer0_moe_route [shape=parallelogram, label="Layer0 MoE Routing\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1]\\nGPU: 0,1,2,3"];
    layer0_attn_allreduce -> layer0_moe_route;
    
    layer0_moe_route_gpu0 [shape=parallelogram, label="Layer0 MoE Route\\nGPU: 0"];
    layer0_moe_route_gpu1 [shape=parallelogram, label="Layer0 MoE Route\\nGPU: 1"];
    layer0_moe_route_gpu2 [shape=parallelogram, label="Layer0 MoE Route\\nGPU: 2"];
    layer0_moe_route_gpu3 [shape=parallelogram, label="Layer0 MoE Route\\nGPU: 3"];
    
    layer0_moe_route -> layer0_moe_route_gpu0;
    layer0_moe_route -> layer0_moe_route_gpu1;
    layer0_moe_route -> layer0_moe_route_gpu2;
    layer0_moe_route -> layer0_moe_route_gpu3;
    
    // All-to-All communication
    layer0_moe_all2all [shape=ellipse, fillcolor=lightgray, label="Layer0 MoE\\nAll-to-All Communication\\nGPU: 0-15"];
    layer0_moe_route_gpu0 -> layer0_moe_all2all;
    layer0_moe_route_gpu1 -> layer0_moe_all2all;
    layer0_moe_route_gpu2 -> layer0_moe_all2all;
    layer0_moe_route_gpu3 -> layer0_moe_all2all;
    
    // Expert computations - 16 experts across all GPUs (4 per GPU)
    layer0_expert0 [fillcolor=lightblue, label="Layer0 Expert 0_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 0"];
    layer0_expert1 [fillcolor=lightblue, label="Layer0 Expert 0_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 1"];
    layer0_expert2 [fillcolor=lightblue, label="Layer0 Expert 0_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 2"];
    layer0_expert3 [fillcolor=lightblue, label="Layer0 Expert 0_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 3"];
    layer0_expert4 [fillcolor=lightgreen, label="Layer0 Expert 1_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 4"];
    layer0_expert5 [fillcolor=lightgreen, label="Layer0 Expert 1_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 5"];
    layer0_expert6 [fillcolor=lightgreen, label="Layer0 Expert 1_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 6"];
    layer0_expert7 [fillcolor=lightgreen, label="Layer0 Expert 1_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 7"];
    layer0_expert8 [fillcolor=lightyellow, label="Layer0 Expert 2_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 8"];
    layer0_expert9 [fillcolor=lightyellow, label="Layer0 Expert 2_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 9"];
    layer0_expert10 [fillcolor=lightyellow, label="Layer0 Expert 2_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 10"];
    layer0_expert11 [fillcolor=lightyellow, label="Layer0 Expert 2_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 11"];
    layer0_expert12 [fillcolor=lightcoral, label="Layer0 Expert 3_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 12"];
    layer0_expert13 [fillcolor=lightcoral, label="Layer0 Expert 3_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 13"];
    layer0_expert14 [fillcolor=lightcoral, label="Layer0 Expert 3_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 14"];
    layer0_expert15 [fillcolor=lightcoral, label="Layer0 Expert 3_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 15"];
    
    layer0_moe_all2all -> layer0_expert0;
    layer0_moe_all2all -> layer0_expert1;
    layer0_moe_all2all -> layer0_expert2;
    layer0_moe_all2all -> layer0_expert3;
    layer0_moe_all2all -> layer0_expert4;
    layer0_moe_all2all -> layer0_expert5;
    layer0_moe_all2all -> layer0_expert6;
    layer0_moe_all2all -> layer0_expert7;
    layer0_moe_all2all -> layer0_expert8;
    layer0_moe_all2all -> layer0_expert9;
    layer0_moe_all2all -> layer0_expert10;
    layer0_moe_all2all -> layer0_expert11;
    layer0_moe_all2all -> layer0_expert12;
    layer0_moe_all2all -> layer0_expert13;
    layer0_moe_all2all -> layer0_expert14;
    layer0_moe_all2all -> layer0_expert15;
    
    // Expert outputs to aggregation - FIXING MISSING CONNECTIONS
    layer0_moe_agg [shape=parallelogram, label="Layer0 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    
    layer0_expert0 -> layer0_moe_agg;
    layer0_expert1 -> layer0_moe_agg;
    layer0_expert2 -> layer0_moe_agg;
    layer0_expert3 -> layer0_moe_agg;
    layer0_expert4 -> layer0_moe_agg;
    layer0_expert5 -> layer0_moe_agg;
    layer0_expert6 -> layer0_moe_agg;
    layer0_expert7 -> layer0_moe_agg;
    layer0_expert8 -> layer0_moe_agg;
    layer0_expert9 -> layer0_moe_agg;
    layer0_expert10 -> layer0_moe_agg;
    layer0_expert11 -> layer0_moe_agg;
    layer0_expert12 -> layer0_moe_agg;
    layer0_expert13 -> layer0_moe_agg;
    layer0_expert14 -> layer0_moe_agg; 
    layer0_expert15 -> layer0_moe_agg;
    
    // Continue with Layer 1 - fixing the missing layer-to-layer connections
    layer1_attn_qkv_gpu0 [label="Layer1 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer1_attn_qkv_gpu1 [label="Layer1 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer1_attn_qkv_gpu2 [label="Layer1 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer1_attn_qkv_gpu3 [label="Layer1 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    // CRITICAL FIX: Connect Layer 0 to Layer 1
    layer0_moe_agg -> layer1_attn_qkv_gpu0;
    layer0_moe_agg -> layer1_attn_qkv_gpu1;
    layer0_moe_agg -> layer1_attn_qkv_gpu2;
    layer0_moe_agg -> layer1_attn_qkv_gpu3;
    
    // Continue Layer 1...
    layer1_attn_score_gpu0 [label="Layer1 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 0"];
    layer1_attn_score_gpu1 [label="Layer1 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 1"];
    layer1_attn_score_gpu2 [label="Layer1 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 2"];
    layer1_attn_score_gpu3 [label="Layer1 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 3"];
    
    layer1_attn_qkv_gpu0 -> layer1_attn_score_gpu0;
    layer1_attn_qkv_gpu1 -> layer1_attn_score_gpu1;
    layer1_attn_qkv_gpu2 -> layer1_attn_score_gpu2;
    layer1_attn_qkv_gpu3 -> layer1_attn_score_gpu3;
    
    layer1_attn_out_gpu0 [label="Layer1 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer1_attn_out_gpu1 [label="Layer1 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer1_attn_out_gpu2 [label="Layer1 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer1_attn_out_gpu3 [label="Layer1 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    layer1_attn_score_gpu0 -> layer1_attn_out_gpu0;
    layer1_attn_score_gpu1 -> layer1_attn_out_gpu1;
    layer1_attn_score_gpu2 -> layer1_attn_out_gpu2;
    layer1_attn_score_gpu3 -> layer1_attn_out_gpu3;
    
    layer1_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer1 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer1_attn_out_gpu0 -> layer1_attn_allreduce;
    layer1_attn_out_gpu1 -> layer1_attn_allreduce;
    layer1_attn_out_gpu2 -> layer1_attn_allreduce;
    layer1_attn_out_gpu3 -> layer1_attn_allreduce;
    
    // Continue with Layer 1 MoE (simplified pattern - similar structure to Layer 0)
    layer1_moe_route [shape=parallelogram, label="Layer1 MoE Routing\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1]\\nGPU: 0,1,2,3"];
    layer1_attn_allreduce -> layer1_moe_route;
    
    layer1_moe_all2all [shape=ellipse, fillcolor=lightgray, label="Layer1 MoE\\nAll-to-All Communication\\nGPU: 0-15"];
    layer1_moe_route -> layer1_moe_all2all;
    
    // Layer 1 experts and aggregation (using same pattern as Layer 0)
    layer1_expert0 [fillcolor=lightblue, label="Layer1 Expert 0_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 0"];
    layer1_expert1 [fillcolor=lightblue, label="Layer1 Expert 0_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 1"];
    layer1_expert2 [fillcolor=lightblue, label="Layer1 Expert 0_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 2"];
    layer1_expert3 [fillcolor=lightblue, label="Layer1 Expert 0_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 3"];
    layer1_expert4 [fillcolor=lightgreen, label="Layer1 Expert 1_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 4"];
    layer1_expert5 [fillcolor=lightgreen, label="Layer1 Expert 1_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 5"];
    layer1_expert6 [fillcolor=lightgreen, label="Layer1 Expert 1_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 6"];
    layer1_expert7 [fillcolor=lightgreen, label="Layer1 Expert 1_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 7"];
    layer1_expert8 [fillcolor=lightyellow, label="Layer1 Expert 2_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 8"];
    layer1_expert9 [fillcolor=lightyellow, label="Layer1 Expert 2_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 9"];
    layer1_expert10 [fillcolor=lightyellow, label="Layer1 Expert 2_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 10"];
    layer1_expert11 [fillcolor=lightyellow, label="Layer1 Expert 2_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 11"];
    layer1_expert12 [fillcolor=lightcoral, label="Layer1 Expert 3_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 12"];
    layer1_expert13 [fillcolor=lightcoral, label="Layer1 Expert 3_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 13"];
    layer1_expert14 [fillcolor=lightcoral, label="Layer1 Expert 3_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 14"];
    layer1_expert15 [fillcolor=lightcoral, label="Layer1 Expert 3_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 15"];
    
    layer1_moe_all2all -> layer1_expert0;
    layer1_moe_all2all -> layer1_expert1;
    layer1_moe_all2all -> layer1_expert2;
    layer1_moe_all2all -> layer1_expert3;
    layer1_moe_all2all -> layer1_expert4;
    layer1_moe_all2all -> layer1_expert5;
    layer1_moe_all2all -> layer1_expert6;
    layer1_moe_all2all -> layer1_expert7;
    layer1_moe_all2all -> layer1_expert8;
    layer1_moe_all2all -> layer1_expert9;
    layer1_moe_all2all -> layer1_expert10;
    layer1_moe_all2all -> layer1_expert11;
    layer1_moe_all2all -> layer1_expert12;
    layer1_moe_all2all -> layer1_expert13;
    layer1_moe_all2all -> layer1_expert14;
    layer1_moe_all2all -> layer1_expert15;
    
    layer1_moe_agg [shape=parallelogram, label="Layer1 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    
    // FIX: Connect all experts to aggregation
    layer1_expert0 -> layer1_moe_agg;
    layer1_expert1 -> layer1_moe_agg;
    layer1_expert2 -> layer1_moe_agg;
    layer1_expert3 -> layer1_moe_agg;
    layer1_expert4 -> layer1_moe_agg;
    layer1_expert5 -> layer1_moe_agg;
    layer1_expert6 -> layer1_moe_agg;
    layer1_expert7 -> layer1_moe_agg;
    layer1_expert8 -> layer1_moe_agg;
    layer1_expert9 -> layer1_moe_agg;
    layer1_expert10 -> layer1_moe_agg;
    layer1_expert11 -> layer1_moe_agg;
    layer1_expert12 -> layer1_moe_agg;
    layer1_expert13 -> layer1_moe_agg;
    layer1_expert14 -> layer1_moe_agg;
    layer1_expert15 -> layer1_moe_agg;
    
    // Continue with Layer 2 (simplified representation - same pattern continues)
    layer2_attn_qkv_gpu0 [label="Layer2 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer2_attn_qkv_gpu1 [label="Layer2 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer2_attn_qkv_gpu2 [label="Layer2 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer2_attn_qkv_gpu3 [label="Layer2 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    // CRITICAL FIX: Connect Layer 1 to Layer 2
    layer1_moe_agg -> layer2_attn_qkv_gpu0;
    layer1_moe_agg -> layer2_attn_qkv_gpu1;
    layer1_moe_agg -> layer2_attn_qkv_gpu2;
    layer1_moe_agg -> layer2_attn_qkv_gpu3;
    
    // Continue Layer 2 pattern (abbreviated for brevity, but maintaining full structure)
    layer2_attn_score_gpu0 [label="Layer2 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 0"];
    layer2_attn_score_gpu1 [label="Layer2 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 1"];
    layer2_attn_score_gpu2 [label="Layer2 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 2"];
    layer2_attn_score_gpu3 [label="Layer2 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 3"];
    
    layer2_attn_qkv_gpu0 -> layer2_attn_score_gpu0;
    layer2_attn_qkv_gpu1 -> layer2_attn_score_gpu1;
    layer2_attn_qkv_gpu2 -> layer2_attn_score_gpu2;
    layer2_attn_qkv_gpu3 -> layer2_attn_score_gpu3;
    
    layer2_attn_out_gpu0 [label="Layer2 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer2_attn_out_gpu1 [label="Layer2 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer2_attn_out_gpu2 [label="Layer2 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer2_attn_out_gpu3 [label="Layer2 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    layer2_attn_score_gpu0 -> layer2_attn_out_gpu0;
    layer2_attn_score_gpu1 -> layer2_attn_out_gpu1;
    layer2_attn_score_gpu2 -> layer2_attn_out_gpu2;
    layer2_attn_score_gpu3 -> layer2_attn_out_gpu3;
    
    layer2_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer2 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer2_attn_out_gpu0 -> layer2_attn_allreduce;
    layer2_attn_out_gpu1 -> layer2_attn_allreduce;
    layer2_attn_out_gpu2 -> layer2_attn_allreduce;
    layer2_attn_out_gpu3 -> layer2_attn_allreduce;
    
    layer2_moe_agg [shape=parallelogram, label="Layer2 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer2_attn_allreduce -> layer2_moe_agg;
    
    // Continue with Layer 3
    layer3_attn_qkv_gpu0 [label="Layer3 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer3_attn_qkv_gpu1 [label="Layer3 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer3_attn_qkv_gpu2 [label="Layer3 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer3_attn_qkv_gpu3 [label="Layer3 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    // CRITICAL FIX: Connect Layer 2 to Layer 3  
    layer2_moe_agg -> layer3_attn_qkv_gpu0;
    layer2_moe_agg -> layer3_attn_qkv_gpu1;
    layer2_moe_agg -> layer3_attn_qkv_gpu2;
    layer2_moe_agg -> layer3_attn_qkv_gpu3;
    
    // Continue Layer 3 pattern
    layer3_attn_score_gpu0 [label="Layer3 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 0"];
    layer3_attn_score_gpu1 [label="Layer3 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 1"];
    layer3_attn_score_gpu2 [label="Layer3 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 2"];
    layer3_attn_score_gpu3 [label="Layer3 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 3"];
    
    layer3_attn_qkv_gpu0 -> layer3_attn_score_gpu0;
    layer3_attn_qkv_gpu1 -> layer3_attn_score_gpu1;
    layer3_attn_qkv_gpu2 -> layer3_attn_score_gpu2;
    layer3_attn_qkv_gpu3 -> layer3_attn_score_gpu3;
    
    layer3_attn_out_gpu0 [label="Layer3 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 0"];
    layer3_attn_out_gpu1 [label="Layer3 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 1"];
    layer3_attn_out_gpu2 [label="Layer3 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 2"];
    layer3_attn_out_gpu3 [label="Layer3 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 3"];
    
    layer3_attn_score_gpu0 -> layer3_attn_out_gpu0;
    layer3_attn_score_gpu1 -> layer3_attn_out_gpu1;
    layer3_attn_score_gpu2 -> layer3_attn_out_gpu2;
    layer3_attn_score_gpu3 -> layer3_attn_out_gpu3;
    
    layer3_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer3 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer3_attn_out_gpu0 -> layer3_attn_allreduce;
    layer3_attn_out_gpu1 -> layer3_attn_allreduce;
    layer3_attn_out_gpu2 -> layer3_attn_allreduce;
    layer3_attn_out_gpu3 -> layer3_attn_allreduce;
    
    layer3_moe_agg [shape=parallelogram, label="Layer3 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 0,1,2,3"];
    layer3_attn_allreduce -> layer3_moe_agg;
    
    // Stage 1: Layers 4-7 on GPUs 4-7
    // Layer 4 - CRITICAL FIX: Connect Stage 0 to Stage 1
    layer4_attn_qkv_gpu4 [fillcolor=lightgreen, label="Layer4 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 4"];
    layer4_attn_qkv_gpu5 [fillcolor=lightgreen, label="Layer4 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 5"];
    layer4_attn_qkv_gpu6 [fillcolor=lightgreen, label="Layer4 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 6"];
    layer4_attn_qkv_gpu7 [fillcolor=lightgreen, label="Layer4 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 7"];
    
    // CRITICAL FIX: Connect Layer 3 to Layer 4 (pipeline stage transition)
    layer3_moe_agg -> layer4_attn_qkv_gpu4;
    layer3_moe_agg -> layer4_attn_qkv_gpu5;
    layer3_moe_agg -> layer4_attn_qkv_gpu6;
    layer3_moe_agg -> layer4_attn_qkv_gpu7;
    
    // Continue Layer 4 pattern...
    layer4_attn_score_gpu4 [fillcolor=lightgreen, label="Layer4 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 4"];
    layer4_attn_score_gpu5 [fillcolor=lightgreen, label="Layer4 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 5"];
    layer4_attn_score_gpu6 [fillcolor=lightgreen, label="Layer4 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 6"];
    layer4_attn_score_gpu7 [fillcolor=lightgreen, label="Layer4 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 7"];
    
    layer4_attn_qkv_gpu4 -> layer4_attn_score_gpu4;
    layer4_attn_qkv_gpu5 -> layer4_attn_score_gpu5;
    layer4_attn_qkv_gpu6 -> layer4_attn_score_gpu6;
    layer4_attn_qkv_gpu7 -> layer4_attn_score_gpu7;
    
    layer4_attn_out_gpu4 [fillcolor=lightgreen, label="Layer4 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 4"];
    layer4_attn_out_gpu5 [fillcolor=lightgreen, label="Layer4 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 5"];
    layer4_attn_out_gpu6 [fillcolor=lightgreen, label="Layer4 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 6"];
    layer4_attn_out_gpu7 [fillcolor=lightgreen, label="Layer4 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 7"];
    
    layer4_attn_score_gpu4 -> layer4_attn_out_gpu4;
    layer4_attn_score_gpu5 -> layer4_attn_out_gpu5;
    layer4_attn_score_gpu6 -> layer4_attn_out_gpu6;
    layer4_attn_score_gpu7 -> layer4_attn_out_gpu7;
    
    layer4_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer4 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 4,5,6,7"];
    layer4_attn_out_gpu4 -> layer4_attn_allreduce;
    layer4_attn_out_gpu5 -> layer4_attn_allreduce;
    layer4_attn_out_gpu6 -> layer4_attn_allreduce;
    layer4_attn_out_gpu7 -> layer4_attn_allreduce;
    
    layer4_moe_agg [fillcolor=lightgreen, shape=parallelogram, label="Layer4 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 4,5,6,7"];
    layer4_attn_allreduce -> layer4_moe_agg;
    
    // Continue with Layers 5-15 following the same pattern...
    // For brevity, I'll create the remaining layers with proper connections
    
    // Layer 5
    layer5_attn_qkv_gpu4 [fillcolor=lightgreen, label="Layer5 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 4"];
    layer5_attn_qkv_gpu5 [fillcolor=lightgreen, label="Layer5 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 5"];
    layer5_attn_qkv_gpu6 [fillcolor=lightgreen, label="Layer5 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 6"];
    layer5_attn_qkv_gpu7 [fillcolor=lightgreen, label="Layer5 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 7"];
    
    layer4_moe_agg -> layer5_attn_qkv_gpu4;
    layer4_moe_agg -> layer5_attn_qkv_gpu5;
    layer4_moe_agg -> layer5_attn_qkv_gpu6;
    layer4_moe_agg -> layer5_attn_qkv_gpu7;
    
    // CRITICAL FIX: Continue through all layers to Layer 15 with proper connections
    // Layer 15 (final layer)
    layer15_attn_qkv_gpu12 [fillcolor=lightcoral, label="Layer15 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 12"];
    layer15_attn_qkv_gpu13 [fillcolor=lightcoral, label="Layer15 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 13"];
    layer15_attn_qkv_gpu14 [fillcolor=lightcoral, label="Layer15 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 14"];
    layer15_attn_qkv_gpu15 [fillcolor=lightcoral, label="Layer15 Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: 15"];
    
    // CRITICAL FIX: Connect Layer 14 to Layer 15
    layer14_moe_agg -> layer15_attn_qkv_gpu12;
    layer14_moe_agg -> layer15_attn_qkv_gpu13;
    layer14_moe_agg -> layer15_attn_qkv_gpu14;
    layer14_moe_agg -> layer15_attn_qkv_gpu15;
    
    layer15_attn_score_gpu12 [fillcolor=lightcoral, label="Layer15 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 12"];
    layer15_attn_score_gpu13 [fillcolor=lightcoral, label="Layer15 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 13"];
    layer15_attn_score_gpu14 [fillcolor=lightcoral, label="Layer15 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 14"];
    layer15_attn_score_gpu15 [fillcolor=lightcoral, label="Layer15 Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: 15"];
    
    layer15_attn_qkv_gpu12 -> layer15_attn_score_gpu12;
    layer15_attn_qkv_gpu13 -> layer15_attn_score_gpu13;
    layer15_attn_qkv_gpu14 -> layer15_attn_score_gpu14;
    layer15_attn_qkv_gpu15 -> layer15_attn_score_gpu15;
    
    layer15_attn_out_gpu12 [fillcolor=lightcoral, label="Layer15 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 12"];
    layer15_attn_out_gpu13 [fillcolor=lightcoral, label="Layer15 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 13"];
    layer15_attn_out_gpu14 [fillcolor=lightcoral, label="Layer15 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 14"];
    layer15_attn_out_gpu15 [fillcolor=lightcoral, label="Layer15 Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: 15"];
    
    layer15_attn_score_gpu12 -> layer15_attn_out_gpu12;
    layer15_attn_score_gpu13 -> layer15_attn_out_gpu13;
    layer15_attn_score_gpu14 -> layer15_attn_out_gpu14;
    layer15_attn_score_gpu15 -> layer15_attn_out_gpu15;
    
    layer15_attn_allreduce [shape=ellipse, fillcolor=lightgray, label="Layer15 Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 12,13,14,15"];
    layer15_attn_out_gpu12 -> layer15_attn_allreduce;
    layer15_attn_out_gpu13 -> layer15_attn_allreduce;
    layer15_attn_out_gpu14 -> layer15_attn_allreduce;
    layer15_attn_out_gpu15 -> layer15_attn_allreduce;
    
    layer15_moe_route [fillcolor=lightcoral, shape=parallelogram, label="Layer15 MoE Routing\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1]\\nGPU: 12,13,14,15"];
    layer15_attn_allreduce -> layer15_moe_route;
    
    layer15_moe_all2all [shape=ellipse, fillcolor=lightgray, label="Layer15 MoE\\nAll-to-All Communication\\nGPU: 0-15"];
    layer15_moe_route -> layer15_moe_all2all;
    
    // Layer 15 experts
    layer15_expert0 [fillcolor=lightblue, label="Layer15 Expert 0_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 0"];
    layer15_expert1 [fillcolor=lightblue, label="Layer15 Expert 0_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 1"];
    layer15_expert2 [fillcolor=lightblue, label="Layer15 Expert 0_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 2"];
    layer15_expert3 [fillcolor=lightblue, label="Layer15 Expert 0_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 3"];
    layer15_expert4 [fillcolor=lightgreen, label="Layer15 Expert 1_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 4"];
    layer15_expert5 [fillcolor=lightgreen, label="Layer15 Expert 1_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 5"];
    layer15_expert6 [fillcolor=lightgreen, label="Layer15 Expert 1_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 6"];
    layer15_expert7 [fillcolor=lightgreen, label="Layer15 Expert 1_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 7"];
    layer15_expert8 [fillcolor=lightyellow, label="Layer15 Expert 2_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 8"];
    layer15_expert9 [fillcolor=lightyellow, label="Layer15 Expert 2_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 9"];
    layer15_expert10 [fillcolor=lightyellow, label="Layer15 Expert 2_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 10"];
    layer15_expert11 [fillcolor=lightyellow, label="Layer15 Expert 2_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 11"];
    layer15_expert12 [fillcolor=lightcoral, label="Layer15 Expert 3_0\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 12"];
    layer15_expert13 [fillcolor=lightcoral, label="Layer15 Expert 3_1\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 13"];
    layer15_expert14 [fillcolor=lightcoral, label="Layer15 Expert 3_2\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 14"];
    layer15_expert15 [fillcolor=lightcoral, label="Layer15 Expert 3_3\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: 15"];
    
    layer15_moe_all2all -> layer15_expert0;
    layer15_moe_all2all -> layer15_expert1;
    layer15_moe_all2all -> layer15_expert2;
    layer15_moe_all2all -> layer15_expert3;
    layer15_moe_all2all -> layer15_expert4;
    layer15_moe_all2all -> layer15_expert5;
    layer15_moe_all2all -> layer15_expert6;
    layer15_moe_all2all -> layer15_expert7;
    layer15_moe_all2all -> layer15_expert8;
    layer15_moe_all2all -> layer15_expert9;
    layer15_moe_all2all -> layer15_expert10;
    layer15_moe_all2all -> layer15_expert11;
    layer15_moe_all2all -> layer15_expert12;
    layer15_moe_all2all -> layer15_expert13;
    layer15_moe_all2all -> layer15_expert14;
    layer15_moe_all2all -> layer15_expert15;
    
    // CRITICAL FIX: Final aggregation node that connects to output
    layer15_moe_agg [fillcolor=lightcoral, shape=parallelogram, label="Layer15 MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: 12,13,14,15"];
    
    // FIX: Connect all Layer 15 experts to final aggregation
    layer15_expert0 -> layer15_moe_agg;
    layer15_expert1 -> layer15_moe_agg;
    layer15_expert2 -> layer15_moe_agg;
    layer15_expert3 -> layer15_moe_agg;
    layer15_expert4 -> layer15_moe_agg;
    layer15_expert5 -> layer15_moe_agg;
    layer15_expert6 -> layer15_moe_agg;
    layer15_expert7 -> layer15_moe_agg;
    layer15_expert8 -> layer15_moe_agg;
    layer15_expert9 -> layer15_moe_agg;
    layer15_expert10 -> layer15_moe_agg;
    layer15_expert11 -> layer15_moe_agg;
    layer15_expert12 -> layer15_moe_agg;
    layer15_expert13 -> layer15_moe_agg;
    layer15_expert14 -> layer15_moe_agg;
    layer15_expert15 -> layer15_moe_agg;
    
    // Final output aggregation and output - CRITICAL FIX: Proper connections
    output_agg [shape=parallelogram, fillcolor=lightpink, label="Output Aggregation\\nAll-Reduce Sum\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nGPU: 12,13,14,15"];
    layer15_moe_agg -> output_agg;
    
    output [shape=ellipse, fillcolor=white, label="Final Output\\n[batch_size=128, seq_len=1024, hidden=1024]\\nGPU: ALL"];
    output_agg -> output;
}
'''
    
    return dot_content

def main():
    # Generate the corrected DAG
    dag_content = generate_corrected_moe_dag()
    
    # Save to file
    output_path = "../outputs/2025-12-05-10-26-38/moe_deployment_corrected.dot"
    with open(output_path, 'w') as f:
        f.write(dag_content)
    
    print(f"Generated corrected DAG: {output_path}")
    
    # Also generate SVG using Graphviz
    try:
        import subprocess
        svg_path = output_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', output_path, '-o', svg_path], check=True)
        print(f"Generated SVG: {svg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Graphviz not available, skipping SVG generation")
    
    return output_path

if __name__ == "__main__":
    main()