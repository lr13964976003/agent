#!/usr/bin/env python3
"""
LLM Parallel Strategy DAG Generator
Generates comprehensive DAGs for current and optimal parallel strategies
"""

import os
import subprocess

def create_current_strategy_dag():
    """Create DAG for EP64-TP8-PP2-DP2 strategy (2048 GPUs)"""
    
    dot_content = '''// Current Strategy: EP64-TP8-PP2-DP2
// Total GPUs: 2048 (64*8*2*2)
digraph CurrentStrategy {
    rankdir=TB;
    nodesep=0.5;
    ranksep=1.0;
    
    // Node styles
    node [fontname="Arial", fontsize=10];
    
    // Input
    input [label="Input\\n[BS=128, SL=1024, H=1024]\\nDP2: 64 seqs per replica", shape=ellipse, fillcolor=white, style=filled];
    
    // Pipeline Stage 0 Cluster (Layers 0-7, GPUs 0-1023)
    subgraph cluster_stage0 {
        label="Pipeline Stage 0 (Layers 0-7)\\nGPUs: 0-1023";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 0 Attention with TP8 decomposition
        subgraph cluster_layer0_attn {
            label="Layer 0 Attention (TP8)";
            style=dashed;
            fillcolor=lightblue;
            
            l0_attn_qkv [label="L0-Attention-QKV\\nTP8[0-7]\\nIn: [64,1024,1024]\\nOut: [64,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_score [label="L0-Attention-Score\\nTP8[0-7]\\nIn: [64,16,1024,64]\\nOut: [64,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_out [label="L0-Attention-Output\\nTP8[0-7]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_allreduce [label="L0-Attention-AllReduce\\nTP8[0-7]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 0 MoE with EP64
        subgraph cluster_layer0_moe {
            label="Layer 0 MoE (EP64)";
            style=dashed;
            fillcolor=lightyellow;
            
            l0_route [label="L0-Expert-Routing\\nEP64[0-63]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l0_dispatch [label="L0-Expert-Dispatch\\nEP64[0-63]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
            
            // Expert groups (64 experts total)
            l0_expert_0_3 [label="L0-Experts-0-3\\nEP64[0-63]\\n(4 experts)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_4_7 [label="L0-Experts-4-7\\nEP64[64-127]\\n(4 experts)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_8_15 [label="L0-Experts-8-15\\nEP64[128-255]\\n(8 experts)\\nIn: [8,1024,1024]\\nOut: [8,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_16_31 [label="L0-Experts-16-31\\nEP64[256-511]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_32_47 [label="L0-Experts-32-47\\nEP64[512-767]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_48_63 [label="L0-Experts-48-63\\nEP64[768-1023]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l0_combine [label="L0-Expert-Combine\\nEP64[0-63]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l0_combine_alltoall [label="L0-Expert-Combine\\nEP64[0-63]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
    }
    
    // Pipeline Stage 1 Cluster (Layers 8-15, GPUs 1024-2047)
    subgraph cluster_stage1 {
        label="Pipeline Stage 1 (Layers 8-15)\\nGPUs: 1024-2047";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 8 Attention with TP8 (similar to layer 0)
        subgraph cluster_layer8_attn {
            label="Layer 8 Attention (TP8)";
            style=dashed;
            fillcolor=lightblue;
            
            l8_attn_qkv [label="L8-Attention-QKV\\nTP8[1024-1031]\\nIn: [64,1024,1024]\\nOut: [64,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_score [label="L8-Attention-Score\\nTP8[1024-1031]\\nIn: [64,16,1024,64]\\nOut: [64,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_out [label="L8-Attention-Output\\nTP8[1024-1031]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_allreduce [label="L8-Attention-AllReduce\\nTP8[1024-1031]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 8 MoE with EP64 (similar to layer 0)
        subgraph cluster_layer8_moe {
            label="Layer 8 MoE (EP64)";
            style=dashed;
            fillcolor=lightyellow;
            
            l8_route [label="L8-Expert-Routing\\nEP64[1024-1087]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l8_dispatch [label="L8-Expert-Dispatch\\nEP64[1024-1087]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
            
            // Expert groups for layer 8 (similar pattern)
            l8_expert_0_3 [label="L8-Experts-0-3\\nEP64[1024-1087]\\n(4 experts)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_4_7 [label="L8-Experts-4-7\\nEP64[1088-1151]\\n(4 experts)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_8_15 [label="L8-Experts-8-15\\nEP64[1152-1279]\\n(8 experts)\\nIn: [8,1024,1024]\\nOut: [8,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_16_31 [label="L8-Experts-16-31\\nEP64[1280-1599]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_32_47 [label="L8-Experts-32-47\\nEP64[1600-1919]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_48_63 [label="L8-Experts-48-63\\nEP64[1920-2047]\\n(16 experts)\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l8_combine [label="L8-Expert-Combine\\nEP64[1024-1087]\\nIn: [64,1024,1024]\\nOut: [64,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l8_combine_alltoall [label="L8-Expert-Combine\\nEP64[1024-1087]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
    }
    
    // Output
    output [label="Output\\n[BS=128, SL=1024, H=1024]", shape=ellipse, fillcolor=white, style=filled];
    
    // Edges - Stage 0
    input -> l0_attn_qkv;
    l0_attn_qkv -> l0_attn_allreduce;
    l0_attn_allreduce -> l0_attn_score;
    l0_attn_score -> l0_attn_out;
    l0_attn_out -> l0_route;
    
    // MoE routing and dispatch
    l0_route -> l0_dispatch;
    l0_dispatch -> l0_expert_0_3;
    l0_dispatch -> l0_expert_4_7;
    l0_dispatch -> l0_expert_8_15;
    l0_dispatch -> l0_expert_16_31;
    l0_dispatch -> l0_expert_32_47;
    l0_dispatch -> l0_expert_48_63;
    
    // Expert to combine (dashed lines for gate selection)
    l0_expert_0_3 -> l0_combine [style=dashed];
    l0_expert_4_7 -> l0_combine [style=dashed];
    l0_expert_8_15 -> l0_combine;
    l0_expert_16_31 -> l0_combine;
    l0_expert_32_47 -> l0_combine;
    l0_expert_48_63 -> l0_combine;
    
    l0_combine -> l0_combine_alltoall;
    l0_combine_alltoall -> l8_attn_qkv [lhead=cluster_stage1];
    
    // Edges - Stage 1
    l8_attn_qkv -> l8_attn_allreduce;
    l8_attn_allreduce -> l8_attn_score;
    l8_attn_score -> l8_attn_out;
    l8_attn_out -> l8_route;
    
    l8_route -> l8_dispatch;
    l8_dispatch -> l8_expert_0_3;
    l8_dispatch -> l8_expert_4_7;
    l8_dispatch -> l8_expert_8_15;
    l8_dispatch -> l8_expert_16_31;
    l8_dispatch -> l8_expert_32_47;
    l8_dispatch -> l8_expert_48_63;
    
    l8_expert_0_3 -> l8_combine [style=dashed];
    l8_expert_4_7 -> l8_combine [style=dashed];
    l8_expert_8_15 -> l8_combine;
    l8_expert_16_31 -> l8_combine;
    l8_expert_32_47 -> l8_combine;
    l8_expert_48_63 -> l8_combine;
    
    l8_combine -> l8_combine_alltoall;
    l8_combine_alltoall -> output;
}'''
    
    return dot_content

def create_optimal_strategy_dag():
    """Create DAG for EP32-TP4-PP4-DP8 strategy (512 GPUs)"""
    
    dot_content = '''// Optimal Strategy: EP32-TP4-PP4-DP8
// Total GPUs: 512 (32*4*4*8)
digraph OptimalStrategy {
    rankdir=TB;
    nodesep=0.5;
    ranksep=1.0;
    
    // Node styles
    node [fontname="Arial", fontsize=10];
    
    // Input with DP8
    input [label="Input\\n[BS=128, SL=1024, H=1024]\\nDP8: 16 seqs per GPU", shape=ellipse, fillcolor=white, style=filled];
    
    // Pipeline Stage 0 Cluster (Layers 0-3, GPUs 0-127)
    subgraph cluster_stage0 {
        label="Pipeline Stage 0 (Layers 0-3)\\nGPUs: 0-127";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 0 Attention with TP4 decomposition
        subgraph cluster_layer0_attn {
            label="Layer 0 Attention (TP4)";
            style=dashed;
            fillcolor=lightblue;
            
            l0_attn_qkv [label="L0-Attention-QKV\\nTP4[0-3]\\nIn: [16,1024,1024]\\nOut: [16,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_score [label="L0-Attention-Score\\nTP4[0-3]\\nIn: [16,16,1024,64]\\nOut: [16,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_out [label="L0-Attention-Output\\nTP4[0-3]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_attn_allreduce [label="L0-Attention-AllReduce\\nTP4[0-3]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 0 MoE with EP32
        subgraph cluster_layer0_moe {
            label="Layer 0 MoE (EP32)";
            style=dashed;
            fillcolor=lightyellow;
            
            l0_route [label="L0-Expert-Routing\\nEP32[0-31]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l0_dispatch [label="L0-Expert-Dispatch\\nEP32[0-31]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
            
            // Expert groups (32 experts total, 2 experts per GPU group)
            l0_expert_0_1 [label="L0-Experts-0,1\\nEP32[0-3]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_2_3 [label="L0-Experts-2,3\\nEP32[4-7]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_4_5 [label="L0-Experts-4,5\\nEP32[8-11]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_6_7 [label="L0-Experts-6,7\\nEP32[12-15]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_8_9 [label="L0-Experts-8,9\\nEP32[16-19]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_10_11 [label="L0-Experts-10,11\\nEP32[20-23]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_12_13 [label="L0-Experts-12,13\\nEP32[24-27]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_14_15 [label="L0-Experts-14,15\\nEP32[28-31]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            // Continue with remaining experts (16-31) - abbreviated
            l0_expert_16_23 [label="L0-Experts-16-23\\nEP32[32-63]\\n(8 experts total)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l0_expert_24_31 [label="L0-Experts-24-31\\nEP32[64-95]\\n(8 experts total)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l0_combine [label="L0-Expert-Combine\\nEP32[0-31]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l0_combine_alltoall [label="L0-Expert-Combine\\nEP32[0-31]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
    }
    
    // Pipeline Stage 1 Cluster (Layers 4-7, GPUs 128-255)
    subgraph cluster_stage1 {
        label="Pipeline Stage 1 (Layers 4-7)\\nGPUs: 128-255";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 4 Attention (similar to layer 0)
        subgraph cluster_layer4_attn {
            label="Layer 4 Attention (TP4)";
            style=dashed;
            fillcolor=lightblue;
            
            l4_attn_qkv [label="L4-Attention-QKV\\nTP4[128-131]\\nIn: [16,1024,1024]\\nOut: [16,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_attn_score [label="L4-Attention-Score\\nTP4[128-131]\\nIn: [16,16,1024,64]\\nOut: [16,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_attn_out [label="L4-Attention-Output\\nTP4[128-131]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_attn_allreduce [label="L4-Attention-AllReduce\\nTP4[128-131]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 4 MoE (similar to layer 0)
        subgraph cluster_layer4_moe {
            label="Layer 4 MoE (EP32)";
            style=dashed;
            fillcolor=lightyellow;
            
            l4_route [label="L4-Expert-Routing\\nEP32[128-159]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l4_dispatch [label="L4-Expert-Dispatch\\nEP32[128-159]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
            
            l4_expert_0_1 [label="L4-Experts-0,1\\nEP32[128-131]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_expert_2_3 [label="L4-Experts-2,3\\nEP32[132-135]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_expert_4_5 [label="L4-Experts-4,5\\nEP32[136-139]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_expert_6_7 [label="L4-Experts-6,7\\nEP32[140-143]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l4_expert_8_15 [label="L4-Experts-8-15\\nEP32[144-159]\\n(8 experts total)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l4_expert_16_31 [label="L4-Experts-16-31\\nEP32[160-191]\\n(16 experts total)\\nIn: [8,1024,1024]\\nOut: [8,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l4_combine [label="L4-Expert-Combine\\nEP32[128-159]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l4_combine_alltoall [label="L4-Expert-Combine\\nEP32[128-159]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
    }
    
    // Pipeline Stage 2 Cluster (Layers 8-11, GPUs 256-383)
    subgraph cluster_stage2 {
        label="Pipeline Stage 2 (Layers 8-11)\\nGPUs: 256-383";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 8 Attention (similar pattern)
        subgraph cluster_layer8_attn {
            label="Layer 8 Attention (TP4)";
            style=dashed;
            fillcolor=lightblue;
            
            l8_attn_qkv [label="L8-Attention-QKV\\nTP4[256-259]\\nIn: [16,1024,1024]\\nOut: [16,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_score [label="L8-Attention-Score\\nTP4[256-259]\\nIn: [16,16,1024,64]\\nOut: [16,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_out [label="L8-Attention-Output\\nTP4[256-259]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_attn_allreduce [label="L8-Attention-AllReduce\\nTP4[256-259]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 8 MoE (similar pattern)
        subgraph cluster_layer8_moe {
            label="Layer 8 MoE (EP32)不远万里";
            style=dashed;
            fillcolor=lightyellow;
            
            l8_route [label="L8-Expert-Routing\\nEP32[256-287]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l8_dispatch [label="L8-Expert-Dispatch\\nEP32[256-287]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
            
            l8_expert_0_1 [label="L8-Experts-0,1\\nEP32[256-259]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_2_3 [label="L8-Experts-2,3\\nEP32[260-263]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_4_5 [label="L8-Experts-4,5\\nEP32[264-267]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_6_7 [label="L8-Experts-6,7\\nEP32[268-271]\\n(2 experts)\\nIn: [1,1024,1024]\\nOut: [1,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l8_expert_8_15 [label="L8-Experts-8-15\\nEP32[272-287]\\n(8 experts total)\\nIn: [4,1024,1024]\\nOut: [4,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l8_expert_16_31 [label="L8-Experts-16-31\\nEP32[288-383]\\n(16 experts total)\\nIn: [8,1024,1024]\\nOut: [8,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            
            l8_combine [label="L8-Expert-Combine\\nEP32[256-287]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=parallelogram, fillcolor=lightyellow, style=filled];
            l8_combine_alltoall [label="L8-Expert-Combine\\nEP32[256-287]\\nAll-to-All", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
    }
    
    // Pipeline Stage 3 Cluster (Layers 12-15, GPUs 384-511)
    subgraph cluster_stage3 {
        label="Pipeline Stage 3 (Layers 12-15)\\nGPUs: 384-511";
        style=rounded;
        fillcolor=lightgrey;
        
        // Layer 12 Attention (similar pattern)
        subgraph cluster_layer12_attn {
            label="Layer 12 Attention (TP4)";
            style=dashed;
            fillcolor=lightblue;
            
            l12_attn_qkv [label="L12-Attention-QKV\\nTP4[384-387]\\nIn: [16,1024,1024]\\nOut: [16,1024,3072]", shape=rectangle, fillcolor=lightblue, style=filled];
            l12_attn_score [label="L12-Attention-Score\\nTP4[384-387]\\nIn: [16,16,1024,64]\\nOut: [16,16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l12_attn_out [label="L12-Attention-Output\\nTP4[384-387]\\nIn: [16,1024,1024]\\nOut: [16,1024,1024]", shape=rectangle, fillcolor=lightblue, style=filled];
            l12_attn_allreduce [label="L12-Attention-AllReduce\\nTP4[384-387]", shape=ellipse, fillcolor=lightgreen, style=filled];
        }
        
        // Layer 12 MoE (similar pattern)
        subgraph cluster_layer12_moe {
            label="Layer 12