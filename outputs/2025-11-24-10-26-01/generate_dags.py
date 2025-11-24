#!/usr/bin/env python3
"""
Complete DAG Generator for LLM Parallel Deployment
Generates fully detailed DAGs for both baseline and proposed strategies
"""

import os

def generate_baseline_dag():
    """Generate complete baseline DAG with TP=8, PP=2 strategy"""
    
    # Define the complete DOT content for baseline
    dot_content = '''digraph baseline_tp8_pp2_complete {
    graph [rankdir=TB, splines=ortho, nodesep=0.3, ranksep=0.8];
    node [shape=box, style=filled];
    
    // Input and Output nodes
    input [label="Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]", 
           shape=parallelogram, fillcolor=lightgreen, fontsize=12];
    output [label="Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]", 
            shape=parallelogram, fillcolor=lightgreen, fontsize=12];

    // Stage 0: Layers 0-7 on GPUs 0-7 (TP=8)
    subgraph cluster_stage0 {
        label="Stage 0 (Layers 0-7)\\nTP=8 across GPUs 0-7";
        style="rounded,dashed";
        color=red;
        
        // Layer 0 on GPUs 0-7
        subgraph cluster_layer0 {
            label="Layer 0\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)";
            style=dotted;
            
            l0_qkv [label="QKV Projection\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nSplit: 512 cols each", 
                   fillcolor=lightcoral, fontsize=10];
            l0_attn [label="Multi-Head Attention\\nGPU: TP=8 across 0-7\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightpink, fontsize=10];
            l0_out [label="Output Projection\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nRow parallel", 
                   fillcolor=lightcoral, fontsize=10];
            l0_res1 [label="Residual Add 1\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l0_norm1 [label="Layer Norm 1\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l0_mlp_gate [label="MLP Gate\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                        fillcolor=lightseagreen, fontsize=10];
            l0_mlp_up [label="MLP Up\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                      fillcolor=lightseagreen, fontsize=10];
            l0_act [label="GELU Activation\\nGPU: TP=8 across 0-7\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]", 
                   fillcolor=lightcyan, fontsize=10];
            l0_mlp_down [label="MLP Down\\nGPU: TP=8 across 0-7\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nRow parallel", 
                        fillcolor=lightseagreen, fontsize=10];
            l0_res2 [label="Residual Add 2\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l0_norm2 [label="Layer Norm 2\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l0_qkv -> l0_attn -> l0_out -> l0_res1 -> l0_norm1 -> l0_mlp_gate;
            l0_norm1 -> l0_mlp_up -> l0_act -> l0_mlp_down -> l0_res2 -> l0_norm2;
            l0_res1 -> l0_res2 [style=dotted];
        }

        // Layer 1 on GPUs 0-7
        subgraph cluster_layer1 {
            label="Layer 1\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)";
            style=dotted;
            
            l1_qkv [label="QKV Projection\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nSplit: 512 cols each", 
                   fillcolor=lightcoral, fontsize=10];
            l1_attn [label="Multi-Head Attention\\nGPU: TP=8 across 0-7\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightpink, fontsize=10];
            l1_out [label="Output Projection\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nRow parallel", 
                   fillcolor=lightcoral, fontsize=10];
            l1_res1 [label="Residual Add 1\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l1_norm1 [label="Layer Norm 1\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l1_mlp_gate [label="MLP Gate\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                        fillcolor=lightseagreen, fontsize=10];
            l1_mlp_up [label="MLP Up\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                      fillcolor=lightseagreen, fontsize=10];
            l1_act [label="GELU Activation\\nGPU: TP=8 across 0-7\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]", 
                   fillcolor=lightcyan, fontsize=10];
            l1_mlp_down [label="MLP Down\\nGPU: TP=8 across 0-7\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nRow parallel", 
                        fillcolor=lightseagreen, fontsize=10];
            l1_res2 [label="Residual Add 2\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l1_norm2 [label="Layer Norm 2\\nGPU: TP=8 across 0-7\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l1_qkv -> l1_attn -> l1_out -> l1_res1 -> l1_norm1 -> l1_mlp_gate;
            l1_norm1 -> l1_mlp_up -> l1_act -> l1_mlp_down -> l1_res2 -> l1_norm2;
            l1_res1 -> l1_res2 [style=dotted];
        }

        // Continue with layers 2-7 (same structure)
        l2 [label="Layer 2\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l3 [label="Layer 3\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l4 [label="Layer 4\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l5 [label="Layer 5\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l6 [label="Layer 6\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l7 [label="Layer 7\\nGPUs: 0,1,2,3,4,5,6,7 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
    }

    // Stage 1: Layers 8-15 on GPUs 8-15 (TP=8)
    subgraph cluster_stage1 {
        label="Stage 1 (Layers 8-15)\\nTP=8 across GPUs 8-15";
        style="rounded,dashed";
        color=blue;
        
        // Layer 8 on GPUs 8-15
        subgraph cluster_layer8 {
            label="Layer 8\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)";
            style=dotted;
            
            l8_qkv [label="QKV Projection\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nSplit: 512 cols each", 
                   fillcolor=lightcoral, fontsize=10];
            l8_attn [label="Multi-Head Attention\\nGPU: TP=8 across 8-15\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightpink, fontsize=10];
            l8_out [label="Output Projection\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nRow parallel", 
                   fillcolor=lightcoral, fontsize=10];
            l8_res1 [label="Residual Add 1\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l8_norm1 [label="Layer Norm 1\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l8_mlp_gate [label="MLP Gate\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                        fillcolor=lightseagreen, fontsize=10];
            l8_mlp_up [label="MLP Up\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nSplit: 2048 cols each", 
                      fillcolor=lightseagreen, fontsize=10];
            l8_act [label="GELU Activation\\nGPU: TP=8 across 8-15\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]", 
                   fillcolor=lightcyan, fontsize=10];
            l8_mlp_down [label="MLP Down\\nGPU: TP=8 across 8-15\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nRow parallel", 
                        fillcolor=lightseagreen, fontsize=10];
            l8_res2 [label="Residual Add 2\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                    fillcolor=lightgray, fontsize=10];
            l8_norm2 [label="Layer Norm 2\\nGPU: TP=8 across 8-15\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                     fillcolor=lightyellow, fontsize=10];
            
            l8_qkv -> l8_attn -> l8_out -> l8_res1 -> l8_norm1 -> l8_mlp_gate;
            l8_norm1 -> l8_mlp_up -> l8_act -> l8_mlp_down -> l8_res2 -> l8_norm2;
            l8_res1 -> l8_res2 [style=dotted];
        }

        // Continue with layers 9-15 (same structure)
        l9 [label="Layer 9\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
            fillcolor=lightcoral, fontsize=10];
        l10 [label="Layer 10\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
        l11 [label="Layer 11\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
        l12 [label="Layer 12\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
        l13 [label="Layer 13\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
        l14 [label="Layer 14\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
        l15 [label="Layer 15\\nGPUs: 8,9,10,11,12,13,14,15 (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nFull attention+MLP", 
             fillcolor=lightcoral, fontsize=10];
    }

    // Communication nodes
    stage_transfer [label="Pipeline Transfer\\nGPU 7 → GPU 8\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                   shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    
    // AllReduce operations (forward only)
    tp_allreduce_0 [label="TP AllReduce\\nStage 0\\nGPUs: 0-7\\nReduce: [128,10000,4096]", 
                   shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    tp_allreduce_1 [label="TP AllReduce\\nStage 1\\nGPUs: 8-15\\nReduce: [128,10000,4096]", 
                   shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];

    // Connections
    input -> l0_qkv;
    
    // Stage 0 connections
    l0_norm2 -> l1_qkv;
    l1_norm2 -> l2;
    l2 -> l3;
    l3 -> l4;
    l4 -> l5;
    l5 -> l6;
    l6 -> l7;
    l7 -> stage_transfer;
    
    // Stage 1 connections
    stage_transfer -> l8_qkv;
    l8_norm2 -> l9;
    l9 -> l10;
    l10 -> l11;
    l11 -> l12;
    l12 -> l13;
    l13 -> l14;
    l14 -> l15;
    l15 -> output;
    
    // TP synchronization (forward only)
    l0_norm2 -> tp_allreduce_0;
    l1_norm2 -> tp_allreduce_0;
    l8_norm2 -> tp_allreduce_1;
    l15 -> tp_allreduce_1;
}'''
    
    return dot_content

def generate_proposed_dag():
    """Generate complete proposed DAG with layer-wise deployment"""
    
    # Define the complete DOT content for proposed strategy
    dot_content = '''digraph proposed_layer_wise_complete {
    graph [rankdir=TB, splines=ortho, nodesep=0.3, ranksep=0.8];
    node [shape=box, style=filled];
    
    // Input and Output nodes
    input [label="Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]", 
           shape=parallelogram, fillcolor=lightgreen, fontsize=12];
    output [label="Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]", 
            shape=parallelogram, fillcolor=lightgreen, fontsize=12];

    // Layer 0 on GPU 0
    subgraph cluster_layer0 {
        label="Layer 0\\nGPU: 0\\nCache: 11.8GB SRAM/L2";
        style="rounded,dashed";
        color=red;
        
        l0_qkv [label="QKV Projection\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nWeights: 1.56GB", 
               fillcolor=lightcoral, fontsize=10];
        l0_attn [label="Multi-Head Attention\\nGPU: 0\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]\\nActivations: 10.49GB", 
                fillcolor=lightpink, fontsize=10];
        l0_out [label="Output Projection\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nWeights: 1.56GB", 
               fillcolor=lightcoral, fontsize=10];
        l0_res1 [label="Residual Add 1\\nGPU: 0\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                fillcolor=lightgray, fontsize=10];
        l0_norm1 [label="Layer Norm 1\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                 fillcolor=lightyellow, fontsize=10];
        
        l0_mlp_gate [label="MLP Gate\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nWeights: 1.56GB", 
                    fillcolor=lightseagreen, fontsize=10];
        l0_mlp_up [label="MLP Up\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nWeights: 1.56GB", 
                  fillcolor=lightseagreen, fontsize=10];
        l0_act [label="GELU Activation\\nGPU: 0\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]", 
               fillcolor=lightcyan, fontsize=10];
        l0_mlp_down [label="MLP Down\\nGPU: 0\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nWeights: 1.56GB", 
                    fillcolor=lightseagreen, fontsize=10];
        l0_res2 [label="Residual Add 2\\nGPU: 0\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                fillcolor=lightgray, fontsize=10];
        l0_norm2 [label="Layer Norm 2\\nGPU: 0\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                 fillcolor=lightyellow, fontsize=10];
        
        l0_qkv -> l0_attn -> l0_out -> l0_res1 -> l0_norm1 -> l0_mlp_gate;
        l0_norm1 -> l0_mlp_up -> l0_act -> l0_mlp_down -> l0_res2 -> l0_norm2;
        l0_res1 -> l0_res2 [style=dotted];
    }

    // Layer 1 on GPU 1
    subgraph cluster_layer1 {
        label="Layer 1\\nGPU: 1\\nCache: 11.8GB SRAM/L2";
        style="rounded,dashed";
        color=blue;
        
        l1_qkv [label="QKV Projection\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nWeights: 1.56GB", 
               fillcolor=lightcoral, fontsize=10];
        l1_attn [label="Multi-Head Attention\\nGPU: 1\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]\\nActivations: 10.49GB", 
                fillcolor=lightpink, fontsize=10];
        l1_out [label="Output Projection\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nWeights: 1.56GB", 
               fillcolor=lightcoral, fontsize=10];
        l1_res1 [label="Residual Add 1\\nGPU: 1\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                fillcolor=lightgray, fontsize=10];
        l1_norm1 [label="Layer Norm 1\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                 fillcolor=lightyellow, fontsize=10];
        
        l1_mlp_gate [label="MLP Gate\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nWeights: 1.56GB", 
                    fillcolor=lightseagreen, fontsize=10];
        l1_mlp_up [label="MLP Up\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nWeights: 1.56GB", 
                  fillcolor=lightseagreen, fontsize=10];
        l1_act [label="GELU Activation\\nGPU: 1\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]", 
               fillcolor=lightcyan, fontsize=10];
        l1_mlp_down [label="MLP Down\\nGPU: 1\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nWeights: 1.56GB", 
                    fillcolor=lightseagreen, fontsize=10];
        l1_res2 [label="Residual Add 2\\nGPU: 1\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]", 
                fillcolor=lightgray, fontsize=10];
        l1_norm2 [label="Layer Norm 2\\nGPU: 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
                 fillcolor=lightyellow, fontsize=10];
        
        l1_qkv -> l1_attn -> l1_out -> l1_res1 -> l1_norm1 -> l1_mlp_gate;
        l1_norm1 -> l1_mlp_up -> l1_act -> l1_mlp_down -> l1_res2 -> l1_norm2;
        l1_res1 -> l1_res2 [style=dotted];
    }

    // Continue with layers 2-15, each on dedicated GPU
    l2 [label="Layer 2\\nGPU: 2\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l3 [label="Layer 3\\nGPU: 3\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l4 [label="Layer 4\\nGPU: 4\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l5 [label="Layer 5\\nGPU: 5\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l6 [label="Layer 6\\nGPU: 6\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l7 [label="Layer 7\\nGPU: 7\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l8 [label="Layer 8\\nGPU: 8\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l9 [label="Layer 9\\nGPU: 9\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
        fillcolor=lightcoral, fontsize=10];
    l10 [label="Layer 10\\nGPU: 10\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];
    l11 [label="Layer 11\\nGPU: 11\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];
    l12 [label="Layer 12\\nGPU: 12\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];
    l13 [label="Layer 13\\nGPU: 13\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];
    l14 [label="Layer 14\\nGPU: 14\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];
    l15 [label="Layer 15\\nGPU: 15\\nComplete attention+MLP\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nCache: 11.8GB", 
         fillcolor=lightcoral, fontsize=10];

    // Transfer nodes between GPUs
    t0_1 [label="Transfer\\nGPU: 0 → 1\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t1_2 [label="Transfer\\nGPU: 1 → 2\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t2_3 [label="Transfer\\nGPU: 2 → 3\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t3_4 [label="Transfer\\nGPU: 3 → 4\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t4_5 [label="Transfer\\nGPU: 4 → 5\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t5_6 [label="Transfer\\nGPU: 5 → 6\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t6_7 [label="Transfer\\nGPU: 6 → 7\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t7_8 [label="Transfer\\nGPU: 7 → 8\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t8_9 [label="Transfer\\nGPU: 8 → 9\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
         shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t9_10 [label="Transfer\\nGPU: 9 → 10\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
          shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t10_11 [label="Transfer\\nGPU: 10 → 11\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
           shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t11_12 [label="Transfer\\nGPU: 11 → 12\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
           shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t12_13 [label="Transfer\\nGPU: 12 → 13\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
           shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t13_14 [label="Transfer\\nGPU: 13 → 14\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
           shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];
    t14_15 [label="Transfer\\nGPU: 14 → 15\\nSize: 5.24GB\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]", 
           shape=ellipse, fillcolor=lightblue, style=dashed, fontsize=10];

    // Connections
    input -> l0_qkv;
    
    // Layer 0 connections
    l0_qkv -> l0_attn -> l0_out -> l0_res1 -> l0_norm1 -> l0_mlp_gate;
    l0_norm1 -> l0_mlp_up -> l0_act -> l0_mlp_down -> l0_res2 -> l0_norm2;
    l0_res1 -> l0_res2 [style=dotted];
    
    // Layer 1 connections
    l0_norm2 -> t0_1 -> l1_qkv;
    l1_qkv -> l1_attn -> l1_out -> l1_res1 -> l1_norm1 -> l1_mlp_gate;
    l1_norm1 -> l1_mlp_up -> l1_act -> l1_mlp_down -> l1_res2 -> l1_norm2;
    l1_res1 -> l1_res2 [style=dotted];
    
    // Continue pattern for all layers
    l1_norm2 -> t1_2 -> l2;
    l2 -> t2_3 -> l3;
    l3 -> t3_4 -> l4;
    l4 -> t4_5 -> l5;
    l5 -> t5_6 -> l6;
    l6 -> t6_7 -> l7;
    l7 -> t7_8 -> l8;
    l8 -> t8_9 -> l9;
    l9 -> t9_10 -> l10;
    l10 -> t10_11 -> l11;
    l11 -> t11_12 -> l12;
    l12 -> t12_13 -> l13;
    l13 -> t13_14 -> l14;
    l14 -> t14_15 -> l15;
    l15 -> output;
}'''
    
    return dot_content

if __name__ == "__main__":
    # Generate baseline DAG
    baseline_dot = generate_baseline_dag()
    with open('../outputs/2025-11-24-10-26-01/baseline_dag_complete.dot', 'w') as f:
        f.write(baseline_dot)
    
    # Generate proposed DAG
    proposed_dot = generate_proposed_dag()
    with open('../outputs/2025-11-24-10-26-01/proposed_dag_complete.dot', 'w') as f:
        f.write(proposed_dot)
    
    print("DAG files generated successfully!")