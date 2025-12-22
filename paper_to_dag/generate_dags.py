#!/usr/bin/env python3
"""
Generate two complete model deployment DAGs for LLM inference
Current strategy: EP16 × TP4 × PP4 × DP4 = 1024 GPUs
New strategy: EP64 × TP8 × PP2 × DP2 = 2048 GPUs
"""

import os
import subprocess
from datetime import datetime

def create_current_strategy_dag():
    """Generate DAG for current strategy: EP16 × TP4 × PP4 × DP4 = 1024 GPUs"""
    
    dot_content = '''digraph CurrentStrategy {
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Graph styling
    graph [bgcolor=white, fontname="Arial", fontsize=12];
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Input node
    Input [shape=ellipse, label="Input\\nInput: [batch_size=128, seq_len=128-10240, dim=1024]\\nOutput: [batch_size=128, seq_len=128-10240, dim=1024]", fillcolor=lightblue];
    
    // Data Parallel split (DP4)
    DP_Split [shape=parallelogram, label="DP Split\\n4-way Data Parallel\\nInput: [batch_size=128, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgreen];
    
    // Pipeline stages (PP4)
    subgraph cluster_pipeline {
        label="Pipeline Parallelism (PP4)";
        style=filled;
        fillcolor=lightyellow;
        
        // Stage 0: Layers 0-3
        subgraph cluster_stage0 {
            label="Stage 0: Layers 0-3 (GPU 0-255)";
            style=filled;
            fillcolor=lightcoral;
            
            // Tensor Parallel within stage
            subgraph cluster_tp_stage0 {
                label="TP4 within Stage 0";
                style=filled;
                fillcolor=lightpink;
                
                // Layer 0
                Layer0_TP0 [label="Layer 0 TP0\\nAttention (QKV)\\nGPU: 0-63\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP1 [label="Layer 0 TP1\\nAttention (QKV)\\nGPU: 64-127\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP2 [label="Layer 0 TP2\\nAttention (QKV)\\nGPU: 128-191\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP3 [label="Layer 0 TP3\\nAttention (QKV)\\nGPU: 192-255\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                
                // All-Reduce for TP
                AR_Layer0 [shape=ellipse, label="All-Reduce\\nTP4 Reduction\\nGPU: 0-255\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Expert Parallel for MoE
                subgraph cluster_ep_layer0 {
                    label="EP16: 4 experts per GPU";
                    style=filled;
d fillcolor=lightcyan;
                    
                    Expert_L0_GPU0 [label="Expert 0-3\\nGPU: 0\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L0_GPU1 [label="Expert 4-7\\nGPU: 1\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L0_GPU15 [label="Expert 60-63\\nGPU: 15\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
                }
                
                // All-to-All for EP
                A2A_Layer0 [shape=ellipse, label="All-to-All\\nEP16 Dispatch\\nGPU: 0-255\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Routing (gate) with dashed line
                Routing_L0 [shape=parallelogram, style=dashed, label="Routing Gate\\nSelect experts\\nGPU: 0-255\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: routing decisions", fillcolor=yellow];
            }
        }
        
        // Similar structure for other stages
        subgraph cluster_stage1 {
            label="Stage 1: Layers 4-7 (GPU 256-511)";
            style=filled;
            fillcolor=lightcoral;
            
            Layer4_TP0 [label="Layer 4 TP0\\nAttention (QKV)\\nGPU: 256-319\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer4_TP1 [label="Layer 4 TP1\\nAttention (QKV)\\nGPU: 320-383\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer4_TP2 [label="Layer 4 TP2\\nAttention (QKV)\\nGPU: 384-447\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer4_TP3 [label="Layer 4 TP3\\nAttention (QKV)\\nGPU: 448-511\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
        }
        
        subgraph cluster_stage2 {
            label="Stage 2: Layers 8-11 (GPU 512-767)";
            style=filled;
            fillcolor=lightcoral;
            
            Layer8_TP0 [label="Layer 8 TP0\\nAttention (QKV)\\nGPU: 512-575\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer8_TP1 [label="Layer 8 TP1\\nAttention (QKV)\\nGPU: 576-639\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer8_TP2 [label="Layer 8 TP2\\nAttention (QKV)\\nGPU: 640-703\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer8_TP3 [label="Layer 8 TP3\\nAttention (QKV)\\nGPU: 704-767\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
        }
        
        subgraph cluster_stage3 {
            label="Stage 3: Layers 12-15 (GPU 768-1023)";
            style=filled;
            fillcolor=lightcoral;
            
            Layer12_TP0 [label="Layer 12 TP0\\nAttention (QKV)\\nGPU: 768-831\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer12_TP1 [label="Layer 12 TP1\\nAttention (QKV)\\nGPU: 832-895\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer12_TP2 [label="Layer 12 TP2\\nAttention (QKV)\\nGPU: 896-959\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
            Layer12_TP3 [label="Layer 12 TP3\\nAttention (QKV)\\nGPU: 960-1023\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=white];
        }
    }
    
    // Pipeline communication
    PP_Comm0 [shape=ellipse, label="Pipeline Comm\\nStage 0→1\\nGPU: 255→256\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgray];
    PP_Comm1 [shape=ellipse, label="Pipeline Comm\\nStage 1→2\\nGPU: 511→512\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgray];
    PP_Comm2 [shape=ellipse, label="Pipeline Comm\\nStage 2→3\\nGPU: 767→768\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=32, seq_len=?, dim=1024]", fillcolor=lightgray];
    
    // DP aggregation
    DP_Aggregate [shape=parallelogram, label="DP Aggregate\\n4-way reduction\\nInput: [batch_size=32, seq_len=?, dim=1024]\\nOutput: [batch_size=128, seq_len=?, dim=1024]", fillcolor=lightgreen];
    
    // Output
    Output [shape=ellipse, label="Output\\nInput: [batch_size=128, seq_len=?, dim=1024]\\nOutput: [batch_size=128, seq_len=?, dim=1024]", fillcolor=lightblue];
    
    // Connections
    Input -> DP_Split;
    DP_Split -> Layer0_TP0;
    DP_Split -> Layer0_TP1;
    DP_Split -> Layer0_TP2;
    DP_Split -> Layer0_TP3;
    
    Layer0_TP0 -> AR_Layer0;
    Layer0_TP1 -> AR_Layer0;
    Layer0_TP2 -> AR_Layer0;
    Layer0_TP3 -> AR_Layer0;
    
    AR_Layer0 -> Routing_L0;
    Routing_L0 -> A2A_Layer0;
    A2A_Layer0 -> Expert_L0_GPU0;
    A2A_Layer0 -> Expert_L0_GPU1;
    A2A_Layer0 -> Expert_L0_GPU15;
    
    Expert_L0_GPU0 -> PP_Comm0;
    Expert_L0_GPU1 -> PP_Comm0;
    Expert_L0_GPU15 -> PP_Comm0;
    PP_Comm0 -> Layer4_TP0;
    
    // Simplified connections for remaining layers
    Layer4_TP0 -> PP_Comm1;
    Layer4_TP1 -> PP_Comm1;
    Layer4_TP2 -> PP_Comm1;
    Layer4_TP3 -> PP_Comm1;
    
    PP_Comm1 -> Layer8_TP0;
    Layer8_TP0 -> PP_Comm2;
    Layer8_TP1 -> PP_Comm2;
    Layer8_TP2 -> PP_Comm2;
    Layer8_TP3 -> PP_Comm2;
    
    PP_Comm2 -> Layer12_TP0;
    Layer12_TP0 -> DP_Aggregate;
    Layer12_TP1 -> DP_Aggregate;
    Layer12_TP2 -> DP_Aggregate;
    Layer12_TP3 -> DP_Aggregate;
    
    DP_Aggregate -> Output;
}'''
    
    return dot_content

def create_new_strategy_dag():
    """Generate DAG for new strategy: EP64 × TP8 × PP2 × DP2 = 2048 GPUs"""
    
    dot_content = '''digraph NewStrategy {
    rankdir=TB;
    node [shape=rectangle, style=filled];
    
    // Graph styling
    graph [bgcolor=white, fontname="Arial", fontsize=12];
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Input node
    Input [shape=ellipse, label="Input\\nInput: [batch_size=128, seq_len=128-10240, dim=1024]\\nOutput: [batch_size=128, seq_len=128-10240, dim=1024]", fillcolor=lightblue];
    
    // Data Parallel split (DP2)
    DP_Split [shape=parallelogram, label="DP Split\\n2-way Data Parallel\\nInput: [batch_size=128, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgreen];
    
    // Pipeline stages (PP2)
    subgraph cluster_pipeline {
        label="Pipeline Parallelism (PP2)";
        style=filled;
        fillcolor=lightyellow;
        
        // Stage 0: Layers 0-7
        subgraph cluster_stage0 {
            label="Stage 0: Layers 0-7 (GPU 0-1023)";
            style=filled;
            fillcolor=lightcoral;
            
            // Tensor Parallel within stage (TP8)
            subgraph cluster_tp_stage0 {
                label="TP8 within Stage 0";
                style=filled;
                fillcolor=lightpink;
                
                // Layer 0 with TP8
                Layer0_TP0 [label="Layer 0 TP0\\nAttention (QKV)\\nGPU: 0-127\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP1 [label="Layer 0 TP1\\nAttention (QKV)\\nGPU: 128-255\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP2 [label="Layer 0 TP2\\nAttention (QKV)\\nGPU: 256-383\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP3 [label="Layer 0 TP3\\nAttention (QKV)\\nGPU: 384-511\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP4 [label="Layer 0 TP4\\nAttention (QKV)\\nGPU: 512-639\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP5 [label="Layer 0 TP5\\nAttention (QKV)\\nGPU: 640-767\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP6 [label="Layer 0 TP6\\nAttention (QKV)\\nGPU: 768-895\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer0_TP7 [label="Layer 0 TP7\\nAttention (QKV)\\nGPU: 896-1023\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                
                // All-Reduce for TP8
                AR_Layer0 [shape=ellipse, label="All-Reduce\\nTP8 Reduction\\nGPU: 0-1023\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Expert Parallel for MoE (EP64)
                subgraph cluster_ep_layer0 {
                    label="EP64: 1 expert per GPU";
                    style=filled;
                    fillcolor=lightcyan;
                    
                    Expert_L0_GPU0 [label="Expert 0\\nGPU: 0\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L0_GPU1 [label="Expert 1\\nGPU: 1\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L0_GPU63 [label="Expert 63\\nGPU: 63\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                }
                
                // All-to-All for EP64
                A2A_Layer0 [shape=ellipse, label="All-to-All\\nEP64 Dispatch\\nGPU: 0-1023\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Routing (gate) with dashed line
                Routing_L0 [shape=parallelogram, style=dashed, label="Routing Gate\\nSelect experts\\nGPU: 0-1023\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: routing decisions", fillcolor=yellow];
            }
        }
        
        // Stage 1: Layers 8-15
        subgraph cluster_stage1 {
            label="Stage 1: Layers 8-15 (GPU 1024-2047)";
            style=filled;
            fillcolor=lightcoral;
            
            // Tensor Parallel within stage (TP8)
            subgraph cluster_tp_stage1 {
                label="TP8 within Stage 1";
                style=filled;
                fillcolor=lightpink;
                
                // Layer 8 with TP8
                Layer8_TP0 [label="Layer 8 TP0\\nAttention (QKV)\\nGPU: 1024-1151\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP1 [label="Layer 8 TP1\\nAttention (QKV)\\nGPU: 1152-1279\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP2 [label="Layer 8 TP2\\nAttention (QKV)\\nGPU: 1280-1407\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP3 [label="Layer 8 TP3\\nAttention (QKV)\\nGPU: 1408-1535\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP4 [label="Layer 8 TP4\\nAttention (QKV)\\nGPU: 1536-1663\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP5 [label="Layer 8 TP5\\nAttention (QKV)\\nGPU: 1664-1791\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP6 [label="Layer 8 TP6\\nAttention (QKV)\\nGPU: 1792-1919\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                Layer8_TP7 [label="Layer 8 TP7\\nAttention (QKV)\\nGPU: 1920-2047\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                
                // All-Reduce for TP8
                AR_Layer8 [shape=ellipse, label="All-Reduce\\nTP8 Reduction\\nGPU: 1024-2047\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Expert Parallel for MoE (EP64)
                subgraph cluster_ep_layer8 {
                    label="EP64: 1 expert per GPU";
                    style=filled;
                    fillcolor=lightcyan;
                    
                    Expert_L8_GPU1024 [label="Expert 0\\nGPU: 1024\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L8_GPU1025 [label="Expert 1\\nGPU: 1025\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                    Expert_L8_GPU1087 [label="Expert 63\\nGPU: 1087\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=white];
                }
                
                // All-to-All for EP64
                A2A_Layer8 [shape=ellipse, label="All-to-All\\nEP64 Dispatch\\nGPU: 1024-2047\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgray];
                
                // Routing (gate) with dashed line
                Routing_L8 [shape=parallelogram, style=dashed, label="Routing Gate\\nSelect experts\\nGPU: 1024-2047\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: routing decisions", fillcolor=yellow];
            }
        }
    }
    
    // Pipeline communication
    PP_Comm0 [shape=ellipse, label="Pipeline Comm\\nStage 0→1\\nGPU: 1023→1024\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=64, seq_len=?, dim=1024]", fillcolor=lightgray];
    
    // DP aggregation
    DP_Aggregate [shape=parallelogram, label="DP Aggregate\\n2-way reduction\\nInput: [batch_size=64, seq_len=?, dim=1024]\\nOutput: [batch_size=128, seq_len=?, dim=1024]", fillcolor=lightgreen];
    
    // Output
    Output [shape=ellipse, label="Output\\nInput: [batch_size=128, seq_len=?, dim=1024]\\nOutput: [batch_size=128, seq_len=?, dim=1024]", fillcolor=lightblue];
    
    // Connections
    Input -> DP_Split;
    DP_Split -> Layer0_TP0;
    DP_Split -> Layer0_TP1;
    DP_Split -> Layer0_TP2;
    DP_Split -> Layer0_TP3;
    DP_Split -> Layer0_TP4;
    DP_Split -> Layer0_TP5;
    DP_Split -> Layer0_TP6;
    DP_Split -> Layer0_TP7;
    
    Layer0_TP0 -> AR_Layer0;
    Layer0_TP1 -> AR_Layer0;
    Layer0_TP2 -> AR_Layer0;
    Layer0_TP3 -> AR_Layer0;
    Layer0_TP4 -> AR_Layer0;
    Layer0_TP5 -> AR_Layer0;
    Layer0_TP6 -> AR_Layer0;
    Layer0_TP7 -> AR_Layer0;
    
    AR_Layer0 -> Routing_L0;
    Routing_L0 -> A2A_Layer0;
    A2A_Layer0 -> Expert_L0_GPU0;
    A2A_Layer0 -> Expert_L0_GPU1;
    A2A_Layer0 -> Expert_L0_GPU63;
    
    Expert_L0_GPU0 -> PP_Comm0;
    Expert_L0_GPU1 -> PP_Comm0;
    Expert_L0_GPU63 -> PP_Comm0;
    PP_Comm0 -> Layer8_TP0;
    
    // Simplified connections for remaining layers
    Layer8_TP0 -> AR_Layer8;
    Layer8_TP1 -> AR_Layer8;
    Layer8_TP2 -> AR_Layer8;
    Layer8_TP3 -> AR_Layer8;
    Layer8_TP4 -> AR_Layer8;
    Layer8_TP5 -> AR_Layer8;
    Layer8_TP6 -> AR_Layer8;
    Layer8_TP7 -> AR_Layer8;
    
    AR_Layer8 -> Routing_L8;
    Routing_L8 -> A2A_Layer8;
    A2A_Layer8 -> Expert_L8_GPU1024;
    A2A_Layer8 -> Expert_L8_GPU1025;
    A2A_Layer8 -> Expert_L8_GPU1087;
    
    Expert_L8_GPU1024 -> DP_Aggregate;
    Expert_L8_GPU1025 -> DP_Aggregate;
    Expert_L8_GPU1087 -> DP_Aggregate;
    
    DP_Aggregate -> Output;
}'''
    
    return dot_content

def main():
    # Create output directory
    output_dir = "../outputs/2025-12-22-17-24-15"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate current strategy DAG
    current_dag = create_current_strategy_dag()
    current_dot_path = os.path.join(output_dir, "current_strategy_dag.dot")
    current_svg_path = os.path.join(output_dir, "current_strategy_dag.svg")
    
    with open(current_dot_path, 'w') as f:
        f.write(current_dag)
    
    # Generate SVG for current strategy
    subprocess.run(['dot', '-Tsvg', current_dot_path, '-o', current_svg_path], check=True)
    
    # Generate new strategy DAG
    new_dag = create_new_strategy_dag()
    new_dot_path = os.path.join(output_dir, "new_strategy_dag.dot")
    new_svg_path = os.path.join(output_dir, "new_strategy_dag.svg")
    
    with open(new_dot_path, 'W') as f:
        f.write(new_dag)
    
    # Generate SVG for new strategy
    subprocess.run(['dot', '-Tsvg', new_dot_path, '-o', new_svg_path], check=True)
    
    # Create submission paths JSON
    submission_paths = {
        "current_strategy_dag_dot": current_dot_path,
        "current_strategy_dag_svg": current_svg_path,
        "new_strategy_dag_dot": new_dot_path,
        "new_strategy_dag_svg": new_svg_path
    }
    
    submission_path = os.path.join(output_dir, "submission_paths.json")
    with open(submission_path, 'w') as f:
        import json
        json.dump(submission_paths, f, indent=2)
    
    print(f"Generated DAGs:")
    print(f"Current strategy DAG: {current_dot_path}")
    print(f"Current strategy SVG: {current_svg_path}")
    print(f"New strategy DAG: {new_dot_path}")
    print(f"New strategy SVG: {new_svg_path}")
    print(f"Submission paths: {submission_path}")

if __name__ == "__main__":
    main()