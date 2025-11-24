#!/usr/bin/env python3
"""
Baseline Dense Transformer DAG Generator
Tensor Parallel (8-way) + Pipeline Parallel (2-way) across 16 GPUs
"""

import os

def generate_baseline_dag():
    """Generate Graphviz DOT code for baseline tensor+pipeline parallel"""
    
    dot_content = '''digraph baseline_dense_transformer {
    rankdir=TB;
    bgcolor="#f8f9fa";
    
    // Graph attributes
    node [shape=rectangle, style="rounded,filled", fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=10];
    
    // Input node
    Input [label="Input\nSequence\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
    
    // Pipeline Stage 0 (devices 0-7)
    subgraph cluster_stage0 {
        label="Pipeline Stage 0 (Devices 0-7)";
        style="rounded,dashed";
        fillcolor="#fff3e0";
        
        // Layer 0
        Split0 [label="Split\nAcross 8 GPUs", shape=parallelogram, fillcolor="#ffecb3"];
        
        L0_QKV_proj [label="QKV Projection\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 4096]\nTP=8", fillcolor="#c8e6c9"];
        L0_QKV_allgather [label="All-Gather\nQKV across TP\nSize: 6250*4096*2", shape=ellipse, fillcolor="#ffe082"];
        L0_Attention [label="Multi-Head Attention\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 4096]\n32 heads, 128 dim", fillcolor="#c8e6c9"];
        L0_Output_proj [label="Output Projection\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 4096]\nTP=8", fillcolor="#c8e6c9"];
        L0_Residual1 [label="Add\nResidual", shape=diamond, fillcolor="#ffccbc"];
        
        L0_MLP_gate [label="MLP Gate\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 16384]\nTP=8", fillcolor="#c8e6c9"];
        L0_MLP_up [label="MLP Up\nInput: [128, 6250, 4096]\nOutput: [128, 6250, 16384]\nTP=8", fillcolor="#c8e6c9"];
        L0_MLP_down [label="MLP Down\nInput: [128, 6250, 16384]\nOutput: [128, 6250, 4096]\nTP=8", fillcolor="#c8e6c9"];
        L0_Residual2 [label="Add\nResidual", shape=diamond, fillcolor="#ffccbc"];
        
        // Layer 1-7 (abbreviated)
        Note1 [label="Layers 1-7\n(Repeated 7 times)\n16 layers total", shape=note, fillcolor="#f5f5f5"];
    }
    
    // Pipeline communication
    SendStage0 [label="Send to\nStage 1", shape=ellipse, fillcolor="#ff8a65"];
    
    // Pipeline Stage 1 (devices 8-15)
    subgraph cluster_stage1 {
        label="Pipeline Stage 1 (Devices 8-15)";
        style="rounded,dashed";
        fillcolor="#e8f5e9";
        
        RecvStage1 [label="Receive from\nStage 0", shape=ellipse, fillcolor="#ff8a65"];
        
        // Layer 8-15
        Note2 [label="Layers 8-15\n(Same as Layer 0)\n8 layers per stage", shape=note, fillcolor="#f5f5f5"];
        
        Merge1 [label="Merge\nFrom 8 GPUs", shape=parallelogram, fillcolor="#ffecb3"];
    }
    
    // Output
    Output [label="Output\nSequence\nB=128, L=100K, D=4096", shape=ellipse, fillcolor="#e3f2fd"];
    
    // Connections
    Input -> Split0;
    Split0 -> L0_QKV_proj;
    L0_QKV_proj -> L0_QKV_allgather;
    L0_QKV_allgather -> L0_Attention;
    L0_Attention -> L0_Output_proj;
    L0_Output_proj -> L0_Residual1;
    L0_Residual1 -> L0_MLP_gate;
    L0_MLP_gate -> L0_MLP_up;
    L0_MLP_up -> L0_MLP_down;
    L0_MLP_down -> L0_Residual2;
    L0_Residual2 -> Note1;
    Note1 -> SendStage0;
    SendStage0 -> RecvStage1;
    RecvStage1 -> Note2;
    Note2 -> Merge1;
    Merge1 -> Output;
    
    // Residual connections
    Split0 -> L0_Residual1 [style=dashed, label="Residual"];
    L0_Residual1 -> L0_Residual2 [style=dashed, label="Residual"];
}
'''
    
    return dot_content

if __name__ == "__main__":
    dag_content = generate_baseline_dag()
    
    # Write DOT file
    with open("../outputs/2025-11-24-16-10-27/baseline_dense_transformer.dot", "w") as f:
        f.write(dag_content)
    
    # Generate SVG using Graphviz
    os.system("dot -Tsvg ../outputs/2025-11-24-16-10-27/baseline_dense_transformer.dot -o ../outputs/2025-11-24-16-10-27/baseline_dense_transformer.svg")
    
    print("Generated baseline DAG files:")
    print("- ../outputs/2025-11-24-16-10-27/baseline_dense_transformer.dot")
    print("- ../outputs/2025-11-24-16-10-27/baseline_dense_transformer.svg")