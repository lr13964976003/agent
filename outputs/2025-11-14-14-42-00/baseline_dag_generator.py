#!/usr/bin/env python3

"""
Baseline DAG Generator (Tensor Parallelism + Pipeline Parallelism)
For 4-layer dense model with TP=8, PP=2 on 16 GPUs
"""

import os

def generate_baseline_dag():
    
    # Model dimensions
    batch_size = 128
    seq_len = 10000
    hidden_size = 16384
    head_count = 32
    head_dim = 128
    layers = 4
    precision = "BF16"
    bytes_per_element = 2
    
    # Tensor parallel dimensions
    tp_size = 8
    pp_size = 2
    
    # Activation dimensions after tensor parallel split
    hidden_per_device = hidden_size // tp_size  # 2048
    
    dot_content = """digraph baseline_deployment {
    rankdir=TB;
    compound=true;
    
    // Graph styling
    node [shape=rectangle, style=filled, fontname="Arial"];
    ellipse [shape=ellipse, style=filled, color=lightgrey];
    parallelogram [shape=parallelogram, style=filled, color=lightblue];
    
    // Input node
    Input [label="Input\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: all GPUs", 
           shape=parallelogram, color=lightblue];
    
    // Stage 0: GPUs 0-7, Layers 0-1
    subgraph cluster_stage0 {
        label="Stage 0\nGPUs: [0,1,2,3,4,5,6,7]\nTensor Parallel Group";
        style=dashed;
        color=blue;
        
        // Layer 0
        Layer0_Input_Split [label="TP Split\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: all GPUs in stage 0", 
                           shape=ellipse, color=lightgrey];
        
        Layer0_Linear1 [label="Linear 1 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                        color=yellow];
        
        Layer0_Activation [label="Activation\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                          color=pink];
        
        Layer0_Linear2 [label="Linear 2 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                        color=yellow];
        
        Layer0_AllReduce [label="TP All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [0,1,2,3,4,5,6,7]", 
                          shape=ellipse, color=lightgrey];
        
        // Layer 1
        Layer1_Input_Split [label="TP Split\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                           shape=ellipse, color=lightgrey];
        
        Layer1_Linear1 [label="Linear 1 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                        color=yellow];
        
        Layer1_Activation [label="Activation\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                          color=pink];
        
        Layer1_Linear2 [label="Linear 2 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [0,1,2,3,4,5,6,7]", 
                        color=yellow];
        
        Layer1_AllReduce [label="TP All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [0,1,2,3,4,5,6,7]", 
                          shape=ellipse, color=lightgrey];
    }
    
    // Pipeline communication between stages
    Stage0_to_Stage1 [label="Pipeline Send\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [0,1,2,3,4,5,6,7] → [8,9,10,11,12,13,14,15]", 
                       shape=ellipse, color=lightgrey];
    
    // Stage 1: GPUs 8-15, Layers 2-3
    subgraph cluster_stage1 {
        label="Stage 1\nGPUs: [8,9,10,11,12,13,14,15]\nTensor Parallel Group";
        style=dashed;
        color=blue;
        
        // Layer 2
        Layer2_Input_Split [label="TP Split\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                           shape=ellipse, color=lightgrey];
        
        Layer2_Linear1 [label="Linear 1 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                        color=yellow];
        
        Layer2_Activation [label="Activation\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                          color=pink];
        
        Layer2_Linear2 [label="Linear 2 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                        color=yellow];
        
        Layer2_AllReduce [label="TP All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [8,9,10,11,12,13,14,15]", 
                          shape=ellipse, color=lightgrey];
        
        // Layer 3
        Layer3_Input_Split [label="TP Split\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                           shape=ellipse, color=lightgrey];
        
        Layer3_Linear1 [label="Linear 1 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                        color=yellow];
        
        Layer3_Activation [label="Activation\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                          color=pink];
        
        Layer3_Linear2 [label="Linear 2 (7.5B params)\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=2048]\nGPU: [8,9,10,11,12,13,14,15]", 
                        color=yellow];
        
        Layer3_AllReduce [label="TP All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden=2048]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [8,9,10,11,12,13,14,15]", 
                          shape=ellipse, color=lightgrey];
    }
    
    // Output aggregation
    Output [label="Output\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [8,9,10,11,12,13,14,15]", 
            shape=parallelogram, color=lightblue];
    
    // Connections
    Input -> Layer0_Input_Split;
    Layer0_Input_Split -> Layer0_Linear1;
    Layer0_Linear1 -> Layer0_Activation;
    Layer0_Activation -> Layer0_Linear2;
    Layer0_Linear2 -> Layer0_AllReduce;
    Layer0_AllReduce -> Layer1_Input_Split;
    Layer1_Input_Split -> Layer1_Linear1;
    Layer1_Linear1 -> Layer1_Activation;
    Layer1_Activation -> Layer1_Linear2;
    Layer1_Linear2 -> Layer1_AllReduce;
    Layer1_AllReduce -> Stage0_to_Stage1;
    Stage0_to_Stage1 -> Layer2_Input_Split;
    Layer2_Input_Split -> Layer2_Linear1;
    Layer2_Linear1 -> Layer2_Activation;
    Layer2_Activation -> Layer2_Linear2;
    Layer2_Linear2 -> Layer2_AllReduce;
    Layer2_AllReduce -> Layer3_Input_Split;
    Layer3_Input_Split -> Layer3_Linear1;
    Layer3_Linear1 -> Layer3_Activation;
    Layer3_Activation -> Layer3_Linear2;
    Layer3_Linear2 -> Layer3_AllReduce;
    Layer3_AllReduce -> Output;
}
"""
    
    # Write DOT file
    with open('../outputs/2025-11-14-14-42-00/baseline_deployment.dot', 'w') as f:
        f.write(dot_content)
    
    # Generate SVG using graphviz
    os.system('dot -Tsvg ../outputs/2025-11-14-14-42-00/baseline_deployment.dot -o ../outputs/2025-11-14-14-42-00/baseline_deployment.svg')
    
    return "baseline_deployment.dot generated successfully"

if __name__ == "__main__":
    result = generate_baseline_dag()
    print(result)