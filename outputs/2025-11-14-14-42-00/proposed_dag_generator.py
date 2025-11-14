#!/usr/bin/env python3

"""
Proposed DAG Generator (Layer-wise Cache-Optimized Deployment)
For 4-layer dense model with 4 GPU groups × 4 GPUs each = 16 GPUs
"""

import os

def generate_proposed_dag():
    
    # Model dimensions
    batch_size = 128
    seq_len = 10000
    hidden_size = 16384
    head_count = 32
    head_dim = 128
    layers = 4
    precision = "BF16"
    bytes_per_element = 2
    
    # Cache-optimized dimensions
    chunk_size = 64  # tokens per chunk
    num_chunks = seq_len // chunk_size  # 157 chunks
    activation_chunk_size = 12582912  # 12 MB
    weight_tile_size = 33554432  # 32 MB
    
    # GPU groups
    gpus_per_group = 4
    total_groups = 4
    
    dot_content = """digraph proposed_deployment {
    rankdir=TB;
    compound=true;
    
    // Graph styling
    node [shape=rectangle, style=filled, fontname="Arial"];
    ellipse [shape=ellipse, style=filled, color=lightgrey];
    parallelogram [shape=parallelogram, style=filled, color=lightblue];
    
    // Model-wide dimensions
    dim_note [label="Dimensions:\nBatch: 128\nSeq: 10000\nHidden: 16384\nPrecision: BF16", 
              shape=note, color=lightyellow];
    
    // Input node
    Input [label="Model Input\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: All GPUs (broadcast)", 
           shape=parallelogram, color=lightblue];
    
    // Chunking node
    Chunking [label="Activation Chunking\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, chunk_size=64, hidden=16384] × 157\nGPU: All GPUs", 
              shape=ellipse, color=lightgrey];
    
    // GPU Group 0: Layer 0
    subgraph cluster_group0 {
        label="GPU Group 0\nGPUs: [0,1,2,3]\nLayer 0 Cache-Optimized";
        style=dashed;
        color=green;
        
        // Weight streaming setup
        WeightStream0 [label="Weight Tile Streaming\nInput: Full weights (15GB)\nOutput: Weight tiles (32MB)\nGPU: [0,1,2,3]\nCache: 32MB", 
                       shape=ellipse, color=lightgrey];
        
        // Layer 0 operations (detailed)
        Layer0_Linear1 [label="Linear 1 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [0,1,2,3]\nCache: 32MB weights", 
                        color=yellow];
        
        Layer0_Activation [label="Activation\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [0,1,2,3]", 
                          color=pink];
        
        Layer0_Linear2 [label="Linear 2 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [0,1,2,3]\nCache: 32MB weights", 
                        color=yellow];
        
        // Cache utilization note
        Cache0 [label="Cache Utilization\nWeights: 32MB\nActivations: 12MB\nBuffers: 4MB\nTotal: 48MB/50MB", 
                shape=note, color=lightgreen];
    }
    
    // GPU-to-GPU transfer between groups
    Transfer0to1 [label="Pipeline Transfer\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [0,1,2,3] → [4,5,6,7]\nAsync overlap", 
                  shape=ellipse, color=lightgrey];
    
    // GPU Group 1: Layer 1
    subgraph cluster_group1 {
        label="GPU Group 1\nGPUs: [4,5,6,7]\nLayer 1 Cache-Optimized";
        style=dashed;
        color=green;
        
        WeightStream1 [label="Weight Tile Streaming\nInput: Full weights (15GB)\nOutput: Weight tiles (32MB)\nGPU: [4,5,6,7]\nCache: 32MB", 
                       shape=ellipse, color=lightgrey];
        
        Layer1_Linear1 [label="Linear 1 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [4,5,6,7]\nCache: 32MB weights", 
                        color=yellow];
        
        Layer1_Activation [label="Activation\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [4,5,6,7]", 
                          color=pink];
        
        Layer1_Linear2 [label="Linear 2 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [4,5,6,7]\nCache: 32MB weights", 
                        color=yellow];
        
        Cache1 [label="Cache Utilization\nWeights: 32MB\nActivations: 12MB\nBuffers: 4MB\nTotal: 48MB/50MB", 
                shape=note, color=lightgreen];
    }
    
    Transfer1to2 [label="Pipeline Transfer\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [4,5,6,7] → [8,9,10,11]\nAsync overlap", 
                  shape=ellipse, color=lightgrey];
    
    // GPU Group 2: Layer 2
    subgraph cluster_group2 {
        label="GPU Group 2\nGPUs: [8,9,10,11]\nLayer 2 Cache-Optimized";
        style=dashed;
        color=green;
        
        WeightStream2 [label="Weight Tile Streaming\nInput: Full weights (15GB)\nOutput: Weight tiles (32MB)\nGPU: [8,9,10,11]\nCache: 32MB", 
                       shape=ellipse, color=lightgrey];
        
        Layer2_Linear1 [label="Linear 1 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [8,9,10,11]\nCache: 32MB weights", 
                        color=yellow];
        
        Layer2_Activation [label="Activation\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [8,9,10,11]", 
                          color=pink];
        
        Layer2_Linear2 [label="Linear 2 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [8,9,10,11]\nCache: 32MB weights", 
                        color=yellow];
        
        Cache2 [label="Cache Utilization\nWeights: 32MB\nActivations: 12MB\nBuffers: 4MB\nTotal: 48MB/50MB", 
                shape=note, color=lightgreen];
    }
    
    Transfer2to3 [label="Pipeline Transfer\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [8,9,10,11] → [12,13,14,15]\nAsync overlap", 
                  shape=ellipse, color=lightgrey];
    
    // GPU Group 3: Layer 3
    subgraph cluster_group3 {
        label="GPU Group 3\nGPUs: [12,13,14,15]\nLayer 3 Cache-Optimized";
        style=dashed;
        color=green;
        
        WeightStream3 [label="Weight Tile Streaming\nInput: Full weights (15GB)\nOutput: Weight tiles (32MB)\nGPU: [12,13,14,15]\nCache: 32MB", 
                       shape=ellipse, color=lightgrey];
        
        Layer3_Linear1 [label="Linear 1 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [12,13,14,15]\nCache: 32MB weights", 
                        color=yellow];
        
        Layer3_Activation [label="Activation\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [12,13,14,15]", 
                          color=pink];
        
        Layer3_Linear2 [label="Linear 2 (Tile processing)\nInput: [batch_size=128, chunk=64, hidden=16384]\nOutput: [batch_size=128, chunk=64, hidden=16384]\nGPU: [12,13,14,15]\nCache: 32MB weights", 
                        color=yellow];
        
        Cache3 [label="Cache Utilization\nWeights: 32MB\nActivations: 12MB\nBuffers: 4MB\nTotal: 48MB/50MB", 
                shape=note, color=lightgreen];
    }
    
    // Chunk aggregation
    ChunkAggregation [label="Chunk Aggregation\nInput: [batch_size=128, chunk=64, hidden=16384] × 157\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [12,13,14,15]", 
                      shape=ellipse, color=lightgrey];
    
    // Output
    Output [label="Model Output\nInput: [batch_size=128, seq_len=10000, hidden=16384]\nOutput: [batch_size=128, seq_len=10000, hidden=16384]\nGPU: [12,13,14,15]", 
            shape=parallelogram, color=lightblue];
    
    // Connections with chunking flow
    Input -> Chunking;
    Chunking -> WeightStream0;
    WeightStream0 -> Layer0_Linear1;
    Layer0_Linear1 -> Layer0_Activation;
    Layer0_Activation -> Layer0_Linear2;
    Layer0_Linear2 -> Transfer0to1;
    Transfer0to1 -> WeightStream1;
    WeightStream1 -> Layer1_Linear1;
    Layer1_Linear1 -> Layer1_Activation;
    Layer1_Activation -> Layer1_Linear2;
    Layer1_Linear2 -> Transfer1to2;
    Transfer1to2 -> WeightStream2;
    WeightStream2 -> Layer2_Linear1;
    Layer2_Linear1 -> Layer2_Activation;
    Layer2_Activation -> Layer2_Linear2;
    Layer2_Linear2 -> Transfer2to3;
    Transfer2to3 -> WeightStream3;
    WeightStream3 -> Layer3_Linear1;
    Layer3_Linear1 -> Layer3_Activation;
    Layer3_Activation -> Layer3_Linear2;
    Layer3_Linear2 -> ChunkAggregation;
    ChunkAggregation -> Output;
}
"""
    
    # Write DOT file
    with open('../outputs/2025-11-14-14-42-00/proposed_deployment.dot', 'w') as f:
        f.write(dot_content)
    
    # Generate SVG using graphviz
    os.system('dot -Tsvg ../outputs/2025-11-14-14-42-00/proposed_deployment.dot -o ../outputs/2025-11-14-14-42-00/proposed_deployment.svg')
    
    return "proposed_deployment.dot generated successfully"

if __name__ == "__main__":
    result = generate_proposed_dag()
    print(result)