#!/usr/bin/env python3
"""
Generate properly formatted DAGs for Llama3-405B deployment
"""

import os

def generate_baseline_dag():
    """Generate baseline DAG with tensor parallelism (TP8) only"""
    dot_content = '''digraph Llama3_405B_Baseline_TP8 {
    rankdir=TB;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    
    // Input layer
    input [label="Model Input\\nInput: [batch_size=1, seq_len=128000, d_model=16384]", 
           shape=parallelogram, fillcolor=lightgreen];
    
    // Embedding layer
    embedding [label="Token Embedding\\nInput: [batch_size=1, seq_len=128000]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
               fillcolor=lightyellow];
    input -> embedding;
    
    // Layer normalization
    layer_norm_0 [label="LayerNorm\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
                  fillcolor=pink];
    embedding -> layer_norm_0;
    
    // Attention Layer 1
    subgraph cluster_attention_layer_1 {
        label = "Attention Layer 1 (TP8 across GPUs 0-7)";
        style=dashed;
        
        // Query projection
        q_proj_1 [label="Query Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, heads=128, d_head=128]\\nGPU: all GPUs 0-7", 
                  fillcolor=lightblue];
        
        // Key projection
        k_proj_1 [label="Key Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]\\nGPU: all GPUs 0-7", 
                  fillcolor=lightblue];
        
        // Value projection
        v_proj_1 [label="Value Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]\\nGPU: all GPUs 0-7", 
                  fillcolor=lightblue];
        
        // Multi-head attention
        attention_1 [label="Multi-Head Attention\\nInput Q: [batch_size=1, seq_len=128000, heads=128, d_head=128]\\nInput K/V: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]\\nOutput: [batch_size=1, seq_len=128000, heads=128, d_head=128]\\nGPU: all GPUs 0-7", 
                     fillcolor=orange];
        
        // Output projection
        o_proj_1 [label="Output Projection\\nInput: [batch_size=1, seq_len=128000, heads=128, d_head=128]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]\\nGPU: all GPUs 0-7", 
                  fillcolor=lightblue];
        
        // Residual connections
        residual_add_1 [label="Residual Add\\nInput: [batch_size=1, seq_len=128000, d_model=16384] x 2\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
                       fillcolor=purple];
        
        // Layer norm after attention
        attn_layer_norm_1 [label="Post-Attention LayerNorm\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
                           fillcolor=pink];
        
        // MLP Layer 1
        gate_proj_1 [label="Gate Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, ffn_dim=53248]\\nGPU: all GPUs 0-7", 
                     fillcolor=lightblue];
        
        up_proj_1 [label="Up Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, ffn_dim=53248]\\nGPU: all GPUs 0-7", 
                   fillcolor=lightblue];
        
        silu_1 [label="SiLU Activation\\nInput: [batch_size=1, seq_len=128000, ffn_dim=53248]\\nOutput: [batch_size=1, seq_len=128000, ffn_dim=53248]", 
                fillcolor=yellow];
        
        elem_mul_1 [label="Element-wise Multiply\\nInput: [batch_size=1, seq_len=128000, ffn_dim=53248] x 2\\nOutput: [batch_size=1, seq_len=128000, ffn_dim=53248]", 
                     fillcolor=yellow];
        
        down_proj_1 [label="Down Projection\\nInput: [batch_size=1, seq_len=128000, ffn_dim=53248]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]\\nGPU: all GPUs 0-7", 
                     fillcolor=lightblue];
        
        mlp_residual_1 [label="MLP Residual Add\\nInput: [batch_size=1, seq_len=128000, d_model=16384] x 2\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
                        fillcolor=purple];
    }
    
    // Connect attention layer 1
    layer_norm_0 -> q_proj_1;
    layer_norm_0 -> k_proj_1;
    layer_norm_0 -> v_proj_1;
    
    q_proj_1 -> attention_1;
    k_proj_1 -> attention_1;
    v_proj_1 -> attention_1;
    attention_1 -> o_proj_1;
    o_proj_1 -> residual_add_1;
    layer_norm_0 -> residual_add_1;
    
    residual_add_1 -> attn_layer_norm_1;
    attn_layer_norm_1 -> gate_proj_1;
    attn_layer_norm_1 -> up_proj_1;
    
    gate_proj_1 -> silu_1;
    up_proj_1 -> elem_mul_1;
    silu_1 -> elem_mul_1;
    elem_mul_1 -> down_proj_1;
    down_proj_1 -> mlp_residual_1;
    residual_add_1 -> mlp_residual_1;
    
    // Layer 126 (using similar structure)
    layer_126_input [label="Layer 126 Input\\nInput: [batch_size=1, seq_len=128000, d_model=16384]", style=invis];
    mlp_residual_1 -> layer_126_input [style=dashed];
    
    layer_126_output [label="Layer 126 Output\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", style=invis];
    layer_126_input -> layer_126_output [style=dashed, label="... 124 layers skipped ..."];
    
    // Final layers
    final_layer_norm [label="Final LayerNorm\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, d_model=16384]", 
                      fillcolor=pink];
    output_projection [label="Output Projection\\nInput: [batch_size=1, seq_len=128000, d_model=16384]\\nOutput: [batch_size=1, seq_len=128000, vocab_size=128256]", 
                       fillcolor=lightblue];
    
    layer_126_output -> final_layer_norm;
    final_layer_norm -> output_projection;
    
    // Model output
    output [label="Model Output\\nInput: [batch_size=1, seq_len=128000, vocab_size=128256]", 
            shape=parallelogram, fillcolor=lightgreen];
    output_projection -> output;
}'''
    return dot_content

def generate_context_parallel_dag():
    """Generate context parallel DAG with CP16+TP8"""
    dot_content = '''digraph Llama3_405B_Context_Parallel {
    rankdir=TB;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    
    // Input layer
    input [label="Model Input\\nInput: [batch_size=1, seq_len=1000000, d_model=16384]\\nTotal tokens: 1M", 
           shape=parallelogram, fillcolor=lightgreen];
    
    // Context sharding
    shard_tokens [label="Context Shard\\nInput: [batch_size=1, seq_len=1000000, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, d_model=16384] x 16\\nOperation: Double-chunk load balancing", 
                  shape=ellipse, fillcolor=lightyellow];
    input -> shard_tokens;
    
    // KV cache initialization across 16 nodes
    init_kv [label="Initialize KV Cache\\nTotal: [batch_size=1, seq_len=1000000, kv_heads=8, d_head=128]\\nPer node: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]", 
             shape=ellipse, fillcolor=lightcoral];
    shard_tokens -> init_kv;
    
    // Layer 1 with CP16+TP8
    subgraph cluster_layer_1 {
        label = "Layer 1 (CP16 + TP8)";
        style=dashed;
        
        // Node 0 processing
        subgraph cluster_node_0_layer_1 {
            label = "Node 0 (CP Rank 0)\\nGPUs 0-7, Sequence: 62500 tokens";
            style=dashed;
            
            // Query projection
            q_proj_0_1 [label="Query Projection\\nInput: [batch_size=1, seq_len=62500, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, heads=128, d_head=128]\\nGPU: node_0.gpu_0-7", 
                        fillcolor=lightblue];
            
            // Key projection
            k_proj_0_1 [label="Key Projection\\nInput: [batch_size=1, seq_len=62500, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]\\nGPU: node_0.gpu_0-7", 
                        fillcolor=lightblue];
            
            // Value projection
            v_proj_0_1 [label="Value Projection\\nInput: [batch_size=1, seq_len=62500, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]\\nGPU: node_0.gpu_0-7", 
                        fillcolor=lightblue];
            
            // KV cache store for this node
            kv_store_0_1 [label="KV Cache Store\\nStore: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]\\nGPU: node_0.gpu_0-7", 
                          shape=ellipse, fillcolor=lightcoral];
            
            // Ring communication for KV
            kv_send_0_1 [label="KV Ring Send\\nSend: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]\\nNetwork: node_0->node_1", 
                         shape=ellipse, fillcolor=gray];
            
            kv_recv_15_1 [label="KV Ring Recv\\nRecv: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]\\nNetwork: node_15->node_0", 
                          shape=ellipse, fillcolor=gray];
            
            // Multi-head attention with global KV
            attention_0_1 [label="Multi-Head Attention\\nLocal Q: [batch_size=1, seq_len=62500, heads=128, d_head=128]\\nGlobal K/V: [batch_size=1, seq_len=1000000, kv_heads=8, d_head=128]\\nOutput: [batch_size=1, seq_len=62500, heads=128, d_head=128]\\nGPU: node_0.gpu_0-7", 
                           fillcolor=orange];
            
            o_proj_0_1 [label="Output Projection\\nInput: [batch_size=1, seq_len=62500, heads=128, d_head=128]\\nOutput: [batch_size=1, seq_len=62500, d_model=16384]\\nGPU: node_0.gpu_0-7", 
                        fillcolor=lightblue];
            
            residual_add_0_1 [label="Residual Add\\nInput: [batch_size=1, seq_len=62500, d_model=16384] x 2\\nOutput: [batch_size=1, seq_len=62500, d_model=16384]", 
                              fillcolor=purple];
            
            // MLP components
            gate_proj_0_1 [label="Gate Projection\\nInput: [batch_size=1, seq_len=62500, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, ffn_dim=53248]\\nGPU: node_0.gpu_0-7", 
                           fillcolor=lightblue];
            
            up_proj_0_1 [label="Up Projection\\nInput: [batch_size=1, seq_len=62500, d_model=16384]\\nOutput: [batch_size=1, seq_len=62500, ffn_dim=53248]\\nGPU: node_0.gpu_0-7", 
                         fillcolor=lightblue];
            
            silu_0_1 [label="SiLU Activation\\nInput: [batch_size=1, seq_len=62500, ffn_dim=53248]\\nOutput: [batch_size=1, seq_len=62500, ffn_dim=53248]", 
                      fillcolor=yellow];
            
            elem_mul_0_1 [label="Element-wise Multiply\\nInput: [batch_size=1, seq_len=62500, ffn_dim=53248] x 2\\nOutput: [batch_size=1, seq_len=62500, ffn_dim=53248]", 
                          fillcolor=yellow];
            
            down_proj_0_1 [label="Down Projection\\nInput: [batch_size=1, seq_len=62500, ffn_dim=53248]\\nOutput: [batch_size=1, seq_len=62500, d_model=16384]\\nGPU: node_0.gpu_0-7", 
                           fillcolor=lightblue];
            
            mlp_residual_0_1 [label="MLP Residual Add\\nInput: [batch_size=1, seq_len=62500, d_model=16384] x 2\\nOutput: [batch_size=1, seq_len=62500, d_model=16384]", 
                              fillcolor=purple];
        }
        
        // Node 1 processing (representing other 15 nodes)
        subgraph cluster_node_1_layer_1 {
            label = "Node 1 (CP Rank 1)\\n...\\nNodes 2-15 similar";
            style=dashed;
            node_1_processing [label="Node 1 Layer 1\\nSame operations as Node 0\\nSequence: 62500 tokens", 
                               style=dotted];
        }
    }
    
    // Global token aggregation
    gather_tokens [label="Gather Cross-Shard Results\\nInput: [batch_size=1, seq_len=62500, d_model=16384] x 16\\nOutput: [batch_size=1, seq_len=1000000, d_model=16384]\\nOperation: All-gather across CP ranks", 
                   shape=ellipse, fillcolor=gray];
    
    // Layer 126 processing (similar structure)
    layer_126 [label="Layer 126\\nSame CP16+TP8 structure\\nOperations repeated 126 times", 
               style=invis];
    
    // Final aggregation
    final_gather [label="Final Token Aggregation\\nInput: [batch_size=1, seq_len=62500, d_model=16384] x 16\\nOutput: [batch_size=1, seq_len=1000000, d_model=16384]", 
                  shape=ellipse, fillcolor=gray];
    
    // Final layer norm
    final_layer_norm [label="Final LayerNorm\\nInput: [batch_size=1, seq_len=1000000, d_model=16384]\\nOutput: [batch_size=1, seq_len=1000000, d_model=16384]", 
                      fillcolor=pink];
    
    // Output projection
    output_projection [label="Output Projection\\nInput: [batch_size=1, seq_len=1000000, d_model=16384]\\nOutput: [batch_size=1, seq_len=1000000, vocab_size=128256]", 
                       fillcolor=lightblue];
    
    // Model output
    output [label="Model Output\\nInput: [batch_size=1, seq_len=1000000, vocab_size=128256]", 
            shape=parallelogram, fillcolor=lightgreen];
    
    // Connections
    shard_tokens -> q_proj_0_1;
    shard_tokens -> k_proj_0_1;
    shard_tokens -> v_proj_0_1;
    
    k_proj_0_1 -> kv_store_0_1;
    v_proj_0_1 -> kv_store_0_1;
    
    // KV ring communication
    kv_store_0_1 -> kv_send_0_1;
    kv_recv_15_1 -> attention_0_1;
    
    // Attention path
    q_proj_0_1 -> attention_0_1;
    kv_store_0_1 -> attention_0_1;
    attention_0_1 -> o_proj_0_1;
    o_proj_0_1 -> residual_add_0_1;
    shard_tokens -> residual_add_0_1;
    
    // MLP path
    residual_add_0_1 -> gate_proj_0_1;
    residual_add_0_1 -> up_proj_0_1;
    gate_proj_0_1 -> silu_0_1;
    up_proj_0_1 -> elem_mul_0_1;
    silu_0_1 -> elem_mul_0_1;
    elem_mul_0_1 -> down_proj_0_1;
    down_proj_0_1 -> mlp_residual_0_1;
    residual_add_0_1 -> mlp_residual_0_1;
    
    // Skip layers 2-125
    mlp_residual_0_1 -> layer_126 [style=dashed, label="... 124 layers ..."];
    layer_126 -> final_gather;
    
    final_gather -> final_layer_norm;
    final_layer_norm -> output_projection;
    output_projection -> output;
}'''
    return dot_content

def main():
    output_dir = "../outputs/2025-11-13-11-35-49"
    
    # Generate baseline DAG
    baseline_dag = generate_baseline_dag()
    baseline_path = os.path.join(output_dir, "baseline_tp8.dot")
    with open(baseline_path, 'w') as f:
        f.write(baseline_dag)
    
    # Generate context parallel DAG
    cp_dag = generate_context_parallel_dag()
    cp_path = os.path.join(output_dir, "context_parallel_tp8_cp16.dot")
    with open(cp_path, 'w') as f:
        f.write(cp_dag)
    
    # Generate SVGs
    import subprocess
    try:
        subprocess.run(['dot', '-Tsvg', baseline_path, '-o', baseline_path.replace('.dot', '.svg')], check=True)
        subprocess.run(['dot', '-Tsvg', cp_path, '-o', cp_path.replace('.dot', '.svg')], check=True)
        print("SVG files generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"Graphviz not available, creating DOT files only: {e}")
    
    return [baseline_path, cp_path]

if __name__ == "__main__":
    paths = main()
    print(f"Generated files: {paths}")