#!/usr/bin/env python3
"""
Generate DAGs for Llama3-405B deployment with context parallelism
"""

import os
import textwrap

def generate_baseline_dag():
    """Generate baseline DAG with tensor parallelism (TP8) only"""
    dot_content = '''digraph Llama3_405B_Baseline_TP8 {
    rankdir=TB;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    
    // Input layer
    input [label=<<B>Model Input</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]>, shape=parallelogram, fillcolor=lightgreen];
    
    // Embedding layer (replicated across all TP ranks)
    embedding [label=<<B>Token Embedding</B><br/>Input: [batch_size=1, seq_len=128000]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=lightyellow];
    input -> embedding;
    
    // Layer normalization (replicated)
    layer_norm_0 [label=<<B>LayerNorm</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=pink];
    embedding -> layer_norm_0;
    
    // Process 126 transformer layers
    for (i = 1; i <= 126; i++) {
        // Attention Layer i
        subgraph cluster_attention_layer_i {
            label = <<B>Attention Layer i</B><br/>(TP8 across GPUs 0-7)>;
            style=dashed;
            // Query projection (column parallel)
            q_proj_i [label=<<B>Query Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, heads=128, d_head=128]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
            
            // Key projection (column parallel)
            k_proj_i [label=<<B>Key Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
            
            // Value projection (column parallel)
            v_proj_i [label=<<B>Value Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
            
            // Multi-head attention
            attention_i [label=<<B>Multi-Head Attention</B><br/>Input Q: [batch_size=1, seq_len=128000, heads=128, d_head=128]<br/>Input K/V: [batch_size=1, seq_len=128000, kv_heads=8, d_head=128]<br/>Output: [batch_size=1, seq_len=128000, heads=128, d_head=128]<br/>GPU: all GPUs 0-7>, fillcolor=orange];
            
            // Output projection (row parallel)
            o_proj_i [label=<<B>Output Projection</B><br/>Input: [batch_size=1, seq_len=128000, heads=128, d_head=128]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
            
            // Residual connections
            residual_add_i [label=<<B>Residual Add</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384] × 2<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=purple];
            
            // Layer norm after attention
            attn_layer_norm_i [label=<<B>Post-Attention LayerNorm</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=pink];
            
            // MLP Layer i
            subgraph cluster_mlp_layer_i {
                label = <<B>MLP Layer i</B><BR/>(TP8 across GPUs 0-7)>;
                style=dashed;
                
                // Gate projection (column parallel)
                gate_proj_i [label=<<B>Gate Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, ffn_dim=53248]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
                
                // Up projection (column parallel)
                up_proj_i [label=<<B>Up Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, ffn_dim=53248]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
                
                // SiLU activation
                silu_i [label=<<B>SiLU Activation</B><br/>Input: [batch_size=1, seq_len=128000, ffn_dim=53248]<br/>Output: [batch_size=1, seq_len=128000, ffn_dim=53248]>, fillcolor=yellow];
                
                // Element-wise multiply
                elem_mul_i [label=<<B>Element-wise Multiply</B><br/>Input: [batch_size=1, seq_len=128000, ffn_dim=53248] × 2<br/>Output: [batch_size=1, seq_len=128000, ffn_dim=53248]>, fillcolor=yellow];
                
                // Down projection (row parallel)
                down_proj_i [label=<<B>Down Projection</B><br/>Input: [batch_size=1, seq_len=128000, ffn_dim=53248]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]<br/>GPU: all GPUs 0-7>, fillcolor=lightblue];
                
                // MLP residual
                mlp_residual_i [label=<<B>MLP Residual Add</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384] × 2<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=purple];
            }
        }
        
        // Connect attention layer
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
    }
    
    // Final layer norm and output
    final_layer_norm [label=<<B>Final LayerNorm</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, d_model=16384]>, fillcolor=pink];
    output_projection [label=<<B>Output Projection</B><br/>Input: [batch_size=1, seq_len=128000, d_model=16384]<br/>Output: [batch_size=1, seq_len=128000, vocab_size=128256]>, fillcolor=lightblue];
    
    // Connect last layer to final layers
    mlp_residual_126 -> final_layer_norm;
    final_layer_norm -> output_projection;
    
    // Model output
    output [label=<<B>Model Output</B><br/>Input: [batch_size=1, seq_len=128000, vocab_size=128256]>, shape=parallelogram, fillcolor=lightgreen];
    output_projection -> output;
    
    // Device annotations
    subgraph cluster_device_0 {
        label = "GPU 0 (TP Rank 0)";
        style=dashed;
        q_proj_1; k_proj_1; v_proj_1; attention_1; o_proj_1; gate_proj_1; up_proj_1; down_proj_1;
    }
    
    subgraph cluster_device_1 {
        label = "GPU 1 (TP Rank 1)";
        style=dashed;
        q_proj_1; k_proj_1; v_proj_1; attention_1; o_proj_1; gate_proj_1; up_proj_1; down_proj_1;
    }
    
    // ... (similar for GPUs 2-7)
    subgraph cluster_device_7 {
        label = "GPU 7 (TP Rank 7)";
        style=dashed;
        q_proj_1; k_proj_1; v_proj_1; attention_1; o_proj_1; gate_proj_1; up_proj_1; down_proj_1;
    }
}'''
    return dot_content

def generate_context_parallel_dag():
    """Generate DAG with both Tensor Parallel (TP8) and Context Parallel (CP16)"""
    dot_content = '''digraph Llama3_405B_Context_Parallel {
    rankdir=TB;
    node [shape=rectangle, style=filled, fillcolor=lightblue];
    
    // Input layer - context sharding
    input [label=<<B>Model Input</B><BR/>Input: [batch_size=1, seq_len=1000000, d_model=16384]<BR/>Total tokens: 1M across 16 CP ranks>, shape=parallelogram, fillcolor=lightgreen];
    
    // Context sharding
    shard_tokens [label=<<B>Context Shard</B><BR/>Input: [batch_size=1, seq_len=1000000, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384] × 16<BR/>Operation: Double-chunk load balancing>, shape=ellipse, fillcolor=lightyellow];
    input -> shard_tokens;
    
    // Process 126 transformer layers with CP16 + TP8
    for (i = 1; i <= 126; i++) {
        subgraph cluster_layer_i {
            label = <<B>Layer i</B><BR/>Context Parallel (CP16) + Tensor Parallel (TP8)>;
            style=dashed;
            
            // Process each CP rank
            for (r = 0; r < 16; r++) {
                subgraph cluster_cp_rank_r_layer_i {
                    label = <<B>CP Rank r</B><BR/>Node r, GPUs 0-7><BR/>Sequence: 62500 tokens>;
                    style=dashed;
                    
                    // Token embedding for this shard
                    embedding_r_i [label=<<B>Token Embedding Shard r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightyellow];
                    
                    // Layer normalization
                    layer_norm_r_i [label=<<B>LayerNorm Shard r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]<BR/>GPU: node_r.gpu_0-7>, fillcolor=pink];
                    
                    // Attention components for this shard
                    q_proj_r_i [label=<<B>Query Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, heads=128, d_head=128]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    k_proj_r_i [label=<<B>Key Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    v_proj_r_i [label=<<B>Value Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    // KV cache management
                    kv_store_r_i [label=<<B>KV Cache Store r</B><BR/>Store: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]<BR/>Total storage: 62500 tokens × 8 heads × 128 dim<BR/>GPU: node_r.gpu_0-7>, shape=ellipse, fillcolor=lightcoral];
                    
                    // Ring communication for attention - Pass-KV variant
                    kv_ring_send_r_i [label=<<B>KV Ring Send r→r+1</B><BR/>Send: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]<BR/>Network: node_r→node_(r+1)%16>, shape=ellipse, fillcolor=gray];
                    
                    kv_ring_recv_r_i [label=<<B>KV Ring Recv r-1→r</B><BR/>Recv: [batch_size=1, seq_len=62500, kv_heads=8, d_head=128]<BR/>Network: node_(r-1)%16→node_r>, shape=ellipse, fillcolor=gray];
                    
                    // Multi-head attention with cross-shard KV
                    attention_r_i [label=<<B>Multi-Head Attention r</B><BR/>Local Q: [batch_size=1, seq_len=62500, heads=128, d_head=128]<BR/>Global K/V: [batch_size=1, seq_len=1000000, kv_heads=8, d_head=128]<BR/>Output: [batch_size=1, seq_len=62500, heads=128, d_head=128]<BR/>GPU: node_r.gpu_0-7>, fillcolor=orange];
                    
                    o_proj_r_i [label=<<B>Output Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, heads=128, d_head=128]<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    residual_add_r_i [label=<<B>Residual Add r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384] × 2<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]>, fillcolor=purple];
                    
                    attn_layer_norm_r_i [label=<<B>Post-Attention LayerNorm r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]<BR/>GPU: node_r.gpu_0-7>, fillcolor=pink];
                    
                    // MLP components for this shard
                    gate_proj_r_i [label=<<B>Gate Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, ffn_dim=53248]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    up_proj_r_i [label=<<B>Up Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384]<BR/>Output: [batch_size=1, seq_len=62500, ffn_dim=53248]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    silu_r_i [label=<<B>SiLU Activation r</B><BR/>Input: [batch_size=1, seq_len=62500, ffn_dim=53248]<BR/>Output: [batch_size=1, seq_len=62500, ffn_dim=53248]>, fillcolor=yellow];
                    
                    elem_mul_r_i [label=<<B>Element-wise Multiply r</B><BR/>Input: [batch_size=1, seq_len=62500, ffn_dim=53248] × 2<BR/>Output: [batch_size=1, seq_len=62500, ffn_dim=53248]>, fillcolor=yellow];
                    
                    down_proj_r_i [label=<<B>Down Projection r</B><BR/>Input: [batch_size=1, seq_len=62500, ffn_dim=53248]<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]<BR/>GPU: node_r.gpu_0-7>, fillcolor=lightblue];
                    
                    mlp_residual_r_i [label=<<B>MLP Residual Add r</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384] × 2<BR/>Output: [batch_size=1, seq_len=62500, d_model=16384]>, fillcolor=purple];
                }
            }
            
            // Global token aggregation across CP ranks
            gather_tokens [label=<<B>Gather Cross-Shard Results</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384] × 16<BR/>Output: [batch_size=1, seq_len=1000000, d_model=16384]<BR/>Operation: All-gather across CP ranks>, shape=ellipse, fillcolor=gray];
        }
    }
    
    // Final aggregation
    final_gather [label=<<B>Final Token Aggregation</B><BR/>Input: [batch_size=1, seq_len=62500, d_model=16384] × 16<BR/>Output: [batch_size=1, seq_len=1000000, d_model=16384]>, shape=ellipse, fillcolor=gray];
    
    // Final layer norm
    final_layer_norm [label=<<B>Final LayerNorm</B><BR/>Input: [batch_size=1, seq_len=1000000, d_model=16384]<BR/>Output: [batch_size=1, seq_len=1000000, d_model=16384]>, fillcolor=pink];
    
    // Output projection
    output_projection [label=<<B>Output Projection</B><BR/>Input: [batch_size=1, seq_len=1000000, d_model=16384]<BR/>Output: [batch_size=1, seq_len=1000000, vocab_size=128256]>, fillcolor=lightblue];
    
    // Model output
    output [label=<<B>Model Output</B><BR/>Input: [batch_size=1, seq_len=1000000, vocab_size=128256]>, shape=parallelogram, fillcolor=lightgreen];
    
    // Connect final layer
    final_gather -> final_layer_norm;
    final_layer_norm -> output_projection;
    output_projection -> output;
    
    // Device mapping annotations
    subgraph cluster_all_devices {
        label = "128 GPUs Total: 16 Nodes × 8 GPUs Each";
        style=invis;
        
        for (r = 0; r < 16; r++) {
            subgraph cluster_node_r {
                label = <<B>Node r</B><BR/>8× H100 GPUs>;
                style=dashed;
                
                for (g = 0; g < 8; g++) {
                    gpu_r_g [label = <<B>GPU g</B>>, shape=box, style=filled, fillcolor=lightblue];
                }
            }
        }
    }
}'''
    return dot_content

def main():
    output_dir = "../outputs/2025-11-13-11-35-49"
    
    # Generate baseline DAG
    baseline_dag = generate_baseline_dag()
    baseline_path = os.path.join(output_dir, "llama3_405b_baseline_tp8.dot")
    with open(baseline_path, 'w') as f:
        f.write(baseline_dag)
    
    # Generate context parallel DAG
    cp_dag = generate_context_parallel_dag()
    cp_path = os.path.join(output_dir, "llama3_405b_context_parallel.dot")
    with open(cp_path, 'w') as f:
        f.write(cp_dag)
    
    print(f"Generated DAGs at:")
    print(f"  - {baseline_path}")
    print(f"  - {cp_path}")
    
    # Generate SVGs using graphviz
    import subprocess
    try:
        subprocess.run(['dot', '-Tsvg', baseline_path, '-o', baseline_path.replace('.dot', '.svg')], check=True)
        subprocess.run(['dot', '-Tsvg', cp_path, '-o', cp_path.replace('.dot', '.svg')], check=True)
        print("SVG files generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG: {e}")
    
    return [baseline_path, cp_path]

if __name__ == "__main__":
    main()