#!/usr/bin/env python3

import os

def generate_corrected_dag():
    """Generate a corrected DAG that addresses all the identified issues"""
    
    dot_content = '''// Corrected Layer-wise Deployment Strategy DAG - 3 Representative Layers
// Addresses: inconsistent expert parallelism, GPU format consistency, module repetition
// GPU 0 (Layers 0-1): Start Representative
// GPU 4 (Layers 8-9): Middle Representative with Expert Parallelism  
// GPU 7 (Layers 14-15): End Representative

digraph {
    dpi=300
    rankdir=TB
    size="35,45"
    
    // Single consistent node style definition
    node [fontname=Arial, fontsize=10, shape=rectangle, style=filled, fillcolor=lightblue]
    edge [fontname=Arial, fontsize=9]
    
    // Input node
    input [label="Input\nGPU: Host\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", 
           fillcolor=lightcyan, shape=ellipse]
    
    // GPU 0: Layers 0-1 (Start Representative)
    subgraph cluster_gpu0 {
        label="GPU 0: Layers 0-1 (Start)"
        style="rounded,filled"
        fillcolor=lightgray
        
        gpu0_layernorm0 [label="LayerNorm\nGPU: 0\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu0_mha_qkv [label="MHA Q/K/V Proj\nGPU: 0\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu0_mha_attention [label="MHA Attention\nGPU: 0\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu0_mha_out [label="MHA Out Proj\nGPU: 0\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu0_residual0 [label="Residual Add\nGPU: 0\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu0_layernorm1 [label="LayerNorm\nGPU: 0\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu0_ffn_gate [label="FFN Gate\nGPU: 0\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]"]
        gpu0_ffn_experts [label="FFN Experts (Simple)\nGPU: 0\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]"]
        gpu0_ffn_out [label="FFN Out Proj\nGPU: 0\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu0_residual1 [label="Residual Add\nGPU: 0\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
    }
    
    // Communication: GPU 0 to 1
    comm_gpu0_gpu1 [label="Inter-GPU Comm\nGPU: 1\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightgreen, shape=ellipse]
    
    // GPU 4: Layers 8-9 (Middle Representative with Expert Parallelism)
    subgraph cluster_gpu4 {
        label="GPU 4: Layers 8-9 (Middle - Expert Parallelism Demo)"
        style="rounded,filled"
        fillcolor=lightgray
        
        gpu4_layernorm8 [label="LayerNorm\nGPU: 4\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu4_mha_qkv [label="MHA Q/K/V Proj\nGPU: 4\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu4_mha_attention [label="MHA Attention\nGPU: 4\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu4_mha_out [label="MHA Out Proj\nGPU: 4\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu4_residual8 [label="Residual Add\nGPU: 4\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu4_layernorm9 [label="LayerNorm\nGPU: 4\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu4_ffn_gate [label="FFN Gate (Expert Selection)\nGPU: 4\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]", style=dashed]
        gpu4_split_tokens [label="Split Tokens by Expert\nGPU: 4\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu4_expert0 [label="Expert 0 Processing\nGPU: 4\nInput: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]"]
        gpu4_expert1 [label="Expert 1 Processing\nGPU: 4\nInput: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]"]
        gpu4_aggregate_experts [label="Aggregate Expert Outputs\nGPU: 4\nInput1: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]\nInput2: [batch_size=64, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu4_ffn_out [label="FFN Out Proj\nGPU: 4\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu4_residual9 [label="Residual Add\nGPU: 4\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
    }
    
    // Communication: GPU 4 to 5  
    comm_gpu4_gpu5 [label="Inter-GPU Comm\nGPU: 5\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightgreen, shape=ellipse]
    
    // GPU 7: Layers 14-15 (End Representative)
    subgraph cluster_gpu7 {
        label="GPU 7: Layers 14-15 (End)"
        style="rounded,filled"
        fillcolor=lightgray
        
        gpu7_layernorm14 [label="LayerNorm\nGPU: 7\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu7_mha_qkv [label="MHA Q/K/V Proj\nGPU: 7\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu7_mha_attention [label="MHA Attention\nGPU: 7\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]"]
        gpu7_mha_out [label="MHA Out Proj\nGPU: 7\nInput: [batch_size=128, seq_len=10000, num_heads=32, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu7_residual14 [label="Residual Add\nGPU: 7\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu7_layernorm15 [label="LayerNorm\nGPU: 7\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu7_ffn_gate [label="FFN Gate with Cache Opt\nGPU: 7\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]"]
        gpu7_ffn_cache_load [label="Load Expert Weights\nGPU: 7\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
        gpu7_ffn_experts [label="FFN Experts (Cached)\nGPU: 7\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]"]
        gpu7_ffn_out [label="FFN Out Proj\nGPU: 7\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]"]
        gpu7_residual15 [label="Residual Add\nGPU: 7\nInput1: [batch_size=128, seq_len=10000, hidden_size=16384]\nInput2: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightyellow, shape=parallelogram]
    }
    
    // Output node
    output [label="Output\nGPU: Host\nInput: [batch_size=128, seq_len=10000, hidden_size=16384]\nOutput: [batch_size=128, seq_len=10000, hidden_size=16384]", fillcolor=lightcyan, shape=ellipse]
    
    // Connections - Complete pipeline with representative layers
    input -> gpu0_layernorm0
    gpu0_layernorm0 -> gpu0_mha_qkv
    gpu0_mha_qkv -> gpu0_mha_attention
    gpu0_mha_attention -> gpu0_mha_out
    gpu0_mha_out -> gpu0_residual0
    input -> gpu0_residual0
    gpu0_residual0 -> gpu0_layernorm1
    gpu0_layernorm1 -> gpu0_ffn_gate
    gpu0_ffn_gate -> gpu0_ffn_experts
    gpu0_ffn_experts -> gpu0_ffn_out
    gpu0_ffn_out -> gpu0_residual1
    gpu0_residual0 -> gpu0_residual1
    gpu0_residual1 -> comm_gpu0_gpu1
    
    comm_gpu0_gpu1 -> gpu4_layernorm8 [style=dashed, label="... through GPUs 1-3 ..."]
    
    gpu4_layernorm8 -> gpu4_mha_qkv
    gpu4_mha_qkv -> gpu4_mha_attention
    gpu4_mha_attention -> gpu4_mha_out
    gpu4_mha_out -> gpu4_residual8
    gpu4_layernorm8 -> gpu4_residual8
    gpu4_residual8 -> gpu4_layernorm9
    gpu4_layernorm9 -> gpu4_ffn_gate
    gpu4_ffn_gate -> gpu4_split_tokens [style=dashed]
    gpu4_split_tokens -> gpu4_expert0
    gpu4_split_tokens -> gpu4_expert1
    gpu4_expert0 -> gpu4_aggregate_experts
    gpu4_expert1 -> gpu4_aggregate_experts
    gpu4_aggregate_experts -> gpu4_ffn_out
    gpu4_ffn_out -> gpu4_residual9
    gpu4_residual8 -> gpu4_residual9
    gpu4_residual9 -> comm_gpu4_gpu5
    
    comm_gpu4_gpu5 -> gpu7_layernorm14 [style=dashed, label="... through GPUs 5-6 ..."]
    
    gpu7_layernorm14 -> gpu7_mha_qkv
    gpu7_mha_qkv -> gpu7_mha_attention
    gpu7_mha_attention -> gpu7_mha_out
    gpu7_mha_out -> gpu7_residual14
    gpu7_layernorm14 -> gpu7_residual14
    gpu7_residual14 -> gpu7_layernorm15
    gpu7_layernorm15 -> gpu7_ffn_gate
    gpu7_ffn_gate -> gpu7_ffn_cache_load
    gpu7_ffn_cache_load -> gpu7_ffn_experts
    gpu7_ffn_experts -> gpu7_ffn_out
    gpu7_ffn_out -> gpu7_residual15
    gpu7_residual14 -> gpu7_residual15
    gpu7_residual15 -> output
}'''
    
    return dot_content

def main():
    # Generate the corrected DAG
    dot_content = generate_corrected_dag()
    
    # Save DOT file
    dot_file_path = "../outputs/2025-11-28-17-19-06/corrected_layer_wise_deployment.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dot_content)
    
    # Generate SVG using dot command
    svg_file_path = "../outputs/2025-11-28-17-19-06/corrected_layer_wise_deployment.svg"
    
    try:
        # Use dot command to generate SVG
        os.system(f'dot -Tsvg {dot_file_path} -o {svg_file_path}')
        print(f"SVG generated successfully: {svg_file_path}")
    except Exception as e:
        print(f"Error generating SVG: {e}")
    
    print(f"Files generated:")
    print(f"DOT: {dot_file_path}")
    print(f"SVG: {svg_file_path}")
    
    return dot_file_path, svg_file_path

if __name__ == "__main__":
    main()