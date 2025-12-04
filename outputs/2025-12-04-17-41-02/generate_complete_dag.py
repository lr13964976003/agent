#!/usr/bin/env python3

import os

def generate_complete_dag():
    """Generate a complete DAG for LLM deployment with EP64_TP2_PP1 strategy"""
    
    dot_content = '''// Complete LLM Deployment DAG - EP64_TP2_PP1 Strategy
// All 64 experts implemented across 16 layers with proper GPU assignments
digraph {
    dpi=300;
    rankdir=TB;
    size="40,60";
    node [fontname=Arial, fontsize=10];
    edge [fontname=Arial, fontsize=9];
    
    // Node style definitions
    node [fillcolor=lightblue, shape=rectangle, style=filled]; // Computation
    node [fillcolor=lightgreen, shape=ellipse, style=filled];  // Communication
    node [fillcolor=yellow, shape=parallelogram, style=filled]; // Routing/Aggregation
    
    // Input layer
    subgraph cluster_input {
        bgcolor=lightgray;
        label="Input Layer";
        style=rounded;
        
        input [label="Input Tokens\nGPU: Broadcast to all 128 GPUs\nInput: [batch_size=128, seq_len=1024, hidden=1024]\nOutput: [batch_size=128, seq_len=1024, hidden=1024]", 
               fillcolor=lightcoral, shape=rectangle];
    }
    
    // Generate all 16 layers with complete MoE structure
'''
    
    # Generate each layer
    for layer in range(1, 17):
        dot_content += f'''
    // Layer {layer}
    subgraph cluster_layer{layer} {{
        bgcolor=lightblue;
        label="Layer {layer} - Attention + MoE (64 Experts)";
        style=rounded;
        
        // Attention components
        attn_norm_{layer} [label="Layer Norm (Attention)\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        attn_q_{layer} [label="Q Projection\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_k_{layer} [label="K Projection\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_v_{layer} [label="V Projection\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        attn_score_{layer} [label="Attention Scores\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        attn_out_{layer} [label="Attention Output\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        // MoE Gate - routing decisions
        moe_gate_{layer} [label="MoE Gate (Top-k routing)\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 64] (expert weights)", fillcolor=yellow, shape=parallelogram];
        
        // Generate all 64 experts with proper GPU assignments
'''
        
        # Generate all 64 experts (each expert uses 2 GPUs for tensor parallelism)
        for expert in range(64):
            gpu_start = expert * 2
            gpu_end = gpu_start + 1
            
            dot_content += f'''
        // Expert {expert} - GPUs {gpu_start}-{gpu_end}
        tp_split_{expert}_{layer} [label="TP Split\nGPU: {gpu_start}-{gpu_end}\nInput: [128, 1024, 16] (tokens per expert)\nOutput: [128, 1024, 8] (split for TP)", fillcolor=lightgreen, shape=ellipse];
        
        expert_compute_0_{expert}_{layer} [label="Expert {expert} Compute Part 0\nGPU: {gpu_start}\nInput: [128, 1024, 8]\nOutput: [128, 1024, 1024] (hidden dim)", fillcolor=lightblue, shape=rectangle];
        expert_compute_1_{expert}_{layer} [label="Expert {expert} Compute Part 1\nGPU: {gpu_end}\nInput: [128, 1024, 8]\nOutput: [128, 1024, 1024] (hidden dim)", fillcolor=lightblue, shape=rectangle];
        
        tp_allreduce_{expert}_{layer} [label="TP All-Reduce\nGPU: {gpu_start}-{gpu_end}\nInput: [128, 1024, 1024] x 2 parts\nOutput: [128, 1024, 2048] (combined)", fillcolor=lightgreen, shape=ellipse];
        
        expert_{expert}_{layer} [label="Expert {expert} Output\nGPU: {gpu_start}-{gpu_end}\nInput: [128, 1024, 2048]\nOutput: [128, 1024, 2048] (expert result)", fillcolor=lightblue, shape=rectangle];
'''
        
        # Expert aggregation (collect all expert outputs)
        expert_agg_{layer} [label="Expert Aggregation (Weighted Sum)\nGPU: All 128 GPUs\nInput: [128, 1024, 2048] x 64 experts\nOutput: [128, 1024, 1024] (final output)", fillcolor=yellow, shape=parallelogram];
        
        # Layer normalization after MoE
        layer_norm_{layer} [label="Layer Norm (Post-MoE)\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        
        // Connect attention components
        dot_content += f'''
        // Attention connections
        attn_norm_{layer} -> attn_q_{layer};
        attn_norm_{layer} -> attn_k_{layer};
        attn_norm_{layer} -> attn_v_{layer};
        attn_q_{layer} -> attn_score_{layer} [label="Q matrix"];
        attn_k_{layer} -> attn_score_{layer} [label="K matrix"];
        attn_v_{layer} -> attn_out_{layer} [label="V matrix"];
        attn_score_{layer} -> attn_out_{layer} [label="Attention weights"];
        attn_out_{layer} -> moe_gate_{layer};
        
        // Connect all 64 experts
'''
        
        # Connect each expert
        for expert in range(64):
            dot_content += f'''
        // Expert {expert} connections
        moe_gate_{layer} -> tp_split_{expert}_{layer} [label="Gate selection {expert}", style=dashed];
        tp_split_{expert}_{layer} -> expert_compute_0_{expert}_{layer};
        tp_split_{expert}_{layer} -> expert_compute_1_{expert}_{layer};
        expert_compute_0_{expert}_{layer} -> tp_allreduce_{expert}_{layer};
        expert_compute_1_{expert}_{layer} -> tp_allreduce_{expert}_{layer};
        tp_allreduce_{expert}_{layer} -> expert_{expert}_{layer};
        expert_{expert}_{layer} -> expert_agg_{layer} [label="Expert {expert} output"];
'''
        
        dot_content += f'''
        // Final connections
        expert_agg_{layer} -> layer_norm_{layer};
    }}
'''
    
    # Output layer
    dot_content += '''
    // Output layer
    subgraph cluster_output {
        bgcolor=lightgray;
        label="Output Layer";
        style=rounded;
        
        output_norm [label="Final Layer Norm\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, 1024]", fillcolor=lightblue, shape=rectangle];
        output_proj [label="Output Projection\nGPU: All 128 GPUs\nInput: [128, 1024, 1024]\nOutput: [128, 1024, vocab_size]", fillcolor=lightblue, shape=rectangle];
        output [label="Output Tokens\nGPU: All 128 GPUs\nInput: [128, 1024, vocab_size]\nOutput: [128, 1024]", fillcolor=lightcoral, shape=rectangle];
    }
    
    // Connect layers
    input -> attn_norm_1;
'''
    
    # Connect all layers sequentially
    for layer in range(1, 16):
        dot_content += f'''    layer_norm_{layer} -> attn_norm_{layer+1};
'''
    
    # Connect final layer to output
    dot_content += '''    layer_norm_16 -> output_norm;
    output_norm -> output_proj;
    output_proj -> output;
}'''
    
    return dot_content

def main():
    # Generate the complete DAG
    dag_content = generate_complete_dag()
    
    # Write to DOT file
    dot_file_path = "../outputs/2025-12-04-17-41-02/llm_deployment_complete_dag.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag_content)
    
    print(f"Complete DAG saved to: {dot_file_path}")
    
    # Generate SVG image using Graphviz
    try:
        import subprocess
        svg_file_path = "../outputs/2025-12-04-17-41-02/llm_deployment_complete_dag.svg"
        subprocess.run(['dot', '-Tsvg', dot_file_path, '-o', svg_file_path], check=True)
        print(f"SVG image saved to: {svg_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG: {e}")
        print("Make sure Graphviz is installed: apt-get install graphviz")
    except FileNotFoundError:
        print("Graphviz not found. Please install it to generate SVG images.")
    
    # Also generate a PNG for easier viewing
    try:
        png_file_path = "../outputs/2025-12-04-17-41-02/llm_deployment_complete_dag.png"
        subprocess.run(['dot', '-Tpng', '-Gdpi=300', dot_file_path, '-o', png_file_path], check=True)
        print(f"PNG image saved to: {png_file_path}")
    except:
        pass

if __name__ == "__main__":
    main()