#!/usr/bin/env python3
"""
Generate simplified MoE deployment DAG showing representative layers.
This provides a more comprehensible view of the overall architecture.
"""

import os

def generate_simplified_dag():
    """Generate a simplified DAG showing key layers"""
    
    dot_content = """digraph MoE_Simplified_Deployment {
    rankdir=TB;
    bgcolor="#f8f9fa";
    node [shape=record, style=rounded, fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Graph metadata
    label="Large-Scale Cross-Node Expert Parallelism (Simplified View)\\n61-Layer MoE Transformer (3 Dense + 58 MoE)\\n3,715 GPUs across 488 Nodes\\nOne Expert per GPU Strategy";
    labelloc=t;
    fontsize=16;
    fontname="Arial Bold";
    
    // Input node
    input [shape=ellipse, style=filled, fillcolor="#e3f2fd", 
           label="Input\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: Host"];
    
    // Dense Layer 0 (Representative)
    dense_0_mha [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 0 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3712 (Node 464, GPU 0)"];
    
    dense_0_ffn [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 0 FFN\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3712 (Node 464, GPU 0)"];
    
    dense_0_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                     label="Dense Layer 0 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3712 (Node 464, GPU 0)"];
    
    // Dense Layer 1 (Representative)
    dense_1_mha [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 1 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3713 (Node 464, GPU 1)"];
    
    dense_1_ffn [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 1 FFN\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3713 (Node 464, GPU 1)"];
    
    dense_1_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                     label="Dense Layer 1 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3713 (Node 464, GPU 1)"];
    
    // Dense Layer 2 (Representative)
    dense_2_mha [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 2 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3714 (Node 464, GPU 2)"];
    
    dense_2_ffn [shape=rectangle, style=filled, fillcolor="#fff3e0",
                label="Dense Layer 2 FFN\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3714 (Node 464, GPU 2)"];
    
    dense_2_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                     label="Dense Layer 2 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3714 (Node 464, GPU 2)"];
    
    // MoE Layer 3 (First MoE Layer - Detailed)
    moe_3_mha [shape=rectangle, style=filled, fillcolor="#e1f5fe",
              label="MoE Layer 3 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 0 (Node 0, GPU 0)"];
    
    moe_3_gate [shape=parallelogram, style=filled, fillcolor="#fff9c4",
               label="MoE Layer 3 Gate\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, top_k=2]\\nGPU: 0 (Node 0, GPU 0)"];
    
    // Expert cluster for Layer 3 (showing representative experts)
    moe_3_experts [shape=rectangle, style=filled, fillcolor="#f3e5f5",
                  label="MoE Layer 3 Expert Cluster (64 experts)\\nGPU Range: 0-63 (Nodes 0-7)\\nEach expert on separate GPU\\nTop-2 experts selected per token"];
    
    moe_3_agg [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
              label="MoE Layer 3 Expert Aggregation\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 0 (Node 0, GPU 0)"];
    
    moe_3_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                   label="MoE Layer 3 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 0 (Node 0, GPU 0)"];
    
    // MoE Layer 4 (Representative)
    moe_4_mha [shape=rectangle, style=filled, fillcolor="#e1f5fe",
              label="MoE Layer 4 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 64 (Node 8, GPU 0)"];
    
    moe_4_gate [shape=parallelogram, style=filled, fillcolor="#fff9c4",
               label="MoE Layer 4 Gate\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, top_k=2]\\nGPU: 64 (Node 8, GPU 0)"];
    
    moe_4_experts [shape=rectangle, style=filled, fillcolor="#f3e5f5",
                  label="MoE Layer 4 Expert Cluster (64 experts)\\nGPU Range: 64-127 (Nodes 8-15)\\nEach expert on separate GPU\\nTop-2 experts selected per token"];
    
    moe_4_agg [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
              label="MoE Layer 4 Expert Aggregation\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 64 (Node 8, GPU 0)"];
    
    moe_4_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                   label="MoE Layer 4 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 64 (Node 8, GPU 0)"];
    
    // MoE Layer 60 (Last MoE Layer - Representative)
    moe_60_mha [shape=rectangle, style=filled, fillcolor="#e1f5fe",
               label="MoE Layer 60 MHA (All Components)\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3648 (Node 456, GPU 0)"];
    
    moe_60_gate [shape=parallelogram, style=filled, fillcolor="#fff9c4",
                label="MoE Layer 60 Gate\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, top_k=2]\\nGPU: 3648 (Node 456, GPU 0)"];
    
    moe_60_experts [shape=rectangle, style=filled, fillcolor="#f3e5f5",
                   label="MoE Layer 60 Expert Cluster (64 experts)\\nGPU Range: 3648-3711 (Nodes 456-463)\\nEach expert on separate GPU\\nTop-2 experts selected per token"];
    
    moe_60_agg [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
               label="MoE Layer 60 Expert Aggregation\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3648 (Node 456, GPU 0)"];
    
    moe_60_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                    label="MoE Layer 60 Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: 3648 (Node 456, GPU 0)"];
    
    // Intermediate layers indicator
    intermediate_layers [shape=ellipse, style=dashed, fillcolor="#ffffff",
                        label="... MoE Layers 5-59 (56 layers total) ...\\nFollowing same pattern:\\nEach layer uses 64 GPUs\\nTotal: 3,584 GPUs for intermediate MoE layers"];
    
    // Output
    output [shape=ellipse, style=filled, fillcolor="#e8f5e8",
            label="Output\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: Host"];
    """
    
    # Add edges
    dot_content += """
    // Connections
    input -> dense_0_mha [label="hidden=7168"];
    dense_0_mha -> dense_0_ffn [label="hidden=7168"];
    dense_0_ffn -> dense_0_residual [label="hidden=7168"];
    
    dense_0_residual -> dense_1_mha [label="hidden=7168"];
    dense_1_mha -> dense_1_ffn [label="hidden=7168"];
    dense_1_ffn -> dense_1_residual [label="hidden=7168"];
    
    dense_1_residual -> dense_2_mha [label="hidden=7168"];
    dense_2_mha -> dense_2_ffn [label="hidden=7168"];
    dense_2_ffn -> dense_2_residual [label="hidden=7168"];
    
    dense_2_residual -> moe_3_mha [label="hidden=7168"];
    moe_3_mha -> moe_3_gate [label="hidden=7168"];
    moe_3_gate -> moe_3_experts [style=dashed, label="token routing"];
    moe_3_experts -> moe_3_agg [label="expert outputs"];
    moe_3_agg -> moe_3_residual [label="hidden=7168"];
    
    moe_3_residual -> moe_4_mha [label="hidden=7168"];
    moe_4_mha -> moe_4_gate [label="hidden=7168"];
    moe_4_gate -> moe_4_experts [style=dashed, label="token routing"];
    moe_4_experts -> moe_4_agg [label="expert outputs"];
    moe_4_agg -> moe_4_residual [label="hidden=7168"];
    
    moe_4_residual -> intermediate_layers [label="hidden=7168"];
    intermediate_layers -> moe_60_mha [style=dashed, label="..."];
    
    moe_60_mha -> moe_60_gate [label="hidden=7168"];
    moe_60_gate -> moe_60_experts [style=dashed, label="token routing"];
    moe_60_experts -> moe_60_agg [label="expert outputs"];
    moe_60_agg -> moe_60_residual [label="hidden=7168"];
    
    moe_60_residual -> output [label="hidden=7168"];
    """
    
    dot_content += "}\n"
    
    return dot_content

def main():
    """Main function to generate and save the simplified DAG"""
    
    # Create output directory if it doesn't exist
    output_dir = "../outputs/2025-11-26-11-58-22"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    print("Generating simplified MoE deployment DAG...")
    dag_content = generate_simplified_dag()
    
    # Save the DOT file
    dot_file = os.path.join(output_dir, "moe_simplified_deployment.dot")
    with open(dot_file, 'w') as f:
        f.write(dag_content)
    
    print(f"Simplified DAG saved to: {dot_file}")
    print(f"File size: {len(dag_content)} characters")
    
    # Generate SVG visualization using graphviz
    try:
        import subprocess
        svg_file = os.path.join(output_dir, "moe_simplified_deployment.svg")
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        print(f"SVG visualization saved to: {svg_file}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: graphviz not available for SVG generation")
        print("To generate visualization, install graphviz and run:")
        print(f"dot -Tsvg {dot_file} -o {svg_file}")

if __name__ == "__main__":
    main()