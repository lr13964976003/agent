#!/usr/bin/env python3
"""
Generate complete MoE deployment DAG for large-scale cross-node expert parallelism.
This script creates a comprehensive DAG showing all 3,715 GPUs with detailed operator-level nodes.
"""

import os

def generate_moe_dag():
    """Generate the complete MoE deployment DAG"""
    
    dot_content = """digraph MoE_Large_Scale_Deployment {
    rankdir=TB;
    bgcolor="#f8f9fa";
    node [shape=record, style=rounded, fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Graph metadata
    label="Large-Scale Cross-Node Expert Parallelism DAG\\n61-Layer MoE Transformer (3 Dense + 58 MoE)\\n3,715 GPUs across 488 Nodes\\nOne Expert per GPU Strategy";
    labelloc=t;
    fontsize=16;
    fontname="Arial Bold";
    
    // Define node styles
    stylesheet=""
    
    // Input node
    input [shape=ellipse, style=filled, fillcolor="#e3f2fd", 
           label="Input\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: Host"];
    
    """
    
    # Generate dense layers (0-2)
    for layer_id in range(3):
        gpu_id = 3712 + layer_id
        node_id = 464
        gpu_within_node = layer_id
        
        dot_content += f"""
    // Dense Layer {layer_id}
    dense_{layer_id}_mha_q [shape=rectangle, style=filled, fillcolor="#fff3e0",
                           label="Dense Layer {layer_id} MHA Q Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_mha_k [shape=rectangle, style=filled, fillcolor="#fff3e0",
                           label="Dense Layer {layer_id} MHA K Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_mha_v [shape=rectangle, style=filled, fillcolor="#fff3e0",
                           label="Dense Layer {layer_id} MHA V Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_mha_attn [shape=rectangle, style=filled, fillcolor="#fff3e0",
                              label="Dense Layer {layer_id} MHA Attention\\nInput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_mha_out [shape=rectangle, style=filled, fillcolor="#fff3e0",
                             label="Dense Layer {layer_id} MHA Output Projection\\nInput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_ffn [shape=rectangle, style=filled, fillcolor="#fff3e0",
                         label="Dense Layer {layer_id} FFN\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    
    dense_{layer_id}_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                              label="Dense Layer {layer_id} Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    """
    
    # Generate MoE layers (3-60)
    for layer_id in range(3, 61):
        base_gpu = (layer_id - 3) * 64
        base_node = (layer_id - 3) * 8
        
        # MHA components for this layer
        mha_gpu = base_gpu  # Use first GPU of the layer for MHA
        mha_node = base_node
        
        dot_content += f"""
    // MoE Layer {layer_id} - MHA components
    moe_{layer_id}_mha_q [shape=rectangle, style=filled, fillcolor="#e1f5fe",
                         label="MoE Layer {layer_id} MHA Q Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    
    moe_{layer_id}_mha_k [shape=rectangle, style=filled, fillcolor="#e1f5fe",
                         label="MoE Layer {layer_id} MHA K Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    
    moe_{layer_id}_mha_v [shape=rectangle, style=filled, fillcolor="#e1f5fe",
                         label="MoE Layer {layer_id} MHA V Projection\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    
    moe_{layer_id}_mha_attn [shape=rectangle, style=filled, fillcolor="#e1f5fe",
                            label="MoE Layer {layer_id} MHA Attention\\nInput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nOutput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    
    moe_{layer_id}_mha_out [shape=rectangle, style=filled, fillcolor="#e1f5fe",
                           label="MoE Layer {layer_id} MHA Output Projection\\nInput: [batch_size=?, seq_len=?, heads=128, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    """
        
        # Gate for expert selection
        dot_content += f"""
    moe_{layer_id}_gate [shape=parallelogram, style=filled, fillcolor="#fff9c4",
                        label="MoE Layer {layer_id} Gate\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, top_k=2]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    """
        
        # Expert computations (64 experts across 64 GPUs)
        for expert_id in range(64):
            gpu_id = base_gpu + expert_id
            node_id = base_node + (expert_id // 8)
            gpu_within_node = expert_id % 8
            
            dot_content += f"""
    moe_{layer_id}_expert_{expert_id} [shape=rectangle, style=filled, fillcolor="#f3e5f5",
                                      label="MoE Layer {layer_id} Expert {expert_id} FFN\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {gpu_id} (Node {node_id}, GPU {gpu_within_node})"];
    """
        
        # Expert aggregation
        dot_content += f"""
    moe_{layer_id}_expert_agg [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                              label="MoE Layer {layer_id} Expert Aggregation\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    
    moe_{layer_id}_residual [shape=parallelogram, style=filled, fillcolor="#e8f5e8",
                            label="MoE Layer {layer_id} Residual Add\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: {mha_gpu} (Node {mha_node}, GPU 0)"];
    """
    
    # Output node
    dot_content += """
    // Output
    output [shape=ellipse, style=filled, fillcolor="#e8f5e8",
            label="Output\\nInput: [batch_size=?, seq_len=?, hidden=7168]\\nOutput: [batch_size=?, seq_len=?, hidden=7168]\\nGPU: Host"];
    
    """
    
    # Add edges - Input to first dense layer
    dot_content += """
    // Connections - Input to Dense Layer 0
    input -> dense_0_mha_q [label="hidden=7168"];
    input -> dense_0_mha_k [label="hidden=7168"];
    input -> dense_0_mha_v [label="hidden=7168"];
    """
    
    # Dense layer connections
    for layer_id in range(3):
        dot_content += f"""
    // Dense Layer {layer_id} connections
    dense_{layer_id}_mha_q -> dense_{layer_id}_mha_attn [label="Q: [heads=128, d_k=128]"];
    dense_{layer_id}_mha_k -> dense_{layer_id}_mha_attn [label="K: [heads=128, d_k=128]"];
    dense_{layer_id}_mha_v -> dense_{layer_id}_mha_attn [label="V: [heads=128, d_k=128]"];
    dense_{layer_id}_mha_attn -> dense_{layer_id}_mha_out [label="[heads=128, d_k=128]"];
    dense_{layer_id}_mha_out -> dense_{layer_id}_ffn [label="hidden=7168"];
    dense_{layer_id}_ffn -> dense_{layer_id}_residual [label="hidden=7168"];
    """
        
        if layer_id == 0:
            dot_content += f"dense_{layer_id}_residual -> dense_1_mha_q [label=\"hidden=7168\"];\n"
        elif layer_id == 1:
            dot_content += f"dense_{layer_id}_residual -> dense_2_mha_q [label=\"hidden=7168\"];\n"
        else:
            dot_content += f"dense_{layer_id}_residual -> moe_3_mha_q [label=\"hidden=7168\"];\n"
    
    # MoE layer connections
    for layer_id in range(3, 61):
        dot_content += f"""
    // MoE Layer {layer_id} connections
    moe_{layer_id}_mha_q -> moe_{layer_id}_mha_attn [label="Q: [heads=128, d_k=128]"];
    moe_{layer_id}_mha_k -> moe_{layer_id}_mha_attn [label="K: [heads=128, d_k=128]"];
    moe_{layer_id}_mha_v -> moe_{layer_id}_mha_attn [label="V: [heads=128, d_k=128]"];
    moe_{layer_id}_mha_attn -> moe_{layer_id}_mha_out [label="[heads=128, d_k=128]"];
    moe_{layer_id}_mha_out -> moe_{layer_id}_gate [label="hidden=7168"];
    """
        
        # Gate to experts (dashed lines for routing)
        for expert_id in range(64):
            dot_content += f"moe_{layer_id}_gate -> moe_{layer_id}_expert_{expert_id} [style=dashed, label=\"tokens\"];\n"
        
        # Experts to aggregation
        for expert_id in range(64):
            dot_content += f"moe_{layer_id}_expert_{expert_id} -> moe_{layer_id}_expert_agg [label=\"hidden=7168\"];\n"
        
        dot_content += f"""
    moe_{layer_id}_expert_agg -> moe_{layer_id}_residual [label="hidden=7168"];
    """
        
        # Layer to layer connections
        if layer_id < 60:
            dot_content += f"moe_{layer_id}_residual -> moe_{layer_id+1}_mha_q [label=\"hidden=7168\"];\n"
        else:
            dot_content += f"moe_{layer_id}_residual -> output [label=\"hidden=7168\"];\n"
    
    dot_content += "}\n"
    
    return dot_content

def main():
    """Main function to generate and save the DAG"""
    
    # Create output directory if it doesn't exist
    output_dir = "../outputs/2025-11-26-11-58-22"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    print("Generating large-scale MoE deployment DAG...")
    dag_content = generate_moe_dag()
    
    # Save the DOT file
    dot_file = os.path.join(output_dir, "moe_large_scale_deployment.dot")
    with open(dot_file, 'w') as f:
        f.write(dag_content)
    
    print(f"DAG saved to: {dot_file}")
    print(f"File size: {len(dag_content)} characters")
    print(f"Estimated nodes: {dag_content.count('[')}")
    print(f"Estimated edges: {dag_content.count('->')}")
    
    # Generate SVG visualization using graphviz
    try:
        import subprocess
        svg_file = os.path.join(output_dir, "moe_large_scale_deployment.svg")
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        print(f"SVG visualization saved to: {svg_file}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: graphviz not available for SVG generation")
        print("To generate visualization, install graphviz and run:")
        print(f"dot -Tsvg {dot_file} -o {svg_file}")

if __name__ == "__main__":
    main()