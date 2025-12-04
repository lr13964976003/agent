#!/usr/bin/env python3

import os
import graphviz

def generate_moe_deployment_dag():
    """
    Generate a comprehensive DAG for 30B MoE model deployment with EP8-TP4-PP4 configuration
    """
    
    # Create the main graph
    dot_content = """
digraph MoE_Deployment_EP8_TP4_PP4 {
    rankdir=TB;
    bgcolor=white;
    node [shape=box, style=filled, fillcolor=lightblue];
    edge [color=black, arrowhead=normal];
    
    // Graph attributes
    graph [fontname="Arial", fontsize=12, ranksep=1.2, nodesep=0.8];
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Define node shapes
    node [shape=box]; // Computation nodes (default)
    
    // Input node
    input [shape=ellipse, label="Input\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]", fillcolor=lightgreen];
    
    // EP Groups - 8 groups, each with 16 GPUs
    """
    
    # Generate EP groups
    for ep_group in range(8):
        dot_content += f"""
    // EP Group {ep_group} - GPUs [{ep_group*16}-{ep_group*16+15}]
    subgraph cluster_ep{ep_group} {{
        label="EP Group {ep_group}\\nGPUs [{ep_group*16}-{ep_group*16+15}]";
        style=filled;
        fillcolor=lightyellow;
        color=blue;
        penwidth=2;
        """
        
        # Generate PP stages within each EP group
        for pp_stage in range(4):
            gpu_base = ep_group * 16 + pp_stage * 4
            dot_content += f"""
        // PP Stage {pp_stage} - GPUs [{gpu_base}-{gpu_base+3}]
        subgraph cluster_pp{ep_group}_{pp_stage} {{
            label="PP Stage {pp_stage}\\nGPUs [{gpu_base}-{gpu_base+3}]";
            style=filled;
            fillcolor=lightcyan;
            color=green;
            penwidth=1.5;
            """
            
            # Generate TP groups within each PP stage
            for tp_gpu in range(4):
                gpu_id = gpu_base + tp_gpu
                
                # Layer processing for this GPU
                for layer in range(pp_stage * 4, (pp_stage + 1) * 4):  # 4 layers per stage
                    
                    # Attention computation
                    attn_name = f"attn_gpu{gpu_id}_layer{layer}"
                    dot_content += f"""
            {attn_name} [label="Attention L{layer} GPU{gpu_id}\\nInput: [batch_size=4, seq_len=1024, heads=16, d_k=64]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]", shape=box, fillcolor=lightblue];"""
                    
                    # Expert routing (gate)
                    gate_name = f"gate_gpu{gpu_id}_layer{layer}"
                    dot_content += f"""
            {gate_name} [label="Gate L{layer} GPU{gpu_id}\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, top_2_experts]", shape=parallelogram, fillcolor=orange, style="filled,dashed", penwidth=2];"""
                    
                    # Expert computation (8 experts per GPU)
                    for expert in range(8):
                        expert_name = f"expert_gpu{gpu_id}_layer{layer}_exp{expert}"
                        dot_content += f"""
            {expert_name} [label="Expert {expert} L{layer} GPU{gpu_id}\\nInput: [batch_size=0.5, seq_len=1024, hidden=1024]\\nOutput: [batch_size=0.5, seq_len=1024, hidden=1024]", shape=box, fillcolor=pink];"""
                    
                    # MLP computation
                    mlp_name = f"mlp_gpu{gpu_id}_layer{layer}"
                    dot_content += f"""
            {mlp_name} [label="MLP L{layer} GPU{gpu_id}\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]", shape=box, fillcolor=lightgreen];"""
            
            dot_content += "\n        }\n"
        dot_content += "\n    }\n"
    
    # Communication nodes
    dot_content += """
    // Communication nodes
    node [shape=ellipse, fillcolor=yellow];
    """
    
    # TP All-reduce communications
    for ep_group in range(8):
        for pp_stage in range(4):
            gpu_base = ep_group * 16 + pp_stage * 4
            for layer in range(pp_stage * 4, (pp_stage + 1) * 4):
                tp_comm_name = f"tp_allreduce_ep{ep_group}_pp{pp_stage}_layer{layer}"
                dot_content += f"""
    {tp_comm_name} [label="TP All-Reduce\\nEP{ep_group} PP{pp_stage} L{layer}\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]"];"""
    
    # EP All-to-all communications
    for ep_group in range(8):
        for layer in range(16):
            ep_comm_name = f"ep_alltoall_ep{ep_group}_layer{layer}"
            dot_content += f"""
    {ep_comm_name} [label="EP All-to-All\\nEP{ep_group} L{layer}\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]"];"""
    
    # PP point-to-point communications
    for ep_group in range(8):
        for pp_stage in range(3):  # 3 connections between 4 stages
            pp_comm_name = f"pp_p2p_ep{ep_group}_stage{pp_stage}_to_{pp_stage+1}"
            dot_content += f"""
    {pp_comm_name} [label="PP P2P\\nEP{ep_group} Stage{pp_stage}→{pp_stage+1}\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]"];"""
    
    # Output node
    dot_content += """
    output [shape=ellipse, label="Output\\nInput: [batch_size=4, seq_len=1024, hidden=1024]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]", fillcolor=lightcoral];
    """
    
    # Edges - Simplified for clarity, showing main flow
    dot_content += """
    // Main data flow edges
    edge [color=black, penwidth=1];
    """
    
    # Input to first layer
    for ep_group in range(8):
        gpu_base = ep_group * 16
        first_layer = 0
        first_attn = f"attn_gpu{gpu_base}_layer{first_layer}"
        dot_content += f"""
    input -> {first_attn};"""
    
    # Connections within layers (simplified)
    for ep_group in range(8):
        for pp_stage in range(4):
            gpu_base = ep_group * 16 + pp_stage * 4
            for layer in range(pp_stage * 4, (pp_stage + 1) * 4):
                for tp_gpu in range(4):
                    gpu_id = gpu_base + tp_gpu
                    
                    attn_name = f"attn_gpu{gpu_id}_layer{layer}"
                    gate_name = f"gate_gpu{gpu_id}_layer{layer}"
                    mlp_name = f"mlp_gpu{gpu_id}_layer{layer}"
                    tp_comm_name = f"tp_allreduce_ep{ep_group}_pp{pp_stage}_layer{layer}"
                    
                    # Attention -> Gate
                    dot_content += f"""
    {attn_name} -> {gate_name};"""
                    
                    # Gate -> Experts (dashed for selection)
                    for expert in range(8):
                        expert_name = f"expert_gpu{gpu_id}_layer{layer}_exp{expert}"
                        dot_content += f"""
    {gate_name} -> {expert_name} [style=dashed, penwidth=2];"""
                    
                    # Experts -> MLP (through aggregation)
                    dot_content += f"""
    {gate_name} -> {mlp_name};"""
                    
                    # MLP -> TP All-reduce
                    dot_content += f"""
    {mlp_name} -> {tp_comm_name};"""
    
    # PP stage connections
    for ep_group in range(8):
        for pp_stage in range(3):
            pp_comm_name = f"pp_p2p_ep{ep_group}_stage{pp_stage}_to_{pp_stage+1}"
            
            # Connect last layer of current stage to first layer of next stage
            current_stage_last_layer = (pp_stage + 1) * 4 - 1
            next_stage_first_layer = (pp_stage + 1) * 4
            
            current_gpu_base = ep_group * 16 + pp_stage * 4
            next_gpu_base = ep_group * 16 + (pp_stage + 1) * 4
            
            for tp_gpu in range(4):
                current_gpu_id = current_gpu_base + tp_gpu
                next_gpu_id = next_gpu_base + tp_gpu
                
                current_mlp = f"mlp_gpu{current_gpu_id}_layer{current_stage_last_layer}"
                next_attn = f"attn_gpu{next_gpu_id}_layer{next_stage_first_layer}"
                
                dot_content += f"""
    {current_mlp} -> {pp_comm_name};
    {pp_comm_name} -> {next_attn};"""
    
    # Final output
    for ep_group in range(8):
        gpu_base = ep_group * 16 + 12  # Last PP stage
        last_layer = 15
        last_mlp = f"mlp_gpu{gpu_base}_layer{last_layer}"
        dot_content += f"""
    {last_mlp} -> output;"""
    
    # Add expert aggregation nodes
    for ep_group in range(8):
        for layer in range(16):
            ep_agg_name = f"ep_agg_ep{ep_group}_layer{layer}"
            dot_content += f"""
    {ep_agg_name} [label="Expert Aggregation\\nEP{ep_group} L{layer}\\nInput: [batch_size=4, seq_len=1024, experts=8]\\nOutput: [batch_size=4, seq_len=1024, hidden=1024]", shape=parallelogram, fillcolor=gold];"""
    
    dot_content += "\n}\n"
    
    return dot_content

if __name__ == "__main__":
    # Generate the DAG
    dag_content = generate_moe_deployment_dag()
    
    # Save to file
    output_dir = "../outputs/2025-12-04-19-42-08"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write DOT file
    dot_file = os.path.join(output_dir, "moe_deployment_dag.dot")
    with open(dot_file, "w") as f:
        f.write(dag_content)
    
    print(f"Generated DAG file: {dot_file}")
    
    # Generate SVG image using graphviz
    try:
        import graphviz as gv
        # Create source from DOT content
        source = gv.Source(dag_content)
        # Render to SVG
        svg_file = os.path.join(output_dir, "moe_deployment_dag.svg")
        source.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
        print(f"Generated SVG image: {svg_file}")
    except ImportError:
        print("graphviz module not available, skipping SVG generation")
    except Exception as e:
        print(f"Error generating SVG: {e}")
    
    # Create submission paths JSON
    submission_paths = {
        "dag_dot_file": dot_file,
        "dag_svg_file": os.path.join(output_dir, "moe_deployment_dag.svg"),
        "timestamp": "2025-12-04-19-42-08"
    }
    
    import json
    json_file = os.path.join(output_dir, "submission_paths.json")
    with open(json_file, "w") as f:
        json.dump(submission_paths, f, indent=2)
    
    print(f"Generated submission paths: {json_file}")
    print("DAG generation completed successfully!")