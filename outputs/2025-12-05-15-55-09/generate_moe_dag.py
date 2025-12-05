#!/usr/bin/env python3
"""
Generate DAG for 30B-MoE model deployment with PP=4, EP=16, DP=8
"""

import graphviz
import os

def create_moe_dag():
    # Create directed graph
    dot = graphviz.Digraph(comment='30B-MoE Deployment DAG')
    dot.attr(rankdir='TB', size='30,40', fontsize='12')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Global configuration
    pp_stages = 4
    ep_groups = 16
    dp_replicas = 8
    layers_per_stage = 4
    experts_per_gpu = 4
    
    # Color scheme
    colors = {
        'input': 'lightgreen',
        'output': 'lightcoral',
        'compute': 'lightblue',
        'comm': 'yellow',
        'router': 'orange',
        'aggregate': 'purple'
    }
    
    # Track nodes for connection
    nodes = {}
    
    # Create input node
    dot.node('input', f'Input\\nBatch: [batch_size=128, seq_len=128-10240]\\nGlobal Batch Split', 
             shape='ellipse', fillcolor=colors['input'])
    
    # Process each pipeline stage
    for stage in range(pp_stages):
        stage_label = f'Pipeline Stage {stage}'
        
        # Process each data parallel replica
        for dp in range(dp_replicas):
            # Process each expert parallel group
            for ep in range(ep_groups):
                gpu_id = stage * ep_groups * dp_replicas + ep * dp_replicas + dp
                
                # Layer processing for this GPU
                for layer_idx in range(layers_per_stage):
                    global_layer = stage * layers_per_stage + layer_idx
                    
                    # Self-Attention Layer
                    attn_node = f'gpu{gpu_id}_layer{global_layer}_attn'
                    dot.node(attn_node, 
                             f'GPU{gpu_id} Layer{global_layer} Attention\\n'
                             f'Input: [batch_size=16, seq_len=?, heads=16, d_k=64]\\n'
                             f'Output: [batch_size=16, seq_len=?, hidden=1024]',
                             fillcolor=colors['compute'])
                    
                    # Attention communication (if needed)
                    if layer_idx == 0 and stage > 0:
                        # Need to receive from previous stage
                        prev_gpu = (stage-1) * ep_groups * dp_replicas + ep * dp_replicas + dp
                        comm_node = f'comm_{prev_gpu}_to_{gpu_id}_layer{global_layer}'
                        dot.node(comm_node, f'Communication\\nGPU{prev_gpu}→GPU{gpu_id}\\n'
                                             f'[batch_size=16, seq_len=?, hidden=1024]',
                                shape='ellipse', fillcolor=colors['comm'])
                        dot.edge(comm_node, attn_node)
                    
                    # MoE Layer
                    moe_node = f'gpu{gpu_id}_layer{global_layer}_moe'
                    dot.node(moe_node,
                             f'GPU{gpu_id} Layer{global_layer} MoE Router\\n'
                             f'Input: [batch_size=16, seq_len=?, hidden=1024]\\n'
                             f'Output: [batch_size=16, seq_len=?, hidden=1024]',
                             shape='parallelogram', fillcolor=colors['router'])
                    
                    # Expert processing
                    for expert_id in range(experts_per_gpu):
                        global_expert = ep * experts_per_gpu + expert_id
                        expert_node = f'gpu{gpu_id}_layer{global_layer}_expert{global_expert}'
                        dot.node(expert_node,
                                 f'GPU{gpu_id} Layer{global_layer} Expert{global_expert}\\n'
                                 f'Input: [batch_size=?, seq_len=?, hidden=1024]\\n'
                                 f'Output: [batch_size=?, seq_len=?, hidden=1024]',
                                 fillcolor=colors['compute'])
                        
                        # Router to expert connection (dashed line for gating)
                        dot.edge(moe_node, expert_node, style='dashed', 
                                label=f'Gating Score Expert{global_expert}')
                        
                        # Expert aggregation
                        agg_node = f'gpu{gpu_id}_layer{global_layer}_agg'
                        if expert_id == 0:  # Create aggregation node only once
                            dot.node(agg_node,
                                     f'GPU{gpu_id} Layer{global_layer} Expert Aggregation\\n'
                                     f'Input: Multiple expert outputs\\n'
                                     f'Output: [batch_size=16, seq_len=?, hidden=1024]',
                                     shape='parallelogram', fillcolor=colors['aggregate'])
                        
                        dot.edge(expert_node, agg_node)
                    
                    # Connect attention to MoE
                    dot.edge(attn_node, moe_node)
                    
                    # Store nodes for final connections
                    if global_layer == 15:  # Last layer
                        if 'final_outputs' not in nodes:
                            nodes['final_outputs'] = []
                        nodes['final_outputs'].append(agg_node)
    
    # Create final output node
    dot.node('output', f'Output\\nBatch: [batch_size=128, seq_len=128-10240]\\nGlobal Batch Merge',
             shape='ellipse', fillcolor=colors['output'])
    
    # Connect final layer outputs to output
    for final_node in nodes.get('final_outputs', []):
        dot.edge(final_node, 'output')
    
    # Add expert parallel communication edges
    for stage in range(pp_stages):
        for dp in range(dp_replicas):
            for ep in range(ep_groups):
                gpu_id = stage * ep_groups * dp_replicas + ep * dp_replicas + dp
                for layer_idx in range(layers_per_stage):
                    global_layer = stage * layers_per_stage + layer_idx
                    
                    # Add all-reduce communication for expert gradients
                    comm_node = f'comm_ep_allreduce_gpu{gpu_id}_layer{global_layer}'
                    dot.node(comm_node,
                             f'Expert Parallel All-Reduce\\nGPU{gpu_id} Layer{global_layer}\\n'
                             f'[batch_size=16, seq_len=?, hidden=1024]',
                             shape='ellipse', fillcolor=colors['comm'])
                    
                    # Connect to expert aggregation
                    agg_node = f'gpu{gpu_id}_layer{global_layer}_agg'
                    dot.edge(agg_node, comm_node)
    
    # Add data parallel communication edges
    for stage in range(pp_stages):
        for ep in range(ep_groups):
            for layer_idx in range(layers_per_stage):
                global_layer = stage * layers_per_stage + layer_idx
                
                # Add all-reduce for data parallel gradients
                dp_comm_node = f'comm_dp_allreduce_stage{stage}_ep{ep}_layer{global_layer}'
                dot.node(dp_comm_node,
                         f'Data Parallel All-Reduce\\nStage{stage} EP{ep} Layer{global_layer}\\n'
                         f'Across 8 DP replicas',
                         shape='ellipse', fillcolor=colors['comm'])
                
                # Connect to all DP replicas in this stage/ep group
                for dp in range(dp_replicas):
                    gpu_id = stage * ep_groups * dp_replicas + ep * dp_replicas + dp
                    agg_node = f'gpu{gpu_id}_layer{global_layer}_agg'
                    dot.edge(agg_node, dp_comm_node)
    
    return dot

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "../outputs/2025-12-05-15-55-09"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "moe_deployment_dag.dot")
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    
    # Render as SVG
    svg_file = os.path.join(output_dir, "moe_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")