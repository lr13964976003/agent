#!/usr/bin/env python3

import graphviz
from pathlib import Path

def create_optimized_dag():
    """Create a complete optimized DAG for MoE model with all connections"""
    
    # Create the main graph
    dot = graphviz.Digraph('optimized_complete_moe')
    dot.graph_attr.update({
        'rankdir': 'TB',
        'ranksep': '1.2',
        'nodesep': '0.3',
        'splines': 'ortho'
    })
    
    # Global node attributes
    dot.node_attr.update({
        'shape': 'rectangle',
        'style': 'filled',
        'fillcolor': 'lightblue'
    })
    
    # Input node
    dot.node('input', 
             'Input Embedding\\n[1024,2048,4096]\\nGPU: All',
             shape='ellipse', fillcolor='lightgreen', height='1.2', width='3.5')
    
    # Output node
    dot.node('output',
             'Final Output\\n[1024,2048,4096]\\nGPU: 8-15',
             shape='ellipse', fillcolor='lightgreen', height='1.2', width='3.5')
    
    # Create Pipeline Stage 0
    with dot.subgraph(name='cluster_pipeline_0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='red')
        
        # Create Layer 0
        _create_layer(stage0, 0, 0, 7)
        
        # Create Layer 1
        _create_layer(stage0, 1, 0, 7)
    
    # Create Pipeline Stage 1
    with dot.subgraph(name='cluster_pipeline_1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='blue')
        
        # Create Layer 2
        _create_layer(stage1, 2, 8, 15)
        
        # Create Layer 3
        _create_layer(stage1, 3, 8, 15)
    
    # Create inter-stage communication
    dot.node('stage0_to_stage1',
             'Pipeline Communication\\nStage 0 → Stage 1\\n[1024,2048,4096]',
             shape='parallelogram', fillcolor='gray', height='1.2', width='4')
    
    # Connect all layers
    _connect_layers(dot)
    
    return dot

def _create_layer(graph, layer_num, start_gpu, end_gpu):
    """Create a complete transformer layer with attention and MoE"""
    
    with graph.subgraph(name=f'cluster_layer_{layer_num}') as layer:
        layer.attr(label=f'Layer {layer_num}', style='rounded')
        
        # Attention Block
        with layer.subgraph(name=f'cluster_l{layer_num}_attention') as attn:
            attn.attr(label='Attention Block', style='rounded', color='lightblue')
            
            # Token split
            split = f'l{layer_num}_token_split'
            attn.node(split,
                     f'Token Split\\n[1024,2048,4096] → 8×[128,2048,4096]\\nGPU: {start_gpu}-{end_gpu}',
                     shape='parallelogram', fillcolor='lightcyan')
            
            # QKV Projections for each GPU
            for gpu in range(8):
                gpu_id = start_gpu + gpu
                qkv = f'l{layer_num}_qkv_gpu{gpu}'
                attn.node(qkv,
                         f'QKV Projection GPU{gpu_id}\\n[128,2048,4096]×[4096,512]\\n→ [128,2048,1536]',
                         fillcolor='yellow')
            
            # Attention computations for each GPU
            for gpu in range(8):
                gpu_id = start_gpu + gpu
                attn_node = f'l{layer_num}_attn_gpu{gpu}'
                attn.node(attn_node,
                         f'Multi-Head Attention GPU{gpu_id}\\n[128,2048,1536]\\n→ [128,2048,512]',
                         fillcolor='yellow')
            
            # Output projections for each GPU
            for gpu in range(8):
                gpu_id = start_gpu + gpu
                out = f'l{layer_num}_out_gpu{gpu}'
                attn.node(out,
                         f'Output Projection GPU{gpu_id}\\n[128,2048,512]×[512,4096]\\n→ [128,2048,4096]',
                         fillcolor='yellow')
            
            # Token gather
            gather = f'l{layer_num}_attn_gather'
            attn.node(gather,
                     f'Token Gather\\n8×[128,2048,4096] → [1024,2048,4096]\\nGPU: {start_gpu}-{end_gpu}',
                     shape='parallelogram', fillcolor='lightcyan')
            
            # Residual add
            attn_res = f'l{layer_num}_attn_res'
            attn.node(attn_res,
                     f'Residual Add\\n[1024,2048,4096]\\nGPU: {start_gpu}-{end_gpu}',
                     fillcolor='orange')
        
        # MoE Block
        with layer.subgraph(name=f'cluster_l{layer_num}_moe') as moe:
            moe.attr(label='MoE Block', style='rounded', color='lightcoral')
            
            # Gate network
            gate = f'l{layer_num}_moe_gate'
            moe.node(gate,
                    f'Gate Network\\n[1024,2048,4096]×[4096,16]\\n→ [1024,2048,16]\\nGPU: {start_gpu}-{end_gpu}',
                    fillcolor='lightcoral')
            
            # Expert routing
            route = f'l{layer_num}_expert_route'
            moe.node(route,
                    f'Expert Routing\\n[1024,2048,4096] → 8 experts\\nGPU: {start_gpu}-{end_gpu}',
                    shape='parallelogram', fillcolor='lightcyan')
            
            # Expert processing for each expert
            for expert in range(8):
                gpu_id = start_gpu + expert
                expert_node = f'l{layer_num}_expert{expert}'
                moe.node(expert_node,
                        f'Expert {expert}\\n[128,2048,4096]×[4096,4096]\\n→ [128,2048,4096]\\nGPU: {gpu_id}',
                        fillcolor='lightpink')
            
            # Expert aggregation
            agg = f'l{layer_num}_moe_agg'
            moe.node(agg,
                    f'Expert Aggregation\\n8×[128,2048,4096] → [1024,2048,4096]\\nGPU: {start_gpu}-{end_gpu}',
                    shape='parallelogram', fillcolor='lightcyan')
            
            # MoE residual
            moe_res = f'l{layer_num}_moe_res'
            moe.node(moe_res,
                    f'Residual Add\\n[1024,2048,4096]\\nGPU: {start_gpu}-{end_gpu}',
                    fillcolor='orange')

def _connect_layers(dot):
    """Connect all nodes within and between layers"""
    
    # Connect input to Layer 0
    dot.edge('input', 'l0_token_split')
    
    # Connect Layer 0 attention block
    for gpu in range(8):
        dot.edge('l0_token_split', f'l0_qkv_gpu{gpu}')
        dot.edge(f'l0_qkv_gpu{gpu}', f'l0_attn_gpu{gpu}')
        dot.edge(f'l0_attn_gpu{gpu}', f'l0_out_gpu{gpu}')
        dot.edge(f'l0_out_gpu{gpu}', 'l0_attn_gather')
    
    # Connect Layer 0 attention to MoE
    dot.edge('l0_attn_gather', 'l0_attn_res')
    dot.edge('input', 'l0_attn_res', label='Residual')
    dot.edge('l0_attn_res', 'l0_moe_gate')
    dot.edge('l0_attn_res', 'l0_expert_route')
    
    # Connect Layer 0 experts
    for expert in range(8):
        dot.edge('l0_expert_route', f'l0_expert{expert}')
        dot.edge(f'l0_expert{expert}', 'l0_moe_agg')
    
    dot.edge('l0_moe_agg', 'l0_moe_res')
    dot.edge('l0_attn_res', 'l0_moe_res', label='Residual')
    
    # Connect Layer 0 to Layer 1
    dot.edge('l0_moe_res', 'l1_token_split')
    dot.edge('l0_moe_res', 'l1_attn_res', label='Residual')
    
    # Connect Layer 1 attention block
    for gpu in range(8):
        dot.edge('l1_token_split', f'l1_qkv_gpu{gpu}')
        dot.edge(f'l1_qkv_gpu{gpu}', f'l1_attn_gpu{gpu}')
        dot.edge(f'l1_attn_gpu{gpu}', f'l1_out_gpu{gpu}')
        dot.edge(f'l1_out_gpu{gpu}', 'l1_attn_gather')
    
    dot.edge('l1_attn_gather', 'l1_attn_res')
    dot.edge('l0_moe_res', 'l1_attn_res', label='Residual')
    dot.edge('l1_attn_res', 'l1_moe_gate')
    dot.edge('l1_attn_res', 'l1_expert_route')
    
    # Connect Layer 1 experts
    for expert in range(8):
        dot.edge('l1_expert_route', f'l1_expert{expert}')
        dot.edge(f'l1_expert{expert}', 'l1_moe_agg')
    
    dot.edge('l1_moe_agg', 'l1_moe_res')
    dot.edge('l1_attn_res', 'l1_moe_res', label='Residual')
    
    # Connect Layer 1 to Layer 2 via pipeline communication
    dot.edge('l1_moe_res', 'stage0_to_stage1')
    dot.edge('stage0_to_stage1', 'l2_token_split')
    dot.edge('stage0_to_stage1', 'l2_attn_res', label='Residual')
    
    # Connect Layer 2 attention block
    for gpu in range(8):
        dot.edge('l2_token_split', f'l2_qkv_gpu{gpu}')
        dot.edge(f'l2_qkv_gpu{gpu}', f'l2_attn_gpu{gpu}')
        dot.edge(f'l2_attn_gpu{gpu}', f'l2_out_gpu{gpu}')
        dot.edge(f'l2_out_gpu{gpu}', 'l2_attn_gather')
    
    dot.edge('l2_attn_gather', 'l2_attn_res')
    dot.edge('stage0_to_stage1', 'l2_attn_res', label='Residual')
    dot.edge('l2_attn_res', 'l2_moe_gate')
    dot.edge('l2_attn_res', 'l2_expert_route')
    
    # Connect Layer 2 experts
    for expert in range(8):
        dot.edge('l2_expert_route', f'l2_expert{expert}')
        dot.edge(f'l2_expert{expert}', 'l2_moe_agg')
    
    dot.edge('l2_moe_agg', 'l2_moe_res')
    dot.edge('l2_attn_res', 'l2_moe_res', label='Residual')
    
    # Connect Layer 2 to Layer 3
    dot.edge('l2_moe_res', 'l3_token_split')
    dot.edge('l2_moe_res', 'l3_attn_res', label='Residual')
    
    # Connect Layer 3 attention block
    for gpu in range(8):
        dot.edge('l3_token_split', f'l3_qkv_gpu{gpu}')
        dot.edge(f'l3_qkv_gpu{gpu}', f'l3_attn_gpu{gpu}')
        dot.edge(f'l3_attn_gpu{gpu}', f'l3_out_gpu{gpu}')
        dot.edge(f'l3_out_gpu{gpu}', 'l3_attn_gather')
    
    dot.edge('l3_attn_gather', 'l3_attn_res')
    dot.edge('l2_moe_res', 'l3_attn_res', label='Residual')
    dot.edge('l3_attn_res', 'l3_moe_gate')
    dot.edge('l3_attn_res', 'l3_expert_route')
    
    # Connect Layer 3 experts
    for expert in range(8):
        dot.edge('l3_expert_route', f'l3_expert{expert}')
        dot.edge(f'l3_expert{expert}', 'l3_moe_agg')
    
    dot.edge('l3_moe_agg', 'l3_moe_res')
    dot.edge('l3_attn_res', 'l3_moe_res', label='Residual')
    
    # Final output
    dot.edge('l3_moe_res', 'output')

def main():
    """Generate the optimized DAG"""
    dot = create_optimized_dag()
    
    # Save files
    output_dir = Path('../outputs/2025-10-16-14-45-55')
    
    # Save DOT file
    dot_file = output_dir / 'optimized_complete_moe.dot'
    with open(dot_file, 'w') as f:
        f.write(dot.source)
    
    # Render to SVG
    svg_file = output_dir / 'optimized_complete_moe.svg'
    dot.render(str(dot_file.with_suffix('')), format='svg', cleanup=False)
    
    print(f"Optimized DAG generated:")
    print(f"DOT: {dot_file}")
    print(f"SVG: {svg_file}")

if __name__ == "__main__":
    main()