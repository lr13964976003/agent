#!/usr/bin/env python3

import graphviz
import os

def create_moe_deployment_dag():
    """
    Create a comprehensive DAG for the 30B MoE model deployment with 512 GPUs.
    Shows all parallel dimensions: PP (4), EP (8), TP (4), DP (4)
    """
    
    # Create directed graph
    dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Communication  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    
    # Input specifications
    batch_size = 128
    seq_len = 1024
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    ffn_hidden = 2048
    vocab_size = 51200
    
    # Create subgraphs for different pipeline stages
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(style='filled', fillcolor='lightgray', label='Pipeline Stage 0 (GPUs 0-127)')
        create_pipeline_stage(c0, 'stage0', 0, 127, batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden)
    
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(style='filled', fillcolor='lightgray', label='Pipeline Stage 1 (GPUs 128-255)')
        create_pipeline_stage(c1, 'stage1', 128, 255, batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden)
    
    with dot.subgraph(name='cluster_pipeline_stage_2') as c2:
        c2.attr(style='filled', fillcolor='lightgray', label='Pipeline Stage 2 (GPUs 256-383)')
        create_pipeline_stage(c2, 'stage2', 256, 383, batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden)
    
    with dot.subgraph(name='cluster_pipeline_stage_3') as c3:
        c3.attr(style='filled', fillcolor='lightgray', label='Pipeline Stage 3 (GPUs 384-511)')
        create_pipeline_stage(c3, 'stage3', 384, 511, batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden)
    
    # Add pipeline communication between stages
    add_pipeline_communication(dot, batch_size, seq_len, hidden_size)
    
    return dot

def create_pipeline_stage(subgraph, stage_name, gpu_start, gpu_end, batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden):
    """Create nodes for a single pipeline stage with 4 layers"""
    
    # Data parallel groups within this pipeline stage
    dp_groups = 4
    gpus_per_dp = (gpu_end - gpu_start + 1) // dp_groups  # 32 GPUs per DP group
    
    # Input node for the stage
    subgraph.node(f'{stage_name}_input', 
                  f'Input\\nInput: [batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # Create 4 layers for this pipeline stage
    for layer in range(4):
        layer_name = f'{stage_name}_layer{layer}'
        create_transformer_layer(subgraph, layer_name, layer, gpu_start, gpus_per_dp, 
                               batch_size//4, seq_len, hidden_size, num_heads, head_dim, ffn_hidden)
        
        # Connect layers within stage
        if layer == 0:
            subgraph.edge(f'{stage_name}_input', f'{layer_name}_attention_start')
        else:
            prev_layer = layer - 1
            subgraph.edge(f'{stage_name}_layer{prev_layer}_mlp_end', f'{layer_name}_attention_start')
    
    # Output node for the stage
    subgraph.node(f'{stage_name}_output', 
                  f'Stage Output\\nInput: [batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # Connect last layer to output
    last_layer = 3
    subgraph.edge(f'{stage_name}_layer{last_layer}_mlp_end', f'{stage_name}_output')

def create_transformer_layer(subgraph, layer_name, layer_idx, gpu_start, gpus_per_dp, 
                           batch_size, seq_len, hidden_size, num_heads, head_dim, ffn_hidden):
    """Create a complete transformer layer with attention and MLP, showing all parallel dimensions"""
    
    # Tensor parallel configuration
    tp_groups = 4
    hidden_per_tp = hidden_size // tp_groups  # 256
    heads_per_tp = num_heads // tp_groups  # 4
    ffn_per_tp = ffn_hidden // tp_groups  # 512
    
    # Expert parallel configuration  
    ep_groups = 8
    experts_per_gpu = 8  # 64 experts / 8 EP groups = 8 per GPU
    
    # Attention Layer - Tensor Parallel
    subgraph.node(f'{layer_name}_attention_start',
                  f'Attention Start (Layer {layer_idx})\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # QKV Projection - Column Parallel
    subgraph.node(f'{layer_name}_qkv_proj',
                  f'QKV Projection (TP-Column)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={heads_per_tp}, d_k={head_dim}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # Attention Computation
    subgraph.node(f'{layer_name}_attention_compute',
                  f'Attention Compute (TP)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, heads={heads_per_tp}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={heads_per_tp}, d_k={head_dim}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # Output Projection - Row Parallel
    subgraph.node(f'{layer_name}_attention_out',
                  f'Attention Output (TP-Row)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, heads={heads_per_tp}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_per_tp}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # Attention All-Reduce
    subgraph.node(f'{layer_name}_attention_allreduce',
                  f'Attention All-Reduce (TP)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_per_tp}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # Residual Connection
    subgraph.node(f'{layer_name}_attention_residual',
                  f'Attention + Residual\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # Layer Norm
    subgraph.node(f'{layer_name}_layernorm2',
                  f'Layer Norm 2\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # MoE Gate - This is where token routing decisions are made
    subgraph.node(f'{layer_name}_moe_gate',
                  f'MoE Gate (Routing Decision)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, top_k=2]',
                  shape='parallelogram', fillcolor='yellow')
    
    # Expert Parallelism - All-to-All Communication for token routing
    subgraph.node(f'{layer_name}_expert_dispatch',
                  f'Expert Dispatch (EP All-to-All)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//ep_groups}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # Expert Computation - Each GPU handles 8 experts
    for expert_group in range(8):  # 8 expert groups
        gpu_base = gpu_start + (expert_group * gpus_per_dp // ep_groups)
        subgraph.node(f'{layer_name}_experts_group{expert_group}',
                      f'Experts Group {expert_group}\\nGPU {gpu_base}-{gpu_base+gpus_per_dp//ep_groups-1}\\n8 Experts per GPU\\nInput: [batch={batch_size//ep_groups}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//ep_groups}, seq={seq_len}, hidden={hidden_size}]',
                      shape='rectangle', fillcolor='lightblue')
    
    # Expert Combine - All-to-All Communication
    subgraph.node(f'{layer_name}_expert_combine',
                  f'Expert Combine (EP All-to-All)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size//ep_groups}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # MLP First Linear - Column Parallel
    subgraph.node(f'{layer_name}_mlp_fc1',
                  f'MLP FC1 (TP-Column)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={ffn_per_tp}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # GELU Activation
    subgraph.node(f'{layer_name}_mlp_gelu',
                  f'MLP GELU\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, ffn={ffn_per_tp}]\\nOutput: [batch={batch_size}, seq={seq_len}, ffn={ffn_per_tp}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # MLP Second Linear - Row Parallel
    subgraph.node(f'{layer_name}_mlp_fc2',
                  f'MLP FC2 (TP-Row)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, ffn={ffn_per_tp}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_per_tp}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # MLP All-Reduce
    subgraph.node(f'{layer_name}_mlp_allreduce',
                  f'MLP All-Reduce (TP)\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_per_tp}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='ellipse', fillcolor='lightgreen')
    
    # Final MLP Residual
    subgraph.node(f'{layer_name}_mlp_end',
                  f'MLP + Residual\\nGPU {gpu_start}-{gpu_start+gpus_per_dp-1}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]',
                  shape='rectangle', fillcolor='lightblue')
    
    # Connect all nodes in sequence
    connections = [
        (f'{layer_name}_attention_start', f'{layer_name}_qkv_proj'),
        (f'{layer_name}_qkv_proj', f'{layer_name}_attention_compute'),
        (f'{layer_name}_attention_compute', f'{layer_name}_attention_out'),
        (f'{layer_name}_attention_out', f'{layer_name}_attention_allreduce'),
        (f'{layer_name}_attention_allreduce', f'{layer_name}_attention_residual'),
        (f'{layer_name}_attention_residual', f'{layer_name}_layernorm2'),
        (f'{layer_name}_layernorm2', f'{layer_name}_moe_gate'),
    ]
    
    # Add dashed line for gate routing decision (special requirement)
    subgraph.edge(f'{layer_name}_moe_gate', f'{layer_name}_expert_dispatch', style='dashed')
    subgraph.edge(f'{layer_name}_expert_dispatch', f'{layer_name}_expert_combine')
    
    # Connect expert groups
    for expert_group in range(8):
        subgraph.edge(f'{layer_name}_expert_dispatch', f'{layer_name}_experts_group{expert_group}')
        subgraph.edge(f'{layer_name}_experts_group{expert_group}', f'{layer_name}_expert_combine')
    
    # Continue with MLP
    connections.extend([
        (f'{layer_name}_expert_combine', f'{layer_name}_mlp_fc1'),
        (f'{layer_name}_mlp_fc1', f'{layer_name}_mlp_gelu'),
        (f'{layer_name}_mlp_gelu', f'{layer_name}_mlp_fc2'),
        (f'{layer_name}_mlp_fc2', f'{layer_name}_mlp_allreduce'),
        (f'{layer_name}_mlp_allreduce', f'{layer_name}_mlp_end'),
    ])
    
    # Add all connections
    for src, dst in connections:
        subgraph.edge(src, dst)

def add_pipeline_communication(dot, batch_size, seq_len, hidden_size):
    """Add communication edges between pipeline stages"""
    
    # Stage 0 -> Stage 1
    dot.edge('stage0_output', 'stage1_input', 
             label=f'Pipeline Communication\\n[batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]')
    
    # Stage 1 -> Stage 2  
    dot.edge('stage1_output', 'stage2_input',
             label=f'Pipeline Communication\\n[batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]')
    
    # Stage 2 -> Stage 3
    dot.edge('stage2_output', 'stage3_input',
             label=f'Pipeline Communication\\n[batch={batch_size//4}, seq={seq_len}, hidden={hidden_size}]')
    
    # Final output
    dot.node('final_output',
             f'Final Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, vocab=51200]',
             shape='ellipse', fillcolor='lightgreen')
    
    dot.edge('stage3_output', 'final_output')

def main():
    # Create the DAG
    dag = create_moe_deployment_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-04-17-13-12/moe_deployment_dag.dot'
    dag.save(dot_file)
    
    # Render as SVG
    svg_file = '../outputs/2025-12-04-17-13-12/moe_deployment_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_file}")
    print(f"SVG rendered to: {svg_file}")
    
    # Also save a simplified version for better readability
    create_simplified_dag()

def create_simplified_dag():
    """Create a simplified DAG showing high-level structure"""
    dot = graphviz.Digraph(comment='30B MoE Model Deployment - Simplified View')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Define styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Input
    dot.node('input', 'Input\\n[batch=128, seq=1024, hidden=1024]', shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline stages (high level)
    for stage in range(4):
        gpu_start = stage * 128
        gpu_end = gpu_start + 127
        
        with dot.subgraph(name=f'cluster_stage{stage}') as c:
            c.attr(style='filled', fillcolor='lightgray', label=f'Pipeline Stage {stage}\\nGPUs {gpu_start}-{gpu_end}')
            
            # 4 layers per stage
            for layer in range(4):
                layer_id = stage * 4 + layer
                c.node(f'stage{stage}_layer{layer}', 
                       f'Layer {layer_id}\\n4 Layers Total\\nTP=4, EP=8, DP=4\\n8 Experts/GPU',
                       shape='rectangle', fillcolor='lightblue')
                
                if layer > 0:
                    c.edge(f'stage{stage}_layer{layer-1}', f'stage{stage}_layer{layer}')
    
    # Connect pipeline stages
    for stage in range(3):
        dot.edge(f'stage{stage}_layer3', f'stage{stage+1}_layer0', 
                label='Pipeline Communication')
    
    # Output
    dot.node('output', 'Output\\n[batch=128, seq=1024, vocab=51200]', shape='ellipse', fillcolor='lightgreen')
    dot.edge('stage3_layer3', 'output')
    
    # Save simplified version
    dot.save('../outputs/2025-12-04-17-13-12/moe_deployment_simplified.dot')
    dot.render('../outputs/2025-12-04-17-13-12/moe_deployment_simplified', format='svg', cleanup=True)

if __name__ == '__main__':
    main()