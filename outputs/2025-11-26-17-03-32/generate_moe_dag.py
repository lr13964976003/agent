#!/usr/bin/env python3

import graphviz
import os

def create_moe_dag():
    """
    Generate a comprehensive DAG for Large-Scale Cross-Node Expert Parallelism MoE model
    Following all requirements:
    - One expert per GPU
    - EP >= 16
    - Detailed operator-level breakdown
    - Communication shown with dashed lines
    - Tensor dimensions specified
    - GPU IDs clearly marked
    """
    
    # Model configuration
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    num_layers = 61
    dense_layers = 3
    moe_layers = 58
    experts_per_layer = 64  # Example configuration
    batch_size = 4
    seq_len = 2048
    
    # Create DAG
    dag = graphviz.Digraph('Large_Scale_Cross_Node_MoE', 
                          filename='large_scale_cross_node_moe_dag',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12',
                              'label': 'Large-Scale Cross-Node Expert Parallelism MoE DAG\n61 Layers (3 Dense + 58 MoE), EP=64, One Expert per GPU',
                              'labelloc': 't'
                          })
    
    # Define node styles
    dag.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dag.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dag.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    
    # Input node
    dag.node('input', 
             f'Input Layer\nGPU: ALL_GPUs\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Process through layers
    prev_node = 'input'
    
    for layer_idx in range(num_layers):
        if layer_idx < dense_layers:
            # Dense layer processing
            layer_prefix = f'dense_layer_{layer_idx}'
            
            # MHA for dense layer
            mha_node = f'{layer_prefix}_mha'
            dag.node(mha_node,
                     f'Dense Layer {layer_idx} - MHA\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # MHA Communication (dashed)
            dag.edge(prev_node, mha_node, style='dashed', label='Token Broadcast')
            
            # MHA Output Aggregation
            mha_agg = f'{layer_prefix}_mha_agg'
            dag.node(mha_agg,
                     f'Dense Layer {layer_idx} - MHA Agg\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     shape='parallelogram', fillcolor='yellow')
            
            dag.edge(mha_node, mha_agg)
            
            # MLP for dense layer
            mlp_node = f'{layer_prefix}_mlp'
            dag.node(mlp_node,
                     f'Dense Layer {layer_idx} - MLP\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     shape='rectangle', fillcolor='lightgreen')
            
            dag.edge(mha_agg, mlp_node)
            
            prev_node = mlp_node
            
        else:
            # MoE layer processing
            layer_prefix = f'moe_layer_{layer_idx}'
            
            # MHA for MoE layer (same as dense)
            mha_node = f'{layer_prefix}_mha'
            dag.node(mha_node,
                     f'MoE Layer {layer_idx} - MHA\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                     shape='rectangle', fillcolor='lightgreen')
            
            dag.edge(prev_node, mha_node, style='dashed', label='Token Broadcast')
            
            # MHA Output Aggregation
            mha_agg = f'{layer_prefix}_mha_agg'
            dag.node(mha_agg,
                     f'MoE Layer {layer_idx} - MHA Agg\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     shape='parallelogram', fillcolor='yellow')
            
            dag.edge(mha_node, mha_agg)
            
            # Gating Network
            gating_node = f'{layer_prefix}_gating'
            dag.node(gating_node,
                     f'MoE Layer {layer_idx} - Gating\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, expert_assignments={experts_per_layer}]',
                     shape='rectangle', fillcolor='lightgreen')
            
            dag.edge(mha_agg, gating_node)
            
            # Expert Processing - Create nodes for each expert
            expert_outputs = []
            for expert_id in range(experts_per_layer):
                gpu_id = expert_id % 128  # Distribute across 128 GPUs
                
                # Expert MLP Layer 1
                expert_mlp1 = f'{layer_prefix}_expert_{expert_id}_mlp1'
                dag.node(expert_mlp1,
                         f'MoE Layer {layer_idx} - Expert {expert_id} MLP1\nGPU: GPU_{gpu_id}\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]',
                         shape='rectangle', fillcolor='lightgreen')
                
                # Routing to expert (dashed)
                dag.edge(gating_node, expert_mlp1, style='dashed', 
                        label=f'Route tokens to Expert {expert_id}')
                
                # Expert Activation
                expert_act = f'{layer_prefix}_expert_{expert_id}_act'
                dag.node(expert_act,
                         f'MoE Layer {layer_idx} - Expert {expert_id} GELU\nGPU: GPU_{gpu_id}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]',
                         shape='rectangle', fillcolor='lightgreen')
                
                dag.edge(expert_mlp1, expert_act)
                
                # Expert MLP Layer 2
                expert_mlp2 = f'{layer_prefix}_expert_{expert_id}_mlp2'
                dag.node(expert_mlp2,
                         f'MoE Layer {layer_idx} - Expert {expert_id} MLP2\nGPU: GPU_{gpu_id}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                         shape='rectangle', fillcolor='lightgreen')
                
                dag.edge(expert_act, expert_mlp2)
                expert_outputs.append(expert_mlp2)
            
            # Expert Output Aggregation
            expert_agg = f'{layer_prefix}_expert_agg'
            dag.node(expert_agg,
                     f'MoE Layer {layer_idx} - Expert Aggregation\nGPU: GPU_0_to_GPU_15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] * {experts_per_layer}\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     shape='parallelogram', fillcolor='yellow')
            
            # Connect all experts to aggregation
            for expert_output in expert_outputs:
                dag.edge(expert_output, expert_agg, style='dashed', label='Expert Output')
            
            prev_node = expert_agg
    
    # Output node
    dag.node('output',
             f'Output Layer\nGPU: ALL_GPUs\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    dag.edge(prev_node, 'output')
    
    return dag

def create_simplified_moe_dag():
    """
    Create a simplified but complete DAG showing the key concepts
    """
    
    # Model configuration
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    batch_size = 4
    seq_len = 2048
    experts_per_layer = 64
    
    dag = graphviz.Digraph('Simplified_Cross_Node_MoE',
                          filename='simplified_cross_node_moe_dag',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12',
                              'label': 'Simplified Cross-Node Expert Parallelism MoE\nShowing One Layer with EP=64',
                              'labelloc': 't'
                          })
    
    # Input
    dag.node('input',
             f'Input Tokens\nGPU: Distributed\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # MHA Processing
    dag.node('mha_q',
             f'MHA - Q Projection\nGPU: GPU_0-15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    dag.node('mha_k',
             f'MHA - K Projection\nGPU: GPU_16-31\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    dag.node('mha_v',
             f'MHA - V Projection\nGPU: GPU_32-47\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    dag.node('mha_attn',
             f'MHA - Attention Computation\nGPU: GPU_0-63\nInput: Q,K,V [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    dag.node('mha_out',
             f'MHA - Output Projection\nGPU: GPU_0-63\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    # Gating
    dag.node('gating',
             f'MoE Gating Network\nGPU: GPU_0-15\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: Expert assignments for each token',
             shape='rectangle', fillcolor='lightgreen')
    
    # Expert processing (showing first 8 experts as example)
    expert_nodes = []
    for i in range(8):
        gpu_id = i * 8  # Distribute across GPUs
        
        expert_node = f'expert_{i}'
        dag.node(expert_node,
                 f'Expert {i}\nGPU: GPU_{gpu_id}\nInput: Routed tokens [batch_size=?, seq_len=?, token_dim={token_dim}]\nOutput: [batch_size=?, seq_len=?, token_dim={token_dim}]',
                 shape='rectangle', fillcolor='lightgreen')
        
        expert_nodes.append(expert_node)
    
    # Expert aggregation
    dag.node('expert_agg',
             f'Expert Aggregation\nGPU: GPU_0-15\nInput: Expert outputs from 64 experts\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', fillcolor='yellow')
    
    # Output
    dag.node('output',
             f'Layer Output\nGPU: Distributed\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Connect nodes
    dag.edge('input', 'mha_q', style='dashed')
    dag.edge('input', 'mha_k', style='dashed')
    dag.edge('input', 'mha_v', style='dashed')
    
    dag.edge('mha_q', 'mha_attn', style='dashed')
    dag.edge('mha_k', 'mha_attn', style='dashed')
    dag.edge('mha_v', 'mha_attn', style='dashed')
    
    dag.edge('mha_attn', 'mha_out')
    dag.edge('mha_out', 'gating')
    
    for expert_node in expert_nodes:
        dag.edge('gating', expert_node, style='dashed', label='Token routing')
        dag.edge(expert_node, 'expert_agg', style='dashed')
    
    dag.edge('expert_agg', 'output')
    
    return dag

def create_detailed_expert_dag():
    """
    Create a detailed DAG showing expert-level breakdown
    """
    
    dag = graphviz.Digraph('Detailed_Expert_Breakdown',
                          filename='detailed_expert_breakdown_dag',
                          format='svg',
                          graph_attr={
                              'rankdir': 'LR',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12',
                              'label': 'Detailed Expert Processing Breakdown\nOne Expert per GPU - Operator Level',
                              'labelloc': 't'
                          })
    
    # Configuration
    batch_size = 4
    seq_len = 2048
    token_dim = 7168
    mlp_hidden = 2048
    
    # Input to expert
    dag.node('expert_input',
             f'Expert Input\nGPU: Specific_GPU\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # MLP Layer 1 (Column Parallel)
    dag.node('mlp1',
             f'Expert MLP1 - Linear\nGPU: Specific_GPU\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]',
             shape='rectangle', fillcolor='lightgreen')
    
    # Activation
    dag.node('activation',
             f'Expert GELU Activation\nGPU: Specific_GPU\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]',
             shape='rectangle', fillcolor='lightgreen')
    
    # MLP Layer 2 (Row Parallel)
    dag.node('mlp2',
             f'Expert MLP2 - Linear\nGPU: Specific_GPU\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen')
    
    # Output
    dag.node('expert_output',
             f'Expert Output\nGPU: Specific_GPU\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightcyan')
    
    # Connect expert internals
    dag.edge('expert_input', 'mlp1')
    dag.edge('mlp1', 'activation')
    dag.edge('activation', 'mlp2')
    dag.edge('mlp2', 'expert_output')
    
    return dag

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-11-26-17-03-32', exist_ok=True)
    
    # Generate all DAGs
    print("Generating comprehensive MoE DAG...")
    comprehensive_dag = create_moe_dag()
    comprehensive_dag.render(directory='../outputs/2025-11-26-17-03-32', cleanup=True)
    
    print("Generating simplified MoE DAG...")
    simplified_dag = create_simplified_moe_dag()
    simplified_dag.render(directory='../outputs/2025-11-26-17-03-32', cleanup=True)
    
    print("Generating detailed expert breakdown DAG...")
    detailed_dag = create_detailed_expert_dag()
    detailed_dag.render(directory='../outputs/2025-11-26-17-03-32', cleanup=True)
    
    # Save DOT files separately
    with open('../outputs/2025-11-26-17-03-32/comprehensive_moe_dag.dot', 'w') as f:
        f.write(comprehensive_dag.source)
    
    with open('../outputs/2025-11-26-17-03-32/simplified_moe_dag.dot', 'w') as f:
        f.write(simplified_dag.source)
        
    with open('../outputs/2025-11-26-17-03-32/detailed_expert_dag.dot', 'w') as f:
        f.write(detailed_dag.source)
    
    print("DAG generation complete!")
    print("Files generated:")
    print("- comprehensive_moe_dag.svg")
    print("- simplified_moe_dag.svg") 
    print("- detailed_expert_breakdown_dag.svg")
    print("- *.dot files for each DAG")