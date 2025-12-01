#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import json
import os

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2 configuration"""
    dot = Digraph(comment='Baseline TP=8 PP=2 MoE Deployment')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightblue')
    
    # Color schemes for different components
    colors = {
        'input': 'lightgreen',
        'mha': 'lightblue', 
        'mlp': 'lightyellow',
        'expert': 'lightcoral',
        'communication': 'lightpink',
        'output': 'lightsteelblue',
        'residual': 'lightgray'
    }
    
    # Model specifications
    batch_size = 128
    seq_len = 10000
    token_dim = 4096
    mha_heads = 32
    head_dim = 128
    mlp_hidden = 16384
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', fillcolor=colors['input'])
    
    # Pipeline Stage 0 (Layers 0-7) - 8 GPUs with TP=8
    for layer in range(8):
        # Multi-Head Attention across 8 GPUs (TP=8)
        for tp_rank in range(8):
            gpu_id = tp_rank
            mha_name = f'mha_layer{layer}_gpu{gpu_id}'
            # MHA with tensor parallelism: heads split across GPUs
            heads_per_gpu = mha_heads // 8
            dot.node(mha_name, 
                     f'MHA Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     fillcolor=colors['mha'])
        
        # MLP with Expert Colocation across 8 GPUs (TP=8)
        for tp_rank in range(8):
            gpu_id = tp_rank
            mlp_name = f'mlp_layer{layer}_gpu{gpu_id}'
            # MLP tensor parallel: column-parallel first linear, row-parallel second linear
            mlp_hidden_per_gpu = mlp_hidden // 8
            dot.node(mlp_name,
                     f'MLP Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     fillcolor=colors['mlp'])
            
            # Experts colocated on each GPU (8 experts per GPU)
            for expert_id in range(8):
                expert_name = f'expert_layer{layer}_gpu{gpu_id}_expert{expert_id}'
                dot.node(expert_name,
                         f'Expert {expert_id} Layer {layer} GPU {gpu_id}\\n' \
                         f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n' \
                         f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                         fillcolor=colors['expert'])
    
    # Pipeline Stage 1 (Layers 8-15) - 8 GPUs with TP=8
    for layer in range(8, 16):
        # Multi-Head Attention across 8 GPUs (TP=8)
        for tp_rank in range(8):
            gpu_id = tp_rank + 8  # Offset by 8 for second pipeline stage
            mha_name = f'mha_layer{layer}_gpu{gpu_id}'
            heads_per_gpu = mha_heads // 8
            dot.node(mha_name, 
                     f'MHA Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     fillcolor=colors['mha'])
        
        # MLP with Expert Colocation across 8 GPUs (TP=8)
        for tp_rank in range(8):
            gpu_id = tp_rank + 8
            mlp_name = f'mlp_layer{layer}_gpu{gpu_id}'
            dot.node(mlp_name,
                     f'MLP Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                     fillcolor=colors['mlp'])
            
            # Experts colocated on each GPU (8 experts per GPU)
            for expert_id in range(8, 16):  # Experts 8-15 for stage 1
                expert_name = f'expert_layer{layer}_gpu{gpu_id}_expert{expert_id}'
                dot.node(expert_name,
                         f'Expert {expert_id} Layer {layer} GPU {gpu_id}\\n' \
                         f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n' \
                         f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                         fillcolor=colors['expert'])
    
    # Communication nodes
    dot.node('tp_comm_0', 'Tensor Parallel All-Reduce\\nStage 0\\nInput: Partial results from 8 GPUs\\nOutput: Combined results', 
             shape='parallelogram', fillcolor=colors['communication'])
    dot.node('pp_comm_0', 'Pipeline Parallel Communication\\nStage 0->1\\nInput: Layer 7 output\\nOutput: Layer 8 input',
             shape='parallelogram', fillcolor=colors['communication'])
    dot.node('tp_comm_1', 'Tensor Parallel All-Reduce\\nStage 1\\nInput: Partial results from 8 GPUs\\nOutput: Combined results',
             shape='parallelogram', fillcolor=colors['communication'])
    
    # Output node
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', fillcolor=colors['output'])
    
    # Connect nodes
    # Input to Layer 0
    for tp_rank in range(8):
        gpu_id = tp_rank
        dot.edge('input', f'mha_layer0_gpu{gpu_id}')
        dot.edge(f'mha_layer0_gpu{gpu_id}', f'mlp_layer0_gpu{gpu_id}')
        
        # Connect experts to MLP
        for expert_id in range(8):
            dot.edge(f'mlp_layer0_gpu{gpu_id}', f'expert_layer0_gpu{gpu_id}_expert{expert_id}')
    
    # Add tensor parallel communication
    for tp_rank in range(8):
        gpu_id = tp_rank
        dot.edge(f'mha_layer0_gpu{gpu_id}', 'tp_comm_0')
        dot.edge(f'expert_layer0_gpu{gpu_id}_expert7', 'tp_comm_0')
    
    # Connect through layers (simplified for demonstration)
    for layer in range(1, 8):
        for tp_rank in range(8):
            gpu_id = tp_rank
            prev_gpu = tp_rank if layer == 0 else tp_rank
            dot.edge(f'mlp_layer{layer-1}_gpu{prev_gpu}', f'mha_layer{layer}_gpu{gpu_id}')
            dot.edge(f'mha_layer{layer}_gpu{gpu_id}', f'mlp_layer{layer}_gpu{gpu_id}')
    
    # Pipeline stage transition
    for tp_rank in range(8):
        dot.edge(f'mlp_layer7_gpu{tp_rank}', 'pp_comm_0')
    for tp_rank in range(8, 16):
        dot.edge('pp_comm_0', f'mha_layer8_gpu{tp_rank}')
    
    # Final output
    for tp_rank in range(8, 16):
        dot.edge(f'mlp_layer15_gpu{tp_rank}', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=16, one expert per GPU"""
    dot = Digraph(comment='Proposed Cross-Node Expert Parallelism EP=16')
    dot.attr(rankdir='TB', size='50,60', dpi='300')
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightblue')
    
    # Color schemes
    colors = {
        'input': 'lightgreen',
        'mha': 'lightblue', 
        'mlp': 'lightyellow',
        'expert': 'lightcoral',
        'gate': 'lightsteelblue',
        'communication': 'lightpink',
        'output': 'lightgray',
        'routing': 'orange'
    }
    
    # Model specifications
    batch_size = 128
    seq_len = 10000
    token_dim = 4096
    mha_heads = 32
    head_dim = 128
    mlp_hidden = 16384
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', fillcolor=colors['input'])
    
    # Create nodes for all 16 layers
    for layer in range(16):
        # Multi-Head Attention (replicated on all GPUs for this layer)
        for gpu_id in range(16):
            mha_name = f'mha_layer{layer}_gpu{gpu_id}'
            dot.node(mha_name, 
                     f'MHA Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]',
                     fillcolor=colors['mha'])
        
        # Gate mechanism (replicated on all GPUs)
        for gpu_id in range(16):
            gate_name = f'gate_layer{layer}_gpu{gpu_id}'
            dot.node(gate_name,
                     f'Gate Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n' \
                     f'Output: [batch_size={batch_size}, seq_len={seq_len}, expert_selections=2]',
                     shape='diamond', fillcolor=colors['gate'])
        
        # One expert per GPU (16 experts total)
        for expert_id in range(16):
            gpu_id = expert_id  # One expert per GPU
            expert_name = f'expert_layer{layer}_gpu{gpu_id}_expert{expert_id}'
            dot.node(expert_name,
                     f'Expert {expert_id} Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: [batch_size=variable, seq_len=variable, token_dim={token_dim}]\\n' \
                     f'Output: [batch_size=variable, seq_len=variable, token_dim={token_dim}]\\n' \
                     f'MLP: {token_dim}->{mlp_hidden}akey={token_dim}',
                     fillcolor=colors['expert'])
        
        # Token routing and aggregation nodes
        for gpu_id in range(16):
            route_name = f'route_layer{layer}_gpu{gpu_id}'
            dot.node(route_name,
                     f'Token Router Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: Tokens with routing decisions\\n' \
                     f'Output: Tokens sent to destination experts',
                     shape='parallelogram', fillcolor=colors['routing'])
            
            aggregate_name = f'aggregate_layer{layer}_gpu{gpu_id}'
            dot.node(aggregate_name,
                     f'Token Aggregator Layer {layer} GPU {gpu_id}\\n' \
                     f'Input: Processed tokens from all experts\\n' \
                     f'Output: Combined token representations',
                     shape='parallelogram', fillcolor=colors['routing'])
        
        # Communication nodes for cross-GPU token transfer
        comm_name = f'comm_layer{layer}'
        dot.node(comm_name,
                 f'Cross-GPU Communication Layer {layer}\\n' \
                 f'Input: Tokens from routing decisions\\n' \
                 f'Output: Tokens delivered to destination GPUs',
                 shape='ellipse', fillcolor=colors['communication'])
    
    # Output node
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', fillcolor=colors['output'])
    
    # Connect nodes
    # Input to Layer 0 MHA
    for gpu_id in range(16):
        dot.edge('input', f'mha_layer0_gpu{gpu_id}')
    
    # Connect through each layer
    for layer in range(16):
        # MHA -> Gate
        for gpu_id in range(16):
            dot.edge(f'mha_layer{layer}_gpu{gpu_id}', f'gate_layer{layer}_gpu{gpu_id}')
        
        # Gate -> Router (dashed line for routing decisions)
        for gpu_id in range(16):
            dot.edge(f'gate_layer{layer}_gpu{gpu_id}', f'route_layer{layer}_gpu{gpu_id}', style='dashed')
        
        # Router -> Communication
        for gpu_id in range(16):
            dot.edge(f'route_layer{layer}_gpu{gpu_id}', f'comm_layer{layer}')
        
        # Communication -> Experts
        for expert_id in range(16):
            gpu_id = expert_id
            dot.edge(f'comm_layer{layer}', f'expert_layer{layer}_gpu{gpu_id}_expert{expert_id}')
        
        # Experts -> Aggregator
        for expert_id in range(16):
            gpu_id = expert_id
            dot.edge(f'expert_layer{layer}_gpu{gpu_id}_expert{expert_id}', f'aggregate_layer{layer}_gpu{gpu_id}')
        
        # Connect to next layer MHA
        if layer < 15:
            for gpu_id in range(16):
                dot.edge(f'aggregate_layer{layer}_gpu{gpu_id}', f'mha_layer{layer+1}_gpu{gpu_id}')
        else:
            # Final layer to output
            for gpu_id in range(16):
                dot.edge(f'aggregate_layer{layer}_gpu{gpu_id}', 'output')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    output_dir = '../outputs/2025-11-29-17-28-50'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    
    # Save baseline DAG
    baseline_dot_path = os.path.join(output_dir, 'baseline_dag.dot')
    baseline_svg_path = os.path.join(output_dir, 'baseline_dag.svg')
    
    with open(baseline_dot_path, 'w') as f:
        f.write(baseline_dag.source)
    
    baseline_dag.render(os.path.join(output_dir, 'baseline_dag'), format='svg', cleanup=True)
    print(f"Baseline DAG saved to: {baseline_dot_path} and {baseline_svg_path}")
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    
    # Save proposed DAG
    proposed_dot_path = os.path.join(output_dir, 'proposed_dag.dot')
    proposed_svg_path = os.path.join(output_dir, 'proposed_dag.svg')
    
    with open(proposed_dot_path, 'w') as f:
        f.write(proposed_dag.source)
    
    proposed_dag.render(os.path.join(output_dir, 'proposed_dag'), format='svg', cleanup=True)
    print(f"Proposed DAG saved to: {proposed_dot_path} and {proposed_svg_path}")
    
    # Create submission summary
    submission = {
        "baseline_dag_dot": baseline_dot_path,
        "baseline_dag_svg": baseline_svg_path,
        "proposed_dag_dot": proposed_dot_path,
        "proposed_dag_svg": proposed_svg_path,
        "generated_at": "2025-11-29T17:28:50Z"
    }
    
    submission_path = os.path.join(output_dir, 'dag_submission.json')
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Submission summary saved to: {submission_path}")
    
    return submission

if __name__ == '__main__':
    result = main()
    print("DAG generation completed successfully!")
    print(json.dumps(result, indent=2))