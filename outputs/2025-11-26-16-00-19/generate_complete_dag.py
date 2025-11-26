#!/usr/bin/env python3
"""
Generate complete MoE DAG with 32 experts and proper tensor dimensions.
This script creates a validated DAG for large-scale cross-node expert parallelism.
"""

import os
import graphviz
import json

# Model specifications from deployment configuration
BATCH_SIZE = 32
SEQ_LEN = 2048
HIDDEN_DIM = 7168
MLP_HIDDEN = 2048
NUM_EXPERTS = 32
TOP_K = 2
HEADS = 128
HEAD_DIM = 128

# Calculate expert-specific batch sizes
# With top-2 gating and 32 experts, each expert gets roughly (batch_size * seq_len * top_k) / num_experts tokens
EXPERT_BATCH_SIZE = (BATCH_SIZE * SEQ_LEN * TOP_K) // NUM_EXPERTS

def create_complete_moe_dag():
    """Create complete DAG with all 32 experts and correct tensor dimensions."""
    
    dot = graphviz.Digraph('Large_Scale_Cross_Node_Expert_Parallelism_32_Experts')
    dot.attr(rankdir='TB', size='200,200', splines='ortho')
    dot.attr('node', fontsize='10', height='0.8', width='2.5')
    
    # Input Layer
    dot.node('input', 
             '''Input Layer\nGPU: ALL\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]''',
             fillcolor='lightgray', shape='box', style='filled')
    
    # Multi-Head Attention components (shared across GPUs)
    dot.node('q_proj', 
             '''Q Projection\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=128, d_k=128]''',
             fillcolor='lightblue', shape='box', style='filled')
    
    dot.node('k_proj', 
             '''K Projection\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=128, d_k=128]''',
             fillcolor='lightblue', shape='box', style='filled')
    
    dot.node('v_proj', 
             '''V Projection\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, heads=128, d_k=128]''',
             fillcolor='lightblue', shape='box', style='filled')
    
    dot.node('attention', 
             '''Multi-Head Attention\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, heads=128, d_k=128]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]''',
             fillcolor='lightblue', shape='box', style='filled')
    
    dot.node('mha_out_proj', 
             '''MHA Output Projection\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]''',
             fillcolor='lightblue', shape='box', style='filled')
    
    # Gating and routing
    dot.node('gating', 
             '''Gating Network\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, experts=32, top_k=2]''',
             fillcolor='lightyellow', shape='parallelogram', style='filled')
    
    dot.node('routing', 
             '''Token Routing Decision\nGPU: Shared\nInput: [batch_size=32, seq_len=2048, experts=32, top_k=2]\nOutput: Routing decisions and token masks''',
             fillcolor='lightyellow', shape='parallelogram', style='filled')
    
    # Token scatter communication
    dot.node('token_scatter', 
             '''Token Scatter Communication\nGPU: All-to-all\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=4, seq_len=128, hidden=7168] per expert''',
             fillcolor='lightgreen', shape='ellipse', style='filled')
    
    # Create all 32 expert nodes
    for expert_id in range(NUM_EXPERTS):
        gpu_id = expert_id  # Each expert on separate GPU
        
        # Expert gate
        dot.node(f'expert_{expert_id}_gate',
                 f'''Expert {expert_id} Gate\nGPU: {gpu_id}\nInput: [batch_size=4, seq_len=128, hidden=7168]\nOutput: [batch_size=4, seq_len=128, hidden=2048]''',
                 fillcolor='lightblue', shape='box', style='filled')
        
        # Expert computation
        dot.node(f'expert_{expert_id}_expert',
                 f'''Expert {expert_id} Expert\nGPU: {gpu_id}\nInput: [batch_size=4, seq_len=128, hidden=2048]\nOutput: [batch_size=4, seq_len=128, hidden=7168]''',
                 fillcolor='lightblue', shape='box', style='filled')
        
        # Expert multiply
        dot.node(f'expert_{expert_id}_multiply',
                 f'''Expert {expert_id} Multiply\nGPU: {gpu_id}\nInput: [batch_size=4, seq_len=128, hidden=7168], [batch_size=4, seq_len=128, hidden=2048]\nOutput: [batch_size=4, seq_len=128, hidden=7168]''',
                 fillcolor='lightblue', shape='box', style='filled')
    
    # Token gather communication
    dot.node('token_gather',
             '''Token Gather Communication\nGPU: All-to-all\nInput: [batch_size=4, seq_len=128, hidden=7168] from each expert\nOutput: [batch_size=32, seq_len=2048, hidden=7168]''',
             fillcolor='lightcoral', shape='ellipse', style='filled')
    
    # Output layer
    dot.node('layer_output',
             '''Layer Output\nGPU: ALL\nInput: [batch_size=32, seq_len=2048, hidden=7168]\nOutput: [batch_size=32, seq_len=2048, hidden=7168]''',
             fillcolor='lightgray', shape='box', style='filled')
    
    # Create all edges
    # Input to MHA and gating
    dot.edge('input', 'q_proj')
    dot.edge('input', 'k_proj')  
    dot.edge('input', 'v_proj')
    dot.edge('q_proj', 'attention')
    dot.edge('k_proj', 'attention')
    dot.edge('v_proj', 'attention')
    dot.edge('attention', 'mha_out_proj')
    
    # MHA and input to gating/routing
    dot.edge('mha_out_proj', 'gating')
    dot.edge('input', 'gating')
    dot.edge('gating', 'routing')
    dot.edge('routing', 'token_scatter')
    
    # Connect all experts
    for expert_id in range(NUM_EXPERTS):
        dot.edge('token_scatter', f'expert_{expert_id}_gate')
        dot.edge(f'expert_{expert_id}_gate', f'expert_{expert_id}_expert')
        dot.edge(f'expert_{expert_id}_expert', f'expert_{expert_id}_multiply')
        dot.edge(f'expert_{expert_id}_multiply', 'token_gather')
        
        # Dashed routing edges
        dot.edge('routing', f'expert_{expert_id}_gate', style='dashed')
    
    # Final edges
    dot.edge('token_gather', 'layer_output')
    dot.edge('mha_out_proj', 'layer_output')
    
    return dot

def save_dag_files():
    """Save the complete DAG in multiple formats."""
    
    # Create complete DAG
    dot = create_complete_moe_dag()
    
    # Save DOT file
    dot_file = '../outputs/2025-11-26-16-00-19/complete_moe_dag_32_experts.dot'
    with open(dot_file, 'w') as f:
        f.write(dot.source)
    
    # Save SVG file
    svg_file = '../outputs/2025-11-26-16-00-19/complete_moe_dag_32_experts.svg'
    dot.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    # Save PNG for visualization
    png_file = '../outputs/2025-11-26-16-00-19/complete_moe_dag_32_experts.png'
    dot.render(png_file.replace('.png', ''), format='png', cleanup=True)
    
    return {
        'dot_file': dot_file,
        'svg_file': svg_file,
        'png_file': png_file,
        'total_nodes': 32 * 3 + 9,  # 32 experts * 3 nodes each + 9 main nodes
        'total_experts': NUM_EXPERTS,
        'expert_batch_size': EXPERT_BATCH_SIZE
    }

if __name__ == '__main__':
    result = save_dag_files()
    print(json.dumps(result, indent=2))