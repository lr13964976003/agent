#!/usr/bin/env python3
"""
Generate DAG for Large-Scale Cross-Node Expert Parallelism (EP=16)
This represents the proposed model with one expert per GPU
"""

import os
from graphviz import Digraph

def create_proposed_dag():
    """Create complete DAG for proposed EP=16 model"""
    
    # Create the DAG
    dot = Digraph('EP16_Cross_Node_Expert_Parallelism')
    dot.attr(rankdir='TB', size='20,30', concentrate='true')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Define node styles
    dot.attr('node', fillcolor='lightblue')  # Computation nodes
    
    # Global parameters
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    mlp_hidden = 32768
    
    # Input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Layer 0 - Attention and Expert routing
    dot.node('layer0_attn', 
             f'Layer 0 Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightyellow')
    
    # Gate for Layer 0
    dot.node('layer0_gate', 
             f'Layer 0 Expert Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightcoral')
    
    # Token routing for Layer 0
    dot.node('layer0_router', 
             f'Layer 0 Token Router\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [distributed_tokens]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='orange')
    
    # Expert 0-15 for Layer 0
    for i in range(16):
        dot.node(f'layer0_expert{i}', 
                 f'Layer 0 Expert {i}\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: {i}',
                 fillcolor='lightblue')
    
    # Expert aggregation for Layer 0
    dot.node('layer0_aggregate', 
             f'Layer 0 Expert Aggregation\\nInput: [distributed_tokens]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightpink')
    
    # Residual connection Layer 0
    dot.node('layer0_residual', 
             f'Layer 0 Residual Add\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightgray')
    
    # Layer 1 - Attention and Expert routing
    dot.node('layer1_attn', 
             f'Layer 1 Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightyellow')
    
    # Gate for Layer 1
    dot.node('layer1_gate', 
             f'Layer 1 Expert Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightcoral')
    
    # Token routing for Layer 1
    dot.node('layer1_router', 
             f'Layer 1 Token Router\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [distributed_tokens]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='orange')
    
    # Expert 0-15 for Layer 1
    for i in range(16):
        dot.node(f'layer1_expert{i}', 
                 f'Layer 1 Expert {i}\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: {i}',
                 fillcolor='lightblue')
    
    # Expert aggregation for Layer 1
    dot.node('layer1_aggregate', 
             f'Layer 1 Expert Aggregation\\nInput: [distributed_tokens]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightpink')
    
    # Residual connection Layer 1
    dot.node('layer1_residual', 
             f'Layer 1 Residual Add\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightgray')
    
    # Layer 2 - Attention and Expert routing
    dot.node('layer2_attn', 
             f'Layer 2 Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightyellow')
    
    # Gate for Layer 2
    dot.node('layer2_gate', 
             f'Layer 2 Expert Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightcoral')
    
    # Token routing for Layer 2
    dot.node('layer2_router', 
             f'Layer 2 Token Router\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [distributed_tokens]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='orange')
    
    # Expert 0-15 for Layer 2
    for i in range(16):
        dot.node(f'layer2_expert{i}', 
                 f'Layer 2 Expert {i}\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: {i}',
                 fillcolor='lightblue')
    
    # Expert aggregation for Layer 2
    dot.node('layer2_aggregate', 
             f'Layer 2 Expert Aggregation\\nInput: [distributed_tokens]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightpink')
    
    # Residual connection Layer 2
    dot.node('layer2_residual', 
             f'Layer 2 Residual Add\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightgray')
    
    # Layer 3 - Attention and Expert routing
    dot.node('layer3_attn', 
             f'Layer 3 Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightyellow')
    
    # Gate for Layer 3
    dot.node('layer3_gate', 
             f'Layer 3 Expert Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightcoral')
    
    # Token routing for Layer 3
    dot.node('layer3_router', 
             f'Layer 3 Token Router\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [distributed_tokens]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='orange')
    
    # Expert 0-15 for Layer 3
    for i in range(16):
        dot.node(f'layer3_expert{i}', 
                 f'Layer 3 Expert {i}\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: {i}',
                 fillcolor='lightblue')
    
    # Expert aggregation for Layer 3
    dot.node('layer3_aggregate', 
             f'Layer 3 Expert Aggregation\\nInput: [distributed_tokens]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightpink')
    
    # Residual connection Layer 3
    dot.node('layer3_residual', 
             f'Layer 3 Residual Add\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             fillcolor='lightgray')
    
    # Output node
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create edges
    # Input to Layer 0
    dot.edge('input', 'layer0_attn')
    dot.edge('layer0_attn', 'layer0_gate')
    dot.edge('layer0_attn', 'layer0_router')
    dot.edge('layer0_gate', 'layer0_router', style='dashed')
    
    # Router to experts
    for i in range(16):
        dot.edge('layer0_router', f'layer0_expert{i}')
    
    # Experts to aggregation
    for i in range(16):
        dot.edge(f'layer0_expert{i}', 'layer0_aggregate')
    
    # Aggregation to residual
    dot.edge('layer0_aggregate', 'layer0_residual')
    dot.edge('layer0_attn', 'layer0_residual')
    
    # Layer 0 to Layer 1
    dot.edge('layer0_residual', 'layer1_attn')
    dot.edge('layer1_attn', 'layer1_gate')
    dot.edge('layer1_attn', 'layer1_router')
    dot.edge('layer1_gate', 'layer1_router', style='dashed')
    
    # Router to experts
    for i in range(16):
        dot.edge('layer1_router', f'layer1_expert{i}')
    
    # Experts to aggregation
    for i in range(16):
        dot.edge(f'layer1_expert{i}', 'layer1_aggregate')
    
    # Aggregation to residual
    dot.edge('layer1_aggregate', 'layer1_residual')
    dot.edge('layer1_attn', 'layer1_residual')
    
    # Layer 1 to Layer 2
    dot.edge('layer1_residual', 'layer2_attn')
    dot.edge('layer2_attn', 'layer2_gate')
    dot.edge('layer2_attn', 'layer2_router')
    dot.edge('layer2_gate', 'layer2_router', style='dashed')
    
    # Router to experts
    for i in range(16):
        dot.edge('layer2_router', f'layer2_expert{i}')
    
    # Experts to aggregation
    for i in range(16):
        dot.edge(f'layer2_expert{i}', 'layer2_aggregate')
    
    # Aggregation to residual
    dot.edge('layer2_aggregate', 'layer2_residual')
    dot.edge('layer2_attn', 'layer2_residual')
    
    # Layer 2 to Layer 3
    dot.edge('layer2_residual', 'layer3_attn')
    dot.edge('layer3_attn', 'layer3_gate')
    dot.edge('layer3_attn', 'layer3_router')
    dot.edge('layer3_gate', 'layer3_router', style='dashed')
    
    # Router to experts
    for i in range(16):
        dot.edge('layer3_router', f'layer3_expert{i}')
    
    # Experts to aggregation
    for i in range(16):
        dot.edge(f'layer3_expert{i}', 'layer3_aggregate')
    
    # Aggregation to residual
    dot.edge('layer3_aggregate', 'layer3_residual')
    dot.edge('layer3_attn', 'layer3_residual')
    
    # Final output
    dot.edge('layer3_residual', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('./outputs/2025-10-13-21-03-41', exist_ok=True)
    
    # Generate the DAG
    dag = create_proposed_dag()
    
    # Save as DOT file
    with open('./outputs/2025-10-13-21-03-41/proposed_ep16_dag.dot', 'w') as f:
        f.write(dag.source)
    
    # Save as SVG
    dag.render('./outputs/2025-10-13-21-03-41/proposed_ep16_dag', format='svg', cleanup=True)
    
    print("Generated proposed EP=16 DAG successfully")