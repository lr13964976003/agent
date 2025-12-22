#!/usr/bin/env python3
"""
DAG Generator for Current Deployment Method: EP64-TP8-PP2-DP2
Generates a detailed DAG showing operator-level parallelism for MoE LLM inference
"""

import graphviz
from graphviz import Digraph
import os

def create_current_deployment_dag():
    """Create DAG for EP64-TP8-PP2-DP2 deployment strategy"""
    
    # Create the directed graph
    dag = Digraph(name='current_deployment_dag')
    dag.attr(rankdir='TB', bgcolor='white', fontname='Arial')
    
    # Define node styles
    dag.attr('node', shape='rectangle', style='filled', fontname='Arial')
    
    # Input node
    dag.node('input', 'Input\\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=4096]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Embedding layer (distributed across TP=8)
    dag.node('embed', 'Embedding Layer\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             shape='rectangle', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (Layers 0-7)
    dag.node('pp0_start', 'PP Stage 0\\nGPU: PP0\\nLayers 0-7', 
             shape='parallelogram', fillcolor='yellow')
    
    # Layer 0 - Attention (TP=8)
    dag.node('layer0_attn_q', 'Layer 0: Q Projection\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=64]', 
             fillcolor='lightcoral')
    dag.node('layer0_attn_k', 'Layer 0: K Projection\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=64]', 
             fillcolor='lightcoral')
    dag.node('layer0_attn_v', 'Layer 0: V Projection\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, heads=32, d_k=64]', 
             fillcolor='lightcoral')
    
    dag.node('layer0_attn_score', 'Layer 0: Attention Score\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=64]\\nOutput: [batch_size=128, seq_len=1024, seq_len=1024, heads=32]', 
             fillcolor='lightcoral')
    
    dag.node('layer0_attn_softmax', 'Layer 0: Softmax\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, seq_len=1024, heads=32]\\nOutput: [batch_size=128, seq_len=1024, seq_len=1024, heads=32]', 
             fillcolor='lightcoral')
    
    dag.node('layer0_attn_out', 'Layer 0: Attention Output\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, heads=32, d_k=64]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             fillcolor='lightcoral')
    
    dag.node('layer0_attn_proj', 'Layer 0: Output Projection\\nGPU: TP0-TP7\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             fillcolor='lightcoral')
    
    # All-Reduce for attention
    dag.node('layer0_attn_ar', 'All-Reduce\\nTP Group: TP0-TP7\\nSize: [batch_size=128, seq_len=1024, hidden_size=512]', 
             shape='ellipse', fillcolor='orange', style='dashed')
    
    # MoE Layer - Gate and Routing
    dag.node('layer0_gate', 'Layer 0: MoE Gate\\nGPU: EP0-EP63\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, num_experts=64]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Expert routing (dashed line for gate selection)
    dag.node('layer0_route', 'Token Routing\\nGPU: EP0-EP63\\nSelect 2 experts per token', 
             shape='ellipse', fillcolor='pink', style='dashed')
    
    # Expert computation (distributed across EP=64, each GPU has 1 expert)
    for expert_id in range(2):  # Show first 2 experts for clarity
        dag.node(f'layer0_expert{expert_id}_up', f'Layer 0: Expert {expert_id} Up-proj\\nGPU: EP{expert_id}\\nInput: [batch_size=64, seq_len=512, hidden_size=512]\\nOutput: [batch_size=64, seq_len=512, hidden_size=2048]', 
                 fillcolor='lightsteelblue')
        dag.node(f'layer0_expert{expert_id}_act', f'Layer 0: Expert {expert_id} Activation\\nGPU: EP{expert_id}\\nInput: [batch_size=64, seq_len=512, hidden_size=2048]\\nOutput: [batch_size=64, seq_len=512, hidden_size=2048]', 
                 fillcolor='lightsteelblue')
        dag.node(f'layer0_expert{expert_id}_down', f'Layer 0: Expert {expert_id} Down-proj\\nGPU: EP{expert_id}\\nInput: [batch_size=64, seq_len=512, hidden_size=2048]\\nOutput: [batch_size=64, seq_len=512, hidden_size=512]', 
                 fillcolor='lightsteelblue')
    
    # All-to-All communication for expert parallelism
    dag.node('layer0_ep_a2a', 'All-to-All\\nEP Group: EP0-EP63\\nToken dispatch/combine', 
             shape='ellipse', fillcolor='orange')
    
    # Expert combine
    dag.node('layer0_expert_combine', 'Expert Combine\\nGPU: EP0-EP63\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Continue with remaining layers in PP Stage 0...
    # For brevity, showing key layers and transitions
    
    # Pipeline transition to Stage 1
    dag.node('pp0_to_pp1', 'Pipeline Transfer\\nPP0 → PP1\\nActivations transfer', 
             shape='ellipse', fillcolor='orange')
    
    dag.node('pp1_start', 'PP Stage 1\\nGPU: PP1\\nLayers 8-15', 
             shape='parallelogram', fillcolor='yellow')
    
    # Final layer (similar structure but showing different GPU allocation)
    dag.node('layer15_attn', 'Layer 15: Attention\\nGPU: TP0-TP7, PP1\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             fillcolor='lightcoral')
    
    dag.node('layer15_moe', 'Layer 15: MoE\\nGPU: EP0-EP63, PP1\\nInput: [batch_size=128, seq_len=1024, hidden_size=512]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=512]', 
             fillcolor='lightsteelblue')
    
    # Final output
    dag.node('output', 'Output\\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=1024, vocab_size=51200]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Build the graph connections
    dag.edge('input', 'embed')
    dag.edge('embed', 'pp0_start')
    
    # Layer 0 attention flow
    dag.edge('pp0_start', 'layer0_attn_q')
    dag.edge('pp0_start', 'layer0_attn_k')
    dag.edge('pp0_start', 'layer0_attn_v')
    dag.edge('layer0_attn_q', 'layer0_attn_score')
    dag.edge('layer0_attn_k', 'layer0_attn_score')
    dag.edge('layer0_attn_v', 'layer0_attn_out')
    dag.edge('layer0_attn_score', 'layer0_attn_softmax')
    dag.edge('layer0_attn_softmax', 'layer0_attn_out')
    dag.edge('layer0_attn_out', 'layer0_attn_proj')
    dag.edge('layer0_attn_proj', 'layer0_attn_ar')
    
    # MoE flow
    dag.edge('layer0_attn_ar', 'layer0_gate')
    dag.edge('layer0_gate', 'layer0_route')
    
    # Expert computation
    for expert_id in range(2):
        dag.edge('layer0_route', f'layer0_expert{expert_id}_up')
        dag.edge(f'layer0_expert{expert_id}_up', f'layer0_expert{expert_id}_act')
        dag.edge(f'layer0_expert{expert_id}_act', f'layer0_expert{expert_id}_down')
    
    dag.edge('layer0_expert0_down', 'layer0_ep_a2a')
    dag.edge('layer0_expert1_down', 'layer0_ep_a2a')
    dag.edge('layer0_ep_a2a', 'layer0_expert_combine')
    
    # Continue to next stages...
    dag.edge('layer0_expert_combine', 'pp0_to_pp1')
    dag.edge('pp0_to_pp1', 'pp1_start')
    dag.edge('pp1_start', 'layer15_attn')
    dag.edge('layer15_attn', 'layer15_moe')
    dag.edge('layer15_moe', 'output')
    
    return dag

def main():
    """Generate the current deployment DAG"""
    
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-12-22-15-40-56', exist_ok=True)
    
    # Generate the DAG
    dag = create_current_deployment_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-22-15-40-56/current_deployment_dag.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    
    # Render as SVG
    svg_file = '../outputs/2025-12-22-15-40-56/current_deployment_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Current deployment DAG generated:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    return dot_file, svg_file

if __name__ == "__main__":
    main()