#!/usr/bin/env python3
"""
LLM Parallel Strategy DAG Generator - Complete Version
Generates a comprehensive DAG for EP-16 × TP-4 × PP-1 configuration
"""

import graphviz
from graphviz import Digraph

def create_complete_parallel_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Complete Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='40,50', dpi='300')
    dot.attr('graph', bgcolor='white', pad='0.5', ranksep='1.0', nodesep='0.5')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal', penwidth='1.5')
    
    # Define GPU groups and colors (using valid Graphviz colors)
    gpu_colors = {
        'EP0': 'lightcoral', 'EP1': 'lightgreen', 'EP2': 'lightblue', 'EP3': 'lightyellow',
        'EP4': 'lightpink', 'EP5': 'lightgray', 'EP6': 'lightsalmon', 'EP7': 'lightseagreen',
        'EP8': 'lightskyblue', 'EP9': 'lightsteelblue', 'EP10': 'lightcyan', 'EP11': 'lightgoldenrodyellow',
        'EP12': 'plum', 'EP13': 'orange', 'EP14': 'purple', 'EP15': 'turquoise'
    }
    
    # Input node
    dot.node('input', 'Input Layer\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]', 
             shape='ellipse', fillcolor='white', style='filled', penwidth='2')
    
    # Token embedding (distributed across all GPUs) - TP splits the hidden dimension
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Token Embedding (TP-4)', style='rounded,dashed', penwidth='2')
        for ep in range(16):
            for tp in range(4):
                gpu_id = ep * 4 + tp
                c.node(f'embed_{ep}_{tp}', 
                      f'Embedding_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                      fillcolor=gpu_colors[f'EP{ep}'])
    
    # Connect input to embeddings
    for ep in range(16):
        for tp in range(4):
            dot.edge('input', f'embed_{ep}_{tp}')
    
    # Expert routing (gate selection) - happens on first GPU of each EP group
    with dot.subgraph(name='cluster_gates') as c:
        c.attr(label='Expert Routing Gates', style='rounded,dashed', penwidth='2')
        for ep in range(16):
            c.node(f'gate_{ep}', 
                  f'Gate_{ep}\\nGPU:{ep*4}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, top_k=2]\\n(Selects top-2 experts)', 
                  shape='parallelogram', fillcolor='gold', style='filled,dashed', penwidth='2')
    
    # Connect input to gates with dashed lines
    for ep in range(16):
        dot.edge('input', f'gate_{ep}', style='dashed', penwidth='2', label='routing')
    
    # Process each transformer layer
    prev_nodes = {}
    for layer in range(16):
        layer_nodes = {}
        
        # Create subgraph for this layer
        with dot.subgraph(name=f'cluster_layer_{layer}') as layer_c:
            layer_c.attr(label=f'Layer {layer}', style='rounded,bold', penwidth='3', bgcolor='lightgray')
            
            # Attention block for each expert
            with layer_c.subgraph(name=f'cluster_layer_{layer}_attention') as attn_c:
                attn_c.attr(label=f'Attention Block', style='rounded,dashed')
                
                for ep in range(16):
                    for tp in range(4):
                        gpu_id = ep * 4 + tp
                        
                        # Layer normalization
                        attn_c.node(f'ln1_{layer}_{ep}_{tp}', 
                                  f'LayerNorm1_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                                  fillcolor=gpu_colors[f'EP{ep}'], shape='diamond')
                        
                        # QKV projection (column parallel)
                        attn_c.node(f'qkv_{layer}_{ep}_{tp}', 
                                  f'QKV_Proj_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\n(Column Parallel)', 
                                  fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # Attention scores computation
                        attn_c.node(f'attn_scores_{layer}_{ep}_{tp}', 
                                  f'Attention_Scores_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, seq_len=128, heads=8]', 
                                  fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # Attention softmax
                        attn_c.node(f'attn_softmax_{layer}_{ep}_{tp}', 
                                  f'Attention_Softmax_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, seq_len=128, heads=8]\\nOutput: [batch_size=128, seq_len=128, seq_len=128, heads=8]', 
                                  fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # Attention weighted sum
                        attn_c.node(f'attn_weighted_{layer}_{ep}_{tp}', 
                                  f'Attention_Weighted_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, seq_len=128, heads=8], [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, heads=8, d_k=64]', 
                                  fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # Attention output projection (row parallel)
                        attn_c.node(f'attn_out_{layer}_{ep}_{tp}', 
                                  f'Attention_Output_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, hidden=128]\\n(Row Parallel)', 
                                  fillcolor=gpu_colors[f'EP{ep}'])
                        
                        layer_nodes[f'attn_out_{layer}_{ep}_{tp}'] = f'attn_out_{layer}_{ep}_{tp}'
                
                # Attention all-reduce (communication within each EP group)
                for ep in range(16):
                    attn_c.node(f'attn_allreduce_{layer}_{ep}', 
                              f'Attention_AllReduce_{layer}_{ep}\\nEP:{ep} (GPUs {ep*4}-{ep*4+3})\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]\\n(All-reduce across TP)', 
                              shape='ellipse', fillcolor='red', style='filled', penwidth='2')
            
            # MLP block for each expert
            with layer_c.subgraph(name=f'cluster_layer_{layer}_mlp') as mlp_c:
                mlp_c.attr(label=f'MLP Block (MoE)', style='rounded,dashed')
                
                for ep in range(16):
                    for tp in range(4):
                        gpu_id = ep * 4 + tp
                        
                        # Layer normalization
                        mlp_c.node(f'ln2_{layer}_{ep}_{tp}', 
                                 f'LayerNorm2_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                                 fillcolor=gpu_colors[f'EP{ep}'], shape='diamond')
                        
                        # MLP first linear (column parallel)
                        mlp_c.node(f'mlp1_{layer}_{ep}_{tp}', 
                                 f'MLP_Linear1_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, ffn=512]\\n(Column Parallel)', 
                                 fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # GELU activation
                        mlp_c.node(f'gelu_{layer}_{ep}_{tp}', 
                                 f'GELU_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, ffn=512]\\nOutput: [batch_size=128, seq_len=128, ffn=512]', 
                                 fillcolor=gpu_colors[f'EP{ep}'])
                        
                        # MLP second linear (row parallel)
                        mlp_c.node(f'mlp2_{layer}_{ep}_{tp}', 
                                 f'MLP_Linear2_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, ffn=512]\\nOutput: [batch_size=128, seq_len=128, hidden=128]\\n(Row Parallel)', 
                                 fillcolor=gpu_colors[f'EP{ep}'])
                        
                        layer_nodes[f'mlp2_{layer}_{ep}_{tp}'] = f'mlp2_{layer}_{ep}_{tp}'
                
                # MLP all-reduce (communication within each EP group)
                for ep in range(16):
                    mlp_c.node(f'mlp_allreduce_{layer}_{ep}', 
                             f'MLP_AllReduce_{layer}_{ep}\\nEP:{ep} (GPUs {ep*4}-{ep*4+3})\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]\\n(All-reduce across TP)', 
                             shape='ellipse', fillcolor='red', style='filled', penwidth='2')
            
            # Expert routing all-to-all communication (between EP groups)
            for ep in range(16):
                layer_c.node(f'expert_route_{layer}_{ep}', 
                           f'Expert_Route_{layer}_{ep}\\nEP:{ep}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]\\n(All-to-all communication)', 
                           shape='ellipse', fillcolor='orange', style='filled', penwidth='2')
    
    # Final layer norm
    with dot.subgraph(name='cluster_final_norm') as c:
        c.attr(label='Final Layer Normalization', style='rounded,dashed')
        for ep in range(16):
            for tp in range(4):
                gpu_id = ep * 4 + tp
                c.node(f'final_norm_{ep}_{tp}', 
                      f'Final_Norm_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                      fillcolor=gpu_colors[f'EP{ep}'], shape='diamond')
    
    # Output projection
    with dot.subgraph(name='cluster_output_proj') as c:
        c.attr(label='Output Projection', style='rounded,dashed')
        for ep in range(16):
            for tp in range(4):
                gpu_id = ep * 4 + tp
                c.node(f'output_proj_{ep}_{tp}', 
                      f'Output_Proj_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]', 
                      fillcolor=gpu_colors[f'EP{ep}'])
    
    # Final all-reduce for output
    dot.node('final_allreduce', 
            'Final_AllReduce\\nAll 64 GPUs\\nInput: [batch_size=128, seq_len=128, vocab=32000]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]\\n(All-reduce across all GPUs)', 
            shape='ellipse', fillcolor='red', style='filled', penwidth='3')
    
    # Output node
    dot.node('output', 
            'Output Layer\\nInput: [batch_size=128, seq_len=128, vocab=32000]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]', 
            shape='ellipse', fillcolor='white', style='filled', penwidth='2')
    
    # Connect final layer to output
    dot.edge('final_allreduce', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the complete DAG
    dag = create_complete_parallel_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-25-09-26-32/llm_parallel_strategy_complete.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-25-09-26-32/llm_parallel_strategy_complete', format='svg', cleanup=True)
    
    print("Complete DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-25-09-26-32/llm_parallel_strategy_complete.dot")
    print(f"SVG file: ../outputs/2025-12-25-09-26-32/llm_parallel_strategy_complete.svg")