#!/usr/bin/env python3
"""
LLM Parallel Strategy DAG Generator
Generates a detailed DAG for EP-16 × TP-4 × PP-1 configuration
"""

import graphviz
from graphviz import Digraph

def create_parallel_dag():
    # Create a new directed graph
    dot = Digraph(comment='LLM Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('graph', bgcolor='white', pad='0.5')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal', penwidth='1.5')
    
    # Define GPU groups and colors
    gpu_colors = {
        'EP0': 'lightcoral', 'EP1': 'lightgreen', 'EP2': 'lightblue', 'EP3': 'lightyellow',
        'EP4': 'lightpink', 'EP5': 'lightgray', 'EP6': 'lightsalmon', 'EP7': 'lightseagreen',
        'EP8': 'lightskyblue', 'EP9': 'lightsteelblue', 'EP10': 'lightcyan', 'EP11': 'lightgoldenrodyellow',
        'EP12': 'lightmagenta', 'EP13': 'lightorange', 'EP14': 'lightpurple', 'EP15': 'lightteal'
    }
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]', 
             shape='ellipse', fillcolor='white', style='filled', penwidth='2')
    
    # Token embedding (distributed across all GPUs)
    for ep in range(16):
        for tp in range(4):
            gpu_id = ep * 4 + tp
            dot.node(f'embed_{ep}_{tp}', 
                    f'Embedding_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                    fillcolor=gpu_colors[f'EP{ep}'])
    
    # Connect input to embeddings
    for ep in range(16):
        for tp in range(4):
            dot.edge('input', f'embed_{ep}_{tp}')
    
    # Expert routing (gate selection)
    for ep in range(16):
        dot.node(f'gate_{ep}', 
                f'Gate_{ep}\\nGPU:{ep*4}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, top_k=2]', 
                shape='parallelogram', fillcolor='gold', style='filled,dashed', penwidth='2')
    
    # Connect embeddings to gates
    for ep in range(16):
        dot.edge('input', f'gate_{ep}', style='dashed', penwidth='2')
    
    # Layer loop (16 layers)
    for layer in range(16):
        # Attention computation for each expert
        for ep in range(16):
            for tp in range(4):
                gpu_id = ep * 4 + tp
                
                # QKV projection (column parallel)
                dot.node(f'qkv_{layer}_{ep}_{tp}', 
                        f'QKV_Proj_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, heads=8, d_k=64]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention scores
                dot.node(f'attn_scores_{layer}_{ep}_{tp}', 
                        f'Attention_Scores_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, seq_len=128, heads=8]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention softmax
                dot.node(f'attn_softmax_{layer}_{ep}_{tp}', 
                        f'Attention_Softmax_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, seq_len=128, heads=8]\\nOutput: [batch_size=128, seq_len=128, seq_len=128, heads=8]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention dropout
                dot.node(f'attn_dropout_{layer}_{ep}_{tp}', 
                        f'Attention_Dropout_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, seq_len=128, heads=8]\\nOutput: [batch_size=128, seq_len=128, seq_len=128, heads=8]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention output (values)
                dot.node(f'attn_values_{layer}_{ep}_{tp}', 
                        f'Attention_Values_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, heads=8, d_k=64]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention weighted sum
                dot.node(f'attn_weighted_{layer}_{ep}_{tp}', 
                        f'Attention_Weighted_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, seq_len=128, heads=8], [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, heads=8, d_k=64]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention output projection (row parallel)
                dot.node(f'attn_out_{layer}_{ep}_{tp}', 
                        f'Attention_Output_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # Attention all-reduce (communication)
                dot.node(f'attn_allreduce_{layer}_{ep}', 
                        f'Attention_AllReduce_{layer}_{ep}\\nEP:{ep}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]', 
                        shape='ellipse', fillcolor='red', style='filled', penwidth='2')
        
        # MLP computation for each expert
        for ep in range(16):
            for tp in range(4):
                gpu_id = ep * 4 + tp
                
                # MLP first linear (column parallel)
                dot.node(f'mlp1_{layer}_{ep}_{tp}', 
                        f'MLP_Linear1_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, ffn=512]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # GELU activation
                dot.node(f'gelu_{layer}_{ep}_{tp}', 
                        f'GELU_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, ffn=512]\\nOutput: [batch_size=128, seq_len=128, ffn=512]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
                
                # MLP second linear (row parallel)
                dot.node(f'mlp2_{layer}_{ep}_{tp}', 
                        f'MLP_Linear2_{layer}_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, ffn=512]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                        fillcolor=gpu_colors[f'EP{ep}'])
            
            # MLP all-reduce (communication)
            dot.node(f'mlp_allreduce_{layer}_{ep}', 
                    f'MLP_AllReduce_{layer}_{ep}\\nEP:{ep}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]', 
                    shape='ellipse', fillcolor='red', style='filled', penwidth='2')
        
        # Expert routing all-to-all communication
        for ep in range(16):
            dot.node(f'expert_route_{layer}_{ep}', 
                    f'Expert_Route_{layer}_{ep}\\nEP:{ep}\\nInput: [batch_size=128, seq_len=128, hidden=512]\\nOutput: [batch_size=128, seq_len=128, hidden=512]', 
                    shape='ellipse', fillcolor='orange', style='filled', penwidth='2')
    
    # Final layer norm
    for ep in range(16):
        for tp in range(4):
            gpu_id = ep * 4 + tp
            dot.node(f'final_norm_{ep}_{tp}', 
                    f'Final_Norm_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, hidden=128]', 
                    fillcolor=gpu_colors[f'EP{ep}'])
    
    # Output projection
    for ep in range(16):
        for tp in range(4):
            gpu_id = ep * 4 + tp
            dot.node(f'output_proj_{ep}_{tp}', 
                    f'Output_Proj_{ep}_{tp}\\nGPU:{gpu_id}\\nInput: [batch_size=128, seq_len=128, hidden=128]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]', 
                    fillcolor=gpu_colors[f'EP{ep}'])
    
    # Final all-reduce for output
    dot.node('final_allreduce', 
            'Final_AllReduce\\nAll GPUs\\nInput: [batch_size=128, seq_len=128, vocab=32000]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]', 
            shape='ellipse', fillcolor='red', style='filled', penwidth='3')
    
    # Output node
    dot.node('output', 
            'Output\\nInput: [batch_size=128, seq_len=128, vocab=32000]\\nOutput: [batch_size=128, seq_len=128, vocab=32000]', 
            shape='ellipse', fillcolor='white', style='filled', penwidth='2')
    
    # Connect final layer to output
    dot.edge('final_allreduce', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the DAG
    dag = create_parallel_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-25-09-26-32/llm_parallel_strategy.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-25-09-26-32/llm_parallel_strategy', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-25-09-26-32/llm_parallel_strategy.dot")
    print(f"SVG file: ../outputs/2025-12-25-09-26-32/llm_parallel_strategy.svg")