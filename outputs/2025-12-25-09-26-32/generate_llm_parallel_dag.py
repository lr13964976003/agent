#!/usr/bin/env python3
"""
DAG Generator for LLM Parallel Strategy: EP-16 × TP-4 × PP-1
This script generates a comprehensive DAG diagram showing the complete flow
of the LLM inference with expert parallelism, tensor parallelism, and communication patterns.
"""

import graphviz
from graphviz import Digraph
import os

def create_llm_parallel_dag():
    """Create a comprehensive DAG for the LLM parallel strategy."""
    
    # Create the directed graph
    dot = Digraph(comment='LLM Parallel Strategy DAG: EP-16 × TP-4 × PP-1')
    dot.attr(rankdir='TB')
    dot.attr(nodesep='0.8')
    dot.attr(ranksep='1.2')
    dot.attr(bgcolor='white')
    
    # Input node
    dot.node('input', 
             'INPUT\\nBatch Size: 128, Seq Len: 128-10240\\nToken Dim: 512, Precision: FP16',
             shape='oval', fillcolor='lightcoral', style='filled')
    
    # Token embedding
    dot.node('embed', 
             'Token Embedding\\nGPU: ALL\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 512]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Add edges
    dot.edge('input', 'embed')
    
    # Create nodes for each layer (16 layers total)
    for layer in range(16):
        with dot.subgraph(name=f'cluster_layer_{layer}') as c:
            c.attr(label=f'Layer {layer}', style='rounded', fillcolor='lightgray', color='black')
            
            # Layer normalization
            c.node(f'ln1_{layer}', 
                   f'LayerNorm 1\\nGPU: ALL\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 512]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # Self-attention with tensor parallelism (4-way)
            c.node(f'attn_qkv_split_{layer}', 
                   f'Split QKV Weights\\nGPU: ALL\\nTP-4 Sharding',
                   shape='parallelogram', fillcolor='lightyellow', style='filled')
            
            # Q projection (TP-4)
            for tp in range(4):
                c.node(f'attn_q_{layer}_tp{tp}', 
                       f'Q Projection\\nGPU: {layer*4 + tp}\\nTP-{tp}\\nInput: [128, seq_len, 128]\\nOutput: [128, seq_len, 128]',
                       shape='rectangle', fillcolor='lightgreen', style='filled')
                
            # K projection (TP-4)
            for tp in range(4):
                c.node(f'attn_k_{layer}_tp{tp}', 
                       f'K Projection\\nGPU: {layer*4 + tp}\\nTP-{tp}\\nInput: [128, seq_len, 128]\\nOutput: [128, seq_len, 128]',
                       shape='rectangle', fillcolor='lightgreen', style='filled')
                
            # V projection (TP-4)
            for tp in range(4):
                c.node(f'attn_v_{layer}_tp{tp}', 
                       f'V Projection\\nGPU: {layer*4 + tp}\\nTP-{tp}\\nInput: [128, seq_len, 128]\\nOutput: [128, seq_len, 128]',
                       shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # All-reduce for QKV
            c.node(f'attn_qkv_allreduce_{layer}', 
                   f'All-Reduce QKV\\nGPU: ALL\\nTP Communication',
                   shape='ellipse', fillcolor='lightblue', style='filled')
            
            # Attention computation
            c.node(f'attn_score_{layer}', 
                   f'Attention Score\\nGPU: ALL\\nInput: [128, 16, seq_len, 32]\\nOutput: [128, 16, seq_len, 32]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            c.node(f'attn_softmax_{layer}', 
                   f'Attention Softmax\\nGPU: ALL\\nInput: [128, 16, seq_len, seq_len]\\nOutput: [128, 16, seq_len, seq_len]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            c.node(f'attn_output_{layer}', 
                   f'Attention Output\\nGPU: ALL\\nInput: [128, 16, seq_len, 32]\\nOutput: [128, seq_len, 512]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # O projection (TP-4)
            for tp in range(4):
                c.node(f'attn_o_{layer}_tp{tp}', 
                       f'O Projection\\nGPU: {layer*4 + tp}\\nTP-{tp}\\nInput: [128, seq_len, 128]\\nOutput: [128, seq_len, 128]',
                       shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # All-reduce for attention output
            c.node(f'attn_out_allreduce_{layer}', 
                   f'All-Reduce Attn Output\\nGPU: ALL\\nTP Communication',
                   shape='ellipse', fillcolor='lightblue', style='filled')
            
            # Residual connection
            c.node(f'residual1_{layer}', 
                   f'Residual Add 1\\nGPU: ALL\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 512]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # Layer normalization 2
            c.node(f'ln2_{layer}', 
                   f'LayerNorm 2\\nGPU: ALL\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 512]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
            
            # Expert routing (gate)
            c.node(f'gate_{layer}', 
                   f'Expert Gate\\nGPU: ALL\\nTop-2 Expert Selection',
                   shape='diamond', fillcolor='orange', style='filled')
            
            # Expert routing communication (all-to-all)
            c.node(f'route_comm_{layer}', 
                   f'All-to-All Routing\\nGPU: ALL\\nEP Communication',
                   shape='ellipse', fillcolor='lightblue', style='dashed')
            
            # Expert computations (16 experts, 1 per GPU group)
            for expert in range(16):
                gpu_group = expert * 4  # Each expert group has 4 GPUs for TP
                
                # Expert MLP first linear (column parallel)
                for tp in range(4):
                    c.node(f'expert_{layer}_exp{expert}_mlp1_tp{tp}', 
                           f'Expert {expert} MLP1\\nGPU: {gpu_group + tp}\\nTP-{tp}\\nInput: [tokens, 256]\\nOutput: [tokens, 512]',
                           shape='rectangle', fillcolor='lightgreen', style='filled')
                
                # Expert MLP activation
                c.node(f'expert_{layer}_exp{expert}_act', 
                       f'Expert {expert} GELU\\nGPU: {gpu_group}-{gpu_group+3}\\nInput: [tokens, 1024]\\nOutput: [tokens, 1024]',
                       shape='rectangle', fillcolor='lightgreen', style='filled')
                
                # Expert MLP second linear (row parallel)
                for tp in range(4):
                    c.node(f'expert_{layer}_exp{expert}_mlp2_tp{tp}', 
                           f'Expert {expert} MLP2\\nGPU: {gpu_group + tp}\\nTP-{tp}\\nInput: [tokens, 512]\\nOutput: [tokens, 256]',
                           shape='rectangle', fillcolor='lightgreen', style='filled')
                
                # Expert output all-reduce
                c.node(f'expert_{layer}_exp{expert}_allreduce', 
                       f'Expert {expert} All-Reduce\\nGPU: {gpu_group}-{gpu_group+3}\\nTP Communication',
                       shape='ellipse', fillcolor='lightblue', style='filled')
            
            # Expert aggregation
            c.node(f'expert_agg_{layer}', 
                   f'Expert Aggregation\\nGPU: ALL\\nWeighted Sum\\nInput: [2, tokens, 512]\\nOutput: [128, seq_len, 512]',
                   shape='parallelogram', fillcolor='lightyellow', style='filled')
            
            # Final residual connection
            c.node(f'residual2_{layer}', 
                   f'Residual Add 2\\nGPU: ALL\\nInput: [128, seq_len, 512]\\nOutput: [128, seq_len, 512]',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Output node
    dot.node('output', 
             'OUTPUT\\nFinal Hidden States\\n[128, seq_len, 512]',
             shape='oval', fillcolor='lightcoral', style='filled')
    
    # Connect layers
    dot.edge('embed', 'ln1_0')
    
    for layer in range(16):
        # Attention path
        dot.edge(f'ln1_{layer}', f'attn_qkv_split_{layer}')
        
        # QKV projections with TP
        for tp in range(4):
            dot.edge(f'attn_qkv_split_{layer}', f'attn_q_{layer}_tp{tp}')
            dot.edge(f'attn_qkv_split_{layer}', f'attn_k_{layer}_tp{tp}')
            dot.edge(f'attn_qkv_split_{layer}', f'attn_v_{layer}_tp{tp}')
        
        # All-reduce after QKV
        for tp in range(4):
            dot.edge(f'attn_q_{layer}_tp{tp}', f'attn_qkv_allreduce_{layer}')
            dot.edge(f'attn_k_{layer}_tp{tp}', f'attn_qkv_allreduce_{layer}')
            dot.edge(f'attn_v_{layer}_tp{tp}', f'attn_qkv_allreduce_{layer}')
        
        dot.edge(f'attn_qkv_allreduce_{layer}', f'attn_score_{layer}')
        dot.edge(f'attn_score_{layer}', f'attn_softmax_{layer}')
        dot.edge(f'attn_softmax_{layer}', f'attn_output_{layer}')
        
        # O projection with TP
        for tp in range(4):
            dot.edge(f'attn_output_{layer}', f'attn_o_{layer}_tp{tp}')
        
        # All-reduce after O projection
        for tp in range(4):
            dot.edge(f'attn_o_{layer}_tp{tp}', f'attn_out_allreduce_{layer}')
        
        dot.edge(f'attn_out_allreduce_{layer}', f'residual1_{layer}')
        dot.edge(f'residual1_{layer}', f'ln2_{layer}')
        
        # Expert path
        dot.edge(f'ln2_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'route_comm_{layer}')
        
        # Expert computations
        for expert in range(16):
            dot.edge(f'route_comm_{layer}', f'expert_{layer}_exp{expert}_mlp1_tp0')
            
            # MLP1 with TP
            for tp in range(4):
                dot.edge(f'expert_{layer}_exp{expert}_mlp1_tp{tp}', f'expert_{layer}_exp{expert}_act')
            
            dot.edge(f'expert_{layer}_exp{expert}_act', f'expert_{layer}_exp{expert}_mlp2_tp0')
            
            # MLP2 with TP
            for tp in range(4):
                dot.edge(f'expert_{layer}_exp{expert}_mlp2_tp{tp}', f'expert_{layer}_exp{expert}_allreduce')
            
            dot.edge(f'expert_{layer}_exp{expert}_allreduce', f'expert_agg_{layer}')
        
        dot.edge(f'expert_agg_{layer}', f'residual2_{layer}')
        
        # Connect to next layer or output
        if layer < 15:
            dot.edge(f'residual2_{layer}', f'ln1_{layer+1}')
        else:
            dot.edge(f'residual2_{layer}', 'output')
    
    return dot

def main():
    """Main function to generate and save the DAG."""
    # Create the DAG
    dag = create_llm_parallel_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-25-09-26-32/llm_parallel_strategy.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    print(f"DOT file saved to: {dot_file}")
    
    # Save as SVG file
    svg_file = '../outputs/2025-12-25-09-26-32/llm_parallel_strategy.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_file}")
    
    # Also save as PNG for easier viewing
    png_file = '../outputs/2025-12-25-09-26-32/llm_parallel_strategy.png'
    dag.render(png_file.replace('.png', ''), format='png', cleanup=True)
    print(f"PNG file saved to: {png_file}")
    
    print("DAG generation completed successfully!")

if __name__ == "__main__":
    main()