#!/usr/bin/env python3
"""
Generate DAG for Qwen3-235B MoE Parallel Strategy
TP=8, PP=1, EP=8, SP=1
"""

import graphviz

def create_moe_dag():
    """Create detailed DAG for MoE model with parallel strategy"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='Qwen3-235B MoE Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded', bgcolor='lightgray')
        c.node('input', 'Input\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
               shape='rectangle', fillcolor='lightcyan')
    
    # Embedding layer (distributed across TP=8)
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Embedding Layer (TP=8)', style='rounded', bgcolor='lightgray')
        for gpu in range(8):
            c.node(f'embed_{gpu}', f'Embedding_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=512]', 
                   shape='rectangle', fillcolor='lightgreen')
        
        # All-gather for embedding output
        c.node('embed_ag', 'All-Gather Embedding\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
               shape='ellipse', fillcolor='lightblue')
    
    # Connect input to embedding
    dot.edge('input', 'embed_0')
    dot.edge('input', 'embed_1')
    dot.edge('input', 'embed_2')
    dot.edge('input', 'embed_3')
    dot.edge('input', 'embed_4')
    dot.edge('input', 'embed_5')
    dot.edge('input', 'embed_6')
    dot.edge('reduce_7')
    
    # Connect embedding to all-gather
    for gpu in range(8):
        dot.edge(f'embed_{gpu}', 'embed_ag')
    
    # Layer pattern - repeat for 94 layers
    for layer in range(1):  # Show pattern for first layer, then indicate repetition
        with dot.subgraph(name=f'cluster_layer_{layer}') as c:
            c.attr(label=f'Layer {layer} (94 total layers)', style='rounded', bgcolor='lightgray')
            
            # LayerNorm (replicated across TP)
            c.node(f'ln1_{layer}', f'LayerNorm1_L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention mechanism (TP=8)
            for gpu in range(8):
                # QKV projection
                c.node(f'qkv_{layer}_gpu{gpu}', f'QKV_Proj_L{layer}_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=192]', 
                       shape='rectangle', fillcolor='lightgreen')
                
                # Attention computation
                c.node(f'attn_{layer}_gpu{gpu}', f'Attention_L{layer}_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=192]\\nOutput: [batch_size=128, seq_len=2048, hidden=64]', 
                       shape='rectangle', fillcolor='lightgreen')
                
                # Output projection
                c.node(f'out_proj_{layer}_gpu{gpu}', f'Out_Proj_L{layer}_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=64]\\nOutput: [batch_size=128, seq_len=2048, hidden=512]', 
                       shape='rectangle', fillcolor='lightgreen')
            
            # All-gather for attention output
            c.node(f'attn_ag_{layer}', f'All-Gather Attention L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='ellipse', fillcolor='lightblue')
            
            # Residual connection
            c.node(f'resid_{layer}', f'Residual_L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Second LayerNorm
            c.node(f'ln2_{layer}', f'LayerNorm2_L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # MoE Gate (distributed)
            for gpu in range(8):
                c.node(f'gate_{layer}_gpu{gpu}', f'Gate_L{layer}_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, expert_scores=128]', 
                       shape='parallelogram', fillcolor='lightyellow')
            
            # Expert selection (Top-K=8)
            c.node(f'select_{layer}', f'Select Top-8 Experts L{layer}\\nInput: [batch_size=128, seq_len=2048, expert_scores=4096]\\nOutput: [batch_size=128, seq_len=2048, selected_experts=8]', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Expert computation (EP=8, 16 experts per GPU)
            for gpu in range(8):
                for expert in range(16):  # 16 experts per GPU
                    c.node(f'expert_{layer}_gpu{gpu}_exp{expert}', f'Expert_L{layer}_GPU{gpu}_Exp{expert}\\nInput: [batch_size=?, seq_len=?, hidden=512]\\nOutput: [batch_size=?, seq_len=?, hidden=512]', 
                           shape='rectangle', fillcolor='lightgreen')
            
            # Expert routing (dashed lines for gate selection)
            for gpu in range(8):
                c.node(f'route_{layer}_gpu{gpu}', f'Route to Experts L{layer}_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, selected_experts=8]\\nOutput: [batch_size=?, seq_len=?, hidden=512]', 
                       shape='parallelogram', fillcolor='lightyellow')
            
            # Expert aggregation
            for gpu in range(8):
                c.node(f'agg_{layer}_gpu{gpu}', f'Aggregate Experts L{layer}_GPU{gpu}\\nInput: [batch_size=?, seq_len=?, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=512]', 
                       shape='parallelogram', fillcolor='lightyellow')
            
            # All-gather for MoE output
            c.node(f'moe_ag_{layer}', f'All-Gather MoE L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='ellipse', fillcolor='lightblue')
            
            # Final residual
            c.node(f'final_resid_{layer}', f'Final Residual_L{layer}\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle', fillcolor='lightgreen')
        
        # Connect layer components
        # From embedding/prev layer to current layer
        if layer == 0:
            dot.edge('embed_ag', f'ln1_{layer}')
        
        # LayerNorm1 -> QKV (split across TP)
        for gpu in range(8):
            dot.edge(f'ln1_{layer}', f'qkv_{layer}_gpu{gpu}')
            dot.edge(f'qkv_{layer}_gpu{gpu}', f'attn_{layer}_gpu{gpu}')
            dot.edge(f'attn_{layer}_gpu{gpu}', f'out_proj_{layer}_gpu{gpu}')
            dot.edge(f'out_proj_{layer}_gpu{gpu}', f'attn_ag_{layer}')
        
        # Attention all-gather -> residual
        dot.edge(f'attn_ag_{layer}', f'resid_{layer}')
        
        # Residual -> LayerNorm2
        dot.edge(f'resid_{layer}', f'ln2_{layer}')
        
        # LayerNorm2 -> Gate (split across TP)
        for gpu in range(8):
            dot.edge(f'ln2_{layer}', f'gate_{layer}_gpu{gpu}')
            dot.edge(f'gate_{layer}_gpu{gpu}', f'select_{layer}')
        
        # Expert selection -> routing (dashed lines for gate selection)
        for gpu in range(8):
            dot.edge(f'select_{layer}', f'route_{layer}_gpu{gpu}', style='dashed')
            
            # Routing to individual experts
            for expert in range(16):
                dot.edge(f'route_{layer}_gpu{gpu}', f'expert_{layer}_gpu{gpu}_exp{expert}')
                dot.edge(f'expert_{layer}_gpu{gpu}_exp{expert}', f'agg_{layer}_gpu{gpu}')
            
            # Aggregation -> All-gather
            dot.edge(f'agg_{layer}_gpu{gpu}', f'moe_ag_{layer}')
        
        # MoE all-gather -> final residual
        dot.edge(f'moe_ag_{layer}', f'final_resid_{layer}')
    
    # Final LayerNorm and Output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='rounded', bgcolor='lightgray')
        
        # Final LayerNorm (distributed across TP)
        for gpu in range(8):
            c.node(f'final_ln_gpu{gpu}', f'Final LayerNorm_GPU{gpu}\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=512]', 
                   shape='rectangle', fillcolor='lightgreen')
        
        # Final all-gather
        c.node('final_ag', 'Final All-Gather\\nInput: [batch_size=128, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
               shape='ellipse', fillcolor='lightblue')
        
        # Output projection
        c.node('output_proj', 'Output Projection\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, vocab=151936]', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Final output
        c.node('output', 'Output\\nInput: [batch_size=128, seq_len=2048, vocab=151936]\\nOutput: [batch_size=128, seq_len=2048, vocab=151936]', 
               shape='rectangle', fillcolor='lightcyan')
    
    # Connect final layer to output
    dot.edge('final_resid_0', 'final_ln_gpu0')
    dot.edge('final_resid_0', 'final_ln_gpu1')
    dot.edge('final_resid_0', 'final_ln_gpu2')
    dot.edge('final_resid_0', 'final_ln_gpu3')
    dot.edge('final_resid_0', 'final_ln_gpu4')
    dot.edge('final_resid_0', 'final_ln_gpu5')
    dot.edge('final_resid_0', 'final_ln_gpu6')
    dot.edge('final_resid_0', 'final_ln_gpu7')
    
    for gpu in range(8):
        dot.edge(f'final_ln_gpu{gpu}', 'final_ag')
    
    dot.edge('final_ag', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

if __name__ == "__main__":
    # Create the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dag.save('./outputs/2025-12-31-10-57-02/moe_parallel_dag.dot')
    
    # Save as SVG image
    dag.render('./outputs/2025-12-31-10-57-02/moe_parallel_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files saved:")
    print("- ./outputs/2025-12-31-10-57-02/moe_parallel_dag.dot")
    print("- ./outputs/2025-12-31-10-57-02/moe_parallel_dag.svg")