import graphviz
import os

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_model_dag', 
                           comment='Baseline 4-layer Dense Model DAG (TP=8, PP=2)',
                           format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,40', splines='ortho')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Communication
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightgray')  # Aggregation/Reduction
    
    # Input node
    dot.node('input', '''<b>Input</b><br/>Input: [batch_size=1024, seq_len=?, vocab_size=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    # Embedding layer - distributed across first 8 GPUs (pipeline stage 0)
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Embedding Layer (Pipeline Stage 0, TP=8)', fontsize='12', style='rounded')
        
        # Embedding across GPUs 0-7
        for i in range(8):
            c.node(f'embed_{i}', 
                   f'<b>Token Embedding</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, vocab=32000]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
        
        # Positional encoding
        for i in range(8):
            c.node(f'pos_enc_{i}', 
                   f'<b>Positional Encoding</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Layer 0 - Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_layer0') as c:
        c.attr(label='Layer 0 (Pipeline Stage 0, TP=8)', fontsize='12', style='rounded')
        
        # Attention components
        for i in range(8):
            # Q projection
            c.node(f'l0_q_proj_{i}', 
                   f'<b>Q Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # K projection
            c.node(f'l0_k_proj_{i}', 
                   f'<b>K Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # V projection
            c.node(f'l0_v_proj_{i}', 
                   f'<b>V Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Flash attention
            c.node(f'l0_attn_{i}', 
                   f'<b>Flash Attention</b><br/>GPU: gpu_{i}<br/>Input: Q,K,V [batch=1024, seq=?, heads=32, d_k=16]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Output projection
            c.node(f'l0_out_proj_{i}', 
                   f'<b>Output Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, heads=32, d_v=16]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Residual connection
            c.node(f'l0_res_{i}', 
                   f'<b>Residual Add</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            # Layer norm
            c.node(f'l0_ln1_{i}', 
                   f'<b>Layer Norm 1</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # FFN components
            c.node(f'l0_ffn_up_{i}', 
                   f'<b>FFN Up Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, ffn_dim=2048]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l0_ffn_down_{i}', 
                   f'<b>FFN Down Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, ffn_dim=2048]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l0_res2_{i}', 
                   f'<b>Residual Add 2</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l0_ln2_{i}', 
                   f'<b>Layer Norm 2</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Communication between pipeline stages
    dot.node('comm_stage0_to_stage1', 
             '<b>Pipeline Communication</b><br/>All GPUs 0-7 to All GPUs 8-15<br/>[batch=1024, seq=?, hidden=512]',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer 1 - Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 (Pipeline Stage 1, TP=8)', fontsize='12', style='rounded')
        
        for i in range(8):
            gpu_id = i + 8
            # Similar structure for layer 1
            c.node(f'l1_q_proj_{i}', 
                   f'<b>Q Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_k_proj_{i}', 
                   f'<b>K Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_v_proj_{i}', 
                   f'<b>V Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_attn_{i}', 
                   f'<b>Flash Attention</b><br/>GPU: gpu_{gpu_id}<br/>Input: Q,K,V [batch=1024, seq=?, heads=32, d_k=16]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_out_proj_{i}', 
                   f'<b>Output Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, heads=32, d_v=16]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_res_{i}', 
                   f'<b>Residual Add</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l1_ln1_{i}', 
                   f'<b>Layer Norm 1</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_ffn_up_{i}', 
                   f'<b>FFN Up Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, ffn_dim=2048]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_ffn_down_{i}', 
                   f'<b>FFN Down Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, ffn_dim=2048]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_res2_{i}', 
                   f'<b>Residual Add 2</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l1_ln2_{i}', 
                   f'<b>Layer Norm 2</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Communication back to stage 0 for layers 2-3
    dot.node('comm_stage1_to_stage0', 
             '<b>Pipeline Communication</b><br/>All GPUs 8-15 to All GPUs 0-7<br/>[batch=1024, seq=?, hidden=512]',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer 2 - Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_layer2') as c:
        c.attr(label='Layer 2 (Pipeline Stage 0, TP=8)', fontsize='12', style='rounded')
        
        for i in range(8):
            c.node(f'l2_q_proj_{i}', 
                   f'<b>Q Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_k_proj_{i}', 
                   f'<b>K Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_v_proj_{i}', 
                   f'<b>V Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_attn_{i}', 
                   f'<b>Flash Attention</b><br/>GPU: gpu_{i}<br/>Input: Q,K,V [batch=1024, seq=?, heads=32, d_k=16]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_out_proj_{i}', 
                   f'<b>Output Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, heads=32, d_v=16]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_res_{i}', 
                   f'<b>Residual Add</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l2_ln1_{i}', 
                   f'<b>Layer Norm 1</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_ffn_up_{i}', 
                   f'<b>FFN Up Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, ffn_dim=2048]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_ffn_down_{i}', 
                   f'<b>FFN Down Projection</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, ffn_dim=2048]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l2_res2_{i}', 
                   f'<b>Residual Add 2</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l2_ln2_{i}', 
                   f'<b>Layer Norm 2</b><br/>GPU: gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Communication to stage 1 for layer 3
    dot.node('comm_stage0_to_stage1_2', 
             '<b>Pipeline Communication</b><br/>All GPUs 0-7 to All GPUs 8-15<br/>[batch=1024, seq=?, hidden=512]',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Layer 3 - Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_layer3') as c:
        c.attr(label='Layer 3 (Pipeline Stage 1, TP=8)', fontsize='12', style='rounded')
        
        for i in range(8):
            gpu_id = i + 8
            c.node(f'l3_q_proj_{i}', 
                   f'<b>Q Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_k_proj_{i}', 
                   f'<b>K Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_k=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_v_proj_{i}', 
                   f'<b>V Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_attn_{i}', 
                   f'<b>Flash Attention</b><br/>GPU: gpu_{gpu_id}<br/>Input: Q,K,V [batch=1024, seq=?, heads=32, d_k=16]<br/>Output: [batch=1024, seq=?, heads=32, d_v=16]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_out_proj_{i}', 
                   f'<b>Output Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, heads=32, d_v=16]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_res_{i}', 
                   f'<b>Residual Add</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l3_ln1_{i}', 
                   f'<b>Layer Norm 1</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_ffn_up_{i}', 
                   f'<b>FFN Up Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, ffn_dim=2048]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_ffn_down_{i}', 
                   f'<b>FFN Down Projection</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, ffn_dim=2048]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l3_res2_{i}', 
                   f'<b>Residual Add 2</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='diamond', fillcolor='lightgray')
            
            c.node(f'l3_ln2_{i}', 
                   f'<b>Layer Norm 2</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Output layer - distributed across GPUs 8-15
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer (Pipeline Stage 1, TP=8)', fontsize='12', style='rounded')
        
        for i in range(8):
            gpu_id = i + 8
            c.node(f'output_{i}', 
                   f'<b>Linear Output</b><br/>GPU: gpu_{gpu_id}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, vocab=4000]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Final output aggregation
    dot.node('output', '''<b>Final Output</b><br/>Input: [batch=1024, seq=?, vocab=32000]<br/>Output: [batch=1024, seq=?, vocab=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    # All-reduce operations for tensor parallelism
    for layer in range(4):
        for op in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'ffn_up', 'ffn_down']:
            dot.node(f'allreduce_{layer}_{op}', 
                     f'<b>All-Reduce</b><br/>TP across 8 GPUs<br/>[batch=1024, seq=?, dim=4096]',
                     shape='parallelogram', fillcolor='lightyellow')
    
    # Create edges
    # Input to embedding
    for i in range(8):
        dot.edge('input', f'embed_{i}')
        dot.edge(f'embed_{i}', f'pos_enc_{i}')
    
    # Layer 0 connections
    for i in range(8):
        dot.edge(f'pos_enc_{i}', f'l0_q_proj_{i}')
        dot.edge(f'pos_enc_{i}', f'l0_k_proj_{i}')
        dot.edge(f'pos_enc_{i}', f'l0_v_proj_{i}')
        dot.edge(f'l0_q_proj_{i}', f'l0_attn_{i}')
        dot.edge(f'l0_k_proj_{i}', f'l0_attn_{i}')
        dot.edge(f'l0_v_proj_{i}', f'l0_attn_{i}')
        dot.edge(f'l0_attn_{i}', f'l0_out_proj_{i}')
        dot.edge(f'l0_out_proj_{i}', f'l0_res_{i}')
        dot.edge(f'pos_enc_{i}', f'l0_res_{i}')  # Residual connection
        dot.edge(f'l0_res_{i}', f'l0_ln1_{i}')
        dot.edge(f'l0_ln1_{i}', f'l0_ffn_up_{i}')
        dot.edge(f'l0_ffn_up_{i}', f'l0_ffn_down_{i}')
        dot.edge(f'l0_ffn_down_{i}', f'l0_res2_{i}')
        dot.edge(f'l0_ln1_{i}', f'l0_res2_{i}')  # Residual connection
        dot.edge(f'l0_res2_{i}', f'l0_ln2_{i}')
    
    # Pipeline communication
    for i in range(8):
        dot.edge(f'l0_ln2_{i}', 'comm_stage0_to_stage1')
    
    # Layer 1 connections
    for i in range(8):
        dot.edge('comm_stage0_to_stage1', f'l1_q_proj_{i}')
        dot.edge('comm_stage0_to_stage1', f'l1_k_proj_{i}')
        dot.edge('comm_stage0_to_stage1', f'l1_v_proj_{i}')
        dot.edge(f'l1_q_proj_{i}', f'l1_attn_{i}')
        dot.edge(f'l1_k_proj_{i}', f'l1_attn_{i}')
        dot.edge(f'l1_v_proj_{i}', f'l1_attn_{i}')
        dot.edge(f'l1_attn_{i}', f'l1_out_proj_{i}')
        dot.edge(f'l1_out_proj_{i}', f'l1_res_{i}')
        dot.edge('comm_stage0_to_stage1', f'l1_res_{i}')  # Residual connection
        dot.edge(f'l1_res_{i}', f'l1_ln1_{i}')
        dot.edge(f'l1_ln1_{i}', f'l1_ffn_up_{i}')
        dot.edge(f'l1_ffn_up_{i}', f'l1_ffn_down_{i}')
        dot.edge(f'l1_ffn_down_{i}', f'l1_res2_{i}')
        dot.edge(f'l1_ln1_{i}', f'l1_res2_{i}')  # Residual connection
        dot.edge(f'l1_res2_{i}', f'l1_ln2_{i}')
    
    # Pipeline communication back
    for i in range(8):
        dot.edge(f'l1_ln2_{i}', 'comm_stage1_to_stage0')
    
    # Layer 2 connections
    for i in range(8):
        dot.edge('comm_stage1_to_stage0', f'l2_q_proj_{i}')
        dot.edge('comm_stage1_to_stage0', f'l2_k_proj_{i}')
        dot.edge('comm_stage1_to_stage0', f'l2_v_proj_{i}')
        dot.edge(f'l2_q_proj_{i}', f'l2_attn_{i}')
        dot.edge(f'l2_k_proj_{i}', f'l2_attn_{i}')
        dot.edge(f'l2_v_proj_{i}', f'l2_attn_{i}')
        dot.edge(f'l2_attn_{i}', f'l2_out_proj_{i}')
        dot.edge(f'l2_out_proj_{i}', f'l2_res_{i}')
        dot.edge('comm_stage1_to_stage0', f'l2_res_{i}')  # Residual connection
        dot.edge(f'l2_res_{i}', f'l2_ln1_{i}')
        dot.edge(f'l2_ln1_{i}', f'l2_ffn_up_{i}')
        dot.edge(f'l2_ffn_up_{i}', f'l2_ffn_down_{i}')
        dot.edge(f'l2_ffn_down_{i}', f'l2_res2_{i}')
        dot.edge(f'l2_ln1_{i}', f'l2_res2_{i}')  # Residual connection
        dot.edge(f'l2_res2_{i}', f'l2_ln2_{i}')
    
    # Pipeline communication to stage 1
    for i in range(8):
        dot.edge(f'l2_ln2_{i}', 'comm_stage0_to_stage1_2')
    
    # Layer 3 connections
    for i in range(8):
        dot.edge('comm_stage0_to_stage1_2', f'l3_q_proj_{i}')
        dot.edge('comm_stage0_to_stage1_2', f'l3_k_proj_{i}')
        dot.edge('comm_stage0_to_stage1_2', f'l3_v_proj_{i}')
        dot.edge(f'l3_q_proj_{i}', f'l3_attn_{i}')
        dot.edge(f'l3_k_proj_{i}', f'l3_attn_{i}')
        dot.edge(f'l3_v_proj_{i}', f'l3_attn_{i}')
        dot.edge(f'l3_attn_{i}', f'l3_out_proj_{i}')
        dot.edge(f'l3_out_proj_{i}', f'l3_res_{i}')
        dot.edge('comm_stage0_to_stage1_2', f'l3_res_{i}')  # Residual connection
        dot.edge(f'l3_res_{i}', f'l3_ln1_{i}')
        dot.edge(f'l3_ln1_{i}', f'l3_ffn_up_{i}')
        dot.edge(f'l3_ffn_up_{i}', f'l3_ffn_down_{i}')
        dot.edge(f'l3_ffn_down_{i}', f'l3_res2_{i}')
        dot.edge(f'l3_ln1_{i}', f'l3_res2_{i}')  # Residual connection
        dot.edge(f'l3_res2_{i}', f'l3_ln2_{i}')
    
    # Output layer connections
    for i in range(8):
        dot.edge(f'l3_ln2_{i}', f'output_{i}')
        dot.edge(f'output_{i}', 'output')
    
    # Save the DAG
    dot.render('../outputs/2025-10-17-08-58-21/baseline_model_dag', cleanup=False)
    
    # Also save as dot file
    dot.format = 'dot'
    dot.render('../outputs/2025-10-17-08-58-21/baseline_model_dag', cleanup=False)

if __name__ == '__main__':
    create_baseline_dag()