#!/usr/bin/env python3
"""
Create detailed DAG for 10B model deployment with optimized parallel strategy
TP=2, PP=1, DP=2, SP=2, EP=1 using 4 GPUs
"""

import graphviz

def create_deployment_dag():
    # Create a new directed graph
    dot = graphviz.Digraph(comment='10B Model Deployment DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]', 
             shape='ellipse', fillcolor='lightcoral')
    
    # DP Split - Routing node
    dot.node('dp_split', 'DP Split\\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=10240, heads=16, d_k=32]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Communication: Input to DP Split
    dot.edge('input', 'dp_split', label='Host to Device')
    
    # DP Replica 1 (GPU0, GPU1)
    with dot.subgraph(name='cluster_dp1') as dp1:
        dp1.attr(label='DP Replica 1: GPU0 + GPU1', style='rounded,filled', fillcolor='lightgray')
        
        # SP Split for Replica 1
        dp1.node('sp_split_1', 'SP Split (Rep1)\\nInput: [batch_size=64, seq_len=10240, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=16, d_k=32]', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 1-16 for Replica 1
        for layer in range(1, 17):
            # Attention operations - split across TP=2
            dp1.node(f'attn_qkv_1_l{layer}', f'Attention QKV (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'attn_qkv_2_l{layer}', f'Attention QKV (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # TP Communication for attention
            dp1.node(f'tp_comm_attn_l{layer}', f'TP All-Reduce (Attn L{layer})\\nGPU0 <-> GPU1\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # Attention output projection
            dp1.node(f'attn_out_1_l{layer}', f'Attention Output (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'attn_out_2_l{layer}', f'Attention Output (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # Another TP Communication
            dp1.node(f'tp_comm_attn_out_l{layer}', f'TP All-Reduce (Attn Out L{layer})\\nGPU0 <-> GPU1\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # MLP operations - split across TP=2
            dp1.node(f'mlp_fc1_1_l{layer}', f'MLP FC1 (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=5120, hidden=512]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'mlp_fc1_2_l{layer}', f'MLP FC1 (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=5120, hidden=512]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            dp1.node(f'mlp_fc2_1_l{layer}', f'MLP FC2 (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, hidden=512]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'mlp_fc2_2_l{layer}', f'MLP FC2 (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, hidden=512]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # TP Communication for MLP
            dp1.node(f'tp_comm_mlp_l{layer}', f'TP All-Reduce (MLP L{layer})\\nGPU0 <-> GPU1\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # LayerNorm operations (replicated)
            dp1.node(f'ln1_1_l{layer}', f'LayerNorm1 (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'ln1_2_l{layer}', f'LayerNorm1 (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            dp1.node(f'ln2_1_l{layer}', f'LayerNorm2 (L{layer})\\nGPU0\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp1.node(f'ln2_2_l{layer}', f'LayerNorm2 (L{layer})\\nGPU1\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
        # SP Merge for Replica 1
        dp1.node('sp_merge_1', 'SP Merge (Rep1)\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=10240, token_dim=512]', 
                shape='parallelogram', fillcolor='lightyellow')
    
    # DP Replica 2 (GPU2, GPU3)  
    with dot.subgraph(name='cluster_dp2') as dp2:
        dp2.attr(label='DP Replica 2: GPU2 + GPU3', style='rounded,filled', fillcolor='lightgray')
        
        # SP Split for Replica 2
        dp2.node('sp_split_2', 'SP Split (Rep2)\\nInput: [batch_size=64, seq_len=10240, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=16, d_k=32]', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 1-16 for Replica 2
        for layer in range(1, 17):
            # Attention operations - split across TP=2
            dp2.node(f'attn_qkv_3_l{layer}', f'Attention QKV (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'attn_qkv_4_l{layer}', f'Attention QKV (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, heads=16, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # TP Communication for attention
            dp2.node(f'tp_comm_attn_gpu23_l{layer}', f'TP All-Reduce (Attn L{layer})\\nGPU2 <-> GPU3\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, heads=8, d_k=32]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # Attention output projection
            dp2.node(f'attn_out_3_l{layer}', f'Attention Output (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'attn_out_4_l{layer}', f'Attention Output (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, heads=8, d_k=32]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # Another TP Communication
            dp2.node(f'tp_comm_attn_out_gpu23_l{layer}', f'TP All-Reduce (Attn Out L{layer})\\nGPU2 <-> GPU3\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # MLP operations - split across TP=2
            dp2.node(f'mlp_fc1_3_l{layer}', f'MLP FC1 (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=5120, hidden=512]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'mlp_fc1_4_l{layer}', f'MLP FC1 (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=5120, hidden=512]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            dp2.node(f'mlp_fc2_3_l{layer}', f'MLP FC2 (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, hidden=512]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'mlp_fc2_4_l{layer}', f'MLP FC2 (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, hidden=512]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # TP Communication for MLP
            dp2.node(f'tp_comm_mlp_gpu23_l{layer}', f'TP All-Reduce (MLP L{layer})\\nGPU2 <-> GPU3\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='ellipse', fillcolor='lightblue')
            
            # LayerNorm operations (replicated)
            dp2.node(f'ln1_3_l{layer}', f'LayerNorm1 (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'ln1_4_l{layer}', f'LayerNorm1 (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            dp2.node(f'ln2_3_l{layer}', f'LayerNorm2 (L{layer})\\nGPU2\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            dp2.node(f'ln2_4_l{layer}', f'LayerNorm2 (L{layer})\\nGPU3\\nInput: [batch_size=64, seq_len=5120, token_dim=256]\\nOutput: [batch_size=64, seq_len=5120, token_dim=256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
        # SP Merge for Replica 2
        dp2.node('sp_merge_2', 'SP Merge (Rep2)\\nInput: [batch_size=64, seq_len=5120, token_dim=512]\\nOutput: [batch_size=64, seq_len=10240, token_dim=512]', 
                shape='parallelogram', fillcolor='lightyellow')
    
    # DP Merge - Aggregation node
    dot.node('dp_merge', 'DP Merge\\nInput: [batch_size=64, seq_len=10240, token_dim=512]\\nOutput: [batch_size=128, seq_len=10240, token_dim=512]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10240, token_dim=512]\\nOutput: [batch_size=128, seq_len=10240, token_dim=512]', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Connect the flow
    # DP Split to SP Splits
    dot.edge('dp_split', 'sp_split_1', label='Replica 1')
    dot.edge('dp_split', 'sp_split_2', label='Replica 2')
    
    # Connect layers for Replica 1
    dot.edge('sp_split_1', 'attn_qkv_1_l1')
    dot.edge('sp_split_1', 'attn_qkv_2_l1')
    
    for layer in range(1, 17):
        # Attention flow
        dot.edge(f'attn_qkv_1_l{layer}', f'tp_comm_attn_l{layer}')
        dot.edge(f'attn_qkv_2_l{layer}', f'tp_comm_attn_l{layer}')
        dot.edge(f'tp_comm_attn_l{layer}', f'attn_out_1_l{layer}')
        dot.edge(f'tp_comm_attn_l{layer}', f'attn_out_2_l{layer}')
        dot.edge(f'attn_out_1_l{layer}', f'tp_comm_attn_out_l{layer}')
        dot.edge(f'attn_out_2_l{layer}', f'tp_comm_attn_out_l{layer}')
        dot.edge(f'tp_comm_attn_out_l{layer}', f'ln1_1_l{layer}')
        dot.edge(f'tp_comm_attn_out_l{layer}', f'ln1_2_l{layer}')
        
        # MLP flow
        dot.edge(f'ln1_1_l{layer}', f'mlp_fc1_1_l{layer}')
        dot.edge(f'ln1_2_l{layer}', f'mlp_fc1_2_l{layer}')
        dot.edge(f'mlp_fc1_1_l{layer}', f'mlp_fc2_1_l{layer}')
        dot.edge(f'mlp_fc1_2_l{layer}', f'mlp_fc2_2_l{layer}')
        dot.edge(f'mlp_fc2_1_l{layer}', f'tp_comm_mlp_l{layer}')
        dot.edge(f'mlp_fc2_2_l{layer}', f'tp_comm_mlp_l{layer}')
        dot.edge(f'tp_comm_mlp_l{layer}', f'ln2_1_l{layer}')
        dot.edge(f'tp_comm_mlp_l{layer}', f'ln2_2_l{layer}')
        
        # Connect to next layer or SP merge
        if layer < 16:
            dot.edge(f'ln2_1_l{layer}', f'attn_qkv_1_l{layer+1}')
            dot.edge(f'ln2_2_l{layer}', f'attn_qkv_2_l{layer+1}')
        else:
            dot.edge(f'ln2_1_l{layer}', 'sp_merge_1')
            dot.edge(f'ln2_2_l{layer}', 'sp_merge_1')
    
    # Connect layers for Replica 2
    dot.edge('sp_split_2', 'attn_qkv_3_l1')
    dot.edge('sp_split_2', 'attn_qkv_4_l1')
    
    for layer in range(1, 17):
        # Attention flow
        dot.edge(f'attn_qkv_3_l{layer}', f'tp_comm_attn_gpu23_l{layer}')
        dot.edge(f'attn_qkv_4_l{layer}', f'tp_comm_attn_gpu23_l{layer}')
        dot.edge(f'tp_comm_attn_gpu23_l{layer}', f'attn_out_3_l{layer}')
        dot.edge(f'tp_comm_attn_gpu23_l{layer}', f'attn_out_4_l{layer}')
        dot.edge(f'attn_out_3_l{layer}', f'tp_comm_attn_out_gpu23_l{layer}')
        dot.edge(f'attn_out_4_l{layer}', f'tp_comm_attn_out_gpu23_l{layer}')
        dot.edge(f'tp_comm_attn_out_gpu23_l{layer}', f'ln1_3_l{layer}')
        dot.edge(f'tp_comm_attn_out_gpu23_l{layer}', f'ln1_4_l{layer}')
        
        # MLP flow
        dot.edge(f'ln1_3_l{layer}', f'mlp_fc1_3_l{layer}')
        dot.edge(f'ln1_4_l{layer}', f'mlp_fc1_4_l{layer}')
        dot.edge(f'mlp_fc1_3_l{layer}', f'mlp_fc2_3_l{layer}')
        dot.edge(f'mlp_fc1_4_l{layer}', f'mlp_fc2_4_l{layer}')
        dot.edge(f'mlp_fc2_3_l{layer}', f'tp_comm_mlp_gpu23_l{layer}')
        dot.edge(f'mlp_fc2_4_l{layer}', f'tp_comm_mlp_gpu23_l{layer}')
        dot.edge(f'tp_comm_mlp_gpu23_l{layer}', f'ln2_3_l{layer}')
        dot.edge(f'tp_comm_mlp_gpu23_l{layer}', f'ln2_4_l{layer}')
        
        # Connect to next layer or SP merge
        if layer < 16:
            dot.edge(f'ln2_3_l{layer}', f'attn_qkv_3_l{layer+1}')
            dot.edge(f'ln2_4_l{layer}', f'attn_qkv_4_l{layer+1}')
        else:
            dot.edge(f'ln2_3_l{layer}', 'sp_merge_2')
            dot.edge(f'ln2_4_l{layer}', 'sp_merge_2')
    
    # Connect to DP merge and output
    dot.edge('sp_merge_1', 'dp_merge')
    dot.edge('sp_merge_2', 'dp_merge')
    dot.edge('dp_merge', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_deployment_dag()
    
    # Save as DOT file
    dag.save('./outputs/2025-12-30-11-49-50/deployment_dag.dot')
    
    # Save as SVG image
    dag.render('./outputs/2025-12-30-11-49-50/deployment_dag', format='svg', cleanup=True)
    
    print("DAG created successfully!")
    print("Files saved:")
    print("- ./outputs/2025-12-30-11-49-50/deployment_dag.dot")
    print("- ./outputs/2025-12-30-11-49-50/deployment_dag.svg")