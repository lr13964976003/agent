#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """
    Create DAG for baseline: TP=8, PP=2 on 16 GPUs
    Model: 4-layer dense transformer
    """
    dot = Digraph(comment='Dense Transformer - Baseline (TP=8, PP=2)')
    dot.attr(rankdir='TB', size='20,20')
    
    # Global parameters
    batch_size = 128
    seq_len = 100000
    hidden_size = 4096
    heads = 32
    head_dim = 128
    mlp_hidden = 32768
    
    # Pipeline Stage 0 (Devices 0-7)
    with dot.subgraph(name='cluster_pipeline_stage0') as c0:
        c0.attr(label='Pipeline Stage 0 (Devices 0-7)', style='rounded', color='blue', bgcolor='lightblue')
        
        # Input for stage 0
        c0.node('input0', 'Input\nInput: [batch=128, seq=100000, d_model=4096]\nGPU: -', 
               shape='ellipse', style='filled', fillcolor='lightyellow')
        
        # Layer 0
        c0.node('embed0', 'Embedding\nInput: [batch=128, seq=100000]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        c0.node('ln0_pre', 'LayerNorm\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # MHA Layer 0 - Split across 8 GPUs
        for i in range(8):
            device_id = i
            c0.node(f'q_proj_0_{i}', f'Q Projection\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, heads=4, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c0.node(f'k_proj_0_{i}', f'K Projection\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, heads=4, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c0.node(f'v_proj_0_{i}', f'V Projection\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, heads=4, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            c0.node(f'attention_0_{i}', f'MHA (4 heads)\nInput: Q[batch=128, seq=100000, heads=4, d_k=128]\nK,V[batch=128, seq=100000, heads=4, d_k=128]\nOutput: [batch=128, seq=100000, heads=4, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled,rounded', fillcolor='gold')
            
            c0.node(f'o_proj_0_{i}', f'O Projection\nInput: [batch=128, seq=100000, heads=4, d_k=128]\nOutput: [batch=128, seq=100000, d_model=512]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        c0.node('all_reduce_0', 'All-Reduce\nInput: 8×[batch=128, seq=100000, d_model=512]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='parallelogram', style='filled', fillcolor='orange')
        
        c0.node('add_0', 'Residual Add\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgray')
        
        # MLP Layer 0
        c0.node('mlp0_gate', 'MLP Gate\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, mlp_hidden=32768]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        c0.node('mlp0_up', 'MLP Up\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, mlp_hidden=32768]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        c0.node('mlp0_down', 'MLP Down\nInput: [batch=128, seq=100000, mlp_hidden=32768]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        
        c0.node('add_1', 'Residual Add\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgray')
        
        # Layer 1 (similar structure)
        c0.node('ln1_pre', 'LayerNorm\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        for i in range(8):
            device_id = i
            c0.node(f'q_proj_1_{i}', f'Q Projection\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c0.node(f'k_proj_1_{i}', f'K Projection\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c0.node(f'v_proj_1_{i}', f'V Projection\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c0.node(f'attention_1_{i}', f'MHA\nGPU: {device_id}', 
                   shape='rectangle', style='filled,rounded', fillcolor='gold')
            c0.node(f'o_proj_1_{i}', f'O Projection\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        c0.node('all_reduce_1', 'All-Reduce\nGPU: all GPUs 0-7', 
               shape='parallelogram', style='filled', fillcolor='orange')
        c0.node('add_2', 'Residual Add\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgray')
        
        c0.node('mlp1_gate', 'MLP Gate\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        c0.node('mlp1_up', 'MLP Up\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        c0.node('mlp1_down', 'MLP Down\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightpink')
        
        c0.node('add_3', 'Residual Add\nGPU: all GPUs 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgray')
        
        # Pipeline communication
        c0.node('send_stage0', 'Send to Stage 1\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: 7→8', 
               shape='ellipse', style='filled', fillcolor='lightsteelblue')
    
    # Pipeline Stage 1 (Devices 8-15)
    with dot.subgraph(name='cluster_pipeline_stage1') as c1:
        c1.attr(label='Pipeline Stage 1 (Devices 8-15)', style='rounded', color='red', bgcolor='lightcoral')
        
        c1.node('recv_stage1', 'Receive from Stage 0\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: 7→8', 
               shape='ellipse', style='filled', fillcolor='lightsteelblue')
        
        # Layer 2 and 3 (similar structure as layer 0/1)
        for layer in [2, 3]:
            c1.node(f'ln{layer}_pre', f'LayerNorm (Layer {layer})\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            
            for i in range(8):
                device_id = 8 + i
                c1.node(f'q_proj_{layer}_{i}', f'Q Proj L{layer}\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightblue')
                c1.node(f'k_proj_{layer}_{i}', f'K Proj L{layer}\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightblue')
                c1.node(f'v_proj_{layer}_{i}', f'V Proj L{layer}\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightblue')
                c1.node(f'attention_{layer}_{i}', f'MHA L{layer}\nGPU: {device_id}', 
                       shape='rectangle', style='filled,rounded', fillcolor='gold')
                c1.node(f'o_proj_{layer}_{i}', f'O Proj L{layer}\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightblue')
            
            c1.node(f'all_reduce_{layer}', f'All-Reduce L{layer}\nGPU: all GPUs 8-15', 
                   shape='parallelogram', style='filled', fillcolor='orange')
            c1.node(f'add_{2*layer}', f'Residual Add L{layer}\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            c1.node(f'mlp{layer}_gate', f'MLP Gate L{layer}\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c1.node(f'mlp{layer}_up', f'MLP Up L{layer}\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c1.node(f'mlp{layer}_down', f'MLP Down L{layer}\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            
            c1.node(f'add_{2*layer+1}', f'Residual Add L{layer}\nGPU: all GPUs 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Output
    dot.node('output', 'Output\nInput: [batch=128, seq=100000, d_model=4096]\nOutput: [batch=128, seq=100000, d_model=4096]\nGPU: all GPUs 8-15', 
            shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Connections for Stage 0
    dot.edge('input0', 'embed0')
    dot.edge('embed0', 'ln0_pre')
    
    # Connect projections to attention
    for i in range(8):
        dot.edge('ln0_pre', f'q_proj_0_{i}')
        dot.edge('ln0_pre', f'k_proj_0_{i}')
        dot.edge('ln0_pre', f'v_proj_0_{i}')
        dot.edge(f'q_proj_0_{i}', f'attention_0_{i}')
        dot.edge(f'k_proj_0_{i}', f'attention_0_{i}')
        dot.edge(f'v_proj_0_{i}', f'attention_0_{i}')
        dot.edge(f'attention_0_{i}', f'o_proj_0_{i}')
    
    # All-reduce connections
    for i in range(8):
        dot.edge(f'o_proj_0_{i}', 'all_reduce_0')
    
    dot.edge('all_reduce_0', 'add_0')
    dot.edge('embed0', 'add_0')  # Residual connection
    
    dot.edge('add_0', 'mlp0_gate')
    dot.edge('add_0', 'mlp0_up')
    dot.edge('mlp0_gate', 'mlp0_down')
    dot.edge('mlp0_up', 'mlp0_down')
    dot.edge('mlp0_down', 'add_1')
    dot.edge('add_0', 'add_1')  # Residual connection
    
    # Layer 1 connections
    dot.edge('add_1', 'ln1_pre')
    for i in range(8):
        dot.edge('ln1_pre', f'q_proj_1_{i}')
        dot.edge('ln1_pre', f'k_proj_1_{i}')
        dot.edge('ln1_pre', f'v_proj_1_{i}')
        dot.edge(f'q_proj_1_{i}', f'attention_1_{i}')
        dot.edge(f'k_proj_1_{i}', f'attention_1_{i}')
        dot.edge(f'v_proj_1_{i}', f'attention_1_{i}')
        dot.edge(f'attention_1_{i}', f'o_proj_1_{i}')
    
    for i in range(8):
        dot.edge(f'o_proj_1_{i}', 'all_reduce_1')
    
    dot.edge('all_reduce_1', 'add_2')
    dot.edge('add_1', 'add_2')  # Residual connection
    
    dot.edge('add_2', 'mlp1_gate')
    dot.edge('add_2', 'mlp1_up')
    dot.edge('mlp1_gate', 'mlp1_down')
    dot.edge('mlp1_up', 'mlp1_down')
    dot.edge('mlp1_down', 'add_3')
    dot.edge('add_2', 'add_3')  # Residual connection
    
    # Pipeline communication
    dot.edge('add_3', 'send_stage0')
    dot.edge('send_stage0', 'recv_stage1')
    
    # Connections for Stage 1 (similar pattern)
    dot.edge('recv_stage1', 'ln2_pre')
    for layer in [2, 3]:
        for i in range(8):
            dot.edge(f'ln{layer}_pre', f'q_proj_{layer}_{i}')
            dot.edge(f'ln{layer}_pre', f'k_proj_{layer}_{i}')
            dot.edge(f'ln{layer}_pre', f'v_proj_{layer}_{i}')
            dot.edge(f'q_proj_{layer}_{i}', f'attention_{layer}_{i}')
            dot.edge(f'k_proj_{layer}_{i}', f'attention_{layer}_{i}')
            dot.edge(f'v_proj_{layer}_{i}', f'attention_{layer}_{i}')
            dot.edge(f'attention_{layer}_{i}', f'o_proj_{layer}_{i}')
            dot.edge(f'o_proj_{layer}_{i}', f'all_reduce_{layer}')
        
        prev_add = 'recv_stage1' if layer == 2 else 'add_7'
        dot.edge(f'all_reduce_{layer}', f'add_{2*layer}')
        dot.edge(prev_add, f'add_{2*layer}')  # Residual connection
        
        dot.edge(f'add_{2*layer}', f'mlp{layer}_gate')
        dot.edge(f'add_{2*layer}', f'mlp{layer}_up')
        dot.edge(f'mlp{layer}_gate', f'mlp{layer}_down')
        dot.edge(f'mlp{layer}_up', f'mlp{layer}_down')
        dot.edge(f'mlp{layer}_down', f'add_{2*layer+1}')
        dot.edge(f'add_{2*layer}', f'add_{2*layer+1}')  # Residual connection
        
        if layer == 2:
            dot.edge(f'add_{2*layer+1}', f'ln3_pre')
        else:
            dot.edge('add_7', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    
    # Save files
    dag.render('../outputs/2025-11-21-16-26-54/baseline_tp_pp_dag', format='svg', cleanup=True)
    
    # Save DOT file
    with open('../outputs/2025-11-21-16-26-54/baseline_tp_pp_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Baseline DAG created successfully!")
    print(f"SVG: ../outputs/2025-11-21-16-26-54/baseline_tp_pp_dag.svg")
    print(f"DOT: ../outputs/2025-11-21-16-26-54/baseline_tp_pp_dag.dot")