#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_ring_attention_sp_dag():
    """
    Create DAG for Ring Attention with Sequence Parallelism
    16 GPUs, SP=16, sequence length split to 6250 tokens per device
    Model: 4-layer dense transformer
    """
    dot = Digraph(comment='Dense Transformer - Ring Attention + Sequence Parallel (SP=16)')
    dot.attr(rankdir='TB', size='25,25', concentrate='true')
    
    # Parameters
    batch_size = 128
    global_seq_len = 100000
    local_seq_len = 6250
    hidden_size = 4096
    heads = 32
    head_dim = 128
    mlp_hidden = 32768
    
    # Create subgraph for each device
    for device_id in range(16):
        start_token = device_id * local_seq_len
        end_token = start_token + local_seq_len - 1
        
        with dot.subgraph(name=f'cluster_device_{device_id}') as c:
            c.attr(label=f'Device {device_id}\nSequence: [{start_token}, {end_token}]\nLocal tokens: {local_seq_len}', 
                   style='rounded', color=f'color{device_id % 8}', fillcolor=f'lightcolor{device_id % 8}')
            
            # Input split
            c.node(f'input_split_{device_id}', 
                   f'Sequence Split\nInput: [batch=128, seq={global_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Embedding (after split)
            c.node(f'embed_{device_id}', 
                   f'Embedding\nInput: [batch=128, seq={local_seq_len}]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Layer 0
            c.node(f'ln0_pre_{device_id}', 
                   f'LayerNorm L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Q,K,V projections for Layer 0
            c.node(f'q_proj_0_{device_id}', 
                   f'Q Projection L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'k_proj_0_{device_id}', 
                   f'K Projection L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'v_proj_0_{device_id}', 
                   f'V Projection L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Ring Attention mechanism
            c.node(f'ring_attention_0_{device_id}', 
                   f'Ring Attention L0\nLocal Q: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nRing K,V: [batch=128, seq={global_seq_len}, heads=32, d_k=128]\nOutput: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nGPU: {device_id}\nStages: 16', 
                   shape='doubleoctagon', style='filled', fillcolor='gold', peripheries='2')
            
            # Ring communication nodes
            for stage in range(16):
                src_device = (device_id - stage) % 16
                next_device = (device_id + 1) % 16
                
                c.node(f'kv_send_0_{device_id}_{stage}', 
                       f'Send K,V\nStage {stage}\nData: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nFrom: {device_id}\nTo: {next_device}', 
                       shape='ellipse', style='filled,dashed', fillcolor='lightsteelblue')
                
                c.node(f'kv_recv_0_{device_id}_{stage}', 
                       f'Recv K,V\nStage {stage}\nData: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nFrom: {src_device}\nTo: {device_id}', 
                       shape='ellipse', style='filled,dashed', fillcolor='lightsteelblue')
            
            c.node(f'o_proj_0_{device_id}', 
                   f'O Projection L0\nInput: [batch=128, seq={local_seq_len}, heads=32, d_k=128]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            c.node(f'add_0_{device_id}', 
                   f'Residual Add L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            # MLP Layer 0
            c.node(f'mlp0_gate_{device_id}', 
                   f'MLP Gate L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, mlp_hidden=32768]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'mlp0_up_{device_id}', 
                   f'MLP Up L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, mlp_hidden=32768]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'mlp0_down_{device_id}', 
                   f'MLP Down L0\nInput: [batch=128, seq={local_seq_len}, mlp_hidden=32768]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            
            c.node(f'add_1_{device_id}', 
                   f'Residual Add L0\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={local_seq_len}, d_model=4096]\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Layer 1
            c.node(f'ln1_pre_{device_id}', 
                   f'LayerNorm L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            c.node(f'q_proj_1_{device_id}', f'Q Proj L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'k_proj_1_{device_id}', f'K Proj L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'v_proj_1_{device_id}', f'V Proj L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            c.node(f'ring_attention_1_{device_id}', 
                   f'Ring Attention L1\nGPU: {device_id}\nStages: 16', 
                   shape='doubleoctagon', style='filled', fillcolor='gold', peripheries='2')
            
            for stage in range(16):
                src_device = (device_id - stage) % 16
                next_device = (device_id + 1) % 16
                
                c.node(f'kv_send_1_{device_id}_{stage}', 
                       f'Send K,V L1\nStage {stage}\nFrom: {device_id}\nTo: {next_device}', 
                       shape='ellipse', style='filled,dashed', fillcolor='lightsteelblue')
                c.node(f'kv_recv_1_{device_id}_{stage}', 
                       f'Recv K,V L1\nStage {stage}\nFrom: {src_device}\nTo: {device_id}', 
                       shape='ellipse', style='filled,dashed', fillcolor='lightsteelblue')
            
            c.node(f'o_proj_1_{device_id}', f'O Proj L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'add_2_{device_id}', f'Residual Add L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            c.node(f'mlp1_gate_{device_id}', f'MLP Gate L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'mlp1_up_{device_id}', f'MLP Up L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'mlp1_down_{device_id}', f'MLP Down L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            
            c.node(f'add_3_{device_id}', f'Residual Add L1\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Layer 2 (simplified - same pattern as Layer 0)
            c.node(f'ln2_pre_{device_id}', f'LayerNorm L2\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            c.node(f'ring_attention_2_{device_id}', f'Ring Attention L2\nGPU: {device_id}\nStages: 16', 
                   shape='doubleoctagon', style='filled', fillcolor='gold', peripheries='2')
            c.node(f'o_proj_2_{device_id}', f'O Proj L2\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'add_4_{device_id}', f'Residual Add L2\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            c.node(f'mlp2_down_{device_id}', f'MLP Down L2\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'add_5_{device_id}', f'Residual Add L2\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Layer 3 (simplified - same pattern as Layer 0)
            c.node(f'ln3_pre_{device_id}', f'LayerNorm L3\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            c.node(f'ring_attention_3_{device_id}', f'Ring Attention L3\nGPU: {device_id}\nStages: 16', 
                   shape='doubleoctagon', style='filled', fillcolor='gold', peripheries='2')
            c.node(f'o_proj_3_{device_id}', f'O Proj L3\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
            c.node(f'add_6_{device_id}', f'Residual Add L3\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            c.node(f'mlp3_down_{device_id}', f'MLP Down L3\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightpink')
            c.node(f'add_7_{device_id}', f'Residual Add L3\nGPU: {device_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Output gather
            c.node(f'output_gather_{device_id}', 
                   f'Gather Output\nInput: [batch=128, seq={local_seq_len}, d_model=4096]\nOutput: [batch=128, seq={global_seq_len}, d_model=4096]\nGPU: {device_id}→All', 
                   shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Global input
    dot.node('global_input', 'Global Input\n[batch=128, seq=100000, d_model=4096]\nGPU: Host', 
            shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Global output
    dot.node('global_output', 'Global Output\n[batch=128, seq=100000, d_model=4096]\nGPU: Host', 
            shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Create connections for each device
    for device_id in range(16):
        # Input split
        dot.edge('global_input', f'input_split_{device_id}')
        dot.edge(f'input_split_{device_id}', f'embed_{device_id}')
        
        # Layer 0 connections
        dot.edge(f'embed_{device_id}', f'ln0_pre_{device_id}')
        dot.edge(f'ln0_pre_{device_id}', f'q_proj_0_{device_id}')
        dot.edge(f'ln0_pre_{device_id}', f'k_proj_0_{device_id}')
        dot.edge(f'ln0_pre_{device_id}', f'v_proj_0_{device_id}')
        
        # Ring attention pattern
        dot.edge(f'q_proj_0_{device_id}', f'ring_attention_0_{device_id}')
        dot.edge(f'k_proj_0_{device_id}', f'ring_attention_0_{device_id}')
        dot.edge(f'v_proj_0_{device_id}', f'ring_attention_0_{device_id}')
        
        # Ring communication edges
        for stage in range(16):
            src_device = (device_id - stage) % 16
            next_device = (device_id + 1) % 16
            
            # Receive KV from previous device
            dot.edge(f'k_proj_0_{src_device}', f'kv_recv_0_{device_id}_{stage}', style='dashed')
            dot.edge(f'v_proj_0_{src_device}', f'kv_recv_0_{device_id}_{stage}', style='dashed')
            dot.edge(f'kv_recv_0_{device_id}_{stage}', f'ring_attention_0_{device_id}')
            
            # Send KV to next device
            dot.edge(f'k_proj_0_{device_id}', f'kv_send_0_{device_id}_{stage}')
            dot.edge(f'v_proj_0_{device_id}', f'kv_send_0_{device_id}_{stage}')
            dot.edge(f'kv_send_0_{device_id}_{stage}', f'kv_recv_0_{next_device}_{stage}', style='dashed')
        
        dot.edge(f'ring_attention_0_{device_id}', f'o_proj_0_{device_id}')
        dot.edge(f'o_proj_0_{device_id}', f'add_0_{device_id}')
        dot.edge(f'embed_{device_id}', f'add_0_{device_id}')  # Residual
        
        # MLP Layer 0
        dot.edge(f'add_0_{device_id}', f'mlp0_gate_{device_id}')
        dot.edge(f'add_0_{device_id}', f'mlp0_up_{device_id}')
        dot.edge(f'mlp0_gate_{device_id}', f'mlp0_down_{device_id}')
        dot.edge(f'mlp0_up_{device_id}', f'mlp0_down_{device_id}')
        dot.edge(f'mlp0_down_{device_id}', f'add_1_{device_id}')
        dot.edge(f'add_0_{device_id}', f'add_1_{device_id}')  # Residual
        
        # Layer 1 connections
        dot.edge(f'add_1_{device_id}', f'ln1_pre_{device_id}')
        dot.edge(f'ln1_pre_{device_id}', f'q_proj_1_{device_id}')
        dot.edge(f'ln1_pre_{device_id}', f'k_proj_1_{device_id}')
        dot.edge(f'ln1_pre_{device_id}', f'v_proj_1_{device_id}')
        
        dot.edge(f'q_proj_1_{device_id}', f'ring_attention_1_{device_id}')
        dot.edge(f'k_proj_1_{device_id}', f'ring_attention_1_{device_id}')
        dot.edge(f'v_proj_1_{device_id}', f'ring_attention_1_{device_id}')
        
        # Ring communication for layer 1
        for stage in range(16):
            src_device = (device_id - stage) % 16
            next_device = (device_id + 1) % 16
            
            dot.edge(f'k_proj_1_{src_device}', f'kv_recv_1_{device_id}_{stage}', style='dashed')
            dot.edge(f'v_proj_1_{src_device}', f'kv_recv_1_{device_id}_{stage}', style='dashed')
            dot.edge(f'kv_recv_1_{device_id}_{stage}', f'ring_attention_1_{device_id}')
            dot.edge(f'k_proj_1_{device_id}', f'kv_send_1_{device_id}_{stage}')
            dot.edge(f'v_proj_1_{device_id}', f'kv_send_1_{device_id}_{stage}')
            dot.edge(f'kv_send_1_{device_id}_{stage}', f'kv_recv_1_{next_device}_{stage}', style='dashed')
        
        dot.edge(f'ring_attention_1_{device_id}', f'o_proj_1_{device_id}')
        dot.edge(f'o_proj_1_{device_id}', f'add_2_{device_id}')
        dot.edge(f'add_1_{device_id}', f'add_2_{device_id}')  # Residual
        
        dot.edge(f'add_2_{device_id}', f'mlp1_gate_{device_id}')
        dot.edge(f'add_2_{device_id}', f'mlp1_up_{device_id}')
        dot.edge(f'mlp1_gate_{device_id}', f'mlp1_down_{device_id}')
        dot.edge(f'mlp1_up_{device_id}', f'mlp1_down_{device_id}')
        dot.edge(f'mlp1_down_{device_id}', f'add_3_{device_id}')
        dot.edge(f'add_2_{device_id}', f'add_3_{device_id}')  # Residual
        
        # Layer 2 (simplified connections)
        dot.edge(f'add_3_{device_id}', f'ln2_pre_{device_id}')
        dot.edge(f'ln2_pre_{device_id}', f'ring_attention_2_{device_id}')
        dot.edge(f'ring_attention_2_{device_id}', f'o_proj_2_{device_id}')
        dot.edge(f'o_proj_2_{device_id}', f'add_4_{device_id}')
        dot.edge(f'add_3_{device_id}', f'add_4_{device_id}')  # Residual
        dot.edge(f'add_4_{device_id}', f'mlp2_down_{device_id}')
        dot.edge(f'mlp2_down_{device_id}', f'add_5_{device_id}')
        dot.edge(f'add_4_{device_id}', f'add_5_{device_id}')  # Residual
        
        # Layer 3 (simplified connections)
        dot.edge(f'add_5_{device_id}', f'ln3_pre_{device_id}')
        dot.edge(f'ln3_pre_{device_id}', f'ring_attention_3_{device_id}')
        dot.edge(f'ring_attention_3_{device_id}', f'o_proj_3_{device_id}')
        dot.edge(f'o_proj_3_{device_id}', f'add_6_{device_id}')
        dot.edge(f'add_5_{device_id}', f'add_6_{device_id}')  # Residual
        dot.edge(f'add_6_{device_id}', f'mlp3_down_{device_id}')
        dot.edge(f'mlp3_down_{device_id}', f'add_7_{device_id}')
        dot.edge(f'add_6_{device_id}', f'add_7_{device_id}')  # Residual
        
        # Output gather
        dot.edge(f'add_7_{device_id}', f'output_gather_{device_id}')
        for device_id2 in range(16):
            dot.edge(f'output_gather_{device_id2}', 'global_output')
    
    return dot

if __name__ == "__main__":
    dag = create_ring_attention_sp_dag()
    
    # Save files
    dag.render('../outputs/2025-11-21-16-26-54/ring_attention_sp_dag', format='svg', cleanup=True)
    
    # Save DOT file
    with open('../outputs/2025-11-21-16-26-54/ring_attention_sp_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Ring Attention SP DAG created successfully!")
    print(f"SVG: ../outputs/2025-11-21-16-26-54/ring_attention_sp_dag.svg")
    print(f"DOT: ../outputs/2025-11-21-16-26-54/ring_attention_sp_dag.dot")