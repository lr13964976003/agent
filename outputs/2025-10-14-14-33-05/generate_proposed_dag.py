#!/usr/bin/env python3
"""
Generate complete DAG for the proposed two-level attention partitioning method.
This DAG shows deployment across 16 GPUs with head-level and dimension-level partitioning.
"""

import graphviz

def create_proposed_dag():
    """Create the complete DAG for the proposed two-level partitioning method."""
    
    # Create directed graph
    dot = graphviz.Digraph('Proposed_Two_Level_Partitioning', 
                          comment='Two-Level Attention Partitioning DAG',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'splines': 'ortho',
                              'nodesep': '0.8',
                              'ranksep': '1.2',
                              'bgcolor': 'white'
                          })
    
    # Set node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input node
    dot.node('input', 'Model Input', 
             shape='ellipse', 
             style='filled', 
             fillcolor='lightblue',
             xlabel='Input: [batch_size=1024, seq_len=10000, d_model=8192]')
    
    # Layer 0
    with dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0', style='rounded, filled', fillcolor='lightgray')
        
        # MHA Partitioning
        with layer0.subgraph(name='cluster_mha0') as mha0:
            mha0.attr(label='Multi-Head Attention', style='rounded, filled', fillcolor='lightyellow')
            
            # Input broadcast to all devices
            dot.node('broadcast0', 'Input Broadcast', 
                    shape='parallelogram', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Broadcast to all 16 GPUs')
            
            # 16 parallel attention computations
            for i in range(4):  # head groups
                for j in range(4):  # dimension slices
                    device_id = i * 4 + j
                    
                    # Q projection
                    dot.node(f'q_proj_{device_id}', f'Q Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    # K projection  
                    dot.node(f'k_proj_{device_id}', f'K Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    # V projection
                    dot.node(f'v_proj_{device_id}', f'V Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    # Attention computation
                    dot.node(f'attn_{device_id}', f'Attention\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightgoldenrod',
                            xlabel='Input: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
                    
                    # Residual connections
                    dot.node(f'residual_{device_id}', f'Residual Add\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightblue',
                            xlabel='Input1: [1024, 10000, 512]\nInput2: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
        
        # Layer Norm 1
        dot.node('layernorm1_0', 'Layer Norm 1\nAll GPUs', 
                shape='rectangle', 
                style='filled', 
                fillcolor='lightpink',
                xlabel='Input: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
        
        # MLP
        with layer0.subgraph(name='cluster_mlp0') as mlp0:
            mlp0.attr(label='MLP', style='rounded, filled', fillcolor='lightcyan')
            
            # First linear (column parallel)
            dot.node('mlp_linear1_0', 'First Linear\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 2048]')
            
            # Activation
            dot.node('mlp_gelu_0', 'GELU Activation\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 2048]\nOutput: [1024, 10000, 2048]')
            
            # Second linear (row parallel)
            dot.node('mlp_linear2_0', 'Second Linear\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 2048]\nOutput: [1024, 10000, 8192]')
            
            # Residual
            dot.node('mlp_residual_0', 'MLP Residual\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightblue',
                    xlabel='Input1: [1024, 10000, 8192]\nInput2: [1024, 10000, 8192]\nOutput: [1024, 10000, 8192]')
        
        # Layer Norm 2
        dot.node('layernorm2_0', 'Layer Norm 2\nAll GPUs', 
                shape='rectangle', 
                style='filled', 
                fillcolor='lightpink',
                xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 8192]')
    
    # Hierarchical aggregation for MHA
    # Dimension concatenation within head groups
    for i in range(4):  # head groups
        dot.node(f'dim_concat_{i}', f'Dimension Concat\nGroup {i}', 
                shape='parallelogram', 
                style='filled', 
                fillcolor='lightyellow',
                xlabel='Input: 4×[1024, 10000, 512]\nOutput: [1024, 10000, 2048]')
    
    # Head concatenation across groups
    dot.node('head_concat', 'Head Concat\nAll Groups', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightyellow',
            xlabel='Input: 4×[1024, 10000, 2048]\nOutput: [1024, 10000, 8192]')
    
    # Layer 1
    with dot.subgraph(name='cluster_layer1') as layer1:
        layer1.attr(label='Layer 1', style='rounded, filled', fillcolor='lightgray')
        
        # MHA Partitioning (same as layer 0)
        with layer1.subgraph(name='cluster_mha1') as mha1:
            mha1.attr(label='Multi-Head Attention', style='rounded, filled', fillcolor='lightyellow')
            
            # Input broadcast
            dot.node('broadcast1', 'Input Broadcast\nLayer 1', 
                    shape='parallelogram', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Broadcast to all 16 GPUs')
            
            # 16 parallel attention computations
            for i in range(4):
                for j in range(4):
                    device_id = i * 4 + j
                    
                    dot.node(f'q_proj1_{device_id}', f'Q Projection L1\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    dot.node(f'k_proj1_{device_id}', f'K Projection L1\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    dot.node(f'v_proj1_{device_id}', f'V Projection L1\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 512]')
                    
                    dot.node(f'attn1_{device_id}', f'Attention L1\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightgoldenrod',
                            xlabel='Input: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
                    
                    dot.node(f'residual1_{device_id}', f'Residual Add L1\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightblue',
                            xlabel='Input1: [1024, 10000, 512]\nInput2: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
        
        # Layer Norm 1
        dot.node('layernorm1_1', 'Layer Norm 1 L1\nAll GPUs', 
                shape='rectangle', 
                style='filled', 
                fillcolor='lightpink',
                xlabel='Input: [1024, 10000, 512]\nOutput: [1024, 10000, 512]')
        
        # MLP
        with layer1.subgraph(name='cluster_mlp1') as mlp1:
            mlp1.attr(label='MLP L1', style='rounded, filled', fillcolor='lightcyan')
            
            dot.node('mlp_linear1_1', 'First Linear L1\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 2048]')
            
            dot.node('mlp_gelu_1', 'GELU Activation L1\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 2048]\nOutput: [1024, 10000, 2048]')
            
            dot.node('mlp_linear2_1', 'Second Linear L1\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightgreen',
                    xlabel='Input: [1024, 10000, 2048]\nOutput: [1024, 10000, 8192]')
            
            dot.node('mlp_residual_1', 'MLP Residual L1\nAll GPUs', 
                    shape='rectangle', 
                    style='filled', 
                    fillcolor='lightblue',
                    xlabel='Input1: [1024, 10000, 8192]\nInput2: [1024, 10000, 8192]\nOutput: [1024, 10000, 8192]')
        
        # Layer Norm 2
        dot.node('layernorm2_1', 'Layer Norm 2 L1\nAll GPUs', 
                shape='rectangle', 
                style='filled', 
                fillcolor='lightpink',
                xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 8192]')
    
    # Hierarchical aggregation for Layer 1 MHA
    for i in range(4):
        dot.node(f'dim_concat1_{i}', f'Dimension Concat L1\nGroup {i}', 
                shape='parallelogram', 
                style='filled', 
                fillcolor='lightyellow',
                xlabel='Input: 4×[1024, 10000, 512]\nOutput: [1024, 10000, 2048]')
    
    dot.node('head_concat1', 'Head Concat L1\nAll Groups', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightyellow',
            xlabel='Input: 4×[1024, 10000, 2048]\nOutput: [1024, 10000, 8192]')
    
    # Output
    dot.node('output', 'Model Output', 
             shape='ellipse', 
             style='filled', 
             fillcolor='lightblue',
             xlabel='Output: [batch_size=1024, seq_len=10000, d_model=8192]')
    
    # ==== Create edges ====
    
    # Input to broadcast
    dot.edge('input', 'broadcast0')
    dot.edge('broadcast0', 'broadcast1')
    
    # Layer 0 MHA connections
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            
            # Input to projections
            dot.edge('broadcast0', f'q_proj_{device_id}')
            dot.edge('broadcast0', f'k_proj_{device_id}')
            dot.edge('broadcast0', f'v_proj_{device_id}')
            
            # Projections to attention
            dot.edge(f'q_proj_{device_id}', f'attn_{device_id}')
            dot.edge(f'k_proj_{device_id}', f'attn_{device_id}')
            dot.edge(f'v_proj_{device_id}', f'attn_{device_id}')
            
            # Attention to residual
            dot.edge(f'attn_{device_id}', f'residual_{device_id}')
            
            # Residual to dimension concatenation
            dot.edge(f'residual_{device_id}', f'dim_concat_{i}')
    
    # Dimension concatenation to head concatenation
    for i in range(4):
        dot.edge(f'dim_concat_{i}', 'head_concat')
    
    # Head concatenation to layer norm
    dot.edge('head_concat', 'layernorm1_0')
    
    # Layer norm to MLP
    dot.edge('layernorm1_0', 'mlp_linear1_0')
    dot.edge('mlp_linear1_0', 'mlp_gelu_0')
    dot.edge('mlp_gelu_0', 'mlp_linear2_0')
    dot.edge('mlp_linear2_0', 'mlp_residual_0')
    dot.edge('mlp_residual_0', 'layernorm2_0')
    
    # Layer 0 to Layer 1
    dot.edge('layernorm2_0', 'broadcast1')
    
    # Layer 1 MHA connections
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            
            dot.edge('broadcast1', f'q_proj1_{device_id}')
            dot.edge('broadcast1', f'k_proj1_{device_id}')
            dot.edge('broadcast1', f'v_proj1_{device_id}')
            
            dot.edge(f'q_proj1_{device_id}', f'attn1_{device_id}')
            dot.edge(f'k_proj1_{device_id}', f'attn1_{device_id}')
            dot.edge(f'v_proj1_{device_id}', f'attn1_{device_id}')
            
            dot.edge(f'attn1_{device_id}', f'residual1_{device_id}')
            dot.edge(f'residual1_{device_id}', f'dim_concat1_{i}')
    
    # Layer 1 aggregation
    for i in range(4):
        dot.edge(f'dim_concat1_{i}', 'head_concat1')
    
    dot.edge('head_concat1', 'layernorm1_1')
    dot.edge('layernorm1_1', 'mlp_linear1_1')
    dot.edge('mlp_linear1_1', 'mlp_gelu_1')
    dot.edge('mlp_gelu_1', 'mlp_linear2_1')
    dot.edge('mlp_linear2_1', 'mlp_residual_1')
    dot.edge('mlp_residual_1', 'layernorm2_1')
    dot.edge('layernorm2_1', 'output')
    
    return dot

def create_detailed_mha_dag():
    """Create a detailed DAG showing the two-level partitioning for MHA specifically."""
    
    dot = graphviz.Digraph('Detailed_MHA_Partitioning', 
                          comment='Detailed MHA Two-Level Partitioning',
                          format='svg',
                          graph_attr={
                              'rankdir': 'LR',
                              'splines': 'ortho',
                              'nodesep': '0.5',
                              'ranksep': '1.0'
                          })
    
    # Input
    dot.node('input_detailed', 'Input X', 
             shape='ellipse', 
             style='filled', 
             fillcolor='lightblue',
             xlabel='[1024, 10000, 8192]')
    
    # 16 devices for MHA
    devices = []
    for i in range(4):  # head groups
        for j in range(4):  # dimension slices
            device_id = i * 4 + j
            
            # Create device cluster
            with dot.subgraph(name=f'cluster_device_{device_id}') as device:
                device.attr(label=f'Device {device_id}\nHead Group {i}, Dim Slice {j}', 
                          style='rounded, filled', 
                          fillcolor='lightcyan')
                
                # Q projection
                dot.node(f'q_detailed_{device_id}', f'Q Proj\nW_Q[{i*512}:{i*512+512}, {j*128}:{j*128+128}]', 
                        shape='rectangle', 
                        style='filled', 
                        fillcolor='lightcoral',
                        xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 128]')
                
                # K projection
                dot.node(f'k_detailed_{device_id}', f'K Proj\nW_K[{i*512}:{i*512+512}, {j*128}:{j*128+128}]', 
                        shape='rectangle', 
                        style='filled', 
                        fillcolor='lightcoral',
                        xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 128]')
                
                # V projection
                dot.node(f'v_detailed_{device_id}', f'V Proj\nW_V[{i*512}:{i*512+512}, {j*128}:{j*128+128}]', 
                        shape='rectangle', 
                        style='filled', 
                        fillcolor='lightcoral',
                        xlabel='Input: [1024, 10000, 8192]\nOutput: [1024, 10000, 128]')
                
                # Attention computation
                dot.node(f'attn_detailed_{device_id}', f'Attention\nHeads {i*4}-{(i+1)*4-1}\nDims {j*128}-{(j+1)*128-1}', 
                        shape='rectangle', 
                        style='filled', 
                        fillcolor='lightgoldenrod',
                        xlabel='Input: [1024, 10000, 128]\nOutput: [1024, 10000, 128]')
                
                # Connect within device
                dot.edge(f'q_detailed_{device_id}', f'attn_detailed_{device_id}')
                dot.edge(f'k_detailed_{device_id}', f'attn_detailed_{device_id}')
                dot.edge(f'v_detailed_{device_id}', f'attn_detailed_{device_id}')
    
    # Hierarchical aggregation
    for i in range(4):
        dot.node(f'dim_concat_detailed_{i}', f'Dimension Concat\nGroup {i}', 
                shape='parallelogram', 
                style='filled', 
                fillcolor='lightyellow',
                xlabel='Input: 4×[1024, 10000, 128]\nOutput: [1024, 10000, 512]')
    
    dot.node('head_concat_detailed', 'Head Concat\nAll Groups', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightyellow',
            xlabel='Input: 4×[1024, 10000, 512]\nOutput: [1024, 10000, 2048]')
    
    dot.node('final_output', 'Final Output', 
             shape='ellipse', 
             style='filled', 
             fillcolor='lightblue',
             xlabel='[1024, 10000, 8192]')
    
    # Connect input to all devices
    for i in range(16):
        dot.edge('input_detailed', f'q_detailed_{i}')
        dot.edge('input_detailed', f'k_detailed_{i}')
        dot.edge('input_detailed', f'v_detailed_{i}')
    
    # Connect devices to aggregation
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            dot.edge(f'attn_detailed_{device_id}', f'dim_concat_detailed_{i}')
    
    for i in range(4):
        dot.edge(f'dim_concat_detailed_{i}', 'head_concat_detailed')
    
    dot.edge('head_concat_detailed', 'final_output')
    
    return dot

if __name__ == "__main__":
    # Generate main DAG
    main_dag = create_proposed_dag()
    main_dag.render('./outputs/2025-10-14-14-33-05/proposed_two_level_partitioning', format='svg', cleanup=True)
    main_dag.render('./outputs/2025-10-14-14-33-05/proposed_two_level_partitioning', format='dot', cleanup=True)
    
    # Generate detailed MHA DAG
    detailed_dag = create_detailed_mha_dag()
    detailed_dag.render('./outputs/2025-10-14-14-33-05/detailed_mha_partitioning', format='svg', cleanup=True)
    detailed_dag.render('./outputs/2025-10-14-14-33-05/detailed_mha_partitioning', format='dot', cleanup=True)
    
    print("DAGs generated successfully!")
    print("- Main DAG: proposed_two_level_partitioning.svg")
    print("- Detailed MHA DAG: detailed_mha_partitioning.svg")