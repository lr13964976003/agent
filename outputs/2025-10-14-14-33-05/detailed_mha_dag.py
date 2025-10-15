#!/usr/bin/env python3
"""
Generate detailed DAG specifically for MHA two-level partitioning
"""

import graphviz

def create_detailed_mha_dag():
    """Create a detailed DAG showing the two-level partitioning for MHA specifically."""
    
    dot = graphviz.Digraph('Detailed_MHA_Partitioning', 
                          comment='Detailed MHA Two-Level Partitioning',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'splines': 'ortho',
                              'nodesep': '0.5',
                              'ranksep': '1.0',
                              'bgcolor': 'white'
                          })
    
    # Set node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input
    dot.node('input_detailed', 'Input X', 
             shape='ellipse', 
             style='filled', 
             fillcolor='lightblue',
             xlabel='Input: [batch_size=1024, seq_len=10000, d_model=8192]')
    
    # Broadcast node
    dot.node('broadcast_detailed', 'Input Broadcast', 
             shape='parallelogram', 
             style='filled', 
             fillcolor='lightgreen',
             xlabel='Broadcast to all 16 GPUs')
    
    # 16 devices for MHA - organized by head groups and dimension slices
    for i in range(4):  # head groups
        with dot.subgraph(name=f'cluster_group_{i}') as group:
            group.attr(label=f'Head Group {i} (Heads {i*4}-{(i+1)*4-1})', 
                      style='rounded, filled', 
                      fillcolor='lightgray')
            
            for j in range(4):  # dimension slices
                device_id = i * 4 + j
                
                with group.subgraph(name=f'cluster_device_{device_id}') as device:
                    device.attr(label=f'Device {device_id}\nDim Slice {j} (Dims {j*128}-{(j+1)*128-1})', 
                              style='rounded, filled', 
                              fillcolor='lightcyan')
                    
                    # Q projection with specific tensor dimensions
                    dot.node(f'q_detailed_{device_id}', 
                            f'Q Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nWeight: [512, 8192]\nOutput: [1024, 10000, 128]')
                    
                    # K projection
                    dot.node(f'k_detailed_{device_id}', 
                            f'K Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nWeight: [512, 8192]\nOutput: [1024, 10000, 128]')
                    
                    # V projection
                    dot.node(f'v_detailed_{device_id}', 
                            f'V Projection\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightcoral',
                            xlabel='Input: [1024, 10000, 8192]\nWeight: [512, 8192]\nOutput: [1024, 10000, 128]')
                    
                    # Attention computation
                    dot.node(f'attn_detailed_{device_id}', 
                            f'Scaled Dot-Product Attention\nGPU {device_id}', 
                            shape='rectangle', 
                            style='filled', 
                            fillcolor='lightgoldenrod',
                            xlabel='Q: [1024, 10000, 128]\nK: [1024, 10000, 128]\nV: [1024, 10000, 128]\nOutput: [1024, 10000, 128]')
    
    # Hierarchical aggregation
    for i in range(4):
        dot.node(f'dim_concat_detailed_{i}', 
                f'Dimension Concatenation\nGroup {i}', 
                shape='parallelogram', 
                style='filled', 
                fillcolor='lightyellow',
                xlabel='Input: 4×[1024, 10000, 128]\nOutput: [1024, 10000, 512]')
    
    dot.node('head_concat_detailed', 
            'Head Concatenation\nAll Groups', 
            shape='parallelogram', 
            style='filled', 
            fillcolor='lightyellow',
            xlabel='Input: 4×[1024, 10000, 512]\nOutput: [1024, 10000, 2048]')
    
    dot.node('final_projection', 
            'Output Projection\nAll GPUs', 
            shape='rectangle', 
            style='filled', 
            fillcolor='lightgreen',
            xlabel='Input: [1024, 10000, 2048]\nWeight: [8192, 2048]\nOutput: [1024, 10000, 8192]')
    
    dot.node('final_output', 
            'MHA Output', 
            shape='ellipse', 
            style='filled', 
            fillcolor='lightblue',
            xlabel='Output: [batch_size=1024, seq_len=10000, d_model=8192]')
    
    # Connect input to broadcast
    dot.edge('input_detailed', 'broadcast_detailed')
    
    # Connect broadcast to all devices
    for i in range(16):
        dot.edge('broadcast_detailed', f'q_detailed_{i}')
        dot.edge('broadcast_detailed', f'k_detailed_{i}')
        dot.edge('broadcast_detailed', f'v_detailed_{i}')
    
    # Connect within devices
    for i in range(16):
        dot.edge(f'q_detailed_{i}', f'attn_detailed_{i}')
        dot.edge(f'k_detailed_{i}', f'attn_detailed_{i}')
        dot.edge(f'v_detailed_{i}', f'attn_detailed_{i}')
    
    # Connect devices to aggregation
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            dot.edge(f'attn_detailed_{device_id}', f'dim_concat_detailed_{i}')
    
    for i in range(4):
        dot.edge(f'dim_concat_detailed_{i}', 'head_concat_detailed')
    
    dot.edge('head_concat_detailed', 'final_projection')
    dot.edge('final_projection', 'final_output')
    
    return dot

if __name__ == "__main__":
    dag = create_detailed_mha_dag()
    dag.render('./outputs/2025-10-14-14-33-05/detailed_mha_partitioning', format='svg', cleanup=True)
    dag.render('./outputs/2025-10-14-14-33-05/detailed_mha_partitioning', format='dot', cleanup=True)
    print("Detailed MHA DAG generated successfully!")