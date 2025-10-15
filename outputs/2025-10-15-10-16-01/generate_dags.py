#!/usr/bin/env python3
"""
Generate complete DAGs for Helix two-level attention partitioning
"""

import os
from graphviz import Digraph

def create_mha_partition_dag(layer_id, output_dir):
    """Create detailed DAG for multi-head attention with two-level partitioning"""
    
    dot = Digraph(f'mha_layer_{layer_id}_partitioned')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 
             'Input\\nInput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Layer normalization
    dot.node('ln', 
             'LayerNorm\\nInput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='rectangle', fillcolor='lightyellow')
    
    # QKV linear transformations for each partition
    partitions = []
    for head_group in range(4):  # n=4 head groups
        for dim_segment in range(4):  # m=4 dimension segments
            device_id = head_group * 4 + dim_segment
            partition_id = f"hg{head_group}_ds{dim_segment}"
            
            # Q projection
            q_node = f'q_{partition_id}'
            dot.node(q_node,
                     f'Q Linear\\nHead Group {head_group}, Dim Seg {dim_segment}\\n' +
                     f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                     f'Output: [batch_size=1024, seq_len=10000, heads=4, d_k=128]\\n' +
                     f'GPU: {device_id}',
                     shape='rectangle', fillcolor='lightcoral')
            
            # K projection
            k_node = f'k_{partition_id}'
            dot.node(k_node,
                     f'K Linear\\nHead Group {head_group}, Dim Seg {dim_segment}\\n' +
                     f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                     f'Output: [batch_size=1024, seq_len=10000, heads=4, d_k=128]\\n' +
                     f'GPU: {device_id}',
                     shape='rectangle', fillcolor='lightcoral')
            
            # V projection
            v_node = f'v_{partition_id}'
            dot.node(v_node,
                     f'V Linear\\nHead Group {head_group}, Dim Seg {dim_segment}\\n' +
                     f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                     f'Output: [batch_size=1024, seq_len=10000, heads=4, d_v=128]\\n' +
                     f'GPU: {device_id}',
                     shape='rectangle', fillcolor='lightcoral')
            
            # Attention computation
            attn_node = f'attn_{partition_id}'
            dot.node(attn_node,
                     f'Scaled Dot-Product Attention\\nHead Group {head_group}, Dim Seg {dim_segment}\\n' +
                     f'Q: [batch_size=1024, seq_len=10000, heads=4, d_k=128]\\n' +
                     f'K: [batch_size=1024, seq_len=10000, heads=4, d_k=128]\\n' +
                     f'V: [batch_size=1024, seq_len=10000, heads=4, d_v=128]\\n' +
                     f'Output: [batch_size=1024, seq_len=10000, heads=4, d_v=128]\\n' +
                     f'GPU: {device_id}',
                     shape='rectangle', fillcolor='lightpink')
            
            partitions.append((q_node, k_node, v_node, attn_node, head_group, dim_segment, device_id))
    
    # Intra-group concatenation (concatenate dimension segments within each head group)
    group_concat_nodes = []
    for head_group in range(4):
        concat_node = f'concat_group_{head_group}'
        dot.node(concat_node,
                 f'Concatenate Dimension Segments\\nHead Group {head_group}\\n' +
                 f'Input: 4×[batch_size=1024, seq_len=10000, heads=4, d_v=128]\\n' +
                 f'Output: [batch_size=1024, seq_len=10000, heads=4, d_v=512]\\n' +
                 f'GPU: {head_group*4}-{head_group*4+3}',
                 shape='parallelogram', fillcolor='lightsteelblue')
        group_concat_nodes.append(concat_node)
    
    # Final concatenation (concatenate head groups)
    final_concat = 'final_concat'
    dot.node(final_concat,
             'Concatenate Head Groups\\n' +
             'Input: 4×[batch_size=1024, seq_len=10000, heads=4, d_v=512]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPU: all GPUs',
             shape='parallelogram', fillcolor='lightsteelblue')
    
    # Output projection
    output_proj = 'output_proj'
    dot.node(output_proj,
             'Output Linear Projection\\n' +
             'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPU: all GPUs (tensor parallel)',
             shape='rectangle', fillcolor='lightcoral')
    
    # Residual connection
    residual = 'residual'
    dot.node(residual,
             'Residual Add\\n' +
             'Input 1: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Input 2: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPU: all GPUs',
             shape='rectangle', fillcolor='lightgray')
    
    # Connect nodes
    dot.edge('input', 'ln')
    
    # Connect layer norm to all QKV projections
    for q_node, k_node, v_node, attn_node, head_group, dim_segment, device_id in partitions:
        dot.edge('ln', q_node)
        dot.edge('ln', k_node)
        dot.edge('ln', v_node)
        
        dot.edge(q_node, attn_node)
        dot.edge(k_node, attn_node)
        dot.edge(v_node, attn_node)
    
    # Connect attention outputs to group concatenations
    for head_group in range(4):
        for dim_segment in range(4):
            partition_id = f"hg{head_group}_ds{dim_segment}"
            attn_node = f'attn_{partition_id}'
            concat_node = f'concat_group_{head_group}'
            dot.edge(attn_node, concat_node)
    
    # Connect group concatenations to final concatenation
    for concat_node in group_concat_nodes:
        dot.edge(concat_node, final_concat)
    
    # Connect to output projection and residual
    dot.edge(final_concat, output_proj)
    dot.edge(output_proj, residual)
    dot.edge('input', residual)  # Skip connection
    
    # Save DAG
    dot.render(os.path.join(output_dir, f'mha_layer_{layer_id}_partitioned'), format='svg', cleanup=False)
    dot.save(os.path.join(output_dir, f'mha_layer_{layer_id}_partitioned.dot'))

def create_mlp_dag(layer_id, output_dir):
    """Create detailed DAG for MLP with tensor parallelism across 16 devices"""
    
    dot = Digraph(f'mlp_layer_{layer_id}_tensor_parallel')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 
             'MLP Input\\nInput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Layer normalization
    dot.node('ln', 
             'LayerNorm\\nInput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='rectangle', fillcolor='lightyellow')
    
    # FC1 (column parallel) - split across 16 devices
    fc1_nodes = []
    for device_id in range(16):
        fc1_node = f'fc1_device_{device_id}'
        dot.node(fc1_node,
                 f'FC1 Linear (Column Parallel)\\n' +
                 f'Device {device_id}\\n' +
                 f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                 f'Output: [batch_size=1024, seq_len=10000, hidden_dim=2048]\\n' +
                 f'GPU: {device_id}',
                 shape='rectangle', fillcolor='lightcoral')
        fc1_nodes.append(fc1_node)
    
    # Concatenate FC1 outputs
    fc1_concat = 'fc1_concat'
    dot.node(fc1_concat,
             'Concatenate FC1 Outputs\\n' +
             'Input: 16×[batch_size=1024, seq_len=10000, hidden_dim=2048]\\n' +
             'Output: [batch_size=1024, seq_len=10000, hidden_dim=32768]\\n' +
             'GPU: all GPUs',
             shape='parallelogram', fillcolor='lightsteelblue')
    
    # GELU activation
    gelu = 'gelu'
    dot.node(gelu,
             'GELU Activation\\n' +
             'Input: [batch_size=1024, seq_len=10000, hidden_dim=32768]\\n' +
             'Output: [batch_size=1024, seq_len=10000, hidden_dim=32768]\\n' +
             'GPU: all GPUs',
             shape='rectangle', fillcolor='lightyellow')
    
    # FC2 (row parallel) - split across 16 devices
    fc2_nodes = []
    for device_id in range(16):
        fc2_node = f'fc2_device_{device_id}'
        dot.node(fc2_node,
                 f'FC2 Linear (Row Parallel)\\n' +
                 f'Device {device_id}\\n' +
                 f'Input: [batch_size=1024, seq_len=10000, hidden_dim=2048]\\n' +
                 f'Output: [batch_size=1024, seq_len=10000, embed_dim=2048]\\n' +
                 f'GPU: {device_id}',
                 shape='rectangle', fillcolor='lightcoral')
        fc2_nodes.append(fc2_node)
    
    # All-reduce sum for FC2 outputs
    fc2_allreduce = 'fc2_allreduce'
    dot.node(fc2_allreduce,
             'All-Reduce Sum\\n' +
             'Input: 16×[batch_size=1024, seq_len=10000, embed_dim=2048]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPU: all GPUs',
             shape='parallelogram', fillcolor='lightsteelblue')
    
    # Residual connection
    residual = 'residual'
    dot.node(residual,
             'Residual Add\\n' +
             'Input 1: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Input 2: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPU: all GPUs',
             shape='rectangle', fillcolor='lightgray')
    
    # Connect nodes
    dot.edge('input', 'ln')
    
    # Connect layer norm to all FC1 nodes
    for fc1_node in fc1_nodes:
        dot.edge('ln', fc1_node)
    
    # Connect FC1 outputs to concatenation
    for fc1_node in fc1_nodes:
        dot.edge(fc1_node, fc1_concat)
    
    dot.edge(fc1_concat, gelu)
    
    # Split GELU output to FC2 nodes
    for fc2_node in fc2_nodes:
        dot.edge(gelu, fc2_node)
    
    # Connect FC2 outputs to all-reduce
    for fc2_node in fc2_nodes:
        dot.edge(fc2_node, fc2_allreduce)
    
    # Connect to residual
    dot.edge(fc2_allreduce, residual)
    dot.edge('input', residual)  # Skip connection
    
    # Save DAG
    dot.render(os.path.join(output_dir, f'mlp_layer_{layer_id}_tensor_parallel'), format='svg', cleanup=False)
    dot.save(os.path.join(output_dir, f'mlp_layer_{layer_id}_tensor_parallel.dot'))

def create_complete_model_dag(output_dir):
    """Create complete model DAG showing both layers"""
    
    dot = Digraph('complete_helix_model')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Model input
    dot.node('model_input',
             'Model Input\\nInput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Layer 0 MHA
    dot.node('layer0_mha',
             'Layer 0 - Multi-Head Attention\\n' +
             'Two-Level Partitioning (4×4=16 partitions)\\n' +
             'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPUs: 0-15 (partitioned)',
             shape='rectangle', fillcolor='lightcoral')
    
    # Layer 0 MLP
    dot.node('layer0_mlp',
             'Layer 0 - Feed Forward Network\\n' +
             'Tensor Parallel (16-way)\\n' +
             'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPUs: 0-15 (tensor parallel)',
             shape='rectangle', fillcolor='lightcoral')
    
    # Layer 1 MHA
    dot.node('layer1_mha',
             'Layer 1 - Multi-Head Attention\\n' +
             'Two-Level Partitioning (4×4=16 partitions)\\n' +
             'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPUs: 0-15 (partitioned)',
             shape='rectangle', fillcolor='lightcoral')
    
    # Layer 1 MLP
    dot.node('layer1_mlp',
             'Layer 1 - Feed Forward Network\\n' +
             'Tensor Parallel (16-way)\\n' +
             'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
             'GPUs: 0-15 (tensor parallel)',
             shape='rectangle', fillcolor='lightcoral')
    
    # Model output
    dot.node('model_output',
             'Model Output\\nOutput: [batch_size=1024, seq_len=10000, embed_dim=8192]\\nGPU: all GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('model_input', 'layer0_mha')
    dot.edge('layer0_mha', 'layer0_mlp')
    dot.edge('layer0_mlp', 'layer1_mha')
    dot.edge('layer1_mha', 'layer1_mlp')
    dot.edge('layer1_mlp', 'model_output')
    
    # Save DAG
    dot.render(os.path.join(output_dir, 'complete_helix_model'), format='svg', cleanup=False)
    dot.save(os.path.join(output_dir, 'complete_helix_model.dot'))

def create_detailed_communication_dag(output_dir):
    """Create detailed communication pattern DAG"""
    
    dot = Digraph('helix_communication_patterns')
    dot.attr(rankdir='LR', size='30,20')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Device clusters
    for device_id in range(16):
        with dot.subgraph(name=f'cluster_device_{device_id}') as c:
            c.attr(label=f'Device {device_id}', style='dashed', color='blue')
            
            # MHA partition
            c.node(f'mha_part_{device_id}',
                   f'MHA Partition\\n' +
                   f'Head Group {device_id//4}, Dim Seg {device_id%4}\\n' +
                   f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                   f'Output: [batch_size=1024, seq_len=10000, heads=4, d_v=128]',
                   shape='rectangle', fillcolor='lightcoral')
            
            # MLP partition
            c.node(f'mlp_part_{device_id}',
                   f'MLP Partition\\n' +
                   f'Tensor Parallel Slice {device_id}\\n' +
                   f'Input: [batch_size=1024, seq_len=10000, embed_dim=8192]\\n' +
                   f'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Communication nodes
    for head_group in range(4):
        concat_node = f'group_concat_{head_group}'
        dot.node(concat_node,
                 f'Concatenate Group {head_group}\\n' +
                 f'4 devices → 1\\n' +
                 f'Input: 4×[batch_size=1024, seq_len=10000, heads=4, d_v=128]\\n' +
                 f'Output: [batch_size=1024, seq_len=10000, heads=4, d_v=512]',
                 shape='parallelogram', fillcolor='lightsteelblue')
    
    final_concat = 'final_concat_all'
    dot.node(final_concat,
             'Final Concatenation\\n' +
             '4 groups → 1\\n' +
             'Input: 4×[batch_size=1024, seq_len=10000, heads=4, d_v=512]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]',
             shape='parallelogram', fillcolor='lightsteelblue')
    
    # All-reduce for MLP
    all_reduce = 'all_reduce_mlp'
    dot.node(all_reduce,
             'All-Reduce Sum\\n' +
             '16 devices → 1\\n' +
             'Input: 16×[batch_size=1024, seq_len=10000, embed_dim=2048]\\n' +
             'Output: [batch_size=1024, seq_len=10000, embed_dim=8192]',
             shape='parallelogram', fillcolor='lightsteelblue')
    
    # Connect communication patterns
    for device_id in range(16):
        head_group = device_id // 4
        dot.edge(f'mha_part_{device_id}', f'group_concat_{head_group}')
        dot.edge(f'mlp_part_{device_id}', all_reduce)
    
    for head_group in range(4):
        dot.edge(f'group_concat_{head_group}', final_concat)
    
    # Save DAG
    dot.render(os.path.join(output_dir, 'helix_communication_patterns'), format='svg', cleanup=False)
    dot.save(os.path.join(output_dir, 'helix_communication_patterns.dot'))

def main():
    output_dir = "./outputs/2025-10-15-10-16-01"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all DAGs
    print("Generating MHA Layer 0 DAG...")
    create_mha_partition_dag(0, output_dir)
    
    print("Generating MHA Layer 1 DAG...")
    create_mha_partition_dag(1, output_dir)
    
    print("Generating MLP Layer 0 DAG...")
    create_mlp_dag(0, output_dir)
    
    print("Generating MLP Layer 1 DAG...")
    create_mlp_dag(1, output_dir)
    
    print("Generating complete model DAG...")
    create_complete_model_dag(output_dir)
    
    print("Generating communication patterns DAG...")
    create_detailed_communication_dag(output_dir)
    
    print("All DAGs generated successfully!")

if __name__ == "__main__":
    main()