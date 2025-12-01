#!/usr/bin/env python3

import graphviz

def create_helix_two_level_dag():
    """Create DAG for Helix Two-Level Attention Partitioning (4x4 = 16 partitions)"""
    
    dot = graphviz.Digraph('Helix_Two_Level_Attention_Partitioning')
    dot.attr(rankdir='TB', fontsize='12', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    
    # Input projection splitting (Q, K, V)
    dot.node('input_proj_split', 'Input Projection Split\\nSplit Q,K,V across 16 devices\\n[batch_size=128, seq_len=10000, d_model=4096]→16×[batch_size=128, seq_len=10000, 256]', shape='parallelogram', fillcolor='yellow')
    dot.edge('input', 'input_proj_split')
    
    # Create 16 devices for two-level partitioning (4 head groups × 4 dimension slices)
    devices = []
    for head_group in range(4):  # 4 head groups
        for dim_slice in range(4):  # 4 dimension slices
            device_id = head_group * 4 + dim_slice
            
            # Q projection for this partition
            q_proj = f'device_{device_id}_q_proj'
            dot.node(q_proj, f'Device {device_id} Q Projection\\nHead Group {head_group}, Dim Slice {dim_slice}\\nGPU {device_id}\\nW_Q[{head_group},{dim_slice}]: 256×256\\n[batch_size=128, seq_len=10000, 256]→[batch_size=128, seq_len=10000, 256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # K projection for this partition
            k_proj = f'device_{device_id}_k_proj'
            dot.node(k_proj, f'Device {device_id} K Projection\\nHead Group {head_group}, Dim Slice {dim_slice}\\nGPU {device_id}\\nW_K[{head_group},{dim_slice}]: 256×256\\n[batch_size=128, seq_len=10000, 256]→[batch_size=128, seq_len=10000, 256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # V projection for this partition
            v_proj = f'device_{device_id}_v_proj'
            dot.node(v_proj, f'Device {device_id} V Projection\\nHead Group {head_group}, Dim Slice {dim_slice}\\nGPU {device_id}\\nW_V[{head_group},{dim_slice}]: 256×256\\n[batch_size=128, seq_len=10000, 256]→[batch_size=128, seq_len=10000, 256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # Attention computation for this partition
            attn_comp = f'device_{device_id}_attn'
            dot.node(attn_comp, f'Device {device_id} Attention Computation\\nHead Group {head_group}, Dim Slice {dim_slice}\\nGPU {device_id}\\nsoftmax(QK^T/√32)V\\n[batch_size=128, seq_len=10000, 256]→[batch_size=128, seq_len=10000, 256]', 
                    shape='rectangle', fillcolor='lightgreen')
            
            devices.append({
                'id': device_id,
                'head_group': head_group,
                'dim_slice': dim_slice,
                'q_proj': q_proj,
                'k_proj': k_proj,
                'v_proj': v_proj,
                'attn': attn_comp
            })
            
            # Connect input split to projections
            dot.edge('input_proj_split', q_proj)
            dot.edge('input_proj_split', k_proj)
            dot.edge('input_proj_split', v_proj)
            
            # Connect projections to attention
            dot.edge(q_proj, attn_comp)
            dot.edge(k_proj, attn_comp)
            dot.edge(v_proj, attn_comp)
    
    # Intra-group concatenation (4 devices per head group)
    for head_group in range(4):
        intra_group_concat = f'intra_group_{head_group}_concat'
        dot.node(intra_group_concat, f'Intra-Group {head_group} Concatenation\\nConcatenate 4 dimension slices\\nGPU {head_group*4}-{(head_group+1)*4-1}\\n4×[batch_size=128, seq_len=10000, 256]→[batch_size=128, seq_len=10000, 1024]', 
                shape='parallelogram', fillcolor='yellow')
        
        # Connect devices in this head group to intra-group concatenation
        for dim_slice in range(4):
            device_id = head_group * 4 + dim_slice
            device = next(d for d in devices if d['id'] == device_id)
            dot.edge(device['attn'], intra_group_concat)
    
    # Final concatenation across all head groups
    dot.node('final_concat', 'Final Concatenation\\nConcatenate 4 head groups\\nGPU 0-15\\n4×[batch_size=128, seq_len=10000, 1024]→[batch_size=128, seq_len=10000, 4096]', 
            shape='parallelogram', fillcolor='yellow')
    
    # Connect intra-group concatenations to final concatenation
    for head_group in range(4):
        intra_group_concat = f'intra_group_{head_group}_concat'
        dot.edge(intra_group_concat, 'final_concat')
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    dot.edge('final_concat', 'output')
    
    return dot

def create_helix_baseline_dag():
    """Create DAG for Helix Baseline (TP=8, PP=2)"""
    
    dot = graphviz.Digraph('Helix_Baseline_TP8_PP2')
    dot.attr(rankdir='TB', fontsize='12', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    
    # Pipeline Stage 0 (GPU 0-7)
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Layer 0 MHA with tensor parallelism
    dot.node('layer0_mha_tp8', 'Layer 0 MHA\\nTensor Parallel=8\\nGPU 0-7\\n32 heads split across 8 GPUs\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
            shape='rectangle', fillcolor='lightcoral')
    
    # Layer 0 MLP with tensor parallelism
    dot.node('layer0_mlp_tp8', 'Layer 0 MLP\\nTensor Parallel=8\\nGPU 0-7\\nColumn+Row Parallel\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
            shape='rectangle', fillcolor='lightcoral')
    
    dot.edge('input', 'layer0_mha_tp8')
    dot.edge('layer0_mha_tp8', 'layer0_mlp_tp8')
    
    # Add residual connections
    dot.node('residual0', 'Residual Add Layer 0\\nGPU 0-7\\n[batch_size=128, seq_len=10000, d_model=4096] + [batch_size=128, seq_len=10000, d_model=4096]', 
            shape='parallelogram', fillcolor='yellow')
    dot.edge('input', 'residual0')
    dot.edge('layer0_mlp_tp8', 'residual0')
    
    # Pipeline communication to Stage 1
    dot.node('pipe_comm_0_1', 'Pipeline Communication\\nStage 0 → Stage 1\\nGPU 7 → GPU 8', shape='diamond', fillcolor='orange')
    dot.edge('residual0', 'pipe_comm_0_1')
    
    # Pipeline Stage 1 (GPU 8-15)
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightsteelblue')
    
    # Layer 1 MHA with tensor parallelism
    dot.node('layer1_mha_tp8', 'Layer 1 MHA\\nTensor Parallel=8\\nGPU 8-15\\n32 heads split across 8 GPUs\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
            shape='rectangle', fillcolor='lightsteelblue')
    
    # Layer 1 MLP with tensor parallelism
    dot.node('layer1_mlp_tp8', 'Layer 1 MLP\\nTensor Parallel=8\\nGPU 8-15\\nColumn+Row Parallel\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
            shape='rectangle', fillcolor='lightsteelblue')
    
    dot.edge('pipe_comm_0_1', 'layer1_mha_tp8')
    dot.edge('layer1_mha_tp8', 'layer1_mlp_tp8')
    
    # Add residual connections
    dot.node('residual1', 'Residual Add Layer 1\\nGPU 8-15\\n[batch_size=128, seq_len=10000, d_model=4096] + [batch_size=128, seq_len=10000, d_model=4096]', 
            shape='parallelogram', fillcolor='yellow')
    dot.edge('pipe_comm_0_1', 'residual1')
    dot.edge('layer1_mlp_tp8', 'residual1')
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    dot.edge('residual1', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate Helix Two-Level Partitioning DAG
    helix_dag = create_helix_two_level_dag()
    helix_dag.render('helix_two_level_dag', format='dot', cleanup=False)
    helix_dag.render('helix_two_level_dag', format='svg', cleanup=False)
    
    # Generate Helix Baseline DAG
    helix_baseline_dag = create_helix_baseline_dag()
    helix_baseline_dag.render('helix_baseline_tp8_pp2_dag', format='dot', cleanup=False)
    helix_baseline_dag.render('helix_baseline_tp8_pp2_dag', format='svg', cleanup=False)
    
    print("Generated DAGs:")
    print("- helix_two_level_dag.dot & .svg")
    print("- helix_baseline_tp8_pp2_dag.dot & .svg")