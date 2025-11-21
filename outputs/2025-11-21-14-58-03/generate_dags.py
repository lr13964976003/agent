import graphviz
import os

# Create output directory if it doesn't exist
os.makedirs("../outputs/2025-11-21-14-58-03", exist_ok=True)

def create_proposed_method_dag():
    """Create DAG for the proposed two-level attention partitioning method"""
    dot = graphviz.Digraph('Helix_Proposed_Method', 
                          comment='Two-Level Attention Partitioning on 16 GPUs')
    dot.attr(rankdir='TB', size='20,20')
    dot.attr('node', fontsize='10')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create 4x4 grid for the 16 devices
    devices = {}
    for i in range(4):  # head groups
        for j in range(4):  # dimension slices
            device_id = i * 4 + j
            device_name = f'device_{device_id}'
            
            # Input projection nodes for Q, K, V
            dot.node(f'{device_name}_q_proj', 
                     f'Q Projection {device_id}\\nW_Q[{i},{j}]\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=256]', 
                     shape='rectangle', style='filled', fillcolor='lightyellow')
            
            dot.node(f'{device_name}_k_proj', 
                     f'K Projection {device_id}\\nW_K[{i},{j}]\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=256]', 
                     shape='rectangle', style='filled', fillcolor='lightyellow')
            
            dot.node(f'{device_name}_v_proj', 
                     f'V Projection {device_id}\\nW_V[{i},{j}]\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=256]', 
                     shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # Attention computation
            dot.node(f'{device_name}_attn', 
                     f'Attention {device_id}\\nHeadGroup={i}, Slice={j}\\nsoftmax(QK^T/√32)V\\nInput: [batch=128, seq_len=10000, slice=256]\\nOutput: [batch=128, seq_len=10000, slice=256]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Connect projections to attention
            dot.edge(f'{device_name}_q_proj', f'{device_name}_attn')
            dot.edge(f'{device_name}_k_proj', f'{device_name}_attn')
            dot.edge(f'{device_name}_v_proj', f'{device_name}_attn')
            
            devices[device_id] = device_name
    
    # Add connections from input to all projection nodes
    for i in range(4):
        for j in range(4):
            device_id = i * 4 + j
            dot.edge('input', f'device_{device_id}_q_proj')
            dot.edge('input', f'device_{device_id}_k_proj')
            dot.edge('input', f'device_{device_id}_v_proj')
    
    # Add intra-group concatenation (4 groups, 4 devices each)
    for group_id in range(4):
        concat_node = f'group_{group_id}_concat'
        dot.node(concat_node, 
                 f'Group {group_id} Concat\\nIntra-group reduction\\nInput: 4×[batch=128, seq_len=10000, slice=256]\\nOutput: [batch=128, seq_len=10000, group=1024]', 
                 shape='parallelogram', style='filled', fillcolor='orange')
        
        # Connect devices in this group to concat
        for slice_id in range(4):
            device_id = group_id * 4 + slice_id
            dot.edge(f'device_{device_id}_attn', concat_node)
    
    # Final concatenation across all groups
    dot.node('final_concat', 
             'Final Concat\\nInter-group concatenation\\nInput: 4×[batch=128, seq_len=10000, group=1024]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    for group_id in range(4):
        dot.edge(f'group_{group_id}_concat', 'final_concat')
    
    # Output node
    dot.node('output', 'Output\\nFinal Output\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.edge('final_concat', 'output')
    
    # Save files
    dot.render('../outputs/2025-11-21-14-58-03/proposed_method_dag', format='dot', cleanup=False)
    dot.render('../outputs/2025-11-21-14-58-03/proposed_method_dag', format='svg', cleanup=False)
    return '../outputs/2025-11-21-14-58-03/proposed_method_dag.dot'

def create_baseline_method_dag():
    """Create DAG for the baseline method (Tensor + Pipeline parallelism)"""
    dot = graphviz.Digraph('Helix_Baseline_Method', 
                          comment='Tensor Parallelism (8) + Pipeline Parallelism (2)')
    dot.attr(rankdir='TB', size='20,20')
    dot.attr('node', fontsize='10')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Stage 0: First layer with tensor parallelism (8 devices)
    dot.attr('cluster_0', label='Stage 0 (Devices 0-7)', style='dashed')
    
    # Tensor parallel split for first layer
    for tp_id in range(8):
        # Q, K, V projections for tensor parallel slice
        dot.node(f's0_tp{tp_id}_q_proj', 
                 f'S0 Q Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        dot.node(f's0_tp{tp_id}_k_proj', 
                 f'S0 K Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        dot.node(f's0_tp{tp_id}_v_proj', 
                 f'S0 V Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Attention computation for tensor parallel slice
        dot.node(f's0_tp{tp_id}_attn', 
                 f'S0 Attention TP{tp_id}\\nTensor parallel slice\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # MLP components
        dot.node(f's0_tp{tp_id}_mlp1', 
                 f'S0 MLP1 TP{tp_id}\\nColumn parallel\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, ffn=1024]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        
        dot.node(f's0_tp{tp_id}_mlp2', 
                 f'S0 MLP2 TP{tp_id}\\nRow parallel\\nInput: [batch=128, seq_len=10000, ffn=1024]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Residual connection
        dot.node(f's0_tp{tp_id}_residual', 
                 f'S0 Residual TP{tp_id}\\nAdd & Norm\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # Connect nodes
        dot.edge(f's0_tp{tp_id}_q_proj', f's0_tp{tp_id}_attn')
        dot.edge(f's0_tp{tp_id}_k_proj', f's0_tp{tp_id}_attn')
        dot.edge(f's0_tp{tp_id}_v_proj', f's0_tp{tp_id}_attn')
        dot.edge(f's0_tp{tp_id}_attn', f's0_tp{tp_id}_mlp1')
        dot.edge(f's0_tp{tp_id}_mlp1', f's0_tp{tp_id}_mlp2')
        dot.edge(f's0_tp{tp_id}_mlp2', f's0_tp{tp_id}_residual')
        
        # Connections from input
        dot.edge('input', f's0_tp{tp_id}_q_proj')
        dot.edge('input', f's0_tp{tp_id}_k_proj')
        dot.edge('input', f's0_tp{tp_id}_v_proj')
    
    # All-reduce for tensor parallel group 0
    dot.node('s0_allreduce', 
             'Stage 0 All-Reduce\\nTensor parallel reduction\\nInput: 8×[batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    for tp_id in range(8):
        dot.edge(f's0_tp{tp_id}_residual', 's0_allreduce')
    
    # Pipeline communication
    dot.node('pipeline_comm', 
             'Pipeline Communication\\nCross-stage transfer\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='parallelogram', style='filled', fillcolor='purple')
    
    dot.edge('s0_allreduce', 'pipeline_comm')
    
    # Stage 1: Second layer with tensor parallelism (8 devices)
    dot.attr('cluster_1', label='Stage 1 (Devices 8-15)', style='dashed')
    
    for tp_id in range(8):
        # Q, K, V projections for stage 1
        dot.node(f's1_tp{tp_id}_q_proj', 
                 f'S1 Q Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        dot.node(f's1_tp{tp_id}_k_proj', 
                 f'S1 K Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        dot.node(f's1_tp{tp_id}_v_proj', 
                 f'S1 V Projection TP{tp_id}\\nColumn parallel split\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Attention computation
        dot.node(f's1_tp{tp_id}_attn', 
                 f'S1 Attention TP{tp_id}\\nTensor parallel slice\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # MLP components
        dot.node(f's1_tp{tp_id}_mlp1', 
                 f'S1 MLP1 TP{tp_id}\\nColumn parallel\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, ffn=1024]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        
        dot.node(f's1_tp{tp_id}_mlp2', 
                 f'S1 MLP2 TP{tp_id}\\nRow parallel\\nInput: [batch=128, seq_len=10000, ffn=1024]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Residual connection
        dot.node(f's1_tp{tp_id}_residual', 
                 f'S1 Residual TP{tp_id}\\nAdd & Norm\\nInput: [batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, slice=512]', 
                 shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # Connect nodes
        dot.edge(f's1_tp{tp_id}_q_proj', f's1_tp{tp_id}_attn')
        dot.edge(f's1_tp{tp_id}_k_proj', f's1_tp{tp_id}_attn')
        dot.edge(f's1_tp{tp_id}_v_proj', f's1_tp{tp_id}_attn')
        dot.edge(f's1_tp{tp_id}_attn', f's1_tp{tp_id}_mlp1')
        dot.edge(f's1_tp{tp_id}_mlp1', f's1_tp{tp_id}_mlp2')
        dot.edge(f's1_tp{tp_id}_mlp2', f's1_tp{tp_id}_residual')
    
    # All-reduce for tensor parallel group 1
    dot.node('s1_allreduce', 
             'Stage 1 All-Reduce\\nTensor parallel reduction\\nInput: 8×[batch=128, seq_len=10000, slice=512]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    for tp_id in range(8):
        dot.edge(f's1_tp{tp_id}_residual', 's1_allreduce')
    
    # Output node
    dot.node('output', 'Output\\nFinal Output\\nInput: [batch=128, seq_len=10000, hidden=4096]\\nOutput: [batch=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.edge('pipeline_comm', 's1_tp0_q_proj')
    dot.edge('pipeline_comm', 's1_tp0_k_proj')
    dot.edge('pipeline_comm', 's1_tp0_v_proj')
    dot.edge('pipeline_comm', 's1_tp1_q_proj')
    dot.edge('pipeline_comm', 's1_tp1_k_proj')
    dot.edge('pipeline_comm', 's1_tp1_v_proj')
    dot.edge('pipeline_comm', 's1_tp2_q_proj')
    dot.edge('pipeline_comm', 's1_tp2_k_proj')
    dot.edge('pipeline_comm', 's1_tp2_v_proj')
    dot.edge('pipeline_comm', 's1_tp3_q_proj')
    dot.edge('pipeline_comm', 's1_tp3_k_proj')
    dot.edge('pipeline_comm', 's1_tp3_v_proj')
    dot.edge('pipeline_comm', 's1_tp4_q_proj')
    dot.edge('pipeline_comm', 's1_tp4_k_proj')
    dot.edge('pipeline_comm', 's1_tp4_v_proj')
    dot.edge('pipeline_comm', 's1_tp5_q_proj')
    dot.edge('pipeline_comm', 's1_tp5_k_proj')
    dot.edge('pipeline_comm', 's1_tp5_v_proj')
    dot.edge('pipeline_comm', 's1_tp6_q_proj')
    dot.edge('pipeline_comm', 's1_tp6_k_proj')
    dot.edge('pipeline_comm', 's1_tp6_v_proj')
    dot.edge('pipeline_comm', 's1_tp7_q_proj')
    dot.edge('pipeline_comm', 's1_tp7_k_proj')
    dot.edge('pipeline_comm', 's1_tp7_v_proj')
    dot.edge('s1_allreduce', 'output')
    
    # Save files
    dot.render('../outputs/2025-11-21-14-58-03/baseline_method_dag', format='dot', cleanup=False)
    dot.render('../outputs/2025-11-21-14-58-03/baseline_method_dag', format='svg', cleanup=False)
    return '../outputs/2025-11-21-14-58-03/baseline_method_dag.dot'

if __name__ == "__main__":
    proposed_path = create_proposed_method_dag()
    baseline_path = create_baseline_method_dag()
    
    print(f"Generated DAGs:")
    print(f"Proposed method: {proposed_path}")
    print(f"Baseline method: {baseline_path}")