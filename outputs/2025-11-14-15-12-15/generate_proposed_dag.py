import graphviz
from typing import Dict, List, Tuple

# Constants from deployment config
BATCH_SIZE = 128
SEQ_LEN = 10000
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
MLP_HIDDEN_SIZE = 16384

# Create proposed DAG (4 GPUs per layer)
def create_proposed_dag():
    dot = graphviz.Digraph('proposed_dag', 
                          comment='4-layer Dense Model - Proposed Layer-wise Distribution',
                          graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5', 'ranksep': '1.0'})
    
    # Set node attributes for different types
    dot.attr('node', shape='ellipse')  # Communication nodes
    dot.attr('node', shape='rectangle')  # Computation nodes
    dot.attr('node', shape='parallelogram')  # Routing/aggregation nodes
    
    # Input node
    dot.node('input', 'Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Layer 0: GPUs 0-3
    with dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0 (GPUs 0-3)', style='dashed', color='red')
        
        # Input broadcast to 4 GPUs
        dot.node('l0_input_split', 'Input Broadcast\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]×4\nGPU: 0-3', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Multi-Head Attention with tensor parallelism across 4 GPUs
        dot.node('l0_mha_qkv_split', 'QKV Split\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]×4\nGPU: 0-3', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Individual GPU computations
        for gpu in range(4):
            gpu_id = gpu
            with layer0.subgraph(name=f'cluster_l0_gpu{gpu_id}') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_id}', style='dotted', color='green')
                
                # Attention computation
                dot.node(f'l0_mha_qkv_gpu{gpu_id}', f'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l0_mha_attn_gpu{gpu_id}', f'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # FFN computation
                dot.node(f'l0_ffn_up_gpu{gpu_id}', f'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l0_ffn_gate_gpu{gpu_id}', f'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l0_ffn_act_gpu{gpu_id}', f'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l0_ffn_down_gpu{gpu_id}', f'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Residual and LayerNorm
                dot.node(f'l0_residual_gpu{gpu_id}', f'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=1024] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l0_layernorm_gpu{gpu_id}', f'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # All-reduce aggregations
        dot.node('l0_mha_allreduce', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-3', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.node('l0_ffn_allreduce', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-3', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Final aggregation
        dot.node('l0_output', 'Layer 0 Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 0-3', 
                shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline send to Layer 1
    dot.node('send_l0_l1', 'Pipeline Send\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 3→4', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Layer 1: GPUs 4-7 (similar structure)
    with dot.subgraph(name='cluster_layer1') as layer1:
        layer1.attr(label='Layer 1 (GPUs 4-7)', style='dashed', color='red')
        
        dot.node('l1_input_split', 'Input Broadcast\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]×4\nGPU: 4-7', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l1_mha_qkv_split', 'QKV Split\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]×4\nGPU: 4-7', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        for gpu in range(4, 8):
            gpu_id = gpu
            with layer1.subgraph(name=f'cluster_l1_gpu{gpu_id}') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_id}', style='dotted', color='green')
                
                dot.node(f'l1_mha_qkv_gpu{gpu_id}', f'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l1_mha_attn_gpu{gpu_id}', f'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l1_ffn_up_gpu{gpu_id}', f'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l1_ffn_gate_gpu{gpu_id}', f'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l1_ffn_act_gpu{gpu_id}', f'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l1_ffn_down_gpu{gpu_id}', f'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l1_residual_gpu{gpu_id}', f'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=1024] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l1_layernorm_gpu{gpu_id}', f'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node('l1_mha_allreduce', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 4-7', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.node('l1_ffn_allreduce', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 4-7', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l1_output', 'Layer 1 Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 4-7', 
                shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline send to Layer 2
    dot.node('send_l1_l2', 'Pipeline Send\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 7→8', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Layer 2: GPUs 8-11
    with dot.subgraph(name='cluster_layer2') as layer2:
        layer2.attr(label='Layer 2 (GPUs 8-11)', style='dashed', color='red')
        
        dot.node('l2_input_split', 'Input Broadcast\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]×4\nGPU: 8-11', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l2_mha_qkv_split', 'QKV Split\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]×4\nGPU: 8-11', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        for gpu in range(8, 12):
            gpu_id = gpu
            with layer2.subgraph(name=f'cluster_l2_gpu{gpu_id}') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_id}', style='dotted', color='green')
                
                dot.node(f'l2_mha_qkv_gpu{gpu_id}', f'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l2_mha_attn_gpu{gpu_id}', f'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l2_ffn_up_gpu{gpu_id}', f'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l2_ffn_gate_gpu{gpu_id}', f'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l2_ffn_act_gpu{gpu_id}', f'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l2_ffn_down_gpu{gpu_id}', f'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l2_residual_gpu{gpu_id}', f'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=1024] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l2_layernorm_gpu{gpu_id}', f'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node('l2_mha_allreduce', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-11', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.node('l2_ffn_allreduce', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-11', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l2_output', 'Layer 2 Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 8-11', 
                shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline send to Layer 3
    dot.node('send_l2_l3', 'Pipeline Send\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 11→12', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Layer 3: GPUs 12-15
    with dot.subgraph(name='cluster_layer3') as layer3:
        layer3.attr(label='Layer 3 (GPUs 12-15)', style='dashed', color='red')
        
        dot.node('l3_input_split', 'Input Broadcast\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]×4\nGPU: 12-15', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l3_mha_qkv_split', 'QKV Split\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]×4\nGPU: 12-15', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        for gpu in range(12, 16):
            gpu_id = gpu
            with layer3.subgraph(name=f'cluster_l3_gpu{gpu_id}') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_id}', style='dotted', color='green')
                
                dot.node(f'l3_mha_qkv_gpu{gpu_id}', f'MHA QKV Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l3_mha_attn_gpu{gpu_id}', f'MHA Attention\nInput: [batch_size=128, seq_len=10000, num_heads=8, head_dim=128]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l3_ffn_up_gpu{gpu_id}', f'FFN Up Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l3_ffn_gate_gpu{gpu_id}', f'FFN Gate Linear\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l3_ffn_act_gpu{gpu_id}', f'FFN Activation\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l3_ffn_down_gpu{gpu_id}', f'FFN Down Linear\nInput: [batch_size=128, seq_len=10000, mlp_hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'l3_residual_gpu{gpu_id}', f'Residual Add\nInput: [batch_size=128, seq_len=10000, hidden_size=1024] (x2)\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                dot.node(f'l3_layernorm_gpu{gpu_id}', f'LayerNorm\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]\nOutput: [batch_size=128, seq_len=10000, hidden_size=1024]\nGPU: {gpu_id}', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node('l3_mha_allreduce', 'MHA All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 12-15', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.node('l3_ffn_allreduce', 'FFN All-Reduce\nInput: [batch_size=128, seq_len=10000, hidden_size=1024]×4\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 12-15', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        dot.node('l3_output', 'Layer 3 Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 12-15', 
                shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Final output
    dot.node('final_output', 'Final Output\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: 12-15', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect Layer 0
    dot.edge('input', 'l0_input_split')
    dot.edge('l0_input_split', 'l0_mha_qkv_split')
    
    for gpu in range(4):
        dot.edge('l0_mha_qkv_split', f'l0_mha_qkv_gpu{gpu}')
        dot.edge(f'l0_mha_qkv_gpu{gpu}', f'l0_mha_attn_gpu{gpu}')
        dot.edge('l0_input_split', f'l0_ffn_up_gpu{gpu}')
        dot.edge('l0_input_split', f'l0_ffn_gate_gpu{gpu}')
        dot.edge(f'l0_ffn_up_gpu{gpu}', f'l0_ffn_act_gpu{gpu}')
        dot.edge(f'l0_ffn_gate_gpu{gpu}', f'l0_ffn_act_gpu{gpu}')
        dot.edge(f'l0_ffn_act_gpu{gpu}', f'l0_ffn_down_gpu{gpu}')
        dot.edge(f'l0_mha_attn_gpu{gpu}', f'l0_residual_gpu{gpu}')
        dot.edge(f'l0_ffn_down_gpu{gpu}', f'l0_residual_gpu{gpu}')
        dot.edge(f'l0_residual_gpu{gpu}', f'l0_layernorm_gpu{gpu}')
        dot.edge(f'l0_layernorm_gpu{gpu}', 'l0_mha_allreduce')
        dot.edge('l0_mha_allreduce', 'l0_ffn_allreduce')
        dot.edge('l0_ffn_allreduce', 'l0_output')
    
    # Connect Layer 1
    dot.edge('l0_output', 'send_l0_l1')
    dot.edge('send_l0_l1', 'l1_input_split')
    dot.edge('l1_input_split', 'l1_mha_qkv_split')
    
    for gpu in range(4):
        gpu_id = gpu + 4
        dot.edge('l1_mha_qkv_split', f'l1_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l1_mha_qkv_gpu{gpu_id}', f'l1_mha_attn_gpu{gpu_id}')
        dot.edge('l1_input_split', f'l1_ffn_up_gpu{gpu_id}')
        dot.edge('l1_input_split', f'l1_ffn_gate_gpu{gpu_id}')
        dot.edge(f'l1_ffn_up_gpu{gpu_id}', f'l1_ffn_act_gpu{gpu_id}')
        dot.edge(f'l1_ffn_gate_gpu{gpu_id}', f'l1_ffn_act_gpu{gpu_id}')
        dot.edge(f'l1_ffn_act_gpu{gpu_id}', f'l1_ffn_down_gpu{gpu_id}')
        dot.edge(f'l1_mha_attn_gpu{gpu_id}', f'l1_residual_gpu{gpu_id}')
        dot.edge(f'l1_ffn_down_gpu{gpu_id}', f'l1_residual_gpu{gpu_id}')
        dot.edge(f'l1_residual_gpu{gpu_id}', f'l1_layernorm_gpu{gpu_id}')
        dot.edge(f'l1_layernorm_gpu{gpu_id}', 'l1_mha_allreduce')
        dot.edge('l1_mha_allreduce', 'l1_ffn_allreduce')
        dot.edge('l1_ffn_allreduce', 'l1_output')
    
    # Connect Layer 2
    dot.edge('l1_output', 'send_l1_l2')
    dot.edge('send_l1_l2', 'l2_input_split')
    dot.edge('l2_input_split', 'l2_mha_qkv_split')
    
    for gpu in range(4):
        gpu_id = gpu + 8
        dot.edge('l2_mha_qkv_split', f'l2_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l2_mha_qkv_gpu{gpu_id}', f'l2_mha_attn_gpu{gpu_id}')
        dot.edge('l2_input_split', f'l2_ffn_up_gpu{gpu_id}')
        dot.edge('l2_input_split', f'l2_ffn_gate_gpu{gpu_id}')
        dot.edge(f'l2_ffn_up_gpu{gpu_id}', f'l2_ffn_act_gpu{gpu_id}')
        dot.edge(f'l2_ffn_gate_gpu{gpu_id}', f'l2_ffn_act_gpu{gpu_id}')
        dot.edge(f'l2_ffn_act_gpu{gpu_id}', f'l2_ffn_down_gpu{gpu_id}')
        dot.edge(f'l2_mha_attn_gpu{gpu_id}', f'l2_residual_gpu{gpu_id}')
        dot.edge(f'l2_ffn_down_gpu{gpu_id}', f'l2_residual_gpu{gpu_id}')
        dot.edge(f'l2_residual_gpu{gpu_id}', f'l2_layernorm_gpu{gpu_id}')
        dot.edge(f'l2_layernorm_gpu{gpu_id}', 'l2_mha_allreduce')
        dot.edge('l2_mha_allreduce', 'l2_ffn_allreduce')
        dot.edge('l2_ffn_allreduce', 'l2_output')
    
    # Connect Layer 3
    dot.edge('l2_output', 'send_l2_l3')
    dot.edge('send_l2_l3', 'l3_input_split')
    dot.edge('l3_input_split', 'l3_mha_qkv_split')
    
    for gpu in range(4):
        gpu_id = gpu + 12
        dot.edge('l3_mha_qkv_split', f'l3_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l3_mha_qkv_gpu{gpu_id}', f'l3_mha_attn_gpu{gpu_id}')
        dot.edge('l3_input_split', f'l3_ffn_up_gpu{gpu_id}')
        dot.edge('l3_input_split', f'l3_ffn_gate_gpu{gpu_id}')
        dot.edge(f'l3_ffn_up_gpu{gpu_id}', f'l3_ffn_act_gpu{gpu_id}')
        dot.edge(f'l3_ffn_gate_gpu{gpu_id}', f'l3_ffn_act_gpu{gpu_id}')
        dot.edge(f'l3_ffn_act_gpu{gpu_id}', f'l3_ffn_down_gpu{gpu_id}')
        dot.edge(f'l3_mha_attn_gpu{gpu_id}', f'l3_residual_gpu{gpu_id}')
        dot.edge(f'l3_ffn_down_gpu{gpu_id}', f'l3_residual_gpu{gpu_id}')
        dot.edge(f'l3_residual_gpu{gpu_id}', f'l3_layernorm_gpu{gpu_id}')
        dot.edge(f'l3_layernorm_gpu{gpu_id}', 'l3_mha_allreduce')
        dot.edge('l3_mha_allreduce', 'l3_ffn_allreduce')
        dot.edge('l3_ffn_allreduce', 'l3_output')
    
    dot.edge('l3_output', 'final_output')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    dag.render('../outputs/2025-11-14-15-12-15/proposed_dag', format='svg', cleanup=False)
    dag.save('../outputs/2025-11-14-15-12-15/proposed_dag.dot')