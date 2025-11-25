import graphviz
from graphviz import Digraph

def create_baseline_dag():
    dot = Digraph('Baseline_Model_DAG', comment='Dense Transformer with TP=8, PP=2')
    dot.attr(rankdir='TB', size='20,15')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgray')

    # Define all nodes with their input/output dimensions and GPU assignments
    
    # Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_0') as c0:
        c0.attr(label='Stage 0: GPUs 0-7 (Layers 0-1)', style='dashed', color='blue')
        
        # Input for Stage 0
        c0.node('input', 'Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')
        
        # Layer 0
        # Attention - split across 8 GPUs
        for gpu in range(8):
            # Q projection slice
            c0.node(f'l0_q_{gpu}', f'Layer0_Q_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # K projection slice  
            c0.node(f'l0_k_{gpu}', f'Layer0_K_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # V projection slice
            c0.node(f'l0_v_{gpu}', f'Layer0_V_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Attention compute slice
            c0.node(f'l0_attn_{gpu}', f'Layer0_Attention_{gpu}\nInput: Q:[128,10000,512], K:[128,10000,512], V:[128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Output projection slice
            c0.node(f'l0_out_{gpu}', f'Layer0_Output_Projection_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        # Attention reduction across GPUs
        c0.node('l0_reduce', 'Layer0_Attention_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        
        # Residual connection and LayerNorm
        c0.node('l0_add', 'Layer0_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')
        
        # MLP - split across 8 GPUs
        for gpu in range(8):
            c0.node(f'l0_mlp_fc1_{gpu}', f'Layer0_MLP_FC1_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,2048]\nGPU: {gpu}', shape='rectangle')
            c0.node(f'l0_mlp_fc2_{gpu}', f'Layer0_MLP_FC2_{gpu}\nInput: [128,10000,2048]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        c0.node('l0_mlp_reduce', 'Layer0_MLP_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        c0.node('l0_mlp_add', 'Layer0_MLP_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

    # Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_1') as c1:
        c1.attr(label='Stage 1: GPUs 8-15 (Layers 2-3)', style='dashed', color='red')
        
        # Pipeline communication
        c1.node('comm_0_1', 'Pipeline_Communication_0→1\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: 7→8', shape='ellipse', fillcolor='pink', style='dashed')
        
        # Layer 2
        for gpu in range(8, 16):
            actual_gpu = gpu - 8
            # Q projection slice
            c1.node(f'l2_q_{gpu}', f'Layer2_Q_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # K projection slice  
            c1.node(f'l2_k_{gpu}', f'Layer2_K_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # V projection slice
            c1.node(f'l2_v_{gpu}', f'Layer2_V_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Attention compute slice
            c1.node(f'l2_attn_{gpu}', f'Layer2_Attention_{gpu}\nInput: Q:[128,10000,512], K:[128,10000,512], V:[128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Output projection slice
            c1.node(f'l2_out_{gpu}', f'Layer2_Output_Projection_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        c1.node('l2_reduce', 'Layer2_Attention_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        c1.node('l2_add', 'Layer2_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')
        
        # MLP - split across 8 GPUs
        for gpu in range(8, 16):
            actual_gpu = gpu - 8
            c1.node(f'l2_mlp_fc1_{gpu}', f'Layer2_MLP_FC1_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,2048]\nGPU: {gpu}', shape='rectangle')
            c1.node(f'l2_mlp_fc2_{gpu}', f'Layer2_MLP_FC2_{gpu}\nInput: [128,10000,2048]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        c1.node('l2_mlp_reduce', 'Layer2_MLP_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        c1.node('l2_mlp_add', 'Layer2_MLP_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

        # Layer 3
        for gpu in range(8, 16):
            actual_gpu = gpu - 8
            # Q projection slice
            c1.node(f'l3_q_{gpu}', f'Layer3_Q_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # K projection slice  
            c1.node(f'l3_k_{gpu}', f'Layer3_K_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            # V projection slice
            c1.node(f'l3_v_{gpu}', f'Layer3_V_Projection_{gpu}\nInput: [128,10000,4096]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Attention compute slice
            c1.node(f'l3_attn_{gpu}', f'Layer3_Attention_{gpu}\nInput: Q:[128,10000,512], K:[128,10000,512], V:[128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
            
            # Output projection slice
            c1.node(f'l3_out_{gpu}', f'Layer3_Output_Projection_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        c1.node('l3_reduce', 'Layer3_Attention_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        c1.node('l3_add', 'Layer3_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')
        
        # MLP - split across 8 GPUs
        for gpu in range(8, 16):
            actual_gpu = gpu - 8
            c1.node(f'l3_mlp_fc1_{gpu}', f'Layer3_MLP_FC1_{gpu}\nInput: [128,10000,512]\nOutput: [128,10000,2048]\nGPU: {gpu}', shape='rectangle')
            c1.node(f'l3_mlp_fc2_{gpu}', f'Layer3_MLP_FC2_{gpu}\nInput: [128,10000,2048]\nOutput: [128,10000,512]\nGPU: {gpu}', shape='rectangle')
        
        c1.node('l3_mlp_reduce', 'Layer3_MLP_Reduce\nInput: [128,10000,512]×8\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')
        c1.node('l3_mlp_add', 'Layer3_MLP_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')
        c1.node('output', 'Final_Output\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

    # Connect all nodes
    # Input to Layer 0 attention
    for gpu in range(8):
        dot.edge('input', f'l0_q_{gpu}')
        dot.edge('input', f'l0_k_{gpu}')
        dot.edge('input', f'l0_v_{gpu}')
        dot.edge(f'l0_q_{gpu}', f'l0_attn_{gpu}')
        dot.edge(f'l0_k_{gpu}', f'l0_attn_{gpu}')
        dot.edge(f'l0_v_{gpu}', f'l0_attn_{gpu}')
        dot.edge(f'l0_attn_{gpu}', f'l0_out_{gpu}')
    
    for gpu in range(8):
        dot.edge(f'l0_out_{gpu}', 'l0_reduce')
    dot.edge('l0_reduce', 'l0_add')
    dot.edge('input', 'l0_add')
    
    # Layer 0 MLP
    for gpu in range(8):
        dot.edge('l0_add', f'l0_mlp_fc1_{gpu}')
        dot.edge(f'l0_mlp_fc1_{gpu}', f'l0_mlp_fc2_{gpu}')
        dot.edge(f'l0_mlp_fc2_{gpu}', 'l0_mlp_reduce')
    dot.edge('l0_mlp_reduce', 'l0_mlp_add')
    dot.edge('l0_add', 'l0_mlp_add')
    
    # Pipeline communication
    dot.edge('l0_mlp_add', 'comm_0_1')
    
    # Layer 2 attention
    for gpu in range(8, 16):
        dot.edge('comm_0_1', f'l2_q_{gpu}')
        dot.edge('comm_0_1', f'l2_k_{gpu}')
        dot.edge('comm_0_1', f'l2_v_{gpu}')
        dot.edge(f'l2_q_{gpu}', f'l2_attn_{gpu}')
        dot.edge(f'l2_k_{gpu}', f'l2_attn_{gpu}')
        dot.edge(f'l2_v_{gpu}', f'l2_attn_{gpu}')
        dot.edge(f'l2_attn_{gpu}', f'l2_out_{gpu}')
    
    for gpu in range(8, 16):
        dot.edge(f'l2_out_{gpu}', 'l2_reduce')
    dot.edge('l2_reduce', 'l2_add')
    dot.edge('comm_0_1', 'l2_add')
    
    # Layer 2 MLP
    for gpu in range(8, 16):
        dot.edge('l2_add', f'l2_mlp_fc1_{gpu}')
        dot.edge(f'l2_mlp_fc1_{gpu}', f'l2_mlp_fc2_{gpu}')
        dot.edge(f'l2_mlp_fc2_{gpu}', 'l2_mlp_reduce')
    dot.edge('l2_mlp_reduce', 'l2_mlp_add')
    dot.edge('l2_add', 'l2_mlp_add')
    
    # Layer 3 attention
    for gpu in range(8, 16):
        dot.edge('l2_mlp_add', f'l3_q_{gpu}')
        dot.edge('l2_mlp_add', f'l3_k_{gpu}')
        dot.edge('l2_mlp_add', f'l3_v_{gpu}')
        dot.edge(f'l3_q_{gpu}', f'l3_attn_{gpu}')
        dot.edge(f'l3_k_{gpu}', f'l3_attn_{gpu}')
        dot.edge(f'l3_v_{gpu}', f'l3_attn_{gpu}')
        dot.edge(f'l3_attn_{gpu}', f'l3_out_{gpu}')
    
    for gpu in range(8, 16):
        dot.edge(f'l3_out_{gpu}', 'l3_reduce')
    dot.edge('l3_reduce', 'l3_add')
    dot.edge('l2_mlp_add', 'l3_add')
    
    # Layer 3 MLP
    for gpu in range(8, 16):
        dot.edge('l3_add', f'l3_mlp_fc1_{gpu}')
        dot.edge(f'l3_mlp_fc1_{gpu}', f'l3_mlp_fc2_{gpu}')
        dot.edge(f'l3_mlp_fc2_{gpu}', 'l3_mlp_reduce')
    dot.edge('l3_mlp_reduce', 'l3_mlp_add')
    dot.edge('l3_add', 'l3_mlp_add')
    dot.edge('l3_mlp_add', 'output')

    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('../outputs/2025-11-24-20-37-50/baseline_dag', format='svg', cleanup=False)
    dag.render('../outputs/2025-11-24-20-37-50/baseline_dag', format='dot', cleanup=False)
    print("Baseline DAG generated successfully")