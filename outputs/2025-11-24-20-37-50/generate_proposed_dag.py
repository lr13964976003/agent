import graphviz
from graphviz import Digraph

def create_proposed_dag():
    dot = Digraph('Proposed_Model_DAG', comment='Dense Transformer with Two-Level Attention Partitioning')
    dot.attr(rankdir='TB', size='20,15')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgray')

    # Model parameters for clarity
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    heads_total = 32
    head_dim = 128
    heads_per_group = 8  # 32/4 = 8 heads per group
    dim_slice = 32  # 128/4 = 32 dimensions per slice

    # Create a compact representation for each layer
    # Layer 0 - representative layer (showing detail for one layer)
    with dot.subgraph(name='cluster_l0') as c0:
        c0.attr(label='Layer 0: Two-Level Attention Partitioning (m×n=16)', style='dashed', color='blue')
        
        # Input for all partitions
        c0.node('input_l0', 'Input_Layer0\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')
        
        # Create 16 attention partitions (4 head groups × 4 dimension slices)
        for head_group in range(4):  # 4 head groups
            for dim_part in range(4):  # 4 dimension slices
                partition_id = head_group * 4 + dim_part
                gpu = partition_id
                
                with dot.subgraph(name=f'cluster_partition_{partition_id}') as cp:
                    cp.attr(label=f'Partition {partition_id} (GPU {gpu})\nHeads: [{head_group*8}-{head_group*8+7}], Dim: [{dim_part*32}-{dim_part*32+31}]', style='dotted', color='green')
                    
                    # Q projection for this partition
                    cp.node(f'l0_q_{partition_id}', f'Q_Projection_{partition_id}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # K projection for this partition  
                    cp.node(f'l0_k_{partition_id}', f'K_Projection_{partition_id}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # V projection for this partition
                    cp.node(f'l0_v_{partition_id}', f'V_Projection_{partition_id}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # Attention computation for this partition
                    cp.node(f'l0_attn_{partition_id}', f'Attention_{partition_id}\nInput: Q:[128,10000,256], K:[128,10000,256], V:[128,10000,256]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')

    # Layer 0 aggregation stages
    dot.node('l0_stage1_concat', 'Layer0_Dimension_Concatenation\nInput: [128,10000,256]×4×4\nOutput: [128,10000,1024]×4\nGPU: 0,4,8,12', shape='parallelogram', fillcolor='yellow')
    
    dot.node('l0_stage2_concat', 'Layer0_Head_Concatenation\nInput: [128,10000,1024]×4\nOutput: [128,10000,4096]\nGPU: 0', shape='parallelogram', fillcolor='yellow')
    
    dot.node('l0_add', 'Layer0_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

    # Layer 0 MLP - also partitioned across 16 GPUs
    for partition_id in range(16):
        gpu = partition_id
        dot.node(f'l0_mlp_fc1_{partition_id}', f'MLP_FC1_{partition_id}\nInput: [128,10000,256]\nOutput: [128,10000,1024]\nGPU: {gpu}', shape='rectangle')
        dot.node(f'l0_mlp_fc2_{partition_id}', f'MLP_FC2_{partition_id}\nInput: [128,10000,1024]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
    
    dot.node('l0_mlp_stage1_concat', 'Layer0_MLP_Dimension_Concat\nInput: [128,10000,256]×4×4\nOutput: [128,10000,1024]×4\nGPU: 0,4,8,12', shape='parallelogram', fillcolor='yellow')
    dot.node('l0_mlp_stage2_concat', 'Layer0_MLP_Head_Concat\nInput: [128,10000,1024]×4\nOutput: [128,10000,4096]\nGPU: 0', shape='parallelogram', fillcolor='yellow')
    dot.node('l0_mlp_add', 'Layer0_MLP_Add&Norm\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

    # Simplified representation for other layers (Layers 1-3)
    for layer in range(1, 4):
        dot.node(f'layer{layer}_block', f'Layer {layer}\nComplete Layer with Two-Level Partitioning\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: 0-15\n(16 partitions, similar to Layer 0)', shape='rectangle', fillcolor='lightcyan')
        
    # Final output
    dot.node('output', 'Final_Output\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

    # Connect Layer 0 components
    # Input to all Q/K/V projections
    for partition_id in range(16):
        dot.edge('input_l0', f'l0_q_{partition_id}')
        dot.edge('input_l0', f'l0_k_{partition_id}')
        dot.edge('input_l0', f'l0_v_{partition_id}')
        dot.edge(f'l0_q_{partition_id}', f'l0_attn_{partition_id}')
        dot.edge(f'l0_k_{partition_id}', f'l0_attn_{partition_id}')
        dot.edge(f'l0_v_{partition_id}', f'l0_attn_{partition_id}')
        dot.edge(f'l0_attn_{partition_id}', 'l0_stage1_concat')
    
    # Dimension concatenation for each head group
    for head_group in range(4):
        partition_ids = [head_group * 4 + dim_part for dim_part in range(4)]
        for pid in partition_ids:
            dot.edge(f'l0_attn_{pid}', f'l0_stage1_concat')
    
    # Head concatenation
    dot.edge('l0_stage1_concat', 'l0_stage2_concat')
    dot.edge('l0_stage2_concat', 'l0_add')
    dot.edge('input_l0', 'l0_add')

    # MLP connections
    for partition_id in range(16):
        dot.edge('l0_add', f'l0_mlp_fc1_{partition_id}')
        dot.edge(f'l0_mlp_fc1_{partition_id}', f'l0_mlp_fc2_{partition_id}')
        dot.edge(f'l0_mlp_fc2_{partition_id}', 'l0_mlp_stage1_concat')
    
    dot.edge('l0_mlp_stage1_concat', 'l0_mlp_stage2_concat')
    dot.edge('l0_mlp_stage2_concat', 'l0_mlp_add')
    dot.edge('l0_add', 'l0_mlp_add')

    # Connect remaining layers
    dot.edge('l0_mlp_add', 'layer1_block')
    dot.edge('layer1_block', 'layer2_block')
    dot.edge('layer2_block', 'layer3_block')
    dot.edge('layer3_block', 'output')

    return dot

def create_detailed_proposed_dag():
    """Create a more detailed DAG showing the complete flow"""
    dot = Digraph('Detailed_Proposed_Model_DAG', comment='Detailed Two-Level Attention Partitioning')
    dot.attr(rankdir='TB', size='25,20')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgray')

    # Global input
    dot.node('input', 'Model_Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

    # Process each layer individually
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}') as cl:
            cl.attr(label=f'Layer {layer} - Two-Level Attention Partitioning', style='dashed', color='blue')
            
            # Input to this layer
            if layer == 0:
                cl.node(f'll{layer}_input', f'Layer{layer}_Input\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse')
            else:
                cl.node(f'll{layer}_input', f'Layer{layer}_Input\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse')

            # Broadcast input to all 16 partitions
            for partition in range(16):
                gpu = partition
                head_group = partition // 4
                dim_slice = partition % 4
                heads_start = head_group * 8
                heads_end = heads_start + 7
                dim_start = dim_slice * 32
                dim_end = dim_start + 31
                
                with dot.subgraph(name=f'cluster_layer{layer}_partition{partition}') as cp:
                    cp.attr(label=f'Partition {partition} (GPU {gpu})\nHeads: [{heads_start}-{heads_end}], Dim: [{dim_start}-{dim_end}]')
                    
                    # Q projection
                    cp.node(f'q_{layer}_{partition}', f'Q_Projection_L{layer}_P{partition}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # K projection
                    cp.node(f'k_{layer}_{partition}', f'K_Projection_L{layer}_P{partition}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # V projection
                    cp.node(f'v_{layer}_{partition}', f'V_Projection_L{layer}_P{partition}\nInput: [128,10000,4096]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')
                    
                    # Attention computation
                    cp.node(f'attn_{layer}_{partition}', f'Attention_L{layer}_P{partition}\nInput: Q:[128,10000,256], K:[128,10000,256], V:[128,10000,256]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')

            # Aggregation nodes for this layer
            cl.node(f'l{layer}_dim_concat', f'Dimension_Concatenation_L{layer}\nInput: [128,10000,256]×4×4\nOutput: [128,10000,4096]\nGPU: 0', shape='parallelogram', fillcolor='yellow')
            cl.node(f'l{layer}_add', f'Add&Norm_L{layer}_Attention\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

            # MLP partitioning
            for partition in range(16):
                gpu = partition
                cl.node(f'mlp_fc1_{layer}_{partition}', f'MLP_FC1_L{layer}_P{partition}\nInput: [128,10000,256]\nOutput: [128,10000,1024]\nGPU: {gpu}', shape='rectangle')
                cl.node(f'mlp_fc2_{layer}_{partition}', f'MLP_FC2_L{layer}_P{partition}\nInput: [128,10000,1024]\nOutput: [128,10000,256]\nGPU: {gpu}', shape='rectangle')

            cl.node(f'l{layer}_mlp_concat', f'MLP_Concatenation_L{layer}\nInput: [128,10000,256]×16\nOutput: [128,10000,4096]\nGPU: 0', shape='parallelogram', fillcolor='yellow')
            cl.node(f'l{layer}_mlp_add', f'Add&Norm_L{layer}_MLP\nInput: [128,10000,4096], [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='rectangle', fillcolor='lightblue')

    # Output
    dot.node('output', 'Model_Output\nInput: [128,10000,4096]\nOutput: [128,10000,4096]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

    # Connect all nodes
    dot.edge('input', 'll0_input')
    
    for layer in range(4):
        # Connect input to all partitions
        for partition in range(16):
            gpu = partition
            dot.edge(f'll{layer}_input', f'q_{layer}_{partition}')
            dot.edge(f'll{layer}_input', f'k_{layer}_{partition}')
            dot.edge(f'll{layer}_input', f'v_{layer}_{partition}')
            dot.edge(f'q_{layer}_{partition}', f'attn_{layer}_{partition}')
            dot.edge(f'k_{layer}_{partition}', f'attn_{layer}_{partition}')
            dot.edge(f'v_{layer}_{partition}', f'attn_{layer}_{partition}')
            dot.edge(f'attn_{layer}_{partition}', f'l{layer}_dim_concat')
        
        # Attention aggregation
        dot.edge(f'l{layer}_dim_concat', f'l{layer}_add')
        dot.edge(f'll{layer}_input', f'l{layer}_add')
        
        # MLP connections
        for partition in range(16):
            dot.edge(f'l{layer}_add', f'mlp_fc1_{layer}_{partition}')
            dot.edge(f'mlp_fc1_{layer}_{partition}', f'mlp_fc2_{layer}_{partition}')
            dot.edge(f'mlp_fc2_{layer}_{partition}', f'l{layer}_mlp_concat')
        
        dot.edge(f'l{layer}_mlp_concat', f'l{layer}_mlp_add')
        dot.edge(f'l{layer}_add', f'l{layer}_mlp_add')
        
        # Connect to next layer or output
        if layer < 3:
            dot.edge(f'l{layer}_mlp_add', f'll{layer+1}_input')
        else:
            dot.edge(f'l{layer}_mlp_add', 'output')

    return dot

if __name__ == '__main__':
    # Generate standard DAG
    dag1 = create_proposed_dag()
    dag1.render('../outputs/2025-11-24-20-37-50/proposed_dag', format='svg', cleanup=False)
    dag1.render('../outputs/2025-11-24-20-37-50/proposed_dag', format='dot', cleanup=False)
    
    # Generate detailed DAG
    dag2 = create_detailed_proposed_dag()
    dag2.render('../outputs/2025-11-24-20-37-50/proposed_detailed_dag', format='svg', cleanup=False)
    dag2.render('../outputs/2025-11-24-20-37-50/proposed_detailed_dag', format='dot', cleanup=False)
    
    print("Proposed DAGs generated successfully")