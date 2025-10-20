import graphviz
import os

def create_helix_dag():
    """Create complete Helix two-level partitioning DAG for 2-layer transformer"""
    
    # Create graph
    dot = graphviz.Digraph('Helix_Two_Level_Partitioning', 
                          comment='Helix: Two-Level Attention Partitioning on 16 GPUs',
                          format='dot')
    
    # Graph attributes
    dot.attr(rankdir='TB', splines='ortho', compound='true', ranksep='2', nodesep='0.5')
    
    # Node attributes
    dot.attr('node', fontname='Arial', fontsize='10', margin='0.2,0.1')
    
    # Define subgraphs for each GPU cluster (4x4 grid)
    # Each cluster handles 4 heads (16 heads/4 groups) and 128 dimensions (512/4 segments)
    
    # Global Input
    dot.node('input', 'Global Input\\nInput: [batch_size=1024, seq_len=10000, hidden_size=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # ===== LAYER 1 =====
    # Layer Norm 1 (across all GPUs)
    dot.node('ln1', 'Layer Norm 1\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge('input', 'ln1')
    
    # ===== Multi-Head Attention Layer 1 =====
    # QKV Projection - Split across 16 GPUs (4x4 partitioning)
    
    for gpu_id in range(16):
        row = gpu_id // 4  # 0-3
        col = gpu_id % 4   # 0-3
        
        # Each GPU handles 4 heads total (h/n=16/4=4 heads per group)
        # Each GPU handles 128 dimensions per head (d/m=512/4=128)
        
        # Q Projection
        q_node = f'q_proj_{gpu_id}'
        dot.node(q_node, f'Q Projection\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge('ln1', q_node)
        
        # K Projection
        k_node = f'k_proj_{gpu_id}'
        dot.node(k_node, f'K Projection\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge('ln1', k_node)
        
        # V Projection
        v_node = f'v_proj_{gpu_id}'
        dot.node(v_node, f'V Projection\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge('ln1', v_node)
        
        # Scaled Dot-Product Attention
        attn_node = f'attn_{gpu_id}'
        dot.node(attn_node, f'Scaled Dot-Product Attention\\nGPU:{gpu_id}\\nInput: [1024,10000,4,128] x3\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        dot.edge(q_node, attn_node)
        dot.edge(k_node, attn_node)
        dot.edge(v_node, attn_node)
    
    # ===== Attention Output Aggregation =====
    # Step 1: Concatenate dimension slices within each head group (4 GPUs per group)
    for group_id in range(4):  # 4 head groups
        for dim_seg in range(4):  # 4 dimension segments per head
            concat_dim_node = f'concat_dim_{group_id}_{dim_seg}'
            dot.node(concat_dim_node, 
                     f'Concatenate Dimensions\\nGroup:{group_id}, Dim:{dim_seg}\\nInput: [1024,10000,4,128] x4\\nOutput: [1024,10000,4,512]', 
                     shape='parallelogram', style='filled', fillcolor='orange')
            
            # Connect from 4 GPUs in the same dimension segment
            for gpu_offset in range(4):
                gpu_id = group_id * 4 + gpu_offset
                dot.edge(f'attn_{gpu_id}', concat_dim_node)
    
    # Step 2: Concatenate head groups
    for head_group in range(4):
        concat_heads_node = f'concat_heads_{head_group}'
        dot.node(concat_heads_node, 
                 f'Concatenate Head Groups\\nGroup:{head_group}\\nInput: [1024,10000,4,512]\\nOutput: [1024,10000,2048]', 
                 shape='parallelogram', style='filled', fillcolor='gold')
        
        # Connect from dimension concatenation
        for dim_seg in range(4):
            concat_dim_node = f'concat_dim_{head_group}_{dim_seg}'
            dot.edge(concat_dim_node, concat_heads_node)
    
    # Step 3: Final attention output concatenation
    final_concat_attn = 'final_concat_attn'
    dot.node(final_concat_attn, 
             'Final Attention Concatenation\\nInput: [1024,10000,2048] x4\\nOutput: [1024,10000,8192]', 
             shape='parallelogram', style='filled', fillcolor='darkorange')
    
    for head_group in range(4):
        dot.edge(f'concat_heads_{head_group}', final_concat_attn)
    
    # Attention Output Projection (across all GPUs after concatenation)
    attn_out_proj = 'attn_out_proj'
    dot.node(attn_out_proj, 
             'Attention Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(final_concat_attn, attn_out_proj)
    
    # Residual Connection 1
    residual1 = 'residual1'
    dot.node(residual1, 'Residual Add 1\\nInput: [1024,10000,8192] x2\\nOutput: [1024,10000,8192]', 
             shape='ellipse', style='filled', fillcolor='lightpink')
    dot.edge('input', residual1, style='dashed')
    dot.edge(attn_out_proj, residual1)
    
    # ===== MLP Layer 1 =====
    # Layer Norm 2
    ln2 = 'ln2'
    dot.node(ln2, 'Layer Norm 2\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(residual1, ln2)
    
    # MLP - Column Parallel first layer across 16 GPUs
    for gpu_id in range(16):
        mlp_col = f'mlp_col_{gpu_id}'
        dot.node(mlp_col, 
                 f'MLP Column Parallel\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,2048]', 
                 shape='rectangle', style='filled', fillcolor='lightblue')
        dot.edge(ln2, mlp_col)
        
        # GELU Activation
        gelu = f'gelu_{gpu_id}'
        dot.node(gelu, f'GELU Activation\\nGPU:{gpu_id}\\nInput: [1024,10000,2048]\\nOutput: [1024,10000,2048]', 
                 shape='rectangle', style='filled', fillcolor='lightcyan')
        dot.edge(mlp_col, gelu)
        
        # MLP - Row Parallel second layer
        mlp_row = f'mlp_row_{gpu_id}'
        dot.node(mlp_row, 
                 f'MLP Row Parallel\\nGPU:{gpu_id}\\nInput: [1024,10000,2048]\\nOutput: [1024,10000,512]', 
                 shape='rectangle', style='filled', fillcolor='lightblue')
        dot.edge(gelu, mlp_row)
    
    # MLP Output Aggregation
    mlp_concat = 'mlp_concat'
    dot.node(mlp_concat, 
             'MLP Output Concatenation\\nInput: [1024,10000,512] x16\\nOutput: [1024,10000,8192]', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    for gpu_id in range(16):
        dot.edge(f'mlp_row_{gpu_id}', mlp_concat)
    
    # MLP Output Projection
    mlp_out_proj = 'mlp_out_proj'
    dot.node(mlp_out_proj, 
             'MLP Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(mlp_concat, mlp_out_proj)
    
    # Residual Connection 2
    residual2 = 'residual2'
    dot.node(residual2, 'Residual Add 2\\nInput: [1024,10000,8192] x2\\nOutput: [1024,10000,8192]', 
             shape='ellipse', style='filled', fillcolor='lightpink')
    dot.edge(residual1, residual2, style='dashed')
    dot.edge(mlp_out_proj, residual2)
    
    # ===== LAYER 2 ===== (Identical structure to Layer 1)
    # Layer Norm 3
    ln3 = 'ln3'
    dot.node(ln3, 'Layer Norm 3\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(residual2, ln3)
    
    # QKV Projection Layer 2
    for gpu_id in range(16):
        q2_node = f'q2_proj_{gpu_id}'
        dot.node(q2_node, f'Q Projection L2\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge(ln3, q2_node)
        
        k2_node = f'k2_proj_{gpu_id}'
        dot.node(k2_node, f'K Projection L2\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge(ln3, k2_node)
        
        v2_node = f'v2_proj_{gpu_id}'
        dot.node(v2_node, f'V Projection L2\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        dot.edge(ln3, v2_node)
        
        attn2_node = f'attn2_{gpu_id}'
        dot.node(attn2_node, f'Scaled Dot-Product Attention L2\\nGPU:{gpu_id}\\nInput: [1024,10000,4,128] x3\\nOutput: [1024,10000,4,128]', 
                 shape='rectangle', style='filled', fillcolor='lightcoral')
        dot.edge(q2_node, attn2_node)
        dot.edge(k2_node, attn2_node)
        dot.edge(v2_node, attn2_node)
    
    # Attention Output Aggregation Layer 2
    for group_id in range(4):
        for dim_seg in range(4):
            concat2_dim_node = f'concat2_dim_{group_id}_{dim_seg}'
            dot.node(concat2_dim_node, 
                     f'Concatenate Dimensions L2\\nGroup:{group_id}, Dim:{dim_seg}\\nInput: [1024,10000,4,128] x4\\nOutput: [1024,10000,4,512]', 
                     shape='parallelogram', style='filled', fillcolor='orange')
            
            for gpu_offset in range(4):
                gpu_id = group_id * 4 + gpu_offset
                dot.edge(f'attn2_{gpu_id}', concat2_dim_node)
    
    for head_group in range(4):
        concat2_heads_node = f'concat2_heads_{head_group}'
        dot.node(concat2_heads_node, 
                 f'Concatenate Head Groups L2\\nGroup:{head_group}\\nInput: [1024,10000,4,512]\\nOutput: [1024,10000,2048]', 
                 shape='parallelogram', style='filled', fillcolor='gold')
        
        for dim_seg in range(4):
            concat2_dim_node = f'concat2_dim_{head_group}_{dim_seg}'
            dot.edge(concat2_dim_node, concat2_heads_node)
    
    final2_concat_attn = 'final2_concat_attn'
    dot.node(final2_concat_attn, 
             'Final Attention Concatenation L2\\nInput: [1024,10000,2048] x4\\nOutput: [1024,10000,8192]', 
             shape='parallelogram', style='filled', fillcolor='darkorange')
    
    for head_group in range(4):
        dot.edge(f'concat2_heads_{head_group}', final2_concat_attn)
    
    attn2_out_proj = 'attn2_out_proj'
    dot.node(attn2_out_proj, 
             'Attention Output Projection L2\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(final2_concat_attn, attn2_out_proj)
    
    # Residual Connection 3
    residual3 = 'residual3'
    dot.node(residual3, 'Residual Add 3\\nInput: [1024,10000,8192] x2\\nOutput: [1024,10000,8192]', 
             shape='ellipse', style='filled', fillcolor='lightpink')
    dot.edge(residual2, residual3, style='dashed')
    dot.edge(attn2_out_proj, residual3)
    
    # MLP Layer 2
    ln4 = 'ln4'
    dot.node(ln4, 'Layer Norm 4\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(residual3, ln4)
    
    for gpu_id in range(16):
        mlp2_col = f'mlp2_col_{gpu_id}'
        dot.node(mlp2_col, 
                 f'MLP Column Parallel L2\\nGPU:{gpu_id}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,2048]', 
                 shape='rectangle', style='filled', fillcolor='lightblue')
        dot.edge(ln4, mlp2_col)
        
        gelu2 = f'gelu2_{gpu_id}'
        dot.node(gelu2, f'GELU Activation L2\\nGPU:{gpu_id}\\nInput: [1024,10000,2048]\\nOutput: [1024,10000,2048]', 
                 shape='rectangle', style='filled', fillcolor='lightcyan')
        dot.edge(mlp2_col, gelu2)
        
        mlp2_row = f'mlp2_row_{gpu_id}'
        dot.node(mlp2_row, 
                 f'MLP Row Parallel L2\\nGPU:{gpu_id}\\nInput: [1024,10000,2048]\\nOutput: [1024,10000,512]', 
                 shape='rectangle', style='filled', fillcolor='lightblue')
        dot.edge(gelu2, mlp2_row)
    
    mlp2_concat = 'mlp2_concat'
    dot.node(mlp2_concat, 
             'MLP Output Concatenation L2\\nInput: [1024,10000,512] x16\\nOutput: [1024,10000,8192]', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    for gpu_id in range(16):
        dot.edge(f'mlp2_row_{gpu_id}', mlp2_concat)
    
    mlp2_out_proj = 'mlp2_out_proj'
    dot.node(mlp2_out_proj, 
             'MLP Output Projection L2\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(mlp2_concat, mlp2_out_proj)
    
    # Final Residual Connection
    residual4 = 'residual4'
    dot.node(residual4, 'Residual Add 4\\nInput: [1024,10000,8192] x2\\nOutput: [1024,10000,8192]', 
             shape='ellipse', style='filled', fillcolor='lightpink')
    dot.edge(residual3, residual4, style='dashed')
    dot.edge(mlp2_out_proj, residual4)
    
    # Global Output
    output = 'output'
    dot.node(output, 'Global Output\\nInput: [1024,10000,8192]\\nOutput: [batch_size=1024, seq_len=10000, hidden_size=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge(residual4, output)
    
    return dot

def create_helix_partitioning_detailed():
    """Create detailed view showing the partitioning scheme"""
    
    dot = graphviz.Digraph('Helix_Partitioning_Scheme', 
                          comment='Helix Two-Level Partitioning: 4x4 Grid on 16 GPUs',
                          format='dot')
    
    dot.attr(rankdir='LR', splines='ortho', ranksep='1.5')
    dot.attr('node', fontname='Arial', fontsize='12')
    
    # Input dimensions
    dot.node('input_tensor', 'Input Tensor\\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Partitioning explanation
    dot.node('partition_heads', 'Head-Level Partitioning\\n16 heads → 4 groups\\n4 heads per group', 
             shape='rectangle', style='filled', fillcolor='yellow')
    
    dot.node('partition_dims', 'Intra-Head Dimension Partitioning\\n512 dim/head → 4 segments\\n128 dim per segment', 
             shape='rectangle', style='filled', fillcolor='yellow')
    
    # GPU grid visualization
    with dot.subgraph(name='cluster_gpu_grid') as c:
        c.attr(label='16 GPU Grid (4×4)', style='dashed', color='red')
        
        # Create 4x4 grid
        for row in range(4):
            for col in range(4):
                gpu_id = row * 4 + col
                c.node(f'gpu_{gpu_id}', f'GPU {gpu_id}\\nHead Group {row}\\nDim Segment {col}\\n4 heads × 128 dim', 
                      shape='box', style='filled', fillcolor='lightgreen')
    
    # Connect partitioning to GPUs
    dot.edge('input_tensor', 'partition_heads')
    dot.edge('partition_heads', 'partition_dims')
    dot.edge('partition_dims', 'gpu_0')
    
    # Add connections showing the flow
    for i in range(16):
        dot.edge(f'gpu_{i}', 'output_aggregation', ltail='cluster_gpu_grid')
    
    dot.node('output_aggregation', 'Output Aggregation\\nHierarchical Concatenation', 
             shape='parallelogram', style='filled', fillcolor='orange')
    
    dot.node('final_output', 'Final Output\\n[1024, 10000, 8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.edge('output_aggregation', 'final_output')
    
    return dot

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("./generated_docs/helix", exist_ok=True)
    
    # Generate complete DAG
    dag = create_helix_dag()
    dag.render('./generated_docs/helix/helix_complete_dag', format='svg', cleanup=False)
    dag.render('./generated_docs/helix/helix_complete_dag', format='dot', cleanup=False)
    
    # Generate partitioning scheme diagram
    part_dag = create_helix_partitioning_detailed()
    part_dag.render('./generated_docs/helix/helix_partitioning_scheme', format='svg', cleanup=False)
    part_dag.render('./generated_docs/helix/helix_partitioning_scheme', format='dot', cleanup=False)
    
    print("DAGs generated successfully!")
    print("Files saved to ./generated_docs/helix/")
    
    # Verify DAG structure
    try:
        from graphviz import Source
        # Test loading the generated files
        with open('./generated_docs/helix/helix_complete_dag.dot', 'r') as f:
            content = f.read()
            print(f"Complete DAG has {content.count('->')} edges and {content.count('[')} nodes")
    except Exception as e:
        print(f"Verification warning: {e}")