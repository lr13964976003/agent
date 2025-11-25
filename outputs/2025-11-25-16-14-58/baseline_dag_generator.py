import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE DAG with TP=8, PP=2')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Define colors for different types of nodes
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline Stage 0 (Layers 0-7)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-7)', style='dashed', color='blue')
        c.node('stage0_input', 'Stage0 Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
               shape='ellipse', style='filled', fillcolor='lightyellow')
        
        # Layer 0 (representative layer)
        with c.subgraph(name='cluster_layer0') as layer0:
            layer0.attr(label='Layer 0', style='rounded', color='black')
            
            # Multi-Head Attention (8-way tensor parallel)
            layer0.node('layer0_mha_qkv', 'MHA QKV Linear\\nTP=8\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPUs: [0,1,2,3,4,5,6,7]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer0.node('layer0_mha_attn', 'MHA Attention\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPUs: [0,1,2,3,4,5,6,7]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer0.node('layer0_mha_out', 'MHA Output Linear\\nTP=8\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 4096]\\nGPUs: [0,1,2,3,4,5,6,7]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer0.node('layer0_mha_res', 'MHA Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Expert Layer (16 experts on each GPU)
            layer0.node('layer0_gate', 'Expert Gate\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, routing=16]\\nGPUs: [0,1,2,3,4,5,6,7]', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
            layer0.node('layer0_experts', 'Expert Layer\\n16 Experts per GPU\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, 4096]\\nGPUs: [0,1,2,3,4,5,6,7]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer0.node('layer0_exp_res', 'Expert Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Pipeline Stage 1 (Layers 8-15)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 8-15)', style='dashed', color='red')
        c.node('stage1_input', 'Stage1 Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
               shape='ellipse', style='filled', fillcolor='lightyellow')
        
        # Layer 8 (representative layer)
        with c.subgraph(name='cluster_layer8') as layer8:
            layer8.attr(label='Layer 8', style='rounded', color='black')
            
            # Multi-Head Attention (8-way tensor parallel)
            layer8.node('layer8_mha_qkv', 'MHA QKV Linear\\nTP=8\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPUs: [8,9,10,11,12,13,14,15]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer8.node('layer8_mha_attn', 'MHA Attention\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPUs: [8,9,10,11,12,13,14,15]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer8.node('layer8_mha_out', 'MHA Output Linear\\nTP=8\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 4096]\\nGPUs: [8,9,10,11,12,13,14,15]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer8.node('layer8_mha_res', 'MHA Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Expert Layer (16 experts on each GPU)
            layer8.node('layer8_gate', 'Expert Gate\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, routing=16]\\nGPUs: [8,9,10,11,12,13,14,15]', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
            layer8.node('layer8_experts', 'Expert Layer\\n16 Experts per GPU\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, 4096]\\nGPUs: [8,9,10,11,12,13,14,15]', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            layer8.node('layer8_exp_res', 'Expert Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections
    dot.edge('input', 'stage0_input', label='Initial Input')
    
    # Layer 0 connections
    dot.edge('stage0_input', 'layer0_mha_qkv')
    dot.edge('layer0_mha_qkv', 'layer0_mha_attn')
    dot.edge('layer0_mha_attn', 'layer0_mha_out')
    dot.edge('layer0_mha_out', 'layer0_mha_res')
    dot.edge('stage0_input', 'layer0_mha_res', style='dashed', label='residual')
    
    dot.edge('layer0_mha_res', 'layer0_gate')
    dot.edge('layer0_gate', 'layer0_experts')
    dot.edge('layer0_experts', 'layer0_exp_res')
    dot.edge('layer0_mha_res', 'layer0_exp_res', style='dashed', label='residual')
    
    # Pipeline communication (simplified to show concept)
    dot.edge('layer0_exp_res', 'stage1_input', 
             label='Pipeline Send\\n[batch=128, seq=10000, 4096]\\nGPU7→GPU8', 
             style='dotted', color='blue')
    
    # Layer 8 connections
    dot.edge('stage1_input', 'layer8_mha_qkv')
    dot.edge('layer8_mha_qkv', 'layer8_mha_attn')
    dot.edge('layer8_mha_attn', 'layer8_mha_out')
    dot.edge('layer8_mha_out', 'layer8_mha_res')
    dot.edge('stage1_input', 'layer8_mha_res', style='dashed', label='residual')
    
    dot.edge('layer8_mha_res', 'layer8_gate')
    dot.edge('layer8_gate', 'layer8_experts')
    dot.edge('layer8_experts', 'layer8_exp_res')
    dot.edge('layer8_mha_res', 'layer8_exp_res', style='dashed', label='residual')
    
    # Final output
    dot.edge('layer8_exp_res', 'output')
    
    # Add note about repetition
    dot.node('note', 'Note: Layers 1-7 and 9-15\\nrepeat similar patterns\\nwith same device mappings', 
             shape='note', style='dashed')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('../outputs/2025-11-25-16-14-58/baseline_moe_dag', format='svg', cleanup=False)
    
    # Save dot file
    with open('../outputs/2025-11-25-16-14-58/baseline_moe_dag.dot', 'w') as f:
        f.write(dag.source)