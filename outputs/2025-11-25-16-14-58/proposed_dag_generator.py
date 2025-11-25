import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE DAG with EP=16')
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Layer 0 (representative layer with full EP=16)
    with dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0 (16 Experts, 1 per GPU)', style='dashed', color='purple')
        
        # MHA (no TP, single GPU for simplicity)
        layer0.node('layer0_mha_qkv', 'MHA QKV Linear\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPU: 0', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
        layer0.node('layer0_mha_attn', 'MHA Attention\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 32×128]\\nGPU: 0', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
        layer0.node('layer0_mha_out', 'MHA Output Linear\\nInput: [batch=128, seq=10000, 32×128]\\nOutput: [batch=128, seq=10000, 4096]\\nGPU: 0', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
        layer0.node('layer0_mha_res', 'MHA Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Routing and expert distribution
        layer0.node('layer0_gate', 'Expert Gate\\nGating (Top-2)\\nInput: [batch=128, seq=10000, 4096]\\nOutput: [routing=16, per_expert_tokens]\\nGPU: 0', 
                   shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Expert clusters - show 4 representative experts
        with layer0.subgraph(name='cluster_experts') as experts:
            experts.attr(label='16 Experts Across GPUs', style='dotted')
            
            # Show expert 0
            experts.node('layer0_exp0', 'Expert 0\\nInput: [tokens_to_exp0, 4096]\\nOutput: [tokens_from_exp0, 4096]\\nGPU: 0', 
                        shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Show expert 1  
            experts.node('layer0_exp1', 'Expert 1\\nInput: [tokens_to_exp1, 4096]\\nOutput: [tokens_from_exp1, 4096]\\nGPU: 1', 
                        shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Show expert 8
            experts.node('layer0_exp8', 'Expert 8\\nInput: [tokens_to_exp8, 4096]\\nOutput: [tokens_from_exp8, 4096]\\nGPU: 8', 
                        shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Show expert 15
            experts.node('layer0_exp15', 'Expert 15\\nInput: [tokens_to_exp15, 4096]\\nOutput: [tokens_from_exp15, 4096]\\nGPU: 15', 
                         shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Token aggregation
        layer0.node('layer0_aggregate', 'Token Aggregation\\nGather from all experts\\nInput: [per_expert_outputs]\\nOutput: [batch=128, seq=10000, 4096]\\nGPU: 0', 
                   shape='parallelogram', style='filled', fillcolor='lightyellow')
        layer0.node('layer0_exp_res', 'Expert Residual Add\\nInput: [batch=128, seq=10000, 4096]×2\\nOutput: [batch=128, seq=10000, 4096]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Communication nodes for cross-GPU transfers
    with dot.subgraph(name='cluster_communication') as comm:
        comm.attr(label='Cross-GPU Communication (NCCL)', style='dashed', color='red')
        
        # Show representative communications
        comm.node('comm_0_to_1', 'Token Send\\nGPU 0 → GPU 1\\n[variable_tokens, 4096]', 
                 shape='ellipse', style='filled', fillcolor='orange')
        comm.node('comm_0_to_8', 'Token Send\\nGPU 0 → GPU 8\\n[variable_tokens, 4096]', 
                 shape='ellipse', style='filled', fillcolor='orange')
        
        comm.node('comm_1_to_0', 'Expert Result\\nGPU 1 → GPU 0\\n[processed_tokens, 4096]', 
                 shape='ellipse', style='filled', fillcolor='orange')
        comm.node('comm_8_to_0', 'Expert Result\\nGPU 8 → GPU 0\\n[processed_tokens, 4096]', 
                 shape='ellipse', style='filled', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections for Layer 0
    dot.edge('input', 'layer0_mha_qkv')
    dot.edge('layer0_mha_qkv', 'layer0_mha_attn')
    dot.edge('layer0_mha_attn', 'layer0_mha_out')
    dot.edge('layer0_mha_out', 'layer0_mha_res')
    dot.edge('input', 'layer0_mha_res', style='dashed', label='residual')
    
    # Routing connections
    dot.edge('layer0_mha_res', 'layer0_gate')
    dot.edge('layer0_gate', 'layer0_exp0', style='dashed', label='route tokens')
    dot.edge('layer0_gate', 'layer0_exp1', style='dashed', label='route tokens')
    dot.edge('layer0_gate', 'layer0_exp8', style='dashed', label='route tokens')
    dot.edge('layer0_gate', 'layer0_exp15', style='dashed', label='route tokens')
    
    # Communication connections
    dot.edge('layer0_gate', 'comm_0_to_1', style='dotted')
    dot.edge('layer0_gate', 'comm_0_to_8', style='dotted')
    dot.edge('comm_0_to_1', 'layer0_exp1', style='dotted', label='tokens')
    dot.edge('comm_0_to_8', 'layer0_exp8', style='dotted', label='tokens')
    
    dot.edge('layer0_exp1', 'comm_1_to_0', style='dotted')
    dot.edge('layer0_exp8', 'comm_8_to_0', style='dotted')
    dot.edge('comm_1_to_0', 'layer0_aggregate', style='dotted', label='results')
    dot.edge('comm_8_to_0', 'layer0_aggregate', style='dotted', label='results')
    
    # Back to aggregation
    dot.edge('layer0_exp0', 'layer0_aggregate')
    dot.edge('layer0_exp15', 'layer0_aggregate')
    dot.edge('layer0_aggregate', 'layer0_exp_res')
    dot.edge('layer0_mha_res', 'layer0_exp_res', style='dashed', label='residual')
    
    # Continue to next layer (simplified)
    dot.edge('layer0_exp_res', 'output', label='After 16 similar layers')
    
    # Add notes for scalability
    dot.node('note1', 'Note: Layer 1-15\\nrepeat similar patterns\\nwith different GPU mappings\\n(layer_x: GPUs 16-31, etc.)', 
             shape='note', style='dashed')
    dot.node('note2', 'Key Innovation: One expert per GPU\\nMinimizes contention\\nEnables async communication', 
             shape='note', style='filled', fillcolor='lightgray')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    dag.render('../outputs/2025-11-25-16-14-58/proposed_moe_dag', format='svg', cleanup=False)
    
    # Save dot file
    with open('../outputs/2025-11-25-16-14-58/proposed_moe_dag.dot', 'w') as f:
        f.write(dag.source)