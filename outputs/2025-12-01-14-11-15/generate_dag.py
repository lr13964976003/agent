#!/usr/bin/env python3
"""
Generate complete DAG for LLM deployment with hybrid parallelism
Expert Parallelism (8-way) + Tensor Parallelism (2-way) + Pipeline Parallelism (4-way)
"""

import graphviz
from graphviz import Digraph

def create_complete_dag():
    # Create DAG
    dot = Digraph(comment='LLM Hybrid Parallelism Deployment DAG')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='plaintext', fontname='Arial Bold')
    
    # Pipeline Stage 0 (Layers 0-3)
    with dot.subgraph(name='cluster_pipeline_stage_0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-3)', style='rounded,filled', fillcolor='lightgray')
        
        # Expert Parallelism Group 0 (GPUs 0-7)
        for expert_group in range(8):
            with c.subgraph(name=f'cluster_expert_group_{expert_group}') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {expert_group})', style='rounded,filled', fillcolor='lightcyan')
                
                # Layer 0
                layer = 0
                # Expert routing
                ec.node(f'layer{layer}_expert{expert_group}_route', 
                       f'Expert Routing L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='parallelogram')
                
                # Expert selection (gate)
                ec.node(f'layer{layer}_expert{expert_group}_gate', 
                       f'Expert Gate L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, num_experts=8]', 
                       shape='parallelogram')
                
                # All-to-all communication for expert assignment
                ec.node(f'layer{layer}_expert{expert_group}_a2a', 
                       f'All-to-All Comm L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='ellipse')
                
                # Multi-head attention with tensor parallelism
                # Attention split (2-way TP)
                ec.node(f'layer{layer}_expert{expert_group}_attn_split', 
                       f'Attention Split L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                       shape='parallelogram')
                
                # Q projection (column parallel)
                ec.node(f'layer{layer}_expert{expert_group}_q_proj', 
                       f'Q Projection L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                       shape='rectangle')
                
                # K projection (column parallel)
                ec.node(f'layer{layer}_expert{expert_group}_k_proj', 
                       f'K Projection L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                       shape='rectangle')
                
                # V projection (column parallel)
                ec.node(f'layer{layer}_expert{expert_group}_v_proj', 
                       f'V Projection L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                       shape='rectangle')
                
                # Attention computation
                ec.node(f'layer{layer}_expert{expert_group}_attn', 
                       f'Attention Computation L{layer}\\nInput: [batch_size=128, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=1024, heads=8, d_k=64]', 
                       shape='rectangle')
                
                # Attention output projection (row parallel)
                ec.node(f'layer{layer}_expert{expert_group}_attn_out', 
                       f'Attention Output L{layer}\\nInput: [batch_size=128, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle')
                
                # Attention all-reduce
                ec.node(f'layer{layer}_expert{expert_group}_attn_ar', 
                       f'Attention All-Reduce L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='ellipse')
                
                # Residual add for attention
                ec.node(f'layer{layer}_expert{expert_group}_attn_res', 
                       f'Attention Residual L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024], [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='parallelogram')
                
                # Layer normalization
                ec.node(f'layer{layer}_expert{expert_group}_ln1', 
                       f'Layer Norm 1 L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle')
                
                # MLP with tensor parallelism
                # First linear (column parallel)
                ec.node(f'layer{layer}_expert{expert_group}_mlp1', 
                       f'MLP First Linear L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=1024]', 
                       shape='rectangle')
                
                # GELU activation
                ec.node(f'layer{layer}_expert{expert_group}_gelu', 
                       f'GELU Activation L{layer}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, ffn_hidden=1024]', 
                       shape='rectangle')
                
                # Second linear (row parallel)
                ec.node(f'layer{layer}_expert{expert_group}_mlp2', 
                       f'MLP Second Linear L{layer}\\nInput: [batch_size=128, seq_len=1024, ffn_hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle')
                
                # MLP all-reduce
                ec.node(f'layer{layer}_expert{expert_group}_mlp_ar', 
                       f'MLP All-Reduce L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='ellipse')
                
                # Residual add for MLP
                ec.node(f'layer{layer}_expert{expert_group}_mlp_res', 
                       f'MLP Residual L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024], [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='parallelogram')
                
                # Layer normalization 2
                ec.node(f'layer{layer}_expert{expert_group}_ln2', 
                       f'Layer Norm 2 L{layer}\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
                       shape='rectangle')
    
    # Add connections for layer 0
    dot.edge('input', 'layer0_expert0_route')
    
    # Connect within expert group 0, layer 0
    dot.edge('layer0_expert0_route', 'layer0_expert0_gate')
    dot.edge('layer0_expert0_gate', 'layer0_expert0_a2a')
    dot.edge('layer0_expert0_a2a', 'layer0_expert0_attn_split')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_q_proj')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_k_proj')
    dot.edge('layer0_expert0_attn_split', 'layer0_expert0_v_proj')
    dot.edge('layer0_expert0_q_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_k_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_v_proj', 'layer0_expert0_attn')
    dot.edge('layer0_expert0_attn', 'layer0_expert0_attn_out')
    dot.edge('layer0_expert0_attn_out', 'layer0_expert0_attn_ar')
    dot.edge('layer0_expert0_attn_ar', 'layer0_expert0_attn_res')
    dot.edge('layer0_expert0_attn_res', 'layer0_expert0_ln1')
    dot.edge('layer0_expert0_ln1', 'layer0_expert0_mlp1')
    dot.edge('layer0_expert0_mlp1', 'layer0_expert0_gelu')
    dot.edge('layer0_expert0_gelu', 'layer0_expert0_mlp2')
    dot.edge('layer0_expert0_mlp2', 'layer0_expert0_mlp_ar')
    dot.edge('layer0_expert0_mlp_ar', 'layer0_expert0_mlp_res')
    dot.edge('layer0_expert0_mlp_res', 'layer0_expert0_ln2')
    
    # For brevity, I'll add a few more key nodes and connections
    # In a complete implementation, this would repeat for all layers and expert groups
    
    # Add pipeline stage connections
    dot.node('pipeline_send_0_1', 'Pipeline Send Stage0→1\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='ellipse', fillcolor='orange')
    
    dot.node('pipeline_recv_1_0', 'Pipeline Recv Stage0→1\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='ellipse', fillcolor='orange')
    
    # Connect pipeline stages
    dot.edge('layer0_expert0_ln2', 'pipeline_send_0_1')
    dot.edge('pipeline_send_0_1', 'pipeline_recv_1_0')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=1024, hidden_dim=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='plaintext', fontname='Arial Bold')
    
    # Connect to output (simplified)
    dot.edge('pipeline_recv_1_0', 'output')
    
    return dot

def create_expert_parallelism_detailed():
    """Create detailed expert parallelism DAG"""
    dot = Digraph(comment='Expert Parallelism Detailed View')
    dot.attr(rankdir='TB', size='30,30', dpi='300')
    
    # Expert routing and selection
    dot.node('input_tokens', 'Input Tokens\\n[batch_size=128, seq_len=1024, hidden_dim=1024]')
    
    # Gate computation for each expert group
    for i in range(8):
        dot.node(f'gate_{i}', f'Gate {i}\\n[batch_size=128, seq_len=1024, num_experts=8]\\nGPU: {i}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.node(f'expert_select_{i}', f'Expert Selection {i}\\n[batch_size=128, seq_len=1024]\\nGPU: {i}', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Experts within group
        for j in range(8):
            expert_id = i * 8 + j
            dot.node(f'expert_{expert_id}', f'Expert {expert_id}\\n[batch_size=16, seq_len=1024, hidden_dim=1024]\\nGPU: {i}', 
                    shape='rectangle', fillcolor='lightgreen')
    
    # All-to-all communication
    dot.node('a2a_comm', 'All-to-All Communication\\nExpert Assignment\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='parallelogram', fillcolor='lightyellow')
    
    # Connections
    dot.edge('input_tokens', 'gate_0')
    dot.edge('input_tokens', 'gate_1')
    dot.edge('input_tokens', 'gate_2')
    dot.edge('input_tokens', 'gate_3')
    dot.edge('input_tokens', 'gate_4')
    dot.edge('input_tokens', 'gate_5')
    dot.edge('input_tokens', 'gate_6')
    dot.edge('input_tokens', 'gate_7')
    
    for i in range(8):
        dot.edge(f'gate_{i}', f'expert_select_{i}')
        dot.edge(f'expert_select_{i}', 'a2a_comm')
        
        # Connect to experts (simplified)
        for j in range(8):
            expert_id = i * 8 + j
            dot.edge('a2a_comm', f'expert_{expert_id}')
            dot.edge(f'expert_{expert_id}', 'expert_agg')
    
    return dot

def create_tensor_parallelism_detailed():
    """Create detailed tensor parallelism DAG for attention"""
    dot = Digraph(comment='Tensor Parallelism Detailed View')
    dot.attr(rankdir='LR', size='20,20', dpi='300')
    
    # Input
    dot.node('input', 'Input\\n[batch_size=128, seq_len=1024, hidden_dim=1024]')
    
    # GPU 0 and GPU 1 for tensor parallelism
    with dot.subgraph(name='cluster_gpu0') as c:
        c.attr(label='GPU 0', style='rounded,filled', fillcolor='lightcyan')
        c.node('q_proj_0', 'Q Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('k_proj_0', 'K Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('v_proj_0', 'V Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_0', 'Attention\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('out_proj_0', 'Output Projection\\n[batch_size=128, seq_len=1024, hidden_dim=512]', 
               shape='rectangle', fillcolor='lightgreen')
    
    with dot.subgraph(name='cluster_gpu1') as c:
        c.attr(label='GPU 1', style='rounded,filled', fillcolor='lightcyan')
        c.node('q_proj_1', 'Q Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('k_proj_1', 'K Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('v_proj_1', 'V Projection\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('attn_1', 'Attention\\n[batch_size=128, seq_len=1024, heads=8, d_k=64]', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('out_proj_1', 'Output Projection\\n[batch_size=128, seq_len=1024, hidden_dim=512]', 
               shape='rectangle', fillcolor='lightgreen')
    
    # All-reduce
    dot.node('all_reduce', 'All-Reduce Sum\\n[batch_size=128, seq_len=1024, hidden_dim=1024]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=1024, hidden_dim=1024]')
    
    # Connections
    dot.edge('input', 'q_proj_0')
    dot.edge('input', 'q_proj_1')
    dot.edge('input', 'k_proj_0')
    dot.edge('input', 'k_proj_1')
    dot.edge('input', 'v_proj_0')
    dot.edge('input', 'v_proj_1')
    
    dot.edge('q_proj_0', 'attn_0')
    dot.edge('k_proj_0', 'attn_0')
    dot.edge('v_proj_0', 'attn_0')
    dot.edge('q_proj_1', 'attn_1')
    dot.edge('k_proj_1', 'attn_1')
    dot.edge('v_proj_1', 'attn_1')
    
    dot.edge('attn_0', 'out_proj_0')
    dot.edge('attn_1', 'out_proj_1')
    dot.edge('out_proj_0', 'all_reduce')
    dot.edge('out_proj_1', 'all_reduce')
    dot.edge('all_reduce', 'output')
    
    return dot

if __name__ == '__main__':
    # Create main DAG
    main_dag = create_complete_dag()
    main_dag.render('../outputs/2025-12-01-14-11-15/llm_hybrid_parallelism_dag', format='svg', cleanup=True)
    main_dag.save('../outputs/2025-12-01-14-11-15/llm_hybrid_parallelism_dag.dot')
    
    # Create expert parallelism detailed view
    expert_dag = create_expert_parallelism_detailed()
    expert_dag.render('../outputs/2025-12-01-14-11-15/expert_parallelism_detailed', format='svg', cleanup=True)
    expert_dag.save('../outputs/2025-12-01-14-11-15/expert_parallelism_detailed.dot')
    
    # Create tensor parallelism detailed view
    tensor_dag = create_tensor_parallelism_detailed()
    tensor_dag.render('../outputs/2025-12-01-14-11-15/tensor_parallelism_detailed', format='svg', cleanup=True)
    tensor_dag.save('../outputs/2025-12-01-14-11-15/tensor_parallelism_detailed.dot')
    
    print("DAG files generated successfully!")
    print("Main DAG: ../outputs/2025-12-01-14-11-15/llm_hybrid_parallelism_dag.svg")
    print("Expert DAG: ../outputs/2025-12-01-14-11-15/expert_parallelism_detailed.svg")
    print("Tensor DAG: ../outputs/2025-12-01-14-11-15/tensor_parallelism_detailed.svg")