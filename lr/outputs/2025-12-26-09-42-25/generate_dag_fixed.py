#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_parallelism_dag():
    """
    Generate a complete DAG for the hybrid expert-tensor-pipeline parallelism strategy.
    This represents a 64-GPU deployment with:
    - 8-way Expert Parallelism
    - 4-way Tensor Parallelism  
    - 2-way Pipeline Parallelism
    """
    
    # Create the main graph
    dot = Digraph(comment='Hybrid Expert-Tensor-Pipeline Parallelism DAG')
    dot.attr(rankdir='TB', size='200,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define styles for different node types
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
             shape='ellipse', fillcolor='lightgray')
    
    # Pipeline Stage 0: Layers 0-7
    with dot.subgraph(name='cluster_pipeline_0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-7)', style='rounded,filled', fillcolor='lightcyan')
        
        # Expert Groups 0-7 (Pipeline Stage 0)
        for expert_group in range(8):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p0') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {expert_group*4}-{expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightpink')
                
                # Expert routing (gate) - dashed line style
                ec.node(f'gate_e{expert_group}_p0', 
                       f'Gate Selection\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
                       shape='parallelogram')
                
                # Hierarchical All-to-All Communication - Local
                ec.node(f'all2all_local_e{expert_group}_p0',
                       f'Local All-to-All\\nGPU: {expert_group*4}-{expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=2560, hidden=512]',
                       shape='ellipse')
                
                # Tensor parallel groups within expert
                for tensor_group in range(4):
                    gpu_id = expert_group * 4 + tensor_group
                    
                    # Attention computation
                    ec.node(f'attention_e{expert_group}_t{tensor_group}_p0',
                           f'Attention\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, heads=4, d_k=32]\\nOutput: [batch_size=32, seq_len=2560, heads=4, d_k=32]',
                           shape='box')
                    
                    # MoE computation (2 experts per GPU)
                    for expert_id in range(2):
                        ec.node(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0',
                               f'MoE Expert {expert_id}\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2560, hidden=256]\\nOutput: [batch_size=16, seq_len=2560, hidden=256]',
                               shape='box')
                    
                    # Tensor reduction
                    ec.node(f'tensor_reduce_e{expert_group}_t{tensor_group}_p0',
                           f'Tensor Reduction\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, hidden=256]\\nOutput: [batch_size=32, seq_len=2560, hidden=256]',
                           shape='ellipse')
    
    # Pipeline Stage 1: Layers 8-15
    with dot.subgraph(name='cluster_pipeline_1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 8-15)', style='rounded,filled', fillcolor='lightsteelblue')
        
        # Expert Groups 0-7 (Pipeline Stage 1)
        for expert_group in range(8):
            with c.subgraph(name=f'cluster_expert_{expert_group}_p1') as ec:
                ec.attr(label=f'Expert Group {expert_group} (GPU {32+expert_group*4}-{32+expert_group*4+3})', 
                       style='rounded,filled', fillcolor='lightcoral')
                
                # Expert routing (gate)
                ec.node(f'gate_e{expert_group}_p1', 
                       f'Gate Selection\\nGPU: {32+expert_group*4}-{32+expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', 
                       shape='parallelogram')
                
                # Hierarchical All-to-All Communication - Local
                ec.node(f'all2all_local_e{expert_group}_p1',
                       f'Local All-to-All\\nGPU: {32+expert_group*4}-{32+expert_group*4+3}\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=2560, hidden=512]',
                       shape='ellipse')
                
                # Tensor parallel groups within expert
                for tensor_group in range(4):
                    gpu_id = 32 + expert_group * 4 + tensor_group
                    
                    # Attention computation
                    ec.node(f'attention_e{expert_group}_t{tensor_group}_p1',
                           f'Attention\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, heads=4, d_k=32]\\nOutput: [batch_size=32, seq_len=2560, heads=4, d_k=32]',
                           shape='box')
                    
                    # MoE computation (2 experts per GPU)
                    for expert_id in range(2):
                        ec.node(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p1',
                               f'MoE Expert {expert_id}\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2560, hidden=256]\\nOutput: [batch_size=16, seq_len=2560, hidden=256]',
                               shape='box')
                    
                    # Tensor reduction
                    ec.node(f'tensor_reduce_e{expert_group}_t{tensor_group}_p1',
                           f'Tensor Reduction\\nGPU: {gpu_id}\\nInput: [batch_size=32, seq_len=2560, hidden=256]\\nOutput: [batch_size=32, seq_len=2560, hidden=256]',
                           shape='ellipse')
    
    # Global communication nodes
    dot.node('all2all_global_p0', 'Global All-to-All\\nGPU: 0-31\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', shape='ellipse', fillcolor='lightblue')
    dot.node('all2all_global_p1', 'Global All-to-All\\nGPU: 32-63\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', shape='ellipse', fillcolor='lightblue')
    
    # Pipeline transfer nodes
    dot.node('pipeline_transfer', 'Pipeline Transfer\\nGPU: 0-31 → 32-63\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]', shape='ellipse', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, vocab_size=50000]', 
             shape='ellipse', fillcolor='lightgray')
    
    # ===== CONNECTIONS =====
    
    # Input to Pipeline Stage 0
    dot.edge('input', 'all2all_global_p0')
    
    # Connect global all-to-all to each expert group in pipeline 0
    for expert_group in range(8):
        dot.edge('all2all_global_p0', f'gate_e{expert_group}_p0')
    
    # Pipeline Stage 0 internal connections
    for expert_group in range(8):
        # Gate to local all-to-all
        dot.edge(f'gate_e{expert_group}_p0', f'all2all_local_e{expert_group}_p0')
        
        # Local all-to-all to attention (distributed across tensor groups)
        for tensor_group in range(4):
            dot.edge(f'all2all_local_e{expert_group}_p0', f'attention_e{expert_group}_t{tensor_group}_p0')
            
            # Attention to MoE experts
            for expert_id in range(2):
                dot.edge(f'attention_e{expert_group}_t{tensor_group}_p0', f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0')
            
            # MoE experts to tensor reduction
            for expert_id in range(2):
                dot.edge(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0', f'tensor_reduce_e{expert_group}_t{tensor_group}_p0')
    
    # Pipeline Stage 0 to Pipeline Transfer
    for expert_group in range(8):
        for tensor_group in range(4):
            dot.edge(f'tensor_reduce_e{expert_group}_t{tensor_group}_p0', 'pipeline_transfer')
    
    # Pipeline Transfer to Pipeline Stage 1
    dot.edge('pipeline_transfer', 'all2all_global_p1')
    
    # Connect global all-to-all to each expert group in pipeline 1
    for expert_group in range(8):
        dot.edge('all2all_global_p1', f'gate_e{expert_group}_p1')
    
    # Pipeline Stage 1 internal connections (same pattern as stage 0)
    for expert_group in range(8):
        # Gate to local all-to-all
        dot.edge(f'gate_e{expert_group}_p1', f'all2all_local_e{expert_group}_p1')
        
        # Local all-to-all to attention
        for tensor_group in range(4):
            dot.edge(f'all2all_local_e{expert_group}_p1', f'attention_e{expert_group}_t{tensor_group}_p1')
            
            # Attention to MoE experts
            for expert_id in range(2):
                dot.edge(f'attention_e{expert_group}_t{tensor_group}_p1', f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p1')
            
            # MoE experts to tensor reduction
            for expert_id in range(2):
                dot.edge(f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p1', f'tensor_reduce_e{expert_group}_t{tensor_group}_p1')
    
    # Pipeline Stage 1 to Output
    for expert_group in range(8):
        for tensor_group in range(4):
            dot.edge(f'tensor_reduce_e{expert_group}_t{tensor_group}_p1', 'output')
    
    # Add dashed lines for gate selection (expert routing decisions)
    dot.attr('edge', style='dashed')
    for expert_group in range(8):
        for tensor_group in range(4):
            # Dashed connection from gate to specific experts
            for expert_id in range(2):
                dot.edge(f'gate_e{expert_group}_p0', f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p0')
                dot.edge(f'gate_e{expert_group}_p1', f'moe_e{expert_group}_t{tensor_group}_e{expert_id}_p1')
    
    return dot

if __name__ == '__main__':
    dag = create_parallelism_dag()
    
    # Save as DOT file
    dag.save('./outputs/2025-12-26-09-42-25/parallelism_dag.dot')
    
    # Render as SVG image
    dag.render('./outputs/2025-12-26-09-42-25/parallelism_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ./outputs/2025-12-26-09-42-25/parallelism_dag.dot")
    print(f"SVG image: ./outputs/2025-12-26-09-42-25/parallelism_dag.svg")