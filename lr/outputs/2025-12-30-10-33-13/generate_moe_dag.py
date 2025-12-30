#!/usr/bin/env python3

import graphviz

def create_moe_parallelism_dag():
    """
    Create a comprehensive DAG for MoE model with:
    - DP=4 (4 replicas)
    - PP=2 (2 stages per replica)
    - EP=16 (16 experts per stage, 1 per GPU)
    - TP=4 (4-way tensor parallelism for attention)
    - 128 GPUs total
    """
    
    # Create the main graph
    dot = graphviz.Digraph(comment='MoE Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=4, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=4, seq_len=1024, heads=16, d_k=32]', 
             shape='ellipse', fillcolor='lightblue')
    
    # DP level - 4 replicas
    for dp_id in range(4):
        dp_cluster = f'dp_{dp_id}'
        
        # PP Stage 1 (Layers 1-8)
        stage1 = f'stage1_dp{dp_id}'
        
        # Create TP groups for attention in stage 1
        for tp_group in range(4):  # 4 TP groups per stage
            tp_group_id = f'tp{tp_group}_stage1_dp{dp_id}'
            
            # Attention operation with TP=4
            dot.node(f'attn_stage1_tp{tp_group}_dp{dp_id}', 
                    f'Attention Stage1 TP{tp_group}\\nGPU: {dp_id*32 + tp_group*4}-{dp_id*32 + tp_group*4 + 3}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=4, d_k=32]',
                    shape='rectangle', fillcolor='lightgreen')
            
            # AllReduce for TP
            dot.node(f'allreduce_stage1_tp{tp_group}_dp{dp_id}', 
                    f'AllReduce Stage1 TP{tp_group}\\nGPU: {dp_id*32 + tp_group*4}-{dp_id*32 + tp_group*4 + 3}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                    shape='ellipse', fillcolor='lightblue')
        
        # MoE Layer with EP=16 for Stage 1
        dot.node(f'router_stage1_dp{dp_id}', 
                f'Router Stage1\\nGPU: {dp_id*32}-{dp_id*32 + 15}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                shape='parallelogram', fillcolor='lightyellow')
        
        # Expert nodes for Stage 1 (16 experts, 1 per GPU)
        for expert_id in range(16):
            gpu_id = dp_id * 32 + expert_id
            dot.node(f'expert_stage1_{expert_id}_dp{dp_id}', 
                    f'Expert {expert_id} Stage1\\nGPU: {gpu_id}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                    shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation for Stage 1
        dot.node(f'agg_stage1_dp{dp_id}', 
                f'Aggregate Stage1\\nGPU: {dp_id*32}-{dp_id*32 + 15}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                shape='parallelogram', fillcolor='lightyellow')
        
        # PP Stage 2 (Layers 9-16)
        # Create TP groups for attention in stage 2
        for tp_group in range(4):  # 4 TP groups per stage
            tp_group_id = f'tp{tp_group}_stage2_dp{dp_id}'
            
            # Attention operation with TP=4
            dot.node(f'attn_stage2_tp{tp_group}_dp{dp_id}', 
                    f'Attention Stage2 TP{tp_group}\\nGPU: {dp_id*32 + 16 + tp_group*4}-{dp_id*32 + 16 + tp_group*4 + 3}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=4, d_k=32]',
                    shape='rectangle', fillcolor='lightgreen')
            
            # AllReduce for TP
            dot.node(f'allreduce_stage2_tp{tp_group}_dp{dp_id}', 
                    f'AllReduce Stage2 TP{tp_group}\\nGPU: {dp_id*32 + 16 + tp_group*4}-{dp_id*32 + 16 + tp_group*4 + 3}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                    shape='ellipse', fillcolor='lightblue')
        
        # MoE Layer with EP=16 for Stage 2
        dot.node(f'router_stage2_dp{dp_id}', 
                f'Router Stage2\\nGPU: {dp_id*32 + 16}-{dp_id*32 + 31}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                shape='parallelogram', fillcolor='lightyellow')
        
        # Expert nodes for Stage 2 (16 experts, 1 per GPU)
        for expert_id in range(16):
            gpu_id = dp_id * 32 + 16 + expert_id
            dot.node(f'expert_stage2_{expert_id}_dp{dp_id}', 
                    f'Expert {expert_id} Stage2\\nGPU: {gpu_id}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                    shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation for Stage 2
        dot.node(f'agg_stage2_dp{dp_id}', 
                f'Aggregate Stage2\\nGPU: {dp_id*32 + 16}-{dp_id*32 + 31}\\nInput: [batch_size=1, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=16, d_k=32]',
                shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=4, seq_len=1024, heads=16, d_k=32]\\nOutput: [batch_size=4, seq_len=1024, heads=16, d_k=32]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connect the graph
    for dp_id in range(4):
        # Input to first stage
        dot.edge('input', f'attn_stage1_tp0_dp{dp_id}')
        
        # Connect attention TP groups within stage 1
        for tp_group in range(4):
            dot.edge(f'attn_stage1_tp{tp_group}_dp{dp_id}', f'allreduce_stage1_tp{tp_group}_dp{dp_id}')
            if tp_group < 3:
                dot.edge(f'allreduce_stage1_tp{tp_group}_dp{dp_id}', f'attn_stage1_tp{tp_group+1}_dp{dp_id}')
        
        # Connect to router after all TP groups complete
        dot.edge(f'allreduce_stage1_tp3_dp{dp_id}', f'router_stage1_dp{dp_id}')
        
        # Connect router to experts (dashed lines for gate selection)
        for expert_id in range(16):
            dot.edge(f'router_stage1_dp{dp_id}', f'expert_stage1_{expert_id}_dp{dp_id}', style='dashed')
        
        # Connect experts to aggregation
        for expert_id in range(16):
            dot.edge(f'expert_stage1_{expert_id}_dp{dp_id}', f'agg_stage1_dp{dp_id}')
        
        # Connect stage 1 to stage 2 (PP communication)
        dot.edge(f'agg_stage1_dp{dp_id}', f'attn_stage2_tp0_dp{dp_id}')
        
        # Connect attention TP groups within stage 2
        for tp_group in range(4):
            dot.edge(f'attn_stage2_tp{tp_group}_dp{dp_id}', f'allreduce_stage2_tp{tp_group}_dp{dp_id}')
            if tp_group < 3:
                dot.edge(f'allreduce_stage2_tp{tp_group}_dp{dp_id}', f'attn_stage2_tp{tp_group+1}_dp{dp_id}')
        
        # Connect to router after all TP groups complete
        dot.edge(f'allreduce_stage2_tp3_dp{dp_id}', f'router_stage2_dp{dp_id}')
        
        # Connect router to experts (dashed lines for gate selection)
        for expert_id in range(16):
            dot.edge(f'router_stage2_dp{dp_id}', f'expert_stage2_{expert_id}_dp{dp_id}', style='dashed')
        
        # Connect experts to aggregation
        for expert_id in range(16):
            dot.edge(f'expert_stage2_{expert_id}_dp{dp_id}', f'agg_stage2_dp{dp_id}')
        
        # Connect to output
        dot.edge(f'agg_stage2_dp{dp_id}', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the DAG
    dag = create_moe_parallelism_dag()
    
    # Save as DOT file
    dag.save('./outputs/2025-12-30-10-33-13/moe_parallelism_dag.dot')
    
    # Render as SVG
    dag.render('./outputs/2025-12-30-10-33-13/moe_parallelism_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ./outputs/2025-12-30-10-33-13/moe_parallelism_dag.dot")
    print(f"SVG file: ./outputs/2025-12-30-10-33-13/moe_parallelism_dag.svg")