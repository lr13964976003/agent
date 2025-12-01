#!/usr/bin/env python3

import graphviz

def create_expert_parallelism_dag():
    """Create DAG for Cross-Node Expert Parallelism with 16 experts on 16 GPUs"""
    
    dot = graphviz.Digraph('Cross_Node_Expert_Parallelism')
    dot.attr(rankdir='TB', fontsize='12', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    
    # Gating mechanism (routing)
    dot.node('gating', 'Gating Mechanism\\nSelect top-K experts per token\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, num_experts=16]', shape='parallelogram', fillcolor='yellow')
    
    # Token splitting based on expert routing
    dot.node('token_split', 'Token Split by Expert\\nSplit tokens across 16 experts\\n[batch_size=128, seq_len=10000, d_model=4096]→16×[variable_batch, d_model=4096]', shape='parallelogram', fillcolor='yellow')
    
    # Connect input to gating to token split
    dot.edge('input', 'gating')
    dot.edge('gating', 'token_split', style='dashed', label='expert_selection')
    
    # Create 16 expert nodes (one per GPU)
    experts = []
    for i in range(16):
        gpu_id = i
        # Each expert is a complete MLP: 4096->16384->4096
        expert_name = f'expert_{i}_gpu_{gpu_id}'
        dot.node(expert_name, f'Expert {i}\\nGPU {gpu_id}\\nMLP: 4096→16384→4096\\nInput: [variable_batch, d_model=4096]\\nOutput: [variable_batch, d_model=4096]', 
                shape='rectangle', fillcolor='lightgreen')
        experts.append(expert_name)
        
        # Connect token split to each expert
        dot.edge('token_split', expert_name, label=f'expert_{i}_tokens')
    
    # Expert outputs aggregation
    dot.node('expert_agg', 'Expert Output Aggregation\\nCombine outputs from 16 experts\\n16×[variable_batch, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', shape='parallelogram', fillcolor='yellow')
    
    # Connect all experts to aggregation
    for expert in experts:
        dot.edge(expert, 'expert_agg')
    
    # Final output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    dot.edge('expert_agg', 'output')
    
    return dot

def create_expert_parallelism_baseline_dag():
    """Create DAG for Baseline (TP=8, PP=2)"""
    
    dot = graphviz.Digraph('Baseline_TP8_PP2')
    dot.attr(rankdir='TB', fontsize='12', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='orange')  # Communication
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    
    # Pipeline Stage 0 (layers 0-7)
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightcoral')
    for layer in range(8):
        # MHA with tensor parallelism
        mha_name = f'layer_{layer}_mha_tp8'
        dot.node(mha_name, f'Layer {layer} MHA\\nTensor Parallel=8\\nGPU 0-7\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
                shape='rectangle', fillcolor='lightcoral')
        
        # MLP with tensor parallelism (multiple experts colocated)
        mlp_name = f'layer_{layer}_mlp_tp8'
        dot.node(mlp_name, f'Layer {layer} MLP+Experts\\nTensor Parallel=8\\nGPU 0-7\\n16 experts colocated\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
                shape='rectangle', fillcolor='lightcoral')
        
        if layer == 0:
            dot.edge('input', mha_name)
        else:
            prev_mlp = f'layer_{layer-1}_mlp_tp8'
            dot.edge(prev_mlp, mha_name)
        
        dot.edge(mha_name, mlp_name)
    
    # Pipeline communication between stages
    dot.node('stage0_to_stage1', 'Pipeline Communication\\nStage 0 → Stage 1\\nGPU 7 → GPU 8', shape='diamond', fillcolor='orange')
    dot.edge('layer_7_mlp_tp8', 'stage0_to_stage1')
    
    # Pipeline Stage 1 (layers 8-15)
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightsteelblue')
    for layer in range(8, 16):
        # MHA with tensor parallelism
        mha_name = f'layer_{layer}_mha_tp8'
        dot.node(mha_name, f'Layer {layer} MHA\\nTensor Parallel=8\\nGPU 8-15\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
                shape='rectangle', fillcolor='lightsteelblue')
        
        # MLP with tensor parallelism (multiple experts colocated)
        mlp_name = f'layer_{layer}_mlp_tp8'
        dot.node(mlp_name, f'Layer {layer} MLP+Experts\\nTensor Parallel=8\\nGPU 8-15\\n16 experts colocated\\n[batch_size=128, seq_len=10000, d_model=4096]→[batch_size=128, seq_len=10000, d_model=4096]', 
                shape='rectangle', fillcolor='lightsteelblue')
        
        if layer == 8:
            dot.edge('stage0_to_stage1', mha_name)
        else:
            prev_mlp = f'layer_{layer-1}_mlp_tp8'
            dot.edge(prev_mlp, mha_name)
        
        dot.edge(mha_name, mlp_name)
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=10000, d_model=4096]', shape='ellipse', fillcolor='lightblue')
    dot.edge('layer_15_mlp_tp8', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate Expert Parallelism DAG
    expert_dag = create_expert_parallelism_dag()
    expert_dag.render('expert_parallelism_dag', format='dot', cleanup=False)
    expert_dag.render('expert_parallelism_dag', format='svg', cleanup=False)
    
    # Generate Baseline DAG
    baseline_dag = create_expert_parallelism_baseline_dag()
    baseline_dag.render('baseline_tp8_pp2_dag', format='dot', cleanup=False)
    baseline_dag.render('baseline_tp8_pp2_dag', format='svg', cleanup=False)
    
    print("Generated DAGs:")
    print("- expert_parallelism_dag.dot & .svg")
    print("- baseline_tp8_pp2_dag.dot & .svg")