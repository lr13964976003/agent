import graphviz
from typing import Dict, List, Tuple
import os

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2 and colocated experts"""
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE with TP8 PP2')
    dot.attr(rankdir='TB', size='20,20')
    
    # Input specifications
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    num_heads = 16
    head_dim = 512
    num_experts = 16
    expert_hidden = 32768
    
    # Global input
    dot.node('input', f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed')
        
        # Layer 0
        with c0.subgraph(name='cluster_layer_0') as l0:
            l0.attr(label='Layer 0')
            
            # MHA across GPUs 0-7 (Tensor Parallel)
            for gpu_id in range(8):
                # QKV projection (column parallel)
                dot.node(f'l0_mha_qkv_gpu{gpu_id}', 
                        f'QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Attention computation
                dot.node(f'l0_mha_attn_gpu{gpu_id}', 
                        f'Attention\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Output projection (row parallel)
                dot.node(f'l0_mha_out_gpu{gpu_id}', 
                        f'Output Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//8}]',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # All-reduce for attention output
                dot.node(f'l0_mha_allreduce_gpu{gpu_id}', 
                        f'All-Reduce\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                # Residual add
                dot.node(f'l0_res_add_gpu{gpu_id}', 
                        f'Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Experts (8 experts per GPU)
                for expert_id in range(8):
                    expert_num = gpu_id * 8 + expert_id
                    dot.node(f'l0_expert_{expert_num}_gpu{gpu_id}', 
                            f'Expert {expert_num}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Gate
                dot.node(f'l0_gate_gpu{gpu_id}', 
                        f'Gate\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                        shape='parallelogram', style='filled', fillcolor='lightcyan')
                
                # Expert aggregation
                dot.node(f'l0_expert_agg_gpu{gpu_id}', 
                        f'Expert Aggregation\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                # Second residual add
                dot.node(f'l0_res2_gpu{gpu_id}', 
                        f'Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                        shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Layer 1 (same structure)
        with c0.subgraph(name='cluster_layer_1') as l1:
            l1.attr(label='Layer 1')
            for gpu_id in range(8):
                # MHA
                for op in ['qkv', 'attn', 'out', 'allreduce', 'res_add']:
                    l1.node(f'l1_mha_{op}_gpu{gpu_id}', 
                           f'{op.replace("_", " ").title()}\\nGPU {gpu_id}\\nSame dims as layer 0',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Experts
                for expert_id in range(8):
                    expert_num = gpu_id * 8 + expert_id
                    l1.node(f'l1_expert_{expert_num}_gpu{gpu_id}', 
                           f'Expert {expert_num}\\nGPU {gpu_id}\\nSame dims',
                           shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Gate and aggregation
                l1.node(f'l1_gate_gpu{gpu_id}', 'Gate', shape='parallelogram', style='filled', fillcolor='lightcyan')
                l1.node(f'l1_expert_agg_gpu{gpu_id}', 'Expert Aggregation', shape='parallelogram', style='filled', fillcolor='yellow')
                l1.node(f'l1_res2_gpu{gpu_id}', 'Residual Add', shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Pipeline stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed')
        
        # Layer 2
        with c1.subgraph(name='cluster_layer_2') as l2:
            l2.attr(label='Layer 2')
            for gpu_id in range(8, 16):
                # MHA
                for op in ['qkv', 'attn', 'out', 'allreduce', 'res_add']:
                    l2.node(f'l2_mha_{op}_gpu{gpu_id}', 
                           f'{op.replace("_", " ").title()}\\nGPU {gpu_id}\\nSame dims',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Experts
                for expert_id in range(8):
                    expert_num = (gpu_id - 8) * 8 + expert_id
                    l2.node(f'l2_expert_{expert_num}_gpu{gpu_id}', 
                           f'Expert {expert_num + 64}\\nGPU {gpu_id}\\nSame dims',
                           shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Gate and aggregation
                l2.node(f'l2_gate_gpu{gpu_id}', 'Gate', shape='parallelogram', style='filled', fillcolor='lightcyan')
                l2.node(f'l2_expert_agg_gpu{gpu_id}', 'Expert Aggregation', shape='parallelogram', style='filled', fillcolor='yellow')
                l2.node(f'l2_res2_gpu{gpu_id}', 'Residual Add', shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Layer 3
        with c1.subgraph(name='cluster_layer_3') as l3:
            l3.attr(label='Layer 3')
            for gpu_id in range(8, 16):
                # MHA
                for op in ['qkv', 'attn', 'out', 'allreduce', 'res_add']:
                    l3.node(f'l3_mha_{op}_gpu{gpu_id}', 
                           f'{op.replace("_", " ").title()}\\nGPU {gpu_id}\\nSame dims',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Experts
                for expert_id in range(8):
                    expert_num = (gpu_id - 8) * 8 + expert_id
                    l3.node(f'l3_expert_{expert_num}_gpu{gpu_id}', 
                           f'Expert {expert_num + 72}\\nGPU {gpu_id}\\nSame dims',
                           shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Gate and aggregation
                l3.node(f'l3_gate_gpu{gpu_id}', 'Gate', shape='parallelogram', style='filled', fillcolor='lightcyan')
                l3.node(f'l3_expert_agg_gpu{gpu_id}', 'Expert Aggregation', shape='parallelogram', style='filled', fillcolor='yellow')
                l3.node(f'l3_res2_gpu{gpu_id}', 'Residual Add', shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Global output
    dot.node('output', f'Total Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the DAG
    # Input to layer 0
    for gpu_id in range(8):
        dot.edge('input', f'l0_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l0_mha_qkv_gpu{gpu_id}', f'l0_mha_attn_gpu{gpu_id}')
        dot.edge(f'l0_mha_attn_gpu{gpu_id}', f'l0_mha_out_gpu{gpu_id}')
        dot.edge(f'l0_mha_out_gpu{gpu_id}', f'l0_mha_allreduce_gpu{gpu_id}')
        dot.edge(f'l0_mha_allreduce_gpu{gpu_id}', f'l0_res_add_gpu{gpu_id}')
        dot.edge('input', f'l0_res_add_gpu{gpu_id}')  # Residual connection
        
        # MHA to gate
        dot.edge(f'l0_res_add_gpu{gpu_id}', f'l0_gate_gpu{gpu_id}')
        
        # Gate to experts
        for expert_id in range(8):
            expert_num = gpu_id * 8 + expert_id
            dot.edge(f'l0_gate_gpu{gpu_id}', f'l0_expert_{expert_num}_gpu{gpu_id}', style='dashed')
            dot.edge(f'l0_res_add_gpu{gpu_id}', f'l0_expert_{expert_num}_gpu{gpu_id}')
            dot.edge(f'l0_expert_{expert_num}_gpu{gpu_id}', f'l0_expert_agg_gpu{gpu_id}')
        
        # Expert aggregation to residual
        dot.edge(f'l0_expert_agg_gpu{gpu_id}', f'l0_res2_gpu{gpu_id}')
        dot.edge(f'l0_res_add_gpu{gpu_id}', f'l0_res2_gpu{gpu_id}')
        
        # Layer 0 to layer 1
        dot.edge(f'l0_res2_gpu{gpu_id}', f'l1_mha_qkv_gpu{gpu_id}')
        
        # Layer 1 connections (similar to layer 0)
        dot.edge(f'l1_mha_qkv_gpu{gpu_id}', f'l1_mha_attn_gpu{gpu_id}')
        dot.edge(f'l1_mha_attn_gpu{gpu_id}', f'l1_mha_out_gpu{gpu_id}')
        dot.edge(f'l1_mha_out_gpu{gpu_id}', f'l1_mha_allreduce_gpu{gpu_id}')
        dot.edge(f'l1_mha_allreduce_gpu{gpu_id}', f'l1_res_add_gpu{gpu_id}')
        dot.edge(f'l0_res2_gpu{gpu_id}', f'l1_res_add_gpu{gpu_id}')
        
        dot.edge(f'l1_res_add_gpu{gpu_id}', f'l1_gate_gpu{gpu_id}')
        for expert_id in range(8):
            expert_num = gpu_id * 8 + expert_id
            dot.edge(f'l1_gate_gpu{gpu_id}', f'l1_expert_{expert_num}_gpu{gpu_id}', style='dashed')
            dot.edge(f'l1_res_add_gpu{gpu_id}', f'l1_expert_{expert_num}_gpu{gpu_id}')
            dot.edge(f'l1_expert_{expert_num}_gpu{gpu_id}', f'l1_expert_agg_gpu{gpu_id}')
        
        dot.edge(f'l1_expert_agg_gpu{gpu_id}', f'l1_res2_gpu{gpu_id}')
        dot.edge(f'l1_res_add_gpu{gpu_id}', f'l1_res2_gpu{gpu_id}')
    
    # Pipeline communication between stages
    for gpu_id in range(8):
        target_gpu = gpu_id + 8
        dot.edge(f'l1_res2_gpu{gpu_id}', f'l2_mha_qkv_gpu{target_gpu}')
    
    # Layer 2 and 3 connections
    for gpu_id in range(8, 16):
        # Layer 2
        dot.edge(f'l2_mha_qkv_gpu{gpu_id}', f'l2_mha_attn_gpu{gpu_id}')
        dot.edge(f'l2_mha_attn_gpu{gpu_id}', f'l2_mha_out_gpu{gpu_id}')
        dot.edge(f'l2_mha_out_gpu{gpu_id}', f'l2_mha_allreduce_gpu{gpu_id}')
        dot.edge(f'l2_mha_allreduce_gpu{gpu_id}', f'l2_res_add_gpu{gpu_id}')
        dot.edge(f'l2_mha_qkv_gpu{gpu_id}', f'l2_res_add_gpu{gpu_id}')  # Residual
        
        dot.edge(f'l2_res_add_gpu{gpu_id}', f'l2_gate_gpu{gpu_id}')
        for expert_id in range(8):
            expert_num = (gpu_id - 8) * 8 + expert_id
            dot.edge(f'l2_gate_gpu{gpu_id}', f'l2_expert_{expert_num}_gpu{gpu_id}', style='dashed')
            dot.edge(f'l2_res_add_gpu{gpu_id}', f'l2_expert_{expert_num}_gpu{gpu_id}')
            dot.edge(f'l2_expert_{expert_num}_gpu{gpu_id}', f'l2_expert_agg_gpu{gpu_id}')
        
        dot.edge(f'l2_expert_agg_gpu{gpu_id}', f'l2_res2_gpu{gpu_id}')
        dot.edge(f'l2_res_add_gpu{gpu_id}', f'l2_res2_gpu{gpu_id}')
        
        # Layer 3
        dot.edge(f'l2_res2_gpu{gpu_id}', f'l3_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l3_mha_qkv_gpu{gpu_id}', f'l3_mha_attn_gpu{gpu_id}')
        dot.edge(f'l3_mha_attn_gpu{gpu_id}', f'l3_mha_out_gpu{gpu_id}')
        dot.edge(f'l3_mha_out_gpu{gpu_id}', f'l3_mha_allreduce_gpu{gpu_id}')
        dot.edge(f'l3_mha_allreduce_gpu{gpu_id}', f'l3_res_add_gpu{gpu_id}')
        dot.edge(f'l3_mha_qkv_gpu{gpu_id}', f'l3_res_add_gpu{gpu_id}')
        
        dot.edge(f'l3_res_add_gpu{gpu_id}', f'l3_gate_gpu{gpu_id}')
        for expert_id in range(8):
            expert_num = (gpu_id - 8) * 8 + expert_id
            dot.edge(f'l3_gate_gpu{gpu_id}', f'l3_expert_{expert_num}_gpu{gpu_id}', style='dashed')
            dot.edge(f'l3_res_add_gpu{gpu_id}', f'l3_expert_{expert_num}_gpu{gpu_id}')
            dot.edge(f'l3_expert_{expert_num}_gpu{gpu_id}', f'l3_expert_agg_gpu{gpu_id}')
        
        dot.edge(f'l3_expert_agg_gpu{gpu_id}', f'l3_res2_gpu{gpu_id}')
        dot.edge(f'l3_res_add_gpu{gpu_id}', f'l3_res2_gpu{gpu_id}')
        
        # Connect to output
        dot.edge(f'l3_res2_gpu{gpu_id}', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=16 and one expert per GPU"""
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE with EP16')
    dot.attr(rankdir='TB', size='25,25')
    
    # Input specifications
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    num_heads = 16
    head_dim = 512
    num_experts = 16
    expert_hidden = 32768
    
    # Global input
    dot.node('input', f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create nodes for each GPU (0-15)
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer}')
            
            # Shared MHA (replicated across all GPUs)
            for gpu_id in range(16):
                with layer_cluster.subgraph(name=f'cluster_gpu_{gpu_id}_layer_{layer}') as gpu_cluster:
                    gpu_cluster.attr(label=f'GPU {gpu_id} (Node {gpu_id//4})', style='dashed')
                    
                    # MHA (replicated)
                    dot.node(f'l{layer}_mha_qkv_gpu{gpu_id}', 
                            f'QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                            shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    dot.node(f'l{layer}_mha_attn_gpu{gpu_id}', 
                            f'Attention\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
                            shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    dot.node(f'l{layer}_mha_out_gpu{gpu_id}', 
                            f'Output Projection\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    dot.node(f'l{layer}_mha_res_add_gpu{gpu_id}', 
                            f'Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='rectangle', style='filled', fillcolor='lightcoral')
                    
                    # Expert (one per GPU)
                    expert_id = gpu_id
                    dot.node(f'l{layer}_expert_{expert_id}_gpu{gpu_id}', 
                            f'Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='rectangle', style='filled', fillcolor='lightpink')
                    
                    # Gate (distributed)
                    dot.node(f'l{layer}_gate_gpu{gpu_id}', 
                            f'Gate\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                            shape='parallelogram', style='filled', fillcolor='lightcyan')
                    
                    # Expert aggregation (cross-GPU)
                    dot.node(f'l{layer}_expert_agg_gpu{gpu_id}', 
                            f'Expert Aggregation\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='parallelogram', style='filled', fillcolor='yellow')
                    
                    # Final residual add
                    dot.node(f'l{layer}_res2_gpu{gpu_id}', 
                            f'Residual Add\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                            shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Global output
    dot.node('output', f'Total Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect the DAG
    # Input to layer 0 (broadcast to all GPUs)
    for gpu_id in range(16):
        dot.edge('input', f'l0_mha_qkv_gpu{gpu_id}')
        dot.edge(f'l0_mha_qkv_gpu{gpu_id}', f'l0_mha_attn_gpu{gpu_id}')
        dot.edge(f'l0_mha_attn_gpu{gpu_id}', f'l0_mha_out_gpu{gpu_id}')
        dot.edge(f'l0_mha_out_gpu{gpu_id}', f'l0_mha_res_add_gpu{gpu_id}')
        dot.edge('input', f'l0_mha_res_add_gpu{gpu_id}')  # Residual connection
        
        # MHA to gate
        dot.edge(f'l0_mha_res_add_gpu{gpu_id}', f'l0_gate_gpu{gpu_id}')
        
        # Gate to expert (dashed for selection)
        dot.edge(f'l0_gate_gpu{gpu_id}', f'l0_expert_{gpu_id}_gpu{gpu_id}', style='dashed')
        
        # Input to expert
        dot.edge(f'l0_mha_res_add_gpu{gpu_id}', f'l0_expert_{gpu_id}_gpu{gpu_id}')
        
        # Expert to aggregation (cross-GPU communication)
        for target_gpu in range(16):
            dot.edge(f'l0_expert_{gpu_id}_gpu{gpu_id}', f'l0_expert_agg_gpu{target_gpu}')
        
        # Aggregation to residual
        dot.edge(f'l0_expert_agg_gpu{gpu_id}', f'l0_res2_gpu{gpu_id}')
        dot.edge(f'l0_mha_res_add_gpu{gpu_id}', f'l0_res2_gpu{gpu_id}')
    
    # Connect layers
    for layer in range(1, 4):
        prev_layer = layer - 1
        for gpu_id in range(16):
            # Connect previous layer output to current layer
            dot.edge(f'l{prev_layer}_res2_gpu{gpu_id}', f'l{layer}_mha_qkv_gpu{gpu_id}')
            
            # MHA connections
            dot.edge(f'l{layer}_mha_qkv_gpu{gpu_id}', f'l{layer}_mha_attn_gpu{gpu_id}')
            dot.edge(f'l{layer}_mha_attn_gpu{gpu_id}', f'l{layer}_mha_out_gpu{gpu_id}')
            dot.edge(f'l{layer}_mha_out_gpu{gpu_id}', f'l{layer}_mha_res_add_gpu{gpu_id}')
            dot.edge(f'l{prev_layer}_res2_gpu{gpu_id}', f'l{layer}_mha_res_add_gpu{gpu_id}')
            
            # Expert connections
            dot.edge(f'l{layer}_mha_res_add_gpu{gpu_id}', f'l{layer}_gate_gpu{gpu_id}')
            dot.edge(f'l{layer}_gate_gpu{gpu_id}', f'l{layer}_expert_{gpu_id}_gpu{gpu_id}', style='dashed')
            dot.edge(f'l{layer}_mha_res_add_gpu{gpu_id}', f'l{layer}_expert_{gpu_id}_gpu{gpu_id}')
            
            # Cross-GPU expert communication
            for target_gpu in range(16):
                dot.edge(f'l{layer}_expert_{gpu_id}_gpu{gpu_id}', f'l{layer}_expert_agg_gpu{target_gpu}')
            
            dot.edge(f'l{layer}_expert_agg_gpu{gpu_id}', f'l{layer}_res2_gpu{gpu_id}')
            dot.edge(f'l{layer}_mha_res_add_gpu{gpu_id}', f'l{layer}_res2_gpu{gpu_id}')
    
    # Connect final layer to output
    for gpu_id in range(16):
        dot.edge(f'l3_res2_gpu{gpu_id}', 'output')
    
    return dot

def main():
    # Create output directory
    os.makedirs('./outputs/2025-10-13-16-10-29', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('./outputs/2025-10-13-16-10-29/baseline_dag', format='dot', cleanup=False)
    baseline_dag.render('./outputs/2025-10-13-16-10-29/baseline_dag', format='svg', cleanup=False)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('./outputs/2025-10-13-16-10-29/proposed_dag', format='dot', cleanup=False)
    proposed_dag.render('./outputs/2025-10-13-16-10-29/proposed_dag', format='svg', cleanup=False)
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- ./outputs/2025-10-13-16-10-29/baseline_dag.dot")
    print("- ./outputs/2025-10-13-16-10-29/baseline_dag.svg")
    print("- ./outputs/2025-10-13-16-10-29/proposed_dag.dot")
    print("- ./outputs/2025-10-13-16-10-29/proposed_dag.svg")

if __name__ == "__main__":
    main()