#!/usr/bin/env python3

import graphviz
import os

def create_baseline_dag():
    """Create the baseline MoE DAG with TP=8, PP=2"""
    dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', 
                          filename='moe_baseline_tp8_pp2_fixed.dot',
                          comment='MoE Baseline with TP=8, PP=2')
    
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fillcolor='lightblue', shape='rectangle', style='filled')
    
    # Input node
    dot.node('input', 'Input\nbatch_size=128, seq_len=10000, d_model=4096', 
             fillcolor='lightgreen', shape='ellipse')
    
    # Output node
    dot.node('output', 'Output\nbatch_size=128, seq_len=10000, d_model=4096', 
             fillcolor='lightgreen', shape='ellipse')
    
    # Create nodes for all 16 layers with proper tensor parallelism
    for layer in range(16):
        if layer < 8:  # Stage 0 (GPUs 0-7)
            base_gpu = 0
            stage_name = "Stage0"
        else:  # Stage 1 (GPUs 8-15)
            base_gpu = 8
            stage_name = "Stage1"
        
        # Attention nodes for this layer (8-way TP)
        for tp in range(8):
            gpu_id = base_gpu + tp
            attn_node = f'l{layer}_attn_tp{tp}_gpu{gpu_id}'
            dot.node(attn_node, f'Attention\nL{layer} TP{tp}\nGPU{gpu_id}\n'
                                f'In:[128,10000,4,128]\nOut:[128,10000,4,128]',
                     fillcolor='lightcoral')
        
        # MoE nodes for this layer (2 experts per GPU, 8-way TP)
        for tp in range(8):
            gpu_id = base_gpu + tp
            expert_start = tp * 2
            expert_end = expert_start + 1
            moe_node = f'l{layer}_moe_tp{tp}_gpu{gpu_id}'
            dot.node(moe_node, f'MoE\nL{layer} E{expert_start}-{expert_end}\n'
                               f'GPU{gpu_id}\nIn:[128,10000,4096]\nOut:[128,10000,4096]',
                     fillcolor='lightyellow')
        
        # All-reduce node for this layer
        allreduce_node = f'l{layer}_allreduce'
        dot.node(allreduce_node, f'TP All-Reduce\nL{layer} All GPUs\n'
                                 f'In:[128,10000,4096]\nOut:[128,10000,4096]',
                 fillcolor='lightgray', shape='parallelogram')
        
        # Pipeline communication node between stages
        if layer == 7:  # Between stage 0 and stage 1
            pipeline_node = f'pipeline_stage_0_1'
            dot.node(pipeline_node, f'Pipeline Comm\nStage0→Stage1\n'
                                    f'GPU7→GPU8\nIn:[128,10000,4096]\nOut:[128,10000,4096]',
                     fillcolor='lightblue', shape='ellipse')
    
    # Create connections
    # Input to layer 0
    for tp in range(8):
        dot.edge('input', f'l0_attn_tp{tp}_gpu{tp}')
    
    # Connections within each layer
    for layer in range(16):
        if layer < 8:  # Stage 0
            base_gpu = 0
        else:  # Stage 1
            base_gpu = 8
        
        # Attention to MoE connections
        for tp in range(8):
            gpu_id = base_gpu + tp
            dot.edge(f'l{layer}_attn_tp{tp}_gpu{gpu_id}', 
                    f'l{layer}_moe_tp{tp}_gpu{gpu_id}')
        
        # MoE to all-reduce connections
        for tp in range(8):
            gpu_id = base_gpu + tp
            dot.edge(f'l{layer}_moe_tp{tp}_gpu{gpu_id}', 
                    f'l{layer}_allreduce')
        
        # All-reduce to next layer connections
        if layer < 15:  # Not the last layer
            if layer == 7:  # Special case: pipeline boundary
                for tp in range(8):
                    dot.edge(f'l{layer}_allreduce', 'pipeline_stage_0_1')
                    dot.edge('pipeline_stage_0_1', 
                           f'l{layer+1}_attn_tp{tp}_gpu{8+tp}')
            else:
                next_base_gpu = 8 if layer >= 8 else 0
                for tp in range(8):
                    dot.edge(f'l{layer}_allreduce', 
                           f'l{layer+1}_attn_tp{tp}_gpu{next_base_gpu+tp}')
        else:  # Last layer to output
            for tp in range(8):
                dot.edge(f'l{layer}_allreduce', 'output')
    
    return dot

def create_proposed_dag():
    """Create the proposed MoE DAG with EP=16, one expert per GPU"""
    dot = graphviz.Digraph('MoE_Proposed_EP16_One_Expert_Per_GPU',
                          filename='moe_proposed_ep16_one_expert_per_gpu_fixed.dot',
                          comment='MoE Proposed with EP=16, one expert per GPU')
    
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', fillcolor='lightblue', shape='rectangle', style='filled')
    
    # Input node
    dot.node('input', 'Input\nbatch_size=128, seq_len=10000, d_model=4096', 
             fillcolor='lightgreen', shape='ellipse')
    
    # Output node
    dot.node('output', 'Output\nbatch_size=128, seq_len=10000, d_model=4096', 
             fillcolor='lightgreen', shape='ellipse')
    
    # Create nodes for all 16 layers
    for layer in range(16):
        # Attention + Gate node (one per layer)
        attn_gpu = layer * 16  # Each layer starts at a new GPU block
        attn_node = f'l{layer}_attn_gpu{attn_gpu}'
        dot.node(attn_node, f'Attention+Gate\nL{layer} GPU{attn_gpu}\n'
                            f'In:[128,10000,4096]\nOut:[128,10000,4096]',
                 fillcolor='lightcoral')
        
        # Token routing node
        route_node = f'l{layer}_route'
        dot.node(route_node, f'Token Routing\nL{layer} GPU{attn_gpu}\n'
                             f'In:[128,10000,4096]\nOut:[tokens,4096] to experts',
                 fillcolor='lightgray', shape='parallelogram')
        
        # Expert nodes (16 experts per layer, each on a separate GPU)
        for expert in range(16):
            expert_gpu = attn_gpu + expert + 1  # +1 because attention is on attn_gpu
            expert_node = f'l{layer}_expert{expert}_gpu{expert_gpu}'
            dot.node(expert_node, f'Expert{expert}\nL{layer} GPU{expert_gpu}\n'
                                  f'In:[tokens,4096]\nOut:[tokens,4096]',
                     fillcolor='lightyellow')
        
        # Expert aggregation node
        aggregate_node = f'l{layer}_aggregate'
        dot.node(aggregate_node, f'Expert Aggregation\nL{layer} GPU{attn_gpu}\n'
                                  f'In:[tokens,4096] from 16 experts\n'
                                  f'Out:[128,10000,4096]',
                 fillcolor='lightgray', shape='parallelogram')
        
        # Residual add node
        residual_node = f'l{layer}_residual'
        dot.node(residual_node, f'Residual Add\nL{layer} GPU{attn_gpu}\n'
                                f'In:[128,10000,4096] from attention\n'
                                f'In:[128,10000,4096] from experts\n'
                                f'Out:[128,10000,4096]',
                 fillcolor='lightblue')
    
    # Create connections
    # Input to layer 0
    dot.edge('input', 'l0_attn_gpu0')
    
    # Connections within each layer
    for layer in range(16):
        attn_gpu = layer * 16
        attn_node = f'l{layer}_attn_gpu{attn_gpu}'
        route_node = f'l{layer}_route'
        aggregate_node = f'l{layer}_aggregate'
        residual_node = f'l{layer}_residual'
        
        # Attention to routing
        dot.edge(attn_node, route_node)
        
        # Routing to experts
        for expert in range(16):
            expert_gpu = attn_gpu + expert + 1
            expert_node = f'l{layer}_expert{expert}_gpu{expert_gpu}'
            dot.edge(route_node, expert_node)
        
        # Experts to aggregation
        for expert in range(16):
            expert_gpu = attn_gpu + expert + 1
            expert_node = f'l{layer}_expert{expert}_gpu{expert_gpu}'
            dot.edge(expert_node, aggregate_node)
        
        # Aggregation to residual
        dot.edge(aggregate_node, residual_node)
        dot.edge(attn_node, residual_node)
        
        # Residual to next layer or output
        if layer < 15:
            next_attn_gpu = (layer + 1) * 16
            next_attn_node = f'l{layer+1}_attn_gpu{next_attn_gpu}'
            dot.edge(residual_node, next_attn_node)
        else:
            dot.edge(residual_node, 'output')
    
    return dot

if __name__ == "__main__":
    # Create baseline DAG
    baseline_dot = create_baseline_dag()
    baseline_dot.render(directory='../outputs/2025-11-25-16-47-05', cleanup=True)
    print(f"Baseline DAG created: {baseline_dot.filepath}")
    
    # Create proposed DAG
    proposed_dot = create_proposed_dag()
    proposed_dot.render(directory='../outputs/2025-11-25-16-47-05', cleanup=True)
    print(f"Proposed DAG created: {proposed_dot.filepath}")
    
    print("All DAGs created successfully!")