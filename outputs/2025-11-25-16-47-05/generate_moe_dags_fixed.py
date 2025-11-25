#!/usr/bin/env python3
"""
Generate DAG visualizations for MoE deployments:
1. Baseline: TP=8, PP=2, 16 GPUs, 2 experts/GPU
2. Proposed: EP=16, 256 GPUs, 1 expert/GPU
"""

import os
import json
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2"""
    dot = Digraph(name='MoE_Baseline_TP8_PP2')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\nbatch_size=128, seq_len=10000, d_model=4096', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create pipeline stages
    for stage in [0, 1]:
        stage_name = f"stage_{stage}"
        stage_start = stage * 8
        
        # Process 8 layers per stage
        for layer in range(8):
            actual_layer = stage * 8 + layer
            layer_prefix = f"l{actual_layer}"
            
            # Attention across 8 GPUs
            for tp_shard in range(8):
                gpu_id = stage_start + tp_shard
                attn_node = f"{layer_prefix}_attn_tp{tp_shard}_gpu{gpu_id}"
                dot.node(attn_node, 
                       f'Attention\nL{actual_layer} TP{tp_shard}\nGPU{gpu_id}\nIn:[128,10000,4,128]\nOut:[128,10000,4,128]',
                       fillcolor='lightcoral')
            
            # MoE across 8 GPUs, 2 experts per GPU
            for tp_shard in range(8):
                gpu_id = stage_start + tp_shard
                expert_start = tp_shard * 2
                expert_end = expert_start + 1
                moe_node = f"{layer_prefix}_moe_tp{tp_shard}_gpu{gpu_id}"
                dot.node(moe_node,
                       f'MoE\nL{actual_layer} E{expert_start}-{expert_end}\nGPU{gpu_id}\nIn:[128,10000,4096]\nOut:[128,10000,4096]',
                       fillcolor='lightyellow')
    
    # Add connections
    # Input to first layer
    for tp_shard in range(8):
        dot.edge('input', f'l0_attn_tp{tp_shard}_gpu{tp_shard}')
    
    # Within layers and stages
    for stage in [0, 1]:
        stage_start = stage * 8
        for layer in range(8):
            actual_layer = stage * 8 + layer
            layer_prefix = f"l{actual_layer}"
            
            # Attention to MoE within same TP shard
            for tp_shard in range(8):
                gpu_id = stage_start + tp_shard
                dot.edge(f"{layer_prefix}_attn_tp{tp_shard}_gpu{gpu_id}", 
                        f"{layer_prefix}_moe_tp{tp_shard}_gpu{gpu_id}")
            
            # MoE to next attention (with TP all-reduce)
            if actual_layer < 15:  # Not last layer
                next_layer = actual_layer + 1
                next_prefix = f"l{next_layer}"
                
                # TP all-reduce for attention input
                all_reduce_node = f"{layer_prefix}_allreduce"
                dot.node(all_reduce_node, f'TP All-Reduce\nLayer {actual_layer}', 
                        shape='parallelogram', fillcolor='lightgray')
                
                for tp_shard in range(8):
                    gpu_id = stage_start + tp_shard
                    dot.edge(f"{layer_prefix}_moe_tp{tp_shard}_gpu{gpu_id}", all_reduce_node)
                    
                    # Handle pipeline boundary
                    if layer == 7 and stage == 0:
                        # Pipeline communication to stage 1
                        pipe_comm = f"pipeline_stage_{stage}_{stage+1}"
                        dot.node(pipe_comm, 'Pipeline Comm\nStage0→Stage1',
                               shape='parallelogram', fillcolor='orange', style='dashed')
                        dot.edge(all_reduce_node, pipe_comm)
                        for next_tp in range(8):
                            next_gpu = 8 + next_tp
                            dot.edge(pipe_comm, f"{next_prefix}_attn_tp{next_tp}_gpu{next_gpu}")
                    else:
                        # Same stage, next layer
                        next_gpu = (stage_start + tp_shard) if next_layer < 8 else (8 + tp_shard)
                        next_tp_shard = tp_shard
                        dot.edge(all_reduce_node, f"{next_prefix}_attn_tp{next_tp_shard}_gpu{next_gpu}")
    
    # Output node
    dot.node('output', 'Output\nbatch_size=128, seq_len=10000, d_model=4096', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect last layer to output
    for tp_shard in range(8):
        gpu_id = 8 + tp_shard  # Last stage
        dot.edge(f'l15_moe_tp{tp_shard}_gpu{gpu_id}', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=16, 1 expert/GPU"""
    dot = Digraph(name='MoE_Proposed_EP16_OneExpertPerGPU')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\nbatch_size=128, seq_len=10000, d_model=4096', 
             shape='ellipse', fillcolor='lightgreen')
    
    # We'll show representative layers 0, 1, 15 for clarity
    representative_layers = [0, 1, 15]
    
    for layer_idx in representative_layers:
        layer_prefix = f"l{layer_idx}"
        
        # Attention module (single GPU per layer)
        attn_gpu = layer_idx * 16  # GPU 0, 16, 240 for layers 0,1,15
        attn_node = f"{layer_prefix}_attn_gpu{attn_gpu}"
        dot.node(attn_node,
               f'Attention\nL{layer_idx}\nGPU{attn_gpu}\nIn:[128,10000,32,128]\nOut:[128,10000,4096]',
               fillcolor='lightcoral')
        
        # Gating network
        gate_node = f"{layer_prefix}_gate_gpu{attn_gpu}"
        dot.node(gate_node,
               f'Gate\nL{layer_idx}\nGPU{attn_gpu}\nIn:[128,10000,4096]\nOut:[128,10000,2]',
               shape='parallelogram', fillcolor='lightpink')
        
        # Expert routing (all-to-all communication)
        route_node = f"{layer_prefix}_route"
        dot.node(route_node,
               f'Token Routing\nL{layer_idx}\nAll-to-All\n[128,10000,4096]→16 experts',
               shape='parallelogram', fillcolor='lightgray')
        
        # Expert modules (one per GPU)
        for expert_id in range(16):
            expert_gpu = layer_idx * 16 + expert_id + 1
            expert_node = f"{layer_prefix}_expert{expert_id}_gpu{expert_gpu}"
            dot.node(expert_node,
                   f'Expert{expert_id}\nL{layer_idx}\nGPU{expert_gpu}\nIn:[tokens,4096]\nOut:[tokens,4096]',
                   fillcolor='lightyellow')
        
        # Expert aggregation
        agg_node = f"{layer_prefix}_aggregate"
        dot.node(agg_node,
               f'Expert Aggregation\nL{layer_idx}\n16 experts→[128,10000,4096]',
               shape='parallelogram', fillcolor='lightgray')
        
        # Residual connection
        residual_node = f"{layer_prefix}_residual"
        dot.node(residual_node,
               f'Residual Add\nL{layer_idx}\nIn:[128,10000,4096]+[128,10000,4096]\nOut:[128,10000,4096]',
               shape='diamond', fillcolor='lightgreen')
    
    # Add connections
    # Input to layer 0
    dot.edge('input', 'l0_attn_gpu0')
    
    # Layer 0 connections
    dot.edge('l0_attn_gpu0', 'l0_gate_gpu0')
    dot.edge('l0_gate_gpu0', 'l0_route')
    
    # Connect routing to experts
    for expert_id in range(16):
        expert_gpu = 1 + expert_id
        dot.edge('l0_route', f'l0_expert{expert_id}_gpu{expert_gpu}')
    
    # Connect experts to aggregation
    for expert_id in range(16):
        expert_gpu = 1 + expert_id
        dot.edge(f'l0_expert{expert_id}_gpu{expert_gpu}', 'l0_aggregate')
    
    # Residual connection
    dot.edge('l0_aggregate', 'l0_residual')
    dot.edge('l0_attn_gpu0', 'l0_residual')  # Skip connection
    
    # Layer 1 connections
    dot.edge('l0_residual', 'l1_attn_gpu16')
    dot.edge('l1_attn_gpu16', 'l1_gate_gpu16')
    dot.edge('l1_gate_gpu16', 'l1_route')
    
    for expert_id in range(16):
        expert_gpu = 16 + 1 + expert_id
        dot.edge('l1_route', f'l1_expert{expert_id}_gpu{expert_gpu}')
    
    for expert_id in range(16):
        expert_gpu = 16 + 1 + expert_id
        dot.edge(f'l1_expert{expert_id}_gpu{expert_gpu}', 'l1_aggregate')
    
    dot.edge('l1_aggregate', 'l1_residual')
    dot.edge('l1_attn_gpu16', 'l1_residual')
    
    # Connect to layer 15 (skipping middle layers for clarity)
    dot.edge('l1_residual', 'l15_attn_gpu240')
    dot.edge('l15_attn_gpu240', 'l15_gate_gpu240')
    dot.edge('l15_gate_gpu240', 'l15_route')
    
    for expert_id in range(16):
        expert_gpu = 240 + 1 + expert_id
        dot.edge('l15_route', f'l15_expert{expert_id}_gpu{expert_gpu}')
    
    for expert_id in range(16):
        expert_gpu = 240 + 1 + expert_id
        dot.edge(f'l15_expert{expert_id}_gpu{expert_gpu}', 'l15_aggregate')
    
    dot.edge('l15_aggregate', 'l15_residual')
    dot.edge('l15_attn_gpu240', 'l15_residual')
    
    # Output
    dot.node('output', 'Output\nbatch_size=128, seq_len=10000, d_model=4096',
             shape='ellipse', fillcolor='lightgreen')
    dot.edge('l15_residual', 'output')
    
    # Add layer repetition note
    dot.node('note', 'Note: Layers 2-14 follow same pattern\nEach layer uses next 16 GPUs',
             shape='note', fillcolor='white')
    
    return dot

def main():
    # Create output directory
    os.makedirs('../outputs/2025-11-25-16-47-05', exist_ok=True)
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    baseline_dag.save('../outputs/2025-11-25-16-47-05/moe_baseline_tp8_pp2.dot')
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    proposed_dag.save('../outputs/2025-11-25-16-47-05/moe_proposed_ep16_one_expert_per_gpu.dot')
    
    print("DAG generation complete!")
    
    # Generate JSON with paths
    paths = {
        "baseline_dag_dot": "../outputs/2025-11-25-16-47-05/moe_baseline_tp8_pp2.dot",
        "proposed_dag_dot": "../outputs/2025-11-25-16-47-05/moe_proposed_ep16_one_expert_per_gpu.dot"
    }
    
    with open('../outputs/2025-11-25-16-47-05/dag_paths.json', 'w') as f:
        json.dump(paths, f, indent=2)
    
    return paths

if __name__ == "__main__":
    paths = main()
    print(json.dumps(paths, indent=2))