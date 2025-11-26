import os
import graphviz
from typing import Dict, List, Tuple

def create_moe_deployment_dag():
    # Create a new directed graph
    dot = graphviz.Digraph('MOE_Deployment_DAG', format='svg', 
                          graph_attr={'rankdir': 'TB', 'compound': 'true'})
    
    # Define node styles
    dot.attr('node', fontsize='10')
    
    # Model configuration
    batch_size = '?'  # Variable
    seq_len = '?'     # Variable
    token_dim = 7168
    heads = 128
    d_k = 128
    experts_per_layer = 16
    mlp_hidden = 2048
    
    # Create subgraph for pipeline stages (PP=16, one stage per 4 layers)
    pp_degree = 16
    layers_per_stage = 61 // pp_degree  # ~3.8, round to 4 layers per stage
    
    # Generate the complete DAG
    for pp_stage in range(pp_degree):
        with dot.subgraph(name=f'cluster_pp_{pp_stage}') as c:
            c.attr(label=f'Pipeline Stage {pp_stage}', style='dashed')
            
            # Layers in this stage
            start_layer = pp_stage * layers_per_stage
            end_layer = min((pp_stage + 1) * layers_per_stage, 61)
            
            for layer_idx in range(start_layer, end_layer):
                layer_name = f'layer_{layer_idx}'
                
                # For dense layers (first 3)
                if layer_idx < 3:
                    create_dense_layer_nodes(dot, c, layer_name, pp_stage, batch_size, seq_len, token_dim, heads, d_k)
                else:
                    # For MoE layers
                    create_moe_layer_nodes(dot, c, layer_name, pp_stage, layer_idx, experts_per_layer, 
                                         batch_size, seq_len, token_dim, heads, d_k, mlp_hidden)
    
    # Add input node
    dot.node('input', 'Input Layer\nInput: [batch_size=' + str(batch_size) + ', seq_len=' + str(seq_len) + ', heads=' + str(heads) + ', d_k=' + str(d_k) + ']\nGPU: ALL', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Add output node
    dot.node('output', 'Output Layer\nInput: [batch_size=' + str(batch_size) + ', seq_len=' + str(seq_len) + ', heads=' + str(heads) + ', d_k=' + str(token_dim) + ']\nGPU: ALL', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect stages
    for pp_stage in range(pp_degree):
        if pp_stage == 0:
            dot.edge('input', f'pp_{pp_stage}_layer_0_input', ltail='cluster_pp_0')
        elif pp_stage == pp_degree - 1:
            dot.edge(f'pp_{pp_stage-1}_layer_{61-layers_per_stage}_output', 'output', lhead='cluster_pp_last')
        else:
            prev_stage = pp_stage - 1
            prev_layer = (prev_stage + 1) * layers_per_stage - 1
            next_layer = pp_stage * layers_per_stage
            dot.edge(f'pp_{prev_stage}_layer_{prev_layer}_output', 
                    f'pp_{pp_stage}_layer_{next_layer}_input')
    
    # Save the complete DAG
    dot.render('../outputs/2025-11-26-16-25-55/moe_complete_deployment', cleanup=False)
    
    # Also create a detailed layer DAG for one MoE layer
    create_detailed_moe_layer_dag()
    create_tensor_parallel_dag()

def create_dense_layer_nodes(dot, subgraph, layer_name, pp_stage, batch_size, seq_len, token_dim, heads, d_k):
    """Create nodes for a dense layer"""
    layer_idx = int(layer_name.split('_')[1])
    
    # Multi-Head Attention with TP=2
    # Input: [batch_size, seq_len, token_dim]
    # After TP split: [batch_size, seq_len, token_dim/2] on each GPU
    
    # QKV projection (column parallel)
    for tp_rank in range(2):
        gpu_id = pp_stage * 4 + tp_rank  # 2 GPUs per layer for TP
        
        subgraph.node(f'{layer_name}_qkv_proj_{tp_rank}', 
                     f'QKV Projection\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k*3}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Attention computation
    for tp_rank in range(2):
        gpu_id = pp_stage * 4 + tp_rank
        
        subgraph.node(f'{layer_name}_attention_{tp_rank}',
                     f'MHA Computation\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Attention output projection (row parallel)
    for tp_rank in range(2):
        gpu_id = pp_stage * 4 + tp_rank
        
        subgraph.node(f'{layer_name}_attn_out_{tp_rank}',
                     f'Attention Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # All-reduce for attention output
    subgraph.node(f'{layer_name}_attn_allreduce',
                 f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {(pp_stage*4)+0, (pp_stage*4)+1}',
                 shape='parallelogram', style='dashed', fillcolor='lightgray')
    
    # FFN (dense layers use full MLP)
    for tp_rank in range(2):
        gpu_id = pp_stage * 4 + tp_rank
        
        # First linear (column parallel)
        subgraph.node(f'{layer_name}_ffn_1_{tp_rank}',
                     f'FFN Linear 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={2048}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Activation
        subgraph.node(f'{layer_name}_ffn_act_{tp_rank}',
                     f'GELU Activation\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={2048}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={2048}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Second linear (row parallel)
        subgraph.node(f'{layer_name}_ffn_2_{tp_rank}',
                     f'FFN Linear 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={2048}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightblue')
    
    # All-reduce for FFN output
    subgraph.node(f'{layer_name}_ffn_allreduce',
                 f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {(pp_stage*4)+0, (pp_stage*4)+1}',
                 shape='parallelogram', style='dashed', fillcolor='lightgray')
    
    # Add residual connection
    subgraph.node(f'{layer_name}_residual',
                 f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {(pp_stage*4)+0, (pp_stage*4)+1}',
                 shape='parallelogram', style='filled', fillcolor='purple')

def create_moe_layer_nodes(dot, subgraph, layer_name, pp_stage, layer_idx, experts_per_layer, 
                          batch_size, seq_len, token_dim, heads, d_k, mlp_hidden):
    """Create nodes for an MoE layer"""
    
    # Multi-Head Attention (same as dense)
    create_mha_nodes(dot, subgraph, layer_name, pp_stage, batch_size, seq_len, token_dim, heads, d_k)
    
    # Expert routing
    expert_base_gpu = pp_stage * 16 + layer_idx * 16  # Map experts across GPUs
    
    # Gate computation
    gate_gpu = expert_base_gpu % 928
    subgraph.node(f'{layer_name}_gate',
                 f'Expert Gate\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\nGPU: {gate_gpu}',
                 shape='rectangle', style='filled', fillcolor='orange')
    
    # Token split based on routing
    subgraph.node(f'{layer_name}_token_split',
                 f'Token Split\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nGPU: {gate_gpu}',
                 shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Individual experts
    for expert_id in range(experts_per_layer):
        expert_gpu = expert_base_gpu + expert_id
        if expert_gpu >= 928:
            expert_gpu = expert_gpu % 928
            
        # Create expert MLP
        with dot.subgraph(name=f'cluster_expert_{layer_name}_{expert_id}') as exp_cluster:
            exp_cluster.attr(label=f'Expert {expert_id}', style='dotted')
            
            # Expert MLP components
            exp_cluster.node(f'{layer_name}_expert_{expert_id}_linear1',
                           f'Expert {expert_id} Linear 1\nInput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nGPU: {expert_gpu}',
                           shape='rectangle', style='filled', fillcolor='lightblue')
            
            exp_cluster.node(f'{layer_name}_expert_{expert_id}_act',
                           f'Expert {expert_id} GELU\nInput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nGPU: {expert_gpu}',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
            
            exp_cluster.node(f'{layer_name}_expert_{expert_id}_linear2',
                           f'Expert {expert_id} Linear 2\nInput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nGPU: {expert_gpu}',
                           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregation
    subgraph.node(f'{layer_name}_expert_aggregate',
                 f'Expert Aggregate\nInput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {gate_gpu}',
                 shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Communication edges (dashed)
    for expert_id in range(experts_per_layer):
        expert_gpu = expert_base_gpu + expert_id
        if expert_gpu >= 928:
            expert_gpu = expert_gpu % 928
            
        dot.edge(f'{layer_name}_token_split', f'{layer_name}_expert_{expert_id}_linear1', 
                style='dashed', label=f'async send to GPU {expert_gpu}')
        dot.edge(f'{layer_name}_expert_{expert_id}_linear2', f'{layer_name}_expert_aggregate',
                style='dashed', label=f'async recv from GPU {expert_gpu}')
    
    # Add residual connection
    subgraph.node(f'{layer_name}_residual',
                 f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {gate_gpu}',
                 shape='parallelogram', style='filled', fillcolor='purple')

def create_mha_nodes(dot, subgraph, layer_name, pp_stage, batch_size, seq_len, token_dim, heads, d_k):
    """Create Multi-Head Attention nodes"""
    
    # QKV projection
    gpu_id = pp_stage * 2  # First GPU for MHA
    subgraph.node(f'{layer_name}_qkv_proj',
                 f'QKV Projection\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k*3}]\nGPU: {gpu_id}',
                 shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Attention computation with ring attention
    for ring_rank in range(4):  # Ring attention across 4 GPUs
        ring_gpu = gpu_id + ring_rank
        subgraph.node(f'{layer_name}_attention_ring_{ring_rank}',
                     f'Ring Attention {ring_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len//4}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len//4}, heads={heads}, d_k={d_k}]\nGPU: {ring_gpu}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Attention output projection
    subgraph.node(f'{layer_name}_attn_out',
                 f'Attention Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {gpu_id}',
                 shape='rectangle', style='filled', fillcolor='lightyellow')

def create_detailed_moe_layer_dag():
    """Create a detailed DAG for one MoE layer"""
    dot = graphviz.Digraph('MOE_Detailed_Layer', format='svg',
                          graph_attr={'rankdir': 'TB', 'ranksep': '1.5'})
    
    # Model parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # Layer 3 (first MoE layer)
    layer_idx = 3
    
    # Add input
    dot.node('layer_input', f'Layer {layer_idx} Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # MHA components
    dot.node('qkv_proj', f'QKV Projection\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k*3}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    dot.node('mha', f'Multi-Head Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightcoral')
    
    dot.node('attn_out', f'Attention Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Expert routing
    dot.node('gate', f'Expert Gate\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='orange')
    
    # Token split and routing
    dot.node('token_split', f'Token Split\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Individual experts (16 experts, one per GPU)
    for expert_id in range(16):
        expert_gpu = layer_idx * 16 + expert_id
        
        # Expert processing
        dot.node(f'expert_{expert_id}_linear1',
                f'Expert {expert_id} Linear 1\nInput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'expert_{expert_id}_act',
                f'Expert {expert_id} GELU\nInput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'expert_{expert_id}_linear2',
                f'Expert {expert_id} Linear 2\nInput: [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\nOutput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregation
    dot.node('expert_aggregate', f'Expert Aggregate\nInput: [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Residual connection
    dot.node('layer_output', f'Layer {layer_idx} Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect the nodes
    dot.edge('layer_input', 'qkv_proj')
    dot.edge('qkv_proj', 'mha')
    dot.edge('mha', 'attn_out')
    dot.edge('attn_out', 'gate')
    dot.edge('gate', 'token_split')
    
    # Connect to experts
    for expert_id in range(16):
        expert_gpu = layer_idx * 16 + expert_id
        dot.edge('token_split', f'expert_{expert_id}_linear1', style='dashed', 
                label=f'routing to GPU {expert_gpu}')
        dot.edge(f'expert_{expert_id}_linear1', f'expert_{expert_id}_act')
        dot.edge(f'expert_{expert_id}_act', f'expert_{expert_id}_linear2')
        dot.edge(f'expert_{expert_id}_linear2', 'expert_aggregate', style='dashed',
                label=f'return from GPU {expert_gpu}')
    
    dot.edge('expert_aggregate', 'layer_output')
    
    # Save detailed layer DAG
    dot.render('../outputs/2025-11-26-16-25-55/moe_detailed_layer', cleanup=False)

def create_tensor_parallel_dag():
    """Create DAG showing tensor parallelism"""
    dot = graphviz.Digraph('Tensor_Parallel_DAG', format='svg',
                          graph_attr={'rankdir': 'LR'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    mlp_hidden = 2048
    
    # Input
    dot.node('tp_input', f'TP Input\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Column parallel split
    dot.node('col_split', f'Column Split\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] ->\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nGPU: 0,1',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Parallel computations
    dot.node('linear1_gpu0', f'Linear 1 GPU0\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    dot.node('linear1_gpu1', f'Linear 1 GPU1\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 1',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    dot.node('gelu_gpu0', f'GELU GPU0\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    dot.node('gelu_gpu1', f'GELU GPU1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 1',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Row parallel split and computation
    dot.node('row_split0', f'Row Split 0\n[batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    dot.node('row_split1', f'Row Split 1\n[batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nGPU: 1',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    dot.node('linear2_gpu0', f'Linear 2 GPU0\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    dot.node('linear2_gpu1', f'Linear 2 GPU1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={mlp_hidden//2}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}]\nGPU: 1',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # All-reduce
    dot.node('allreduce', f'All-Reduce\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}] +\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//2}] ->\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0,1',
             shape='parallelogram', style='dashed', fillcolor='lightblue')
    
    # Output
    dot.node('tp_output', f'TP Output\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('tp_input', 'col_split')
    dot.edge('col_split', 'linear1_gpu0')
    dot.edge('col_split', 'linear1_gpu1')
    dot.edge('linear1_gpu0', 'gelu_gpu0')
    dot.edge('linear1_gpu1', 'gelu_gpu1')
    dot.edge('gelu_gpu0', 'row_split0')
    dot.edge('gelu_gpu1', 'row_split1')
    dot.edge('row_split0', 'linear2_gpu0')
    dot.edge('row_split1', 'linear2_gpu1')
    dot.edge('linear2_gpu0', 'allreduce')
    dot.edge('linear2_gpu1', 'allreduce')
    dot.edge('allreduce', 'tp_output')
    
    # Save TP DAG
    dot.render('../outputs/2025-11-26-16-25-55/tensor_parallel_dag', cleanup=False)

if __name__ == "__main__":
    create_moe_deployment_dag()
    
    # Create additional DAGs for different aspects
    create_data_parallel_dag()
    create_pipeline_parallel_dag()

def create_data_parallel_dag():
    """Create a DAG showing data parallelism"""
    dot = graphviz.Digraph('Data_Parallel_DAG', format='svg',
                          graph_attr={'rankdir': 'TB'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    dp_degree = 8  # 8-way data parallelism
    micro_batch_size = batch_size // dp_degree
    
    # Input split for DP
    dot.node('input', f'Global Batch\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: ALL',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Split nodes
    for dp_rank in range(dp_degree):
        dot.node(f'split_{dp_rank}',
                f'DP Split {dp_rank}\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] ->\n[batch_size={micro_batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {dp_rank*116}',
                shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Compute nodes for each DP replica
    for dp_rank in range(dp_degree):
        dot.node(f'compute_{dp_rank}',
                f'MoE Layer Compute\n[batch_size={micro_batch_size}, seq_len={seq_len}, token_dim={token_dim}] ->\n[batch_size={micro_batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {dp_rank*116}-{(dp_rank+1)*116-1}',
                shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # All-reduce gradient sync
    dot.node('allreduce', f'All-Reduce Gradients\nAcross all DP replicas\nGPU: 0-927',
             shape='parallelogram', style='dashed', fillcolor='purple')
    
    # Output
    dot.node('output', f'Output Gradients\n[batch_size={micro_batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: ALL',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'allreduce')
    for dp_rank in range(dp_degree):
        dot.edge('allreduce', f'split_{dp_rank}')
        dot.edge(f'split_{dp_rank}', f'compute_{dp_rank}')
        dot.edge(f'compute_{dp_rank}', 'allreduce')
    dot.edge('allreduce', 'output')
    
    # Save DP DAG
    dot.render('../outputs/2025-11-26-16-25-55/data_parallel_dag', cleanup=False)

def create_pipeline_parallel_dag():
    """Create a DAG showing pipeline parallelism"""
    dot = graphviz.Digraph('Pipeline_Parallel_DAG', format='svg',
                          graph_attr={'rankdir': 'LR'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    pp_degree = 16
    layers_per_stage = 4
    
    # Create stages
    for stage in range(pp_degree):
        start_layer = stage * layers_per_stage
        end_layer = min((stage + 1) * layers_per_stage, 61)
        
        if stage == 0:
            # Input stage
            dot.node(f'stage_{stage}', 
                    f'Pipeline Stage {stage}\nLayers {start_layer}-{end_layer-1}\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {stage*58}-{stage*58+57}',
                    shape='rectangle', style='filled', fillcolor='lightblue')
        elif stage == pp_degree - 1:
            # Output stage
            dot.node(f'stage_{stage}',
                    f'Pipeline Stage {stage}\nLayers {start_layer}-{end_layer-1}\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\nGPU: {stage*58}-{stage*58+57}',
                    shape='rectangle', style='filled', fillcolor='lightgreen')
        else:
            dot.node(f'stage_{stage}',
                    f'Pipeline Stage {stage}\nLayers {start_layer}-{end_layer-1}\nGPU: {stage*58}-{stage*58+57}',
                    shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Pipeline connections
    for stage in range(pp_degree - 1):
        dot.edge(f'stage_{stage}', f'stage_{stage+1}', 
                label=f'forward pass\nbatch {stage+1}')
    
    # Add micro-batch annotations
    dot.attr(label='16-stage Pipeline Parallelism\\nMicro-batches flow through stages')
    
    # Save PP DAG
    dot.render('../outputs/2025-11-26-16-25-55/pipeline_parallel_dag', cleanup=False)