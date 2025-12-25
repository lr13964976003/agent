#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_llm_parallel_dag():
    """Create a comprehensive DAG for LLM parallel inference with EP-16 × TP-4 × PP-1"""
    
    # Create the DAG
    dot = Digraph(comment='LLM Parallel Inference DAG - EP-16 × TP-4 × PP-1')
    dot.attr(rankdir='TB', size='50,30', ranksep='1.5', nodesep='0.8')
    dot.attr('node', fontsize='10', height='0.6', width='2.0')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model configuration
    batch_size = 128
    seq_len = 512
    hidden_size = 512
    num_heads = 16
    head_dim = 32
    num_experts = 16
    top_k = 2
    ffn_hidden = 1024
    
    # GPU configuration
    total_gpus = 64
    ep_degree = 16  # Expert Parallelism
    tp_degree = 4   # Tensor Parallelism
    pp_degree = 1   # Pipeline Parallelism
    
    # Each expert group has 4 GPUs for TP
    gpus_per_expert = tp_degree
    total_expert_groups = ep_degree
    
    # Input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
             shape='ellipse', fillcolor='lightcoral')
    
    # Token embedding (distributed across all GPUs)
    for gpu_id in range(total_gpus):
        expert_group = gpu_id // tp_degree
        tp_rank = gpu_id % tp_degree
        
        dot.node(f'embedding_{gpu_id}', 
                 f'Embedding_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_degree}]',
                 shape='box', fillcolor='lightgreen')
    
    # Connect input to all embeddings
    for gpu_id in range(total_gpus):
        dot.edge('input', f'embedding_{gpu_id}')
    
    # Process each layer
    for layer in range(16):
        # Layer input aggregation
        dot.node(f'layer_{layer}_input', 
                 f'Layer{layer}_Input_Aggregate\\nInput: Distributed\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                 shape='parallelogram', fillcolor='lightyellow')
        
        # Connect embeddings to first layer
        if layer == 0:
            for gpu_id in range(total_gpus):
                if gpu_id % tp_degree == 0:  # Only connect from TP rank 0
                    dot.edge(f'embedding_{gpu_id}', f'layer_{layer}_input')
        
        # Attention computation for each expert group
        for expert_group in range(total_expert_groups):
            base_gpu = expert_group * tp_degree
            
            # QKV projection (column parallel)
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_qkv_{gpu_id}', 
                         f'Layer{layer}_QKV_Proj_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//tp_degree}, d_k={head_dim}]',
                         shape='box', fillcolor='lightgreen')
            
            # Attention scores (distributed)
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_attn_scores_{gpu_id}', 
                         f'Layer{layer}_Attention_Scores_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//tp_degree}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, heads={num_heads//tp_degree}, seq_len={seq_len}, seq_len={seq_len}]',
                         shape='box', fillcolor='lightgreen')
            
            # Attention softmax
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_attn_softmax_{gpu_id}', 
                         f'Layer{layer}_Attention_Softmax_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, heads={num_heads//tp_degree}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={num_heads//tp_degree}, seq_len={seq_len}, seq_len={seq_len}]',
                         shape='box', fillcolor='lightgreen')
            
            # Attention output (row parallel)
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_attn_out_{gpu_id}', 
                         f'Layer{layer}_Attention_Output_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, heads={num_heads//tp_degree}, seq_len={seq_len}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_degree}]',
                         shape='box', fillcolor='lightgreen')
            
            # Attention all-reduce
            dot.node(f'layer_{layer}_attn_allreduce_{expert_group}', 
                     f'Layer{layer}_Attention_AllReduce_ExpertGroup{expert_group}\\nInput: Distributed\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                     shape='ellipse', fillcolor='lightblue')
            
            # Connect attention components
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.edge(f'layer_{layer}_input', f'layer_{layer}_qkv_{gpu_id}')
                dot.edge(f'layer_{layer}_qkv_{gpu_id}', f'layer_{layer}_attn_scores_{gpu_id}')
                dot.edge(f'layer_{layer}_attn_scores_{gpu_id}', f'layer_{layer}_attn_softmax_{gpu_id}')
                dot.edge(f'layer_{layer}_attn_softmax_{gpu_id}', f'layer_{layer}_attn_out_{gpu_id}')
                dot.edge(f'layer_{layer}_attn_out_{gpu_id}', f'layer_{layer}_attn_allreduce_{expert_group}')
        
        # Expert routing (gate)
        for expert_group in range(total_expert_groups):
            base_gpu = expert_group * tp_degree
            gpu_id = base_gpu  # Routing typically on first GPU of group
            
            dot.node(f'layer_{layer}_gate_{gpu_id}', 
                     f'Layer{layer}_Expert_Gate_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts={num_experts}]',
                     shape='parallelogram', fillcolor='lightyellow')
        
        # All-to-all communication for expert routing
        dot.node(f'layer_{layer}_alltoall', 
                 f'Layer{layer}_Expert_Routing_AllToAll\\nInput: Token assignments\\nOutput: Routed tokens to expert groups',
                 shape='ellipse', fillcolor='lightblue')
        
        # Expert computation (only top-2 experts active per token)
        for expert_group in range(total_expert_groups):
            base_gpu = expert_group * tp_degree
            
            # Expert MLP - first linear (column parallel)
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_expert1_{gpu_id}', 
                         f'Layer{layer}_Expert{expert_group}_MLP1_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden//tp_degree}]',
                         shape='box', fillcolor='lightgreen')
            
            # Expert activation
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_expert_act_{gpu_id}', 
                         f'Layer{layer}_Expert{expert_group}_GELU_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden//tp_degree}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden//tp_degree}]',
                         shape='box', fillcolor='lightgreen')
            
            # Expert MLP - second linear (row parallel)
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.node(f'layer_{layer}_expert2_{gpu_id}', 
                         f'Layer{layer}_Expert{expert_group}_MLP2_GPU{gpu_id}\\nTP-Rank{tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={ffn_hidden//tp_degree}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size//tp_degree}]',
                         shape='box', fillcolor='lightgreen')
            
            # Expert output all-reduce
            dot.node(f'layer_{layer}_expert_allreduce_{expert_group}', 
                     f'Layer{layer}_Expert{expert_group}_AllReduce\\nInput: Distributed\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                     shape='ellipse', fillcolor='lightblue')
        
        # Expert aggregation (weighted sum)
        dot.node(f'layer_{layer}_expert_agg', 
                 f'Layer{layer}_Expert_Aggregation\\nInput: Expert outputs from top-2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                 shape='parallelogram', fillcolor='lightyellow')
        
        # Layer normalization
        dot.node(f'layer_{layer}_norm', 
                 f'Layer{layer}_LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]',
                 shape='box', fillcolor='lightgreen')
        
        # Connect layer components
        for expert_group in range(total_expert_groups):
            # Connect to gate (dashed line for routing decision)
            base_gpu = expert_group * tp_degree
            gpu_id = base_gpu
            dot.edge(f'layer_{layer}_attn_allreduce_{expert_group}', f'layer_{layer}_gate_{gpu_id}', style='dashed')
            dot.edge(f'layer_{layer}_gate_{gpu_id}', f'layer_{layer}_alltoall', style='dashed')
            
            # Connect to expert computation
            for tp_rank in range(tp_degree):
                gpu_id = base_gpu + tp_rank
                dot.edge(f'layer_{layer}_alltoall', f'layer_{layer}_expert1_{gpu_id}')
                dot.edge(f'layer_{layer}_expert1_{gpu_id}', f'layer_{layer}_expert_act_{gpu_id}')
                dot.edge(f'layer_{layer}_expert_act_{gpu_id}', f'layer_{layer}_expert2_{gpu_id}')
                dot.edge(f'layer_{layer}_expert2_{gpu_id}', f'layer_{layer}_expert_allreduce_{expert_group}')
            
            # Connect to aggregation
            dot.edge(f'layer_{layer}_expert_allreduce_{expert_group}', f'layer_{layer}_expert_agg')
        
        # Connect to layer norm
        dot.edge(f'layer_{layer}_expert_agg', f'layer_{layer}_norm')
        
        # Connect to next layer input
        if layer < 15:
            dot.edge(f'layer_{layer}_norm', f'layer_{layer+1}_input')
    
    # Final output
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size=?]',
             shape='ellipse', fillcolor='lightcoral')
    
    # Connect final layer to output
    dot.edge('layer_15_norm', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the DAG
    dag = create_llm_parallel_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-25-09-26-32/llm_parallel_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-25-09-26-32/llm_parallel_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-25-09-26-32/llm_parallel_dag.dot")
    print(f"SVG file: ../outputs/2025-12-25-09-26-32/llm_parallel_dag.svg")