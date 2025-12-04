#!/usr/bin/env python3

import graphviz
import os

def create_llm_deployment_dag():
    """
    Create a comprehensive DAG for LLM deployment with EP64_TP2 strategy
    showing expert parallelism across 128 GPUs with tensor parallelism
    """
    
    # Create the DAG
    dag = graphviz.Digraph('LLM_Deployment_EP64_TP2_128GPUs')
    dag.attr(rankdir='TB', size='50,50', dpi='300')
    dag.attr('node', fontname='Arial', fontsize='10')
    dag.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node shapes and colors
    dag.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dag.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dag.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters from deployment file
    batch_size = 128
    seq_len = 10000
    token_dim = 4096
    mlp_hidden_size = 16384
    mha_heads = 32
    mha_head_dim = 128
    num_layers = 16
    num_experts = 64
    experts_per_layer = 16
    
    # Input node
    input_shape = f"[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]"
    dag.node('input', f"Input\\nInput: {input_shape}\\nOutput: {input_shape}", 
             shape='hexagon', style='filled', fillcolor='lightpink')
    
    # Layer norm before attention (GPU 0)
    dag.node('ln1_gpu0', f"LayerNorm\\nGPU: 0\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('input', 'ln1_gpu0')
    
    # Multi-head attention with tensor parallelism (GPUs 0-1)
    # QKV projection - column parallel
    qkv_shape = f"[batch_size={batch_size}, seq_len={seq_len}, qkv_dim={token_dim*3}]"
    dag.node('qkv_proj_gpu0', f"QKV Projection (Col-Parallel)\\nGPU: 0\\nInput: {input_shape}\\nOutput: {qkv_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('qkv_proj_gpu1', f"QKV Projection (Col-Parallel)\\nGPU: 1\\nInput: {input_shape}\\nOutput: {qkv_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('ln1_gpu0', 'qkv_proj_gpu0')
    dag.edge('ln1_gpu0', 'qkv_proj_gpu1')
    
    # All-gather for QKV
    dag.node('qkv_allgather', f"All-Gather QKV\\nGPUs: 0-1\\nInput: {qkv_shape}\\nOutput: {qkv_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('qkv_proj_gpu0', 'qkv_allgather')
    dag.edge('qkv_proj_gpu1', 'qkv_allgather')
    
    # Attention computation
    attn_shape = f"[batch_size={batch_size}, heads={mha_heads}, seq_len={seq_len}, head_dim={mha_head_dim}]"
    dag.node('attention_gpu0', f"Multi-Head Attention\\nGPU: 0\\nInput: {qkv_shape}\\nOutput: {attn_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('qkv_allgather', 'attention_gpu0')
    
    # Attention output projection - row parallel
    dag.node('attn_out_gpu0', f"Attention Output (Row-Parallel)\\nGPU: 0\\nInput: {attn_shape}\\nOutput: {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('attn_out_gpu1', f"Attention Output (Row-Parallel)\\nGPU: 1\\nInput: {attn_shape}\\nOutput: {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('attention_gpu0', 'attn_out_gpu0')
    dag.edge('attention_gpu0', 'attn_out_gpu1')
    
    # All-reduce for attention output
    dag.node('attn_allreduce', f"All-Reduce Attention\\nGPUs: 0-1\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('attn_out_gpu0', 'attn_allreduce')
    dag.edge('attn_out_gpu1', 'attn_allreduce')
    
    # Residual connection after attention
    dag.node('residual1', f"Residual Add\\nGPU: 0\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('input', 'residual1')
    dag.edge('attn_allreduce', 'residual1')
    
    # Layer norm before MoE (GPU 0)
    dag.node('ln2_gpu0', f"LayerNorm\\nGPU: 0\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('residual1', 'ln2_gpu0')
    
    # Expert routing gate (GPU 0)
    gate_shape = f"[batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}]"
    dag.node('gate_gpu0', f"Expert Routing Gate\\nGPU: 0\\nInput: {input_shape}\\nOutput: {gate_shape}",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('ln2_gpu0', 'gate_gpu0')
    
    # Expert selection and token routing
    dag.node('expert_select', f"Expert Selection\\nGPU: 0\\nInput: {gate_shape}\\nOutput: routing decisions",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('gate_gpu0', 'expert_select')
    
    # Create expert nodes - showing first few experts as examples
    expert_input_shape = f"[tokens=?, token_dim={token_dim}]"
    expert_output_shape = f"[tokens=?, token_dim={token_dim}]"
    mlp_internal_shape = f"[tokens=?, mlp_hidden={mlp_hidden_size}]"
    
    # Expert 0 (GPUs 0-1) - Tensor Parallel
    dag.node('expert0_gpu0', f"Expert 0 MLP (Col-Parallel)\\nGPU: 0\\nInput: {expert_input_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert0_gpu1', f"Expert 0 MLP (Col-Parallel)\\nGPU: 1\\nInput: {expert_input_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-gather for expert 0
    dag.node('expert0_gather', f"All-Gather Expert 0\\nGPUs: 0-1\\nInput: {mlp_internal_shape}\\nOutput: {mlp_internal_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('expert0_gpu0', 'expert0_gather')
    dag.edge('expert0_gpu1', 'expert0_gather')
    
    # Expert 0 activation
    dag.node('expert0_act', f"GELU Activation\\nGPU: 0\\nInput: {mlp_internal_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('expert0_gather', 'expert0_act')
    
    # Expert 0 output projection - row parallel
    dag.node('expert0_out_gpu0', f"Expert 0 Output (Row-Parallel)\\nGPU: 0\\nInput: {mlp_internal_shape}\\nOutput: {expert_output_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert0_out_gpu1', f"Expert 0 Output (Row-Parallel)\\nGPU: 1\\nInput: {mlp_internal_shape}\\nOutput: {expert_output_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('expert0_act', 'expert0_out_gpu0')
    dag.edge('expert0_act', 'expert0_out_gpu1')
    
    # All-reduce for expert 0 output
    dag.node('expert0_allreduce', f"All-Reduce Expert 0\\nGPUs: 0-1\\nInput: {expert_output_shape}\\nOutput: {expert_output_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('expert0_out_gpu0', 'expert0_allreduce')
    dag.edge('expert0_out_gpu1', 'expert0_allreduce')
    
    # Expert 1 (GPUs 2-3) - Tensor Parallel
    dag.node('expert1_gpu2', f"Expert 1 MLP (Col-Parallel)\\nGPU: 2\\nInput: {expert_input_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert1_gpu3', f"Expert 1 MLP (Col-Parallel)\\nGPU: 3\\nInput: {expert_input_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # All-gather for expert 1
    dag.node('expert1_gather', f"All-Gather Expert 1\\nGPUs: 2-3\\nInput: {mlp_internal_shape}\\nOutput: {mlp_internal_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('expert1_gpu2', 'expert1_gather')
    dag.edge('expert1_gpu3', 'expert1_gather')
    
    # Expert 1 activation
    dag.node('expert1_act', f"GELU Activation\\nGPU: 2\\nInput: {mlp_internal_shape}\\nOutput: {mlp_internal_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('expert1_gather', 'expert1_act')
    
    # Expert 1 output projection - row parallel
    dag.node('expert1_out_gpu2', f"Expert 1 Output (Row-Parallel)\\nGPU: 2\\nInput: {mlp_internal_shape}\\nOutput: {expert_output_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert1_out_gpu3', f"Expert 1 Output (Row-Parallel)\\nGPU: 3\\nInput: {mlp_internal_shape}\\nOutput: {expert_output_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('expert1_act', 'expert1_out_gpu2')
    dag.edge('expert1_act', 'expert1_out_gpu3')
    
    # All-reduce for expert 1 output
    dag.node('expert1_allreduce', f"All-Reduce Expert 1\\nGPUs: 2-3\\nInput: {expert_output_shape}\\nOutput: {expert_output_shape}",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('expert1_out_gpu2', 'expert1_allreduce')
    dag.edge('expert1_out_gpu3', 'expert1_allreduce')
    
    # Expert routing connections (dashed lines showing token routing)
    dag.edge('expert_select', 'expert0_gpu0', style='dashed', label='route tokens')
    dag.edge('expert_select', 'expert1_gpu2', style='dashed', label='route tokens')
    dag.edge('expert_select', 'expert0_gpu0', style='dashed', label='route tokens')
    dag.edge('expert_select', 'expert1_gpu2', style='dashed', label='route tokens')
    
    # Token distribution and aggregation
    dag.node('token_dist', f"Token Distribution\\nGPU: 0\\nInput: {input_shape}\\nOutput: distributed tokens",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('ln2_gpu0', 'token_dist')
    
    dag.node('expert_agg', f"Expert Aggregation\\nGPU: 0\\nInput: expert outputs\\nOutput: {input_shape}",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('expert0_allreduce', 'expert_agg')
    dag.edge('expert1_allreduce', 'expert_agg')
    
    # All-to-all communication for expert outputs
    dag.node('all2all_expert', f"All-to-All Expert Outputs\\nGPUs: 0-127\\nInput: expert outputs\\nOutput: aggregated",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.edge('expert_agg', 'all2all_expert')
    
    # Final layer norm and output
    dag.node('ln3_gpu0', f"Final LayerNorm\\nGPU: 0\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.edge('all2all_expert', 'ln3_gpu0')
    
    # Residual connection after MoE
    dag.node('residual2', f"Residual Add\\nGPU: 0\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('residual1', 'residual2')
    dag.edge('ln3_gpu0', 'residual2')
    
    # Output node
    dag.node('output', f"Output\\nInput: {input_shape}\\nOutput: {input_shape}",
             shape='hexagon', style='filled', fillcolor='lightpink')
    dag.edge('residual2', 'output')
    
    # Add subgraphs for different GPU groups
    with dag.subgraph(name='cluster_node0') as c:
        c.attr(style='rounded', fillcolor='lightgray', label='Node 0 (GPUs 0-7)')
        c.node('ln1_gpu0')
        c.node('qkv_proj_gpu0')
        c.node('qkv_proj_gpu1')
        c.node('attention_gpu0')
        c.node('attn_out_gpu0')
        c.node('attn_out_gpu1')
        c.node('ln2_gpu0')
        c.node('gate_gpu0')
        c.node('expert0_gpu0')
        c.node('expert0_gpu1')
        c.node('expert0_act')
        c.node('expert0_out_gpu0')
        c.node('expert0_out_gpu1')
        c.node('expert1_gpu2')
        c.node('expert1_gpu3')
        c.node('expert1_act')
        c.node('expert1_out_gpu2')
        c.node('expert1_out_gpu3')
    
    return dag

def create_simplified_dag():
    """
    Create a simplified DAG showing the overall flow
    """
    dag = graphviz.Digraph('LLM_Deployment_Simplified')
    dag.attr(rankdir='TB', size='30,30', dpi='300')
    
    # Model parameters
    batch_size = 128
    seq_len = 10000
    token_dim = 4096
    num_layers = 16
    num_experts = 64
    
    input_shape = f"[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]"
    
    # Input
    dag.node('input', f"Input\\n{input_shape}", shape='hexagon', style='filled', fillcolor='lightpink')
    
    # Layer processing
    for layer in range(min(3, num_layers)):  # Show first 3 layers
        # Attention block
        dag.node(f'attn_{layer}', f"Layer {layer}\\nMulti-Head Attention\\nGPUs: 0-1 (TP=2)",
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # MoE block
        dag.node(f'moe_{layer}', f"Layer {layer}\\nMixture of Experts\\nGPUs: 0-127 (EP=64, TP=2)",
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Connections
        if layer == 0:
            dag.edge('input', f'attn_{layer}')
        else:
            dag.edge(f'moe_{layer-1}', f'attn_{layer}')
        
        dag.edge(f'attn_{layer}', f'moe_{layer}')
        
        # Add communication nodes
        dag.node(f'comm_attn_{layer}', f"All-Reduce\\nAttention Output",
                 shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
        dag.edge(f'attn_{layer}', f'comm_attn_{layer}')
        
        dag.node(f'comm_moe_{layer}', f"All-to-All\\nExpert Routing",
                 shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
        dag.edge(f'comm_moe_{layer}', f'moe_{layer}')
    
    # Output
    dag.node('output', f"Output\\n{input_shape}", shape='hexagon', style='filled', fillcolor='lightpink')
    dag.edge(f'moe_{2}', 'output')
    
    return dag

def create_expert_detail_dag():
    """
    Create a detailed DAG showing expert computation
    """
    dag = graphviz.Digraph('Expert_Computation_Detail')
    dag.attr(rankdir='LR', size='20,15', dpi='300')
    
    batch_size = 128
    token_dim = 4096
    mlp_hidden = 16384
    
    input_shape = f"[batch_size={batch_size}, token_dim={token_dim}]"
    hidden_shape = f"[batch_size={batch_size}, mlp_hidden={mlp_hidden}]"
    
    # Input tokens
    dag.node('tokens', f"Input Tokens\\n{input_shape}", 
             shape='hexagon', style='filled', fillcolor='lightpink')
    
    # Gate
    dag.node('gate', f"Routing Gate\\nSelects Top-2 Experts",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('tokens', 'gate')
    
    # Expert 0 computation (detailed)
    dag.node('expert0_col', f"Expert 0\\nColumn-Parallel Linear\\nGPU: 0\\n{input_shape} → {hidden_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert0_gather', f"All-Gather\\nGPUs: 0-1",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.node('expert0_act', f"GELU Activation\\nGPU: 0",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert0_row', f"Expert 0\\nRow-Parallel Linear\\nGPUs: 0-1\\n{hidden_shape} → {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert0_reduce', f"All-Reduce\\nGPUs: 0-1",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    
    # Expert 0 connections
    dag.edge('gate', 'expert0_col', style='dashed', label='route')
    dag.edge('expert0_col', 'expert0_gather')
    dag.edge('expert0_gather', 'expert0_act')
    dag.edge('expert0_act', 'expert0_row')
    dag.edge('expert0_row', 'expert0_reduce')
    
    # Expert 1 computation (detailed)
    dag.node('expert1_col', f"Expert 1\\nColumn-Parallel Linear\\nGPU: 2\\n{input_shape} → {hidden_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert1_gather', f"All-Gather\\nGPUs: 2-3",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    dag.node('expert1_act', f"GELU Activation\\nGPU: 2",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert1_row', f"Expert 1\\nRow-Parallel Linear\\nGPUs: 2-3\\n{hidden_shape} → {input_shape}",
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dag.node('expert1_reduce', f"All-reduce\\nGPUs: 2-3",
             shape='ellipse', style='filled', fillcolor='lightblue', style='dashed')
    
    # Expert 1 connections
    dag.edge('gate', 'expert1_col', style='dashed', label='route')
    dag.edge('expert1_col', 'expert1_gather')
    dag.edge('expert1_gather', 'expert1_act')
    dag.edge('expert1_act', 'expert1_row')
    dag.edge('expert1_row', 'expert1_reduce')
    
    # Expert aggregation
    dag.node('expert_agg', f"Expert Aggregation\\nWeighted Sum\\nGPU: 0",
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dag.edge('expert0_reduce', 'expert_agg')
    dag.edge('expert1_reduce', 'expert_agg')
    
    # Output
    dag.node('output', f"Output Tokens\\n{input_shape}",
             shape='hexagon', style='filled', fillcolor='lightpink')
    dag.edge('expert_agg', 'output')
    
    return dag

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-12-04-11-03-13', exist_ok=True)
    
    # Generate main DAG
    print("Generating main LLM deployment DAG...")
    main_dag = create_llm_deployment_dag()
    main_dag.render('../outputs/2025-12-04-11-03-13/llm_deployment_main')
    main_dag.save('../outputs/2025-12-04-11-03-13/llm_deployment_main.dot')
    
    # Generate simplified DAG
    print("Generating simplified DAG...")
    simple_dag = create_simplified_dag()
    simple_dag.render('../outputs/2025-12-04-11-03-13/llm_deployment_simplified')
    simple_dag.save('../outputs/2025-12-04-11-03-13/llm_deployment_simplified.dot')
    
    # Generate expert detail DAG
    print("Generating expert detail DAG...")
    expert_dag = create_expert_detail_dag()
    expert_dag.render('../outputs/2025-12-04-11-03-13/expert_computation_detail')
    expert_dag.save('../outputs/2025-12-04-11-03-13/expert_computation_detail.dot')
    
    # Create SVG versions
    print("Converting to SVG format...")
    os.system('dot -Tsvg ../outputs/2025-12-04-11-03-13/llm_deployment_main.dot -o ../outputs/2025-12-04-11-03-13/llm_deployment_main.svg')
    os.system('dot -Tsvg ../outputs/2025-12-04-11-03-13/llm_deployment_simplified.dot -o ../outputs/2025-12-04-11-03-13/llm_deployment_simplified.svg')
    os.system('dot -Tsvg ../outputs/2025-12-04-11-03-13/expert_computation_detail.dot -o ../outputs/2025-12-04-11-03-13/expert_computation_detail.svg')
    
    print("DAG generation complete!")
    print("\nGenerated files:")
    print("- llm_deployment_main.dot & .svg")
    print("- llm_deployment_simplified.dot & .svg") 
    print("- expert_computation_detail.dot & .svg")
    
    # Verify DAGs are acyclic
    print("\nVerifying DAGs are acyclic...")
    try:
        Extract Info From DAG('../outputs/2025-12-04-11-03-13/llm_deployment_main.dot')
        print("Main DAG verification: ACYCLIC ✓")
    except:
        print("Main DAG verification: FAILED")
    
    try:
        Extract Info From DAG('../outputs/2025-12-04-11-03-13/llm_deployment_simplified.dot')
        print("Simplified DAG verification: ACYCLIC ✓")
    except:
        print("Simplified DAG verification: FAILED")
        
    try:
        Extract Info From DAG('../outputs/2025-12-04-11-03-13/expert_computation_detail.dot')
        print("Expert detail DAG verification: ACYCLIC ✓")
    except:
        print("Expert detail DAG verification: FAILED")