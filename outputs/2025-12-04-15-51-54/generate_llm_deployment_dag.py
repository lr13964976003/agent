#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def create_llm_deployment_dag():
    """
    Create a comprehensive DAG for LLM deployment with:
    - 128 GPUs in 4 pipeline stages
    - 64 experts with 2-way tensor parallelism
    - 16 transformer layers
    - Detailed operator-level nodes
    """
    
    # Create the main DAG
    dot = Digraph(comment='LLM Deployment DAG - 128 GPUs with Expert, Tensor, and Pipeline Parallelism')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node',shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal', arrowsize='0.5')
    
    # Define shapes for different node types
    computation_attrs = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    communication_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'}
    routing_attrs = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'yellow'}
    input_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightcoral'}
    output_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightpink'}
    
    # Model dimensions
    batch_size = 128
    seq_len = 1024  # representative sequence length
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    ffn_hidden = 2048
    num_experts = 64
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', 
             **input_attrs)
    
    # Create pipeline stages
    for stage in range(4):
        stage_start_gpu = stage * 32
        stage_label = f'Pipeline Stage {stage} (Layers {stage*4}-{(stage+1)*4-1})'
        
        with dot.subgraph(name=f'cluster_stage_{stage}') as stage_subgraph:
            stage_subgraph.attr(label=stage_label, style='rounded,filled', fillcolor='lightgray', rank='same')
            
            # Create experts for this stage (16 experts per stage)
            for expert_idx in range(16):
                global_expert_id = stage * 16 + expert_idx
                gpu_base = stage_start_gpu + expert_idx * 2
                
                expert_label = f'Expert {global_expert_id} (GPUs {gpu_base}-{gpu_base+1})'
                
                with stage_subgraph.subgraph(name=f'cluster_expert_{global_expert_id}') as expert_subgraph:
                    expert_subgraph.attr(label=expert_label, style='rounded,filled', fillcolor='lightcyan')
                    
                    # Create 4 layers for this expert in this pipeline stage
                    for layer in range(4):
                        global_layer = stage * 4 + layer
                        
                        # Layer Norm (shared across tensor parallel group)
                        ln_node = f'ln_stage{stage}_expert{global_expert_id}_layer{layer}'
                        dot.node(ln_node, 
                                f'LayerNorm (Layer {global_layer})\\nGPU: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                                **computation_attrs)
                        
                        # Self Attention - QKV projection (column parallel)
                        qkv_node_gpu0 = f'qkv_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        qkv_node_gpu1 = f'qkv_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(qkv_node_gpu0,
                                f'QKV Projection GPU{gpu_base} (Column Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        dot.node(qkv_node_gpu1,
                                f'QKV Projection GPU{gpu_base+1} (Column Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, qkv_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        # Attention computation (split across GPUs)
                        attn_node_gpu0 = f'attn_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        attn_node_gpu1 = f'attn_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(attn_node_gpu0,
                                f'Self Attention GPU{gpu_base}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        dot.node(attn_node_gpu1,
                                f'Self Attention GPU{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        # Attention output projection (row parallel)
                        attn_out_node_gpu0 = f'attn_out_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        attn_out_node_gpu1 = f'attn_out_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(attn_out_node_gpu0,
                                f'Attention Output GPU{gpu_base} (Row Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        dot.node(attn_out_node_gpu1,
                                f'Attention Output GPU{gpu_base+1} (Row Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, head_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        # Attention all-reduce
                        attn_allreduce = f'attn_allreduce_stage{stage}_expert{global_expert_id}_layer{layer}'
                        dot.node(attn_allreduce,
                                f'Attention All-Reduce\\nGPUs: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                                **communication_attrs)
                        
                        # MoE Gate (routing)
                        gate_node = f'gate_stage{stage}_expert{global_expert_id}_layer{layer}'
                        dot.node(gate_node,
                                f'MoE Gate (Layer {global_layer})\\nGPU: {gpu_base}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                                **routing_attrs)
                        
                        # MoE Expert processing (first linear - column parallel)
                        moe_first_gpu0 = f'moe_first_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        moe_first_gpu1 = f'moe_first_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(moe_first_gpu0,
                                f'MoE First Linear GPU{gpu_base} (Column Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]',
                                **computation_attrs)
                        
                        dot.node(moe_first_gpu1,
                                f'MoE First Linear GPU{gpu_base+1} (Column Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]',
                                **computation_attrs)
                        
                        # GELU activation
                        gelu_node_gpu0 = f'gelu_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        gelu_node_gpu1 = f'gelu_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(gelu_node_gpu0,
                                f'GELU Activation GPU{gpu_base}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]',
                                **computation_attrs)
                        
                        dot.node(gelu_node_gpu1,
                                f'GELU Activation GPU{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]',
                                **computation_attrs)
                        
                        # MoE Expert processing (second linear - row parallel)
                        moe_second_gpu0 = f'moe_second_stage{stage}_expert{global_expert_id}_layer{layer}_gpu0'
                        moe_second_gpu1 = f'moe_second_stage{stage}_expert{global_expert_id}_layer{layer}_gpu1'
                        
                        dot.node(moe_second_gpu0,
                                f'MoE Second Linear GPU{gpu_base} (Row Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        dot.node(moe_second_gpu1,
                                f'MoE Second Linear GPU{gpu_base+1} (Row Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_dim={ffn_hidden//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]',
                                **computation_attrs)
                        
                        # MoE all-reduce
                        moe_allreduce = f'moe_allreduce_stage{stage}_expert{global_expert_id}_layer{layer}'
                        dot.node(moe_allreduce,
                                f'MoE All-Reduce\\nGPUs: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                                **communication_attrs)
                        
                        # Layer output
                        layer_output = f'layer_out_stage{stage}_expert{global_expert_id}_layer{layer}'
                        dot.node(layer_output,
                                f'Layer {global_layer} Output\\nGPU: {gpu_base}-{gpu_base+1}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                                **computation_attrs)
                        
                        # Connect nodes within layer
                        if layer == 0 and expert_idx == 0 and stage == 0:
                            # First layer connects to input
                            dot.edge('input', ln_node)
                        
                        dot.edge(ln_node, qkv_node_gpu0)
                        dot.edge(ln_node, qkv_node_gpu1)
                        
                        dot.edge(qkv_node_gpu0, attn_node_gpu0)
                        dot.edge(qkv_node_gpu1, attn_node_gpu1)
                        
                        dot.edge(attn_node_gpu0, attn_out_node_gpu0)
                        dot.edge(attn_node_gpu1, attn_out_node_gpu1)
                        
                        dot.edge(attn_out_node_gpu0, attn_allreduce)
                        dot.edge(attn_out_node_gpu1, attn_allreduce)
                        
                        dot.edge(attn_allreduce, gate_node)
                        dot.edge(attn_allreduce, moe_first_gpu0)
                        dot.edge(attn_allreduce, moe_first_gpu1)
                        
                        dot.edge(moe_first_gpu0, gelu_node_gpu0)
                        dot.edge(moe_first_gpu1, gelu_node_gpu1)
                        
                        dot.edge(gelu_node_gpu0, moe_second_gpu0)
                        dot.edge(gelu_node_gpu1, moe_second_gpu1)
                        
                        dot.edge(moe_second_gpu0, moe_allreduce)
                        dot.edge(moe_second_gpu1, moe_allreduce)
                        
                        dot.edge(moe_allreduce, layer_output)
                        
                        # Expert routing communication (dashed lines)
                        for target_expert in range(16):
                            if target_expert != expert_idx:
                                target_gpu_base = stage_start_gpu + target_expert * 2
                                dot.edge(gate_node, f'moe_first_stage{stage}_expert{stage*16+target_expert}_layer{layer}_gpu0', 
                                        style='dashed', color='red', 
                                        label=f'expert_routing_to_GPU{target_gpu_base}')
                                dot.edge(gate_node, f'moe_first_stage{stage}_expert{stage*16+target_expert}_layer{layer}_gpu1', 
                                        style='dashed', color='red',
                                        label=f'expert_routing_to_GPU{target_gpu_base+1}')
    
    # Connect pipeline stages
    for stage in range(3):  # Connect stage 0->1, 1->2, 2->3
        for expert_idx in range(16):
            global_expert_id = stage * 16 + expert_idx
            gpu_base = stage * 32 + expert_idx * 2
            
            for layer in range(4):
                global_layer = stage * 4 + layer
                next_stage = stage + 1
                next_global_layer = next_stage * 4
                
                current_output = f'layer_out_stage{stage}_expert{global_expert_id}_layer{layer}'
                next_input = f'ln_stage{next_stage}_expert{global_expert_id}_layer{0}'
                
                if layer == 3:  # Last layer in stage
                    dot.edge(current_output, next_input, 
                            color='blue', style='bold',
                            label=f'pipeline_stage_{stage}_to_{next_stage}')
    
    # Output node
    output_node = 'final_output'
    dot.node(output_node, 
            f'Final Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
            **output_attrs)
    
    # Connect final layer outputs to final output
    for expert_idx in range(16):
        global_expert_id = 3 * 16 + expert_idx  # Stage 3 experts
        final_layer_output = f'layer_out_stage3_expert{global_expert_id}_layer3'
        dot.edge(final_layer_output, output_node, color='green', style='bold')
    
    return dot

if __name__ == '__main__':
    # Create the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-04-15-51-54/llm_deployment_dag.dot'
    dag.save(dot_file_path)
    print(f"DAG saved to {dot_file_path}")
    
    # Save as SVG image
    svg_file_path = '../outputs/2025-12-04-15-51-54/llm_deployment_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG image saved to {svg_file_path}")
    
    # Print summary
    print(f"Total nodes in DAG: {len(dag.body)}")
    print("DAG generation completed successfully!")