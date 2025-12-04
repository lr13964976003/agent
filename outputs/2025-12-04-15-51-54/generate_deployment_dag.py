#!/usr/bin/env python3
"""
Generate a comprehensive DAG for the 128-GPU MoE model deployment
with perfect load balancing using expert, tensor, and pipeline parallelism.
"""

import graphviz
from typing import Dict, List, Tuple
import os

def generate_deployment_dag():
    """Generate the complete deployment DAG for 128 GPUs."""
    
    # Create the main DAG
    dot = graphviz.Digraph('MoE_128GPU_Deployment', 
                          comment='128-GPU MoE Model Deployment DAG',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'splines': 'ortho',
                              'ranksep': '1.0',
                              'nodesep': '0.5',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12'
                          })
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Global input dimensions
    batch_size = "?"
    seq_len = "?"
    hidden_size = 4096
    ffn_hidden_size = 16384
    num_heads = 32
    head_dim = hidden_size // num_heads
    vocab_size = 50000
    num_experts = 64
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor='lightgreen', style='filled,bold')
    
    # Create pipeline stages (4 stages, 32 GPUs each)
    for stage_idx in range(4):
        with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage:
            stage.attr(label=f'Pipeline Stage {stage_idx} (Layers {stage_idx*4}-{stage_idx*4+3})',
                      style='rounded,filled', fillcolor='lightyellow', color='blue')
            
            # Each stage has 16 experts (2 GPUs per expert due to tensor parallelism)
            for expert_idx in range(16):
                global_expert_id = stage_idx * 16 + expert_idx
                gpu_base = stage_idx * 32 + expert_idx * 2
                
                with stage.subgraph(name=f'cluster_expert_{global_expert_id}') as expert:
                    expert.attr(label=f'Expert {global_expert_id}',
                               style='rounded,filled', fillcolor='lightgray', color='red')
                    
                    # GPU 0 of tensor parallel pair
                    gpu0_id = gpu_base
                    # GPU 1 of tensor parallel pair
                    gpu1_id = gpu_base + 1
                    
                    # Process 4 layers for this expert
                    for layer_idx in range(4):
                        global_layer_id = stage_idx * 4 + layer_idx
                        
                        # Create attention computation nodes
                        attn_norm_0 = f'attn_norm_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        attn_q_0 = f'attn_q_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        attn_k_0 = f'attn_k_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        attn_v_0 = f'attn_v_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        attn_out_0 = f'attn_out_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        
                        attn_norm_1 = f'attn_norm_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        attn_q_1 = f'attn_q_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        attn_k_1 = f'attn_k_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        attn_v_1 = f'attn_k_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        attn_out_1 = f'attn_out_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        
                        # Attention normalization (both GPUs)
                        expert.node(attn_norm_0, 
                                  f'LayerNorm GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                  fillcolor='lightblue')
                        expert.node(attn_norm_1,
                                  f'LayerNorm GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                  fillcolor='lightblue')
                        
                        # Attention Q projection (column parallel)
                        expert.node(attn_q_0,
                                  f'Attn-Q GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]',
                                  fillcolor='lightcoral')
                        expert.node(attn_q_1,
                                  f'Attn-Q GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]',
                                  fillcolor='lightcoral')
                        
                        # Attention K,V projection (column parallel)
                        expert.node(attn_k_0,
                                  f'Attn-K/V GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]',
                                  fillcolor='lightcoral')
                        expert.node(attn_k_1,
                                  f'Attn-K/V GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]',
                                  fillcolor='lightcoral')
                        
                        # Attention output projection (row parallel)
                        expert.node(attn_out_0,
                                  f'Attn-Out GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                                  fillcolor='lightcoral')
                        expert.node(attn_out_1,
                                  f'Attn-Out GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//2}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                                  fillcolor='lightcoral')
                        
                        # MLP computation nodes
                        mlp_gate_0 = f'mlp_gate_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        mlp_up_0 = f'mlp_up_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        mlp_down_0 = f'mlp_down_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu0_id}'
                        
                        mlp_gate_1 = f'mlp_gate_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        mlp_up_1 = f'mlp_up_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        mlp_down_1 = f'mlp_down_s{stage_idx}_e{global_expert_id}_l{layer_idx}_gpu{gpu1_id}'
                        
                        # MLP Gate (column parallel)
                        expert.node(mlp_gate_0,
                                  f'MLP-Gate GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]',
                                  fillcolor='lightcyan')
                        expert.node(mlp_gate_1,
                                  f'MLP-Gate GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]',
                                  fillcolor='lightcyan')
                        
                        # MLP Up projection (column parallel)
                        expert.node(mlp_up_0,
                                  f'MLP-Up GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]',
                                  fillcolor='lightcyan')
                        expert.node(mlp_up_1,
                                  f'MLP-Up GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]',
                                  fillcolor='lightcyan')
                        
                        # MLP Down projection (row parallel)
                        expert.node(mlp_down_0,
                                  f'MLP-Down GPU{gpu0_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                                  fillcolor='lightcyan')
                        expert.node(mlp_down_1,
                                  f'MLP-Down GPU{gpu1_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_size={ffn_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                                  fillcolor='lightcyan')
    
    # Add communication nodes and edges
    # All-reduce for attention output
    for stage_idx in range(4):
        for expert_idx in range(16):
            global_expert_id = stage_idx * 16 + expert_idx
            for layer_idx in range(4):
                attn_allreduce = f'attn_allreduce_s{stage_idx}_e{global_expert_id}_l{layer_idx}'
                dot.node(attn_allreduce, 
                        f'All-Reduce Attn\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                        shape='ellipse', fillcolor='yellow', style='filled,dashed')
                
                mlp_allreduce = f'mlp_allreduce_s{stage_idx}_e{global_expert_id}_l{layer_idx}'
                dot.node(mlp_allreduce,
                        f'All-Reduce MLP\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                        shape='ellipse', fillcolor='yellow', style='filled,dashed')
    
    # Expert routing (MoE gate)
    moe_gate = f'moe_gate'
    dot.node(moe_gate,
            f'MoE Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
            shape='parallelogram', fillcolor='orange', style='filled,dashed')
    
    # Expert selection
    expert_select = f'expert_select'
    dot.node(expert_select,
            f'Expert Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, selected_experts=2]',
            shape='parallelogram', fillcolor='orange', style='filled,dashed')
    
    # Add edges (simplified for clarity)
    # Input to first stage
    for expert_idx in range(16):
        dot.edge('input', f'attn_norm_s0_e{expert_idx}_l0_gpu{expert_idx*2}', style='solid')
        dot.edge('input', f'attn_norm_s0_e{expert_idx}_l0_gpu{expert_idx*2+1}', style='solid')
    
    # Add output node
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]',
             shape='ellipse', fillcolor='lightgreen', style='filled,bold')
    
    # Save the DOT file
    dot.save('../outputs/2025-12-04-15-51-54/moe_deployment_dag.dot')
    
    # Render to SVG
    dot.render('../outputs/2025-12-04-15-51-54/moe_deployment_dag', format='svg', cleanup=False)
    
    return '../outputs/2025-12-04-15-51-54/moe_deployment_dag.dot', '../outputs/2025-12-04-15-51-54/moe_deployment_dag.svg'

def generate_simplified_dag():
    """Generate a simplified DAG showing the high-level structure."""
    
    dot = graphviz.Digraph('MoE_128GPU_Simplified',
                          comment='Simplified 128-GPU MoE Deployment DAG',
                          format='svg',
                          graph_attr={
                              'rankdir': 'TB',
                              'splines': 'ortho',
                              'ranksep': '1.5',
                              'nodesep': '0.8',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '14'
                          })
    
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # Input
    dot.node('input', 'Input Tokens\\n[batch_size, seq_len]', 
             shape='ellipse', fillcolor='lightgreen', style='filled,bold')
    
    # Create pipeline stages
    for stage_idx in range(4):
        with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage:
            stage.attr(label=f'Pipeline Stage {stage_idx}\\nLayers {stage_idx*4}-{stage_idx*4+3}\\nGPUs {stage_idx*32}-{(stage_idx+1)*32-1}',
                      style='rounded,filled', fillcolor='lightyellow', color='blue', penwidth='3')
            
            # 16 experts per stage
            for expert_idx in range(16):
                global_expert_id = stage_idx * 16 + expert_idx
                gpu_pair = f'{stage_idx*32 + expert_idx*2}-{stage_idx*32 + expert_idx*2 + 1}'
                
                expert_node = f'expert_{global_expert_id}_stage_{stage_idx}'
                stage.node(expert_node,
                          f'Expert {global_expert_id}\\nGPUs {gpu_pair}\\n(Tensor Parallel)\\n4 Layers',
                          fillcolor='lightcyan', penwidth='2')
    
    # MoE Gate and routing
    dot.node('moe_gate', 'MoE Gate\\nExpert Routing\\n[64 experts, top-2 selection]',
             shape='parallelogram', fillcolor='orange', style='filled,dashed', penwidth='3')
    
    # Communication nodes
    dot.node('allreduce_attn', 'All-Reduce\\nAttention Outputs',
             shape='ellipse', fillcolor='yellow', style='filled,dashed', penwidth='2')
    
    dot.node('allreduce_mlp', 'All-Reduce\\nMLP Outputs',
             shape='ellipse', fillcolor='yellow', style='filled,dashed', penwidth='2')
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\\nTop-2 Expert Combination',
             shape='parallelogram', fillcolor='orange', style='filled,dashed', penwidth='3')
    
    # Output
    dot.node('output', 'Output Logits\\n[vocab_size]',
             shape='ellipse', fillcolor='lightgreen', style='filled,bold')
    
    # Add edges
    dot.edge('input', 'moe_gate', style='solid', penwidth='2')
    
    # Routing to experts (dashed)
    for stage_idx in range(4):
        for expert_idx in range(16):
            global_expert_id = stage_idx * 16 + expert_idx
            expert_node = f'expert_{global_expert_id}_stage_{stage_idx}'
            dot.edge('moe_gate', expert_node, style='dashed', penwidth='2', color='red')
    
    # All-reduce communication
    dot.edge('allreduce_attn', 'allreduce_mlp', style='dashed', penwidth='2', color='blue')
    
    # Expert aggregation
    dot.edge('allreduce_mlp', 'expert_agg', style='solid', penwidth='2')
    
    # Final output
    dot.edge('expert_agg', 'output', style='solid', penwidth='3')
    
    # Save files
    dot.save('../outputs/2025-12-04-15-51-54/moe_simplified_dag.dot')
    dot.render('../outputs/2025-12-04-15-51-54/moe_simplified_dag', format='svg', cleanup=False)
    
    return '../outputs/2025-12-04-15-51-54/moe_simplified_dag.dot', '../outputs/2025-12-04-15-51-54/moe_simplified_dag.svg'

def generate_communication_dag():
    """Generate a DAG focusing on communication patterns."""
    
    dot = graphviz.Digraph('MoE_128GPU_Communication',
                          comment='Communication Patterns in 128-GPU MoE Deployment',
                          format='svg',
                          graph_attr={
                              'rankdir': 'LR',
                              'splines': 'curved',
                              'ranksep': '2.0',
                              'nodesep': '1.0',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '14'
                          })
    
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # Input data
    dot.node('input_data', 'Input Data\\n[batch_size, seq_len, hidden_size]',
             shape='ellipse', fillcolor='lightgreen', style='filled,bold')
    
    # Expert parallelism communication
    dot.node('expert_parallel', 'Expert Parallelism\\n64-way Partitioning',
             shape='parallelogram', fillcolor='orange', style='filled,dashed', penwidth='3')
    
    # Tensor parallelism communication
    dot.node('tensor_parallel', 'Tensor Parallelism\\n2-way Partitioning',
             shape='parallelogram', fillcolor='purple', style='filled,dashed', penwidth='3')
    
    # Pipeline parallelism communication
    dot.node('pipeline_parallel', 'Pipeline Parallelism\\n4-stage Pipeline',
             shape='parallelogram', fillcolor='green', style='filled,dashed', penwidth='3')
    
    # All-reduce operations
    dot.node('allreduce_tensor', 'All-Reduce\\nTensor Parallel',
             shape='ellipse', fillcolor='yellow', style='filled,dashed', penwidth='2')
    
    dot.node('allgather_expert', 'All-Gather\\nExpert Parallel',
             shape='ellipse', fillcolor='yellow', style='filled,dashed', penwidth='2')
    
    dot.node('sendrecv_pipeline', 'Send/Recv\\nPipeline Parallel',
             shape='ellipse', fillcolor='yellow', style='filled,dashed', penwidth='2')
    
    # GPU groups
    for i in range(4):
        dot.node(f'gpu_group_{i}', f'GPU Group {i}\\nGPUs {i*32}-{i*32+31}\\n(Pipeline Stage {i})',
                fillcolor='lightcyan', penwidth='2')
    
    # Add edges
    dot.edge('input_data', 'expert_parallel', style='solid', penwidth='2')
    dot.edge('expert_parallel', 'tensor_parallel', style='dashed', penwidth='2', color='red')
    dot.edge('tensor_parallel', 'pipeline_parallel', style='dashed', penwidth='2', color='blue')
    
    # Communication patterns
    dot.edge('tensor_parallel', 'allreduce_tensor', style='dashed', penwidth='2', color='purple')
    dot.edge('expert_parallel', 'allgather_expert', style='dashed', penwidth='2', color='orange')
    dot.edge('pipeline_parallel', 'sendrecv_pipeline', style='dashed', penwidth='2', color='green')
    
    # Connect to GPU groups
    for i in range(4):
        dot.edge('allreduce_tensor', f'gpu_group_{i}', style='solid', penwidth='2')
        dot.edge('allgather_expert', f'gpu_group_{i}', style='solid', penwidth='2')
        dot.edge('sendrecv_pipeline', f'gpu_group_{i}', style='solid', penwidth='2')
    
    # Cross-stage communication
    for i in range(3):
        dot.edge(f'gpu_group_{i}', f'gpu_group_{i+1}', 
                style='dashed', penwidth='2', color='green', 
                label='Pipeline Send/Recv')
    
    # Save files
    dot.save('../outputs/2025-12-04-15-51-54/moe_communication_dag.dot')
    dot.render('../outputs/2025-12-04-15-51-54/moe_communication_dag', format='svg', cleanup=False)
    
    return '../outputs/2025-12-04-15-51-54/moe_communication_dag.dot', '../outputs/2025-12-04-15-51-54/moe_communication_dag.svg'

if __name__ == '__main__':
    print("Generating comprehensive MoE deployment DAGs...")
    
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-12-04-15-51-54', exist_ok=True)
    
    # Generate all DAGs
    detailed_dot, detailed_svg = generate_deployment_dag()
    simplified_dot, simplified_svg = generate_simplified_dag()
    communication_dot, communication_svg = generate_communication_dag()
    
    print(f"Generated detailed DAG: {detailed_dot}")
    print(f"Generated simplified DAG: {simplified_dot}")
    print(f"Generated communication DAG: {communication_dot}")
    
    # Create summary
    summary = {
        "detailed_dag_dot": detailed_dot,
        "detailed_dag_svg": detailed_svg,
        "simplified_dag_dot": simplified_dot,
        "simplified_dag_svg": simplified_svg,
        "communication_dag_dot": communication_dot,
        "communication_dag_svg": communication_svg,
        "generated_files": [
            detailed_dot,
            detailed_svg.replace('.svg', '.dot'),
            simplified_dot,
            simplified_svg.replace('.svg', '.dot'),
            communication_dot,
            communication_svg.replace('.svg', '.dot')
        ]
    }
    
    # Save summary
    import json
    with open('../outputs/2025-12-04-15-51-54/dag_files_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("All DAGs generated successfully!")
    print("Files saved to: ../outputs/2025-12-04-15-51-54/")