#!/usr/bin/env python3
"""
Generate comprehensive DAG for large-scale cross-node expert parallelism deployment
based on the provided paper and deployment configuration.
"""

import graphviz
import os

# Define output directory
output_dir = "../outputs/2025-11-27-09-36-58"
os.makedirs(output_dir, exist_ok=True)

def create_detailed_layer_dag():
    """Create detailed DAG for 3 representative layers"""
    detailed_dot = graphviz.Digraph(comment='Detailed MoE Deployment - 3 Representative Layers')
    detailed_dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Define precise tensor dimensions
    batch_size = 8
    seq_len = 1024
    token_dim = 7168
    heads = 128
    d_k = 128
    ffn_hidden = 2048
    
    # Layer 0 (Dense Layer)
    with detailed_dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0 (Dense)', style='rounded')
        
        # Input
        layer0.node('layer0_input', 
                   f'Input\nGPU: 0-3\n[{batch_size},{seq_len},{token_dim}]',
                   shape='ellipse', fillcolor='lightgreen')
        
        # Layer Norm
        layer0.node('layer0_ln', 
                   f'LayerNorm\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                   shape='rectangle', fillcolor='lightyellow')
        
        # MHA across 4 GPUs with tensor parallelism
        for gpu_id in range(4):
            # Q projection
            q_proj = f'layer0_q_proj_gpu{gpu_id}'
            layer0.node(q_proj, 
                       f'Q Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # K projection
            k_proj = f'layer0_k_proj_gpu{gpu_id}'
            layer0.node(k_proj, 
                       f'K Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # V projection
            v_proj = f'layer0_v_proj_gpu{gpu_id}'
            layer0.node(v_proj, 
                       f'V Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # Attention computation
            attn = f'layer0_attn_gpu{gpu_id}'
            layer0.node(attn, 
                       f'Attention\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='rectangle', fillcolor='coral')
            
            # FFN with tensor parallelism
            ffn1 = f'layer0_ffn1_gpu{gpu_id}'
            layer0.node(ffn1, 
                       f'FFN1\nGPU: {gpu_id}\n[{batch_size},{seq_len},{ffn_hidden}]',
                       shape='rectangle', fillcolor='lightblue')
            
            ffn2 = f'layer0_ffn2_gpu{gpu_id}'
            layer0.node(ffn2, 
                       f'FFN2\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='rectangle', fillcolor='lightblue')
            
            # All-reduce for tensor parallel
            all_reduce = f'layer0_all_reduce_gpu{gpu_id}'
            layer0.node(all_reduce, 
                       f'AllReduce\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='parallelogram', fillcolor='orange')
            
            # Connect the chain
            layer0.edge('layer0_input', 'layer0_ln')
            layer0.edge('layer0_ln', q_proj)
            layer0.edge('layer0_ln', k_proj)
            layer0.edge('layer0_ln', v_proj)
            layer0.edge(q_proj, attn)
            layer0.edge(k_proj, attn)
            layer0.edge(v_proj, attn)
            layer0.edge(attn, ffn1)
            layer0.edge(ffn1, ffn2)
            layer0.edge(ffn2, all_reduce)
    
    # Layer 30 (MoE Layer)
    with detailed_dot.subgraph(name='cluster_layer30') as layer30:
        layer30.attr(label='Layer 30 (MoE)', style='rounded')
        
        # Input
        layer30.node('layer30_input', 
                    f'Input\nGPU: 0-63\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
        
        # Gate computation
        gate = 'layer30_gate'
        layer30.node(gate, 
                    f'Gate\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='yellow')
        
        # Expert distribution across 64 GPUs
        for expert_id in range(64):
            gpu_id = expert_id
            node_id = expert_id // 8
            
            # Communication for token routing
            comm = f'layer30_comm_{expert_id}'
            layer30.node(comm, 
                        f'Token Transfer\nFrom GPU: 0\nTo GPU: {gpu_id}\n[1,{seq_len},{token_dim}]',
                        shape='ellipse', fillcolor='purple', style='dashed')
            
            # Expert computation
            expert = f'layer30_expert_{expert_id}'
            layer30.node(expert, 
                        f'Expert {expert_id}\nGPU: {gpu_id}\nNode: {node_id}\n[1,{seq_len},{token_dim}]',
                        shape='rectangle', fillcolor='lightblue')
            
            # Expert FFN
            expert_ffn1 = f'layer30_expert_{expert_id}_ffn1'
            layer30.node(expert_ffn1, 
                        f'Expert FFN1\nGPU: {gpu_id}\n[1,{seq_len},{ffn_hidden}]',
                        shape='rectangle', fillcolor='lightblue')
            
            expert_ffn2 = f'layer30_expert_{expert_id}_ffn2'
            layer30.node(expert_ffn2, 
                        f'Expert FFN2\nGPU: {gpu_id}\n[1,{seq_len},{token_dim}]',
                        shape='rectangle', fillcolor='lightblue')
            
            # Aggregation
            agg = f'layer30_agg_{expert_id}'
            layer30.node(agg, 
                        f'Aggregate\nGPU: {gpu_id}\n[1,{seq_len},{token_dim}]',
                        shape='parallelogram', fillcolor='orange')
            
            # Connections with dashed lines for communication
            layer30.edge('layer30_input', gate)
            layer30.edge(gate, comm, style='dashed')
            layer30.edge(comm, expert_ffn1, style='dashed')
            layer30.edge(expert_ffn1, expert_ffn2)
            layer30.edge(expert_ffn2, agg)
    
    # Layer 60 (Final MoE Layer)
    with detailed_dot.subgraph(name='cluster_layer60') as layer60:
        layer60.attr(label='Layer 60 (Final MoE)', style='rounded')
        
        layer60.node('layer60_input', 
                    f'Input\nGPU: 0-63\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
        
        layer60.node('layer60_gate', 
                    f'Gate\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='yellow')
        
        # Final output aggregation
        layer60.node('layer60_final_agg', 
                    f'Final Aggregation\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='orange')
        
        layer60.node('output', 
                    f'Output\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
        
        layer60.edge('layer60_input', 'layer60_gate')
        layer60.edge('layer60_gate', 'layer60_final_agg')
        layer60.edge('layer60_final_agg', 'output')
    
    # Connect layers
    detailed_dot.edge('layer0_all_reduce_gpu3', 'layer30_input')
    detailed_dot.edge('layer30_agg_63', 'layer60_input')
    
    # Save detailed DAG
    detailed_dot_path = os.path.join(output_dir, "detailed_moe_layers.dot")
    detailed_svg_path = os.path.join(output_dir, "detailed_moe_layers.svg")
    
    with open(detailed_dot_path, 'w') as f:
        f.write(detailed_dot.source)
    
    detailed_dot.render(detailed_svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    return detailed_dot_path, detailed_svg_path

# Create the detailed DAG
print("Creating detailed DAG for 3 representative layers...")
detailed_dot_path, detailed_svg_path = create_detailed_layer_dag()

print(f"Detailed DAG saved to: {detailed_dot_path}")
print(f"Detailed SVG saved to: {detailed_svg_path}")

# Create comprehensive main DAG
print("Creating comprehensive main DAG...")
main_dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
main_dot.attr(rankdir='TB', splines='ortho', compound='true')

# Define main components
batch_size = 8
seq_len = 1024
token_dim = 7168
heads = 128
d_k = 128

# Input node
main_dot.node('input', f'Input\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', fillcolor='lightgreen')

# Dense layers (3 layers represented)
for layer_idx in range(3):
    layer_name = f'dense_{layer_idx}'
    
    # Layer norm
    ln_node = f'{layer_name}_ln'
    main_dot.node(ln_node, 
                 f'LayerNorm\nGPU: 0-3\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='rectangle', fillcolor='lightyellow')
    
    # MHA
    mha_node = f'{layer_name}_mha'
    main_dot.node(mha_node, 
                 f'MHA\nGPU: 0-3\n[{batch_size},{seq_len},{heads},{d_k}]', 
                 shape='rectangle', fillcolor='coral')
    
    # FFN with tensor parallelism
    ffn_node = f'{layer_name}_ffn'
    main_dot.node(ffn_node, 
                 f'FFN\nGPU: 0-3\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='rectangle', fillcolor='lightblue')
    
    # All-reduce
    ar_node = f'{layer_name}_allreduce'
    main_dot.node(ar_node, 
                 f'AllReduce\nGPU: 0-3\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='parallelogram', fillcolor='orange')
    
    # Connect within layer
    main_dot.edge(ln_node, mha_node)
    main_dot.edge(mha_node, ffn_node)
    main_dot.edge(ffn_node, ar_node)

# MoE layers (representative 3 layers)
for layer_idx in [30, 31, 60]:
    layer_name = f'moe_{layer_idx}'
    
    # Gate
    gate_node = f'{layer_name}_gate'
    main_dot.node(gate_node, 
                 f'Gate\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='parallelogram', fillcolor='yellow')
    
    # Expert distribution (64 experts across 64 GPUs)
    expert_node = f'{layer_name}_experts'
    main_dot.node(expert_node, 
                 f'Experts (64)\nGPU: 0-63\n[1,{seq_len},{token_dim}] each', 
                 shape='rectangle', fillcolor='lightblue')
    
    # Communication (dashed)
    comm_node = f'{layer_name}_comm'
    main_dot.node(comm_node, 
                 f'Token Routing\nCross-GPU\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='ellipse', fillcolor='purple', style='dashed')
    
    # Aggregation
    agg_node = f'{layer_name}_agg'
    main_dot.node(agg_node, 
                 f'Aggregate\nGPU: 0-63\n[{batch_size},{seq_len},{token_dim}]', 
                 shape='parallelogram', fillcolor='orange')
    
    # Connect with dashed lines for communication
    main_dot.edge(gate_node, comm_node, style='dashed', color='red')
    main_dot.edge(comm_node, expert_node, style='dashed', color='red')
    main_dot.edge(expert_node, agg_node)

# Connect layers
main_dot.edge('input', 'dense_0_ln')
main_dot.edge('dense_2_allreduce', 'moe_30_gate')
main_dot.edge('moe_31_agg', 'moe_60_gate')

# Output
main_dot.node('output', f'Output\n[{batch_size},{seq_len},{token_dim}]', 
             shape='ellipse', fillcolor='lightgreen')
main_dot.edge('moe_60_agg', 'output')

# Save main DAG
main_dot_path = os.path.join(output_dir, "main_moe_deployment.dot")
main_svg_path = os.path.join(output_dir, "main_moe_deployment.svg")

with open(main_dot_path, 'w') as f:
    f.write(main_dot.source)

main_dot.render(main_svg_path.replace('.svg', ''), format='svg', cleanup=True)

print(f"Main DAG saved to: {main_dot_path}")
print(f"Main SVG saved to: {main_svg_path}")

# Create submission JSON
import json
submission = {
    "dag_files": {
        "main_deployment": {
            "dot_file": main_dot_path,
            "svg_file": main_svg_path
        },
        "detailed_layers": {
            "dot_file": detailed_dot_path,
            "svg_file": detailed_svg_path
        }
    },
    "deployment_strategy": {
        "parallel_strategy": "large_scale_cross_node_expert_parallelism",
        "expert_parallelism": 64,
        "tensor_parallelism": 2,
        "gpus_total": 64,
        "representative_layers": 3,
        "node_distribution": "topology_aware",
        "communication_overlap": "asynchronous_cuda_streams",
        "load_balancing": "dynamic_gate_probabilities"
    },
    "tensor_dimensions": {
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "token_dimension": token_dim,
        "attention_heads": heads,
        "head_dimension": d_k,
        "ffn_hidden_size": 2048
    }
}

submission_path = os.path.join(output_dir, "submission.json")
with open(submission_path, 'w') as f:
    json.dump(submission, f, indent=2)

print(f"Submission saved to: {submission_path}")
print("DAG generation complete!")