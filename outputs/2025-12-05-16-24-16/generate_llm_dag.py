#!/usr/bin/env python3
"""
Generate DAG for 30B MoE Model with 3D Parallelism
- Expert Parallelism: 64 experts across 8 GPUs (8 experts per GPU)
- Tensor Parallelism: 2 GPUs for attention and MoE layers
- Pipeline Parallelism: 4 stages with 4 layers each
"""

import graphviz
import os

def create_llm_deployment_dag():
    """Create comprehensive DAG for 30B MoE model deployment"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='30B MoE Model 3D Parallel Deployment DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Global configurations
    num_gpus = 8
    num_layers = 16
    num_experts = 64
    experts_per_gpu = 8
    pipeline_stages = 4
    layers_per_stage = 4
    tensor_parallel_size = 2
    
    # Initialize node counter
    node_id = 0
    
    # Input node
    input_node = f'input_{node_id}'
    dot.node(input_node, f'Input\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='lightblue')
    node_id += 1
    
    # Previous node tracker
    prev_node = input_node
    
    # Generate pipeline stages
    for stage_idx in range(pipeline_stages):
        # Calculate GPU range for this pipeline stage
        start_gpu = stage_idx * (num_gpus // pipeline_stages)  # 2 GPUs per stage
        end_gpu = start_gpu + (num_gpus // pipeline_stages)
        
        with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage:
            stage.attr(label=f'Pipeline Stage {stage_idx} (GPUs {start_gpu}-{end_gpu-1})', 
                      style='rounded,filled', fillcolor='lightgray', fontname='Arial Bold')
            
            # Generate layers within stage
            for layer_idx in range(layers_per_stage):
                global_layer_idx = stage_idx * layers_per_stage + layer_idx
                
                # Create subgraph for layer
                with stage.subgraph(name=f'cluster_layer_{global_layer_idx}') as layer_cluster:
                    layer_cluster.attr(label=f'Layer {global_layer_idx}', 
                                     style='rounded,filled', fillcolor='white', fontname='Arial Bold')
                    
                    # Layer Input Aggregation
                    layer_input = f'layer_{global_layer_idx}_input'
                    layer_cluster.node(layer_input, 
                                     f'Layer Input\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]',
                                     shape='parallelogram', fillcolor='lightyellow')
                    
                    # Connect to previous layer
                    if global_layer_idx == 0:
                        dot.edge(prev_node, layer_input)
                    else:
                        dot.edge(prev_layer_output, layer_input)
                    
                    # Attention Block with Tensor Parallelism
                    # QKV Projection - Tensor Parallel across 2 GPUs
                    qkv_tp0 = f'layer_{global_layer_idx}_qkv_tp0'
                    qkv_tp1 = f'layer_{global_layer_idx}_qkv_tp1'
                    
                    layer_cluster.node(qkv_tp0, 
                                     f'QKV Projection TP0\\nGPU {start_gpu}\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, 16 heads, 64 d_k]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    layer_cluster.node(qkv_tp1, 
                                     f'QKV Projection TP1\\nGPU {start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, 16 heads, 64 d_k]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    # Connect layer input to QKV projections
                    dot.edge(layer_input, qkv_tp0)
                    dot.edge(layer_input, qkv_tp1)
                    
                    # Attention Computation - Split across GPUs
                    attn_tp0 = f'layer_{global_layer_idx}_attn_tp0'
                    attn_tp1 = f'layer_{global_layer_idx}_attn_tp1'
                    
                    layer_cluster.node(attn_tp0, 
                                     f'Attention Score TP0\\nGPU {start_gpu}\\nInput: [batch_size=128, seq_len=1024, 8 heads, 64 d_k]\\nOutput: [batch_size=128, seq_len=1024, 8 heads, 1024]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    layer_cluster.node(attn_tp1, 
                                     f'Attention Score TP1\\nGPU {start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 8 heads, 64 d_k]\\nOutput: [batch_size=128, seq_len=1024, 8 heads, 1024]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    # Connect QKV to attention
                    dot.edge(qkv_tp0, attn_tp0)
                    dot.edge(qkv_tp1, attn_tp1)
                    
                    # Attention Output Communication
                    attn_comm = f'layer_{global_layer_idx}_attn_comm'
                    layer_cluster.node(attn_comm, 
                                     f'Attention Output All-Reduce\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 1024]\\nOutput: [batch_size=128, seq_len=1024, 1024]',
                                     shape='ellipse', fillcolor='lightblue')
                    
                    dot.edge(attn_tp0, attn_comm)
                    dot.edge(attn_tp1, attn_comm)
                    
                    # Attention Output Projection
                    attn_out = f'layer_{global_layer_idx}_attn_out'
                    layer_cluster.node(attn_out, 
                                     f'Attention Output Projection\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 1024]\\nOutput: [batch_size=128, seq_len=1024, 1024]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    dot.edge(attn_comm, attn_out)
                    
                    # MoE Block with Expert Parallelism
                    # Gate Network
                    gate = f'layer_{global_layer_idx}_gate'
                    layer_cluster.node(gate, 
                                     f'MoE Gate Network\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 1024]\\nOutput: [batch_size=128, seq_len=1024, 64 experts]',
                                     shape='parallelogram', fillcolor='lightyellow')
                    
                    dot.edge(attn_out, gate)
                    
                    # Expert Selection (Dashed line)
                    expert_select = f'layer_{global_layer_idx}_expert_select'
                    layer_cluster.node(expert_select, 
                                     f'Expert Selection\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 64 experts]\\nOutput: [batch_size=128, seq_len=1024, top-2 experts]',
                                     shape='parallelogram', fillcolor='lightyellow', style='dashed')
                    
                    dot.edge(gate, expert_select, style='dashed')
                    
                    # Expert Computation - 8 experts per GPU across all 8 GPUs
                    expert_nodes = []
                    for gpu_idx in range(num_gpus):
                        start_expert = gpu_idx * experts_per_gpu
                        end_expert = start_expert + experts_per_gpu
                        
                        # Expert processing on each GPU
                        expert_gpu = f'layer_{global_layer_idx}_experts_gpu{gpu_idx}'
                        layer_cluster.node(expert_gpu, 
                                         f'Experts {start_expert}-{end_expert-1}\\nGPU {gpu_idx}\\nInput: [batch_size=16, seq_len=1024, 1024]\\nOutput: [batch_size=16, seq_len=1024, 2048]',
                                         shape='rectangle', fillcolor='lightgreen')
                        
                        expert_nodes.append(expert_gpu)
                        
                        # Connect expert selection to experts
                        dot.edge(expert_select, expert_gpu)
                    
                    # Expert Communication - All-to-All
                    expert_comm = f'layer_{global_layer_idx}_expert_comm'
                    layer_cluster.node(expert_comm, 
                                     f'Expert All-to-All Communication\\nAll GPUs\\nInput: [batch_size=128, seq_len=1024, 2048]\\nOutput: [batch_size=128, seq_len=1024, 2048]',
                                     shape='ellipse', fillcolor='lightblue')
                    
                    for expert_node in expert_nodes:
                        dot.edge(expert_node, expert_comm)
                    
                    # MoE Output Aggregation
                    moe_out = f'layer_{global_layer_idx}_moe_out'
                    layer_cluster.node(moe_out, 
                                     f'MoE Output Aggregation\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 2048]\\nOutput: [batch_size=128, seq_len=1024, 1024]',
                                     shape='parallelogram', fillcolor='lightyellow')
                    
                    dot.edge(expert_comm, moe_out)
                    
                    # Layer Norm
                    layer_norm = f'layer_{global_layer_idx}_norm'
                    layer_cluster.node(layer_norm, 
                                     f'Layer Normalization\\nGPUs {start_gpu}-{start_gpu+1}\\nInput: [batch_size=128, seq_len=1024, 1024]\\nOutput: [batch_size=128, seq_len=1024, 1024]',
                                     shape='rectangle', fillcolor='lightgreen')
                    
                    dot.edge(moe_out, layer_norm)
                    
                    # Set this as output for next layer
                    prev_layer_output = layer_norm
    
    # Final output node
    output_node = f'output_{node_id}'
    dot.node(output_node, 
             f'Final Output\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]',
             shape='ellipse', fillcolor='lightblue')
    
    dot.edge(prev_layer_output, output_node)
    
    return dot

def main():
    """Main function to generate and save DAG"""
    output_dir = "../outputs/2025-12-05-16-24-16"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "llm_30b_moe_deployment_dag.dot")
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "llm_30b_moe_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated and saved:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    return {
        "dot_file": dot_file,
        "svg_file": svg_file
    }

if __name__ == "__main__":
    result = main()
    print(result)