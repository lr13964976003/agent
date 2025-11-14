#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def create_baseline_tensor_pipeline_dag():
    """Create DAG for baseline tensor parallelism + pipeline parallelism"""
    dot = Digraph('baseline_tensor_pipeline_parallel', 
                  comment='16-layer Transformer with Tensor Parallelism (TP=8) and Pipeline Parallelism (PP=2)')
    
    dot.attr(rankdir='TB', size='20,20')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Input node
    dot.node('input', 'Model Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nAll GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (GPUs 0-7)
    dot.attr('node', fillcolor='lightblue')
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0\nGPUs 0-7 (Tensor Parallel Group 0)')
        
        # Layer 1
        stage0.node('layer1_norm1', 'Layer 1\nInput Layer Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightyellow')
        
        # Attention for Layer 1
        stage0.node('layer1_qkv_proj', 'Layer 1 QKV Projection\nColumn Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 1536]\nPer GPU', fillcolor='lightcoral')
        stage0.node('layer1_attn', 'Layer 1 Multi-Head Attention\nInput: [128, 10000, 32, 128]\nOutput: [128, 10000, 32, 128]\nAll 8 GPUs', fillcolor='lightcoral')
        stage0.node('layer1_attn_out', 'Layer 1 Attention Output\nRow Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 512]\nPer GPU', fillcolor='lightcoral')
        stage0.node('layer1_attn_allreduce', 'Layer 1 All-Reduce\nSum Reduction\nInput: [128, 10000, 512]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgoldenrod')
        stage0.node('layer1_residual1', 'Layer 1 Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgreen')
        
        # MLP for Layer 1
        stage0.node('layer1_norm2', 'Layer 1 MLP Layer Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightyellow')
        stage0.node('layer1_mlp_gate_up', 'Layer 1 MLP Gate+Up\nColumn Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nPer GPU', fillcolor='lightcyan')
        stage0.node('layer1_mlp_activation', 'Layer 1 MLP Activation (GELU)\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightcyan')
        stage0.node('layer1_mlp_down', 'Layer 1 MLP Down\nRow Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 512]\nPer GPU', fillcolor='lightcyan')
        stage0.node('layer1_mlp_allreduce', 'Layer 1 MLP All-Reduce\nSum Reduction\nInput: [128, 10000, 512]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgoldenrod')
        stage0.node('layer1_residual2', 'Layer 1 Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgreen')
        
        # Repeat for layers 2-8 with ellipsis
        stage0.node('layers_2_to_8', 'Layers 2-8\n[Repeated Structure]\n8 layers total\nEach: Same as Layer 1', fillcolor='lightgray')
    
    # Pipeline communication
    dot.node('pipeline_send_recv_1', 'Pipeline Send/Recv\nBetween Stage 0 and 1\nActivation: [128, 10000, 4096]\nBandwidth: 900 Gbps', 
             shape='parallelogram', fillcolor='lightpink')
    
    # Pipeline Stage 1 (GPUs 8-15)
    dot.attr('node', fillcolor='lightblue')
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1\nGPUs 8-15 (Tensor Parallel Group 1)')
        
        # Layer 9
        stage1.node('layer9_norm1', 'Layer 9\nInput Layer Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightyellow')
        
        # Attention for Layer 9 (similar structure)
        stage1.node('layer9_qkv_proj', 'Layer 9 QKV Projection\nColumn Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 1536]\nPer GPU', fillcolor='lightcoral')
        stage1.node('layer9_attn', 'Layer 9 Multi-Head Attention\nInput: [128, 10000, 32, 128]\nOutput: [128, 10000, 32, 128]\nAll 8 GPUs', fillcolor='lightcoral')
        stage1.node('layer9_attn_out', 'Layer 9 Attention Output\nRow Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 512]\nPer GPU', fillcolor='lightcoral')
        stage1.node('layer9_attn_allreduce', 'Layer 9 All-Reduce\nSum Reduction\nInput: [128, 10000, 512]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgoldenrod')
        stage1.node('layer9_residual1', 'Layer 9 Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgreen')
        
        # MLP for Layer 9
        stage1.node('layer9_norm2', 'Layer 9 MLP Layer Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightyellow')
        stage1.node('layer9_mlp_gate_up', 'Layer 9 MLP Gate+Up\nColumn Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nPer GPU', fillcolor='lightcyan')
        stage1.node('layer9_mlp_activation', 'Layer 9 MLP Activation (GELU)\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightcyan')
        stage1.node('layer9_mlp_down', 'Layer 9 MLP Down\nRow Parallel\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 512]\nPer GPU', fillcolor='lightcyan')
        stage1.node('layer9_mlp_allreduce', 'Layer 9 MLP All-Reduce\nSum Reduction\nInput: [128, 10000, 512]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgoldenrod')
        stage1.node('layer9_residual2', 'Layer 9 Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', fillcolor='lightgreen')
        
        # Repeat for layers 10-16 with ellipsis
        stage1.node('layers_10_to_16', 'Layers 10-16\n[Repeated Structure]\n8 layers total\nEach: Same as Layer 9', fillcolor='lightgray')
    
    # Output node
    dot.node('output', 'Model Output\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nAll 8 GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections for baseline
    dot.edge('input', 'layer1_norm1')
    dot.edge('layer1_norm1', 'layer1_qkv_proj')
    dot.edge('layer1_qkv_proj', 'layer1_attn')
    dot.edge('layer1_attn', 'layer1_attn_out')
    dot.edge('layer1_attn_out', 'layer1_attn_allreduce')
    dot.edge('layer1_attn_allreduce', 'layer1_residual1')
    dot.edge('input', 'layer1_residual1')  # Residual connection
    
    dot.edge('layer1_residual1', 'layer1_norm2')
    dot.edge('layer1_norm2', 'layer1_mlp_gate_up')
    dot.edge('layer1_mlp_gate_up', 'layer1_mlp_activation')
    dot.edge('layer1_mlp_activation', 'layer1_mlp_down')
    dot.edge('layer1_mlp_down', 'layer1_mlp_allreduce')
    dot.edge('layer1_mlp_allreduce', 'layer1_residual2')
    dot.edge('layer1_residual1', 'layer1_residual2')  # Residual connection
    
    dot.edge('layer1_residual2', 'layers_2_to_8')
    dot.edge('layers_2_to_8', 'pipeline_send_recv_1')
    dot.edge('pipeline_send_recv_1', 'layer9_norm1')
    dot.edge('layer9_residual2', 'layers_10_to_16')
    dot.edge('layers_10_to_16', 'output')
    
    return dot

def create_proposed_layer_wise_dag():
    """Create DAG for proposed layer-wise parallelism"""
    dot = Digraph('proposed_layer_wise_parallel', 
                  comment='16-layer Transformer with Layer-wise Parallelism (1 layer per GPU)')
    
    dot.attr(rankdir='TB', size='30,30')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Input node
    dot.node('input', 'Model Input\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\nGPU 0', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create nodes for each layer on each GPU
    for layer_id in range(1, 17):
        gpu_id = layer_id - 1
        
        # Each layer has the complete transformer structure
        with dot.subgraph(name=f'cluster_layer{layer_id}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer_id}\nGPU {gpu_id} (L2 Cache Optimized)')
            
            # Input norm
            layer_cluster.node(f'layer{layer_id}_norm1', 
                             f'Layer {layer_id} Input Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightyellow')
            
            # Self-attention components
            layer_cluster.node(f'layer{layer_id}_qkv_proj', 
                             f'Layer {layer_id} QKV Projection\nCompressed: 33.6 MB\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 12288]\nGPU {gpu_id}', 
                             fillcolor='lightcoral')
            layer_cluster.node(f'layer{layer_id}_q_split', 
                             f'Layer {layer_id} Q Split\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 32, 128]\nGPU {gpu_id}', 
                             fillcolor='lightpink')
            layer_cluster.node(f'layer{layer_id}_k_split', 
                             f'Layer {layer_id} K Split\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 32, 128]\nGPU {gpu_id}', 
                             fillcolor='lightpink')
            layer_cluster.node(f'layer{layer_id}_v_split', 
                             f'Layer {layer_id} V Split\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 32, 128]\nGPU {gpu_id}', 
                             fillcolor='lightpink')
            layer_cluster.node(f'layer{layer_id}_attention', 
                             f'Layer {layer_id} Multi-Head Attention\nInput: [128, 10000, 32, 128]\nOutput: [128, 10000, 32, 128]\nGPU {gpu_id}', 
                             fillcolor='lightcoral')
            layer_cluster.node(f'layer{layer_id}_attention_concat', 
                             f'Layer {layer_id} Attention Concat\nInput: [128, 10000, 32, 128]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightcoral')
            layer_cluster.node(f'layer{layer_id}_attention_out', 
                             f'Layer {layer_id} Attention Output\nCompressed: 33.6 MB\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightcoral')
            layer_cluster.node(f'layer{layer_id}_residual1', 
                             f'Layer {layer_id} Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightgreen')
            
            # MLP components
            layer_cluster.node(f'layer{layer_id}_norm2', 
                             f'Layer {layer_id} MLP Norm\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightyellow')
            layer_cluster.node(f'layer{layer_id}_mlp_gate', 
                             f'Layer {layer_id} MLP Gate\nCompressed: 50.3 MB\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 16384]\nGPU {gpu_id}', 
                             fillcolor='lightcyan')
            layer_cluster.node(f'layer{layer_id}_mlp_up', 
                             f'Layer {layer_id} MLP Up\nCompressed: 50.3 MB\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 16384]\nGPU {gpu_id}', 
                             fillcolor='lightcyan')
            layer_cluster.node(f'layer{layer_id}_mlp_activation', 
                             f'Layer {layer_id} MLP Activation (GELU)\nInput: [128, 10000, 16384]\nOutput: [128, 10000, 16384]\nGPU {gpu_id}', 
                             fillcolor='lightcyan')
            layer_cluster.node(f'layer{layer_id}_mlp_down', 
                             f'Layer {layer_id} MLP Down\nCompressed: 50.3 MB\nInput: [128, 10000, 16384]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightcyan')
            layer_cluster.node(f'layer{layer_id}_residual2', 
                             f'Layer {layer_id} Residual Add\nInput1: [128, 10000, 4096]\nInput2: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU {gpu_id}', 
                             fillcolor='lightgreen')
    
    # Output node
    dot.node('output', 'Model Output\nInput: [128, 10000, 4096]\nOutput: [128, 10000, 4096]\nGPU 15', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections for layer-wise parallelism
    prev_node = 'input'
    for layer_id in range(1, 17):
        # Communication between GPUs
        if layer_id > 1:
            gpu_from = layer_id - 2
            gpu_to = layer_id - 1
            dot.node(f'comm_{layer_id-1}_{layer_id}', 
                   f'GPU {gpu_from} → GPU {gpu_to}\nActivation Transfer\n[128, 10000, 4096]\n900 Gbps', 
                   shape='parallelogram', fillcolor='lightpink')
            dot.edge(prev_node, f'comm_{layer_id-1}_{layer_id}')
            prev_node = f'comm_{layer_id-1}_{layer_id}'
        
        # Layer connections
        dot.edge(prev_node, f'layer{layer_id}_norm1')
        dot.edge(f'layer{layer_id}_norm1', f'layer{layer_id}_qkv_proj')
        dot.edge(f'layer{layer_id}_qkv_proj', f'layer{layer_id}_q_split')
        dot.edge(f'layer{layer_id}_qkv_proj', f'layer{layer_id}_k_split')
        dot.edge(f'layer{layer_id}_qkv_proj', f'layer{layer_id}_v_split')
        dot.edge(f'layer{layer_id}_q_split', f'layer{layer_id}_attention')
        dot.edge(f'layer{layer_id}_k_split', f'layer{layer_id}_attention')
        dot.edge(f'layer{layer_id}_v_split', f'layer{layer_id}_attention')
        dot.edge(f'layer{layer_id}_attention', f'layer{layer_id}_attention_concat')
        dot.edge(f'layer{layer_id}_attention_concat', f'layer{layer_id}_attention_out')
        dot.edge(f'layer{layer_id}_attention_out', f'layer{layer_id}_residual1')
        
        # Handle first layer residual connection
        if layer_id == 1:
            dot.edge('input', f'layer1_residual1')
        else:
            dot.edge(f'comm_{layer_id-1}_{layer_id}', f'layer{layer_id}_residual1')
        
        # MLP connections
        dot.edge(f'layer{layer_id}_residual1', f'layer{layer_id}_norm2')
        dot.edge(f'layer{layer_id}_norm2', f'layer{layer_id}_mlp_gate')
        dot.edge(f'layer{layer_id}_norm2', f'layer{layer_id}_mlp_up')
        dot.edge(f'layer{layer_id}_mlp_gate', f'layer{layer_id}_mlp_activation')
        dot.edge(f'layer{layer_id}_mlp_up', f'layer{layer_id}_mlp_activation')
        dot.edge(f'layer{layer_id}_mlp_activation', f'layer{layer_id}_mlp_down')
        dot.edge(f'layer{layer_id}_mlp_down', f'layer{layer_id}_residual2')
        dot.edge(f'layer{layer_id}_residual1', f'layer{layer_id}_residual2')  # MLP residual
        
        prev_node = f'layer{layer_id}_residual2'
    
    dot.edge(prev_node, 'output')
    
    return dot

def main():
    # Create output directory
    os.makedirs('../outputs/2025-11-14-11-57-49', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_tensor_pipeline_dag()
    baseline_dag.render('../outputs/2025-11-14-11-57-49/baseline_tensor_pipeline_parallel', 
                       format='svg', cleanup=False)
    
    # Write DOT file content for baseline
    with open('../outputs/2025-11-14-11-57-49/baseline_tensor_pipeline_parallel.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_layer_wise_dag()
    proposed_dag.render('../outputs/2025-11-14-11-57-49/proposed_layer_wise_parallel', 
                       format='svg', cleanup=False)
    
    # Write DOT file content for proposed
    with open('../outputs/2025-11-14-11-57-49/proposed_layer_wise_parallel.dot', 'w') as f:
        f.write(proposed_dag.source)
    
    # Create summary JSON
    import json
    summary = {
        "generated_dags": [
            "../outputs/2025-11-14-11-57-49/baseline_tensor_pipeline_parallel.dot",
            "../outputs/2025-11-14-11-57-49/proposed_layer_wise_parallel.dot"
        ],
        "generated_images": [
            "../outputs/2025-11-14-11-57-49/baseline_tensor_pipeline_parallel.svg",
            "../outputs/2025-11-14-11-57-49/proposed_layer_wise_parallel.svg"
        ],
        "dag_info": {
            "baseline_tensor_pipeline": {
                "total_gpus": 16,
                "parallelism": "Tensor Parallelism (8-way) + Pipeline Parallelism (2-way)",
                "layers_per_gpu": "8 layers per pipeline stage",
                "memory_usage": "33.55 MB model + 32.77 MB activations per GPU"
            },
            "proposed_layer_wise": {
                "total_gpus": 16,
                "parallelism": "Layer-wise (1 layer per GPU)",
                "layers_per_gpu": "1 complete layer per GPU",
                "memory_usage": "49.5 MB total per GPU (L2 cache optimized)"
            }
        }
    }
    
    with open('../outputs/2025-11-14-11-57-49/dag_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- baseline_tensor_pipeline_parallel.dot")
    print("- baseline_tensor_pipeline_parallel.svg")
    print("- proposed_layer_wise_parallel.dot")
    print("- proposed_layer_wise_parallel.svg")
    print("- dag_summary.json")

if __name__ == "__main__":
    main()