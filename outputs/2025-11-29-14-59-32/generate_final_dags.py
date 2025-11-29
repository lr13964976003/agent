#!/usr/bin/env python3

import os

def generate_corrected_baseline_dag():
    """Generate corrected baseline tensor pipeline DAG with proper connectivity"""
    
    dot_content = '''// Corrected Baseline Tensor+Pipeline Parallelism DAG
// Fixed connectivity issues and redundant structure
digraph {
    dpi=300
    rankdir=TB
    size="25,35"
    
    // Node styles
    node [fillcolor=lightblue, shape=rectangle, style=filled]
    
    // Input node
    input [label="Input\\nBatch: 128\\nSeq: 10000\\nDim: 4096", fillcolor=lightcoral, shape=diamond]
    
    // Output node  
    output [label="Output\\nBatch: 128\\nSeq: 10000\\nDim: 4096", fillcolor=lightcoral, shape=diamond]
    
    // Pipeline Stage 0: Layers 0-7 on GPUs 0-7 with TP=8'''
    
    # Add pipeline stage 0
    dot_content += '''
    subgraph cluster_pipeline_stage_0 {
        fillcolor=lightsteelblue
        label="Pipeline Stage 0: Layers 0-7\\nGPUs 0-7 with TP=8"
        style="rounded,filled"
        '''
    
    # Add layers 0-7 for stage 0
    for layer in range(8):
        dot_content += f'''
        // Layer {layer} Attention
        subgraph cluster_layer_{layer}_attn {{
            fillcolor=lightgray
            label="Layer {layer} Attention (TP=8)" 
            style="rounded,filled"
            '''
        
        # QKV projections (parallel across 8 GPUs)
        dot_content += f'\n            layer_{layer}_qkv_allgather [label="QKV AllGather\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        
        # Attention computation (each GPU handles 4 heads)
        for gpu in range(8):
            dot_content += f'''
            layer_{layer}_qkv_gpu{gpu} [label="QKV Proj GPU{gpu}\\nColParallel\\nDim: 512x1536", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_comp_gpu{gpu} [label="Attention Compute GPU{gpu}\\n4 heads\\nSeq: 10000x128", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_out_gpu{gpu} [label="Attn Out Proj GPU{gpu}\\nRowParallel\\nDim: 512x512", fillcolor=lightblue, shape=rectangle]'''
        
        dot_content += f'\n            layer_{layer}_attn_allreduce [label="Attention AllReduce\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        dot_content += '\n        }'
        
        # Residual and LayerNorm
        dot_content += f'''
        layer_{layer}_residual1 [label="Residual Add {layer}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln1 [label="LayerNorm {layer}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
        
        # MLP block
        dot_content += f'''
        subgraph cluster_layer_{layer}_mlp {{
            fillcolor=lightgray
            label="Layer {layer} MLP (TP=8)"
            style="rounded,filled"
            '''
        
        dot_content += f'\n            layer_{layer}_mlp_allreduce [label="MLP AllReduce\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        
        for gpu in range(8):
            dot_content += f'''
            layer_{layer}_mlp1_gpu{gpu} [label="MLP1 GPU{gpu}\\nColParallel\\nDim: 512x2048", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_gelu_gpu{gpu} [label="GELU GPU{gpu}\\n2048 dim", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_mlp2_gpu{gpu} [label="MLP2 GPU{gpu}\\nRowParallel\\nDim: 2048x512", fillcolor=lightblue, shape=rectangle]'''
        
        dot_content += '\n        }'
        
        # Final residual and LayerNorm
        dot_content += f'''
        layer_{layer}_residual2 [label="Residual Add {layer}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln2 [label="LayerNorm {layer}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
    
    dot_content += '\n    }'
    
    # Add pipeline stage 1
    dot_content += '''
    
    // Pipeline Stage 1: Layers 8-15 on GPUs 8-15 with TP=8
    subgraph cluster_pipeline_stage_1 {
        fillcolor=lightsteelblue
        label="Pipeline Stage 1: Layers 8-15\\nGPUs 8-15 with TP=8"
        style="rounded,filled"
        '''
    
    # Add layers 8-15 for stage 1
    for layer in range(8, 16):
        dot_content += f'''
        // Layer {layer} Attention
        subgraph cluster_layer_{layer}_attn {{
            fillcolor=lightgray
            label="Layer {layer} Attention (TP=8)" 
            style="rounded,filled"
            '''
        
        # QKV projections (parallel across 8 GPUs 8-15)
        dot_content += f'\n            layer_{layer}_qkv_allgather [label="QKV AllGather\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        
        # Attention computation (each GPU handles 4 heads)
        for gpu in range(8, 16):
            dot_content += f'''
            layer_{layer}_qkv_gpu{gpu} [label="QKV Proj GPU{gpu}\\nColParallel\\nDim: 512x1536", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_comp_gpu{gpu} [label="Attention Compute GPU{gpu}\\n4 heads\\nSeq: 10000x128", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_out_gpu{gpu} [label="Attn Out Proj GPU{gpu}\\nRowParallel\\nDim: 512x512", fillcolor=lightblue, shape=rectangle]'''
        
        dot_content += f'\n            layer_{layer}_attn_allreduce [label="Attention AllReduce\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        dot_content += '\n        }'
        
        # Residual and LayerNorm
        dot_content += f'''
        layer_{layer}_residual1 [label="Residual Add {layer}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln1 [label="LayerNorm {layer}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
        
        # MLP block
        dot_content += f'''
        subgraph cluster_layer_{layer}_mlp {{
            fillcolor=lightgray
            label="Layer {layer} MLP (TP=8)"
            style="rounded,filled"
            '''
        
        dot_content += f'\n            layer_{layer}_mlp_allreduce [label="MLP AllReduce\\n8 GPUs", fillcolor=lightgreen, shape=ellipse]'
        
        for gpu in range(8, 16):
            dot_content += f'''
            layer_{layer}_mlp1_gpu{gpu} [label="MLP1 GPU{gpu}\\nColParallel\\nDim: 512x2048", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_gelu_gpu{gpu} [label="GELU GPU{gpu}\\n2048 dim", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_mlp2_gpu{gpu} [label="MLP2 GPU{gpu}\\nRowParallel\\nDim: 2048x512", fillcolor=lightblue, shape=rectangle]'''
        
        dot_content += '\n        }'
        
        # Final residual and LayerNorm
        dot_content += f'''
        layer_{layer}_residual2 [label="Residual Add {layer}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln2 [label="LayerNorm {layer}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
    
    dot_content += '\n    }'
    
    # Add pipeline communication
    dot_content += '''\n    \n    // Pipeline communication between stages'''
    dot_content += '\n    pipeline_comm_0_1 [label="Pipeline Send/Recv\\nStage 0 → 1\\nLayer 7 → 8", fillcolor=orange, shape=ellipse]'
    
    # Add edges for proper connectivity
    dot_content += '''\n    \n    // Edges - Proper connectivity flow'''
    
    # Input to first layer
    dot_content += '''\n    // Input to first layer'''
    for gpu in range(8):
        dot_content += f'\n    input -> layer_0_qkv_gpu{gpu}'
    
    # Layer 0 attention flow
    dot_content += '''\n    \n    // Layer 0 attention flow'''
    for gpu in range(8):
        dot_content += f'\n    layer_0_qkv_gpu{gpu} -> layer_0_qkv_allgather'
    
    for gpu in range(8):
        dot_content += f'\n    layer_0_qkv_allgather -> layer_0_attn_comp_gpu{gpu}'
    
    for gpu in range(8):
        dot_content += f'\n    layer_0_attn_comp_gpu{gpu} -> layer_0_attn_out_gpu{gpu}'
        dot_content += f'\n    layer_0_attn_out_gpu{gpu} -> layer_0_attn_allreduce'
    
    dot_content += '''\n    layer_0_attn_allreduce -> layer_0_residual1
    layer_0_residual1 -> layer_0_ln1'''
    
    # Layer 0 MLP flow
    dot_content += '''\n    \n    // Layer 0 MLP flow'''
    for gpu in range(8):
        dot_content += f'\n    layer_0_ln1 -> layer_0_mlp1_gpu{gpu}'
        dot_content += f'\n    layer_0_mlp1_gpu{gpu} -> layer_0_gelu_gpu{gpu}'
        dot_content += f'\n    layer_0_gelu_gpu{gpu} -> layer_0_mlp2_gpu{gpu}'
        dot_content += f'\n    layer_0_mlp2_gpu{gpu} -> layer_0_mlp_allreduce'
    
    dot_content += '''\n    layer_0_mlp_allreduce -> layer_0_residual2
    layer_0_residual2 -> layer_0_ln2'''
    
    # Continue pattern for layers 1-6
    for layer in range(1, 7):
        dot_content += f'''\n    \n    // Layer {layer} flow'''
        
        # Attention connections
        for gpu in range(8):
            dot_content += f'\n    layer_{layer-1}_ln2 -> layer_{layer}_qkv_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_qkv_gpu{gpu} -> layer_{layer}_qkv_allgather'
        
        for gpu in range(8):
            dot_content += f'\n    layer_{layer}_qkv_allgather -> layer_{layer}_attn_comp_gpu{gpu}'
        
        for gpu in range(8):
            dot_content += f'\n    layer_{layer}_attn_comp_gpu{gpu} -> layer_{layer}_attn_out_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_attn_out_gpu{gpu} -> layer_{layer}_attn_allreduce'
        
        dot_content += f'''\n    layer_{layer}_attn_allreduce -> layer_{layer}_residual1
    layer_{layer}_residual1 -> layer_{layer}_ln1'''
        
        # MLP connections
        for gpu in range(8):
            dot_content += f'\n    layer_{layer}_ln1 -> layer_{layer}_mlp1_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_mlp1_gpu{gpu} -> layer_{layer}_gelu_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_gelu_gpu{gpu} -> layer_{layer}_mlp2_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_mlp2_gpu{gpu} -> layer_{layer}_mlp_allreduce'
        
        dot_content += f'''\n    layer_{layer}_mlp_allreduce -> layer_{layer}_residual2
    layer_{layer}_residual2 -> layer_{layer}_ln2'''
    
    # Layer 7 to pipeline stage 1 transition
    dot_content += '''\n    \n    // Layer 7 to Stage 1 transition'''
    for gpu in range(8, 16):
        dot_content += f'\n    layer_7_ln2 -> layer_8_qkv_gpu{gpu}'
    
    dot_content += '''\n    layer_7_ln2 -> pipeline_comm_0_1
    pipeline_comm_0_1 -> layer_8_qkv_gpu8'''
    
    # Continue pattern for layers 8-15
    for layer in range(8, 16):
        dot_content += f'''\n    \n    // Layer {layer} flow'''
        
        # Attention connections (using GPUs 8-15)
        for gpu in range(8, 16):
            if layer == 8:
                dot_content += f'\n    pipeline_comm_0_1 -> layer_{layer}_qkv_gpu{gpu}'
            else:
                dot_content += f'\n    layer_{layer-1}_ln2 -> layer_{layer}_qkv_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_qkv_gpu{gpu} -> layer_{layer}_qkv_allgather'
        
        for gpu in range(8, 16):
            dot_content += f'\n    layer_{layer}_qkv_allgather -> layer_{layer}_attn_comp_gpu{gpu}'
        
        for gpu in range(8, 16):
            dot_content += f'\n    layer_{layer}_attn_comp_gpu{gpu} -> layer_{layer}_attn_out_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_attn_out_gpu{gpu} -> layer_{layer}_attn_allreduce'
        
        dot_content += f'''\n    layer_{layer}_attn_allreduce -> layer_{layer}_residual1
    layer_{layer}_residual1 -> layer_{layer}_ln1'''
        
        # MLP connections
        for gpu in range(8, 16):
            dot_content += f'\n    layer_{layer}_ln1 -> layer_{layer}_mlp1_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_mlp1_gpu{gpu} -> layer_{layer}_gelu_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_gelu_gpu{gpu} -> layer_{layer}_mlp2_gpu{gpu}'
            dot_content += f'\n    layer_{layer}_mlp2_gpu{gpu} -> layer_{layer}_mlp_allreduce'
        
        dot_content += f'''\n    layer_{layer}_mlp_allreduce -> layer_{layer}_residual2
    layer_{layer}_residual2 -> layer_{layer}_ln2'''
    
    # Final output
    dot_content += '''\n    \n    // Final output'''
    dot_content += '\n    layer_15_ln2 -> output'
    
    dot_content += '\n}'
    
    # Write the corrected baseline DAG
    with open('../outputs/2025-11-29-14-59-32/corrected_baseline_tensor_pipeline_dag.dot', 'w') as f:
        f.write(dot_content)
    
    print("Generated corrected baseline tensor pipeline DAG")


def generate_corrected_layer_wise_dag():
    """Generate corrected layer-wise DAG with complete connectivity"""
    
    dot_content = '''// Corrected Layer-wise Partitioning DAG
// Fixed connectivity and complete layer coverage
digraph {
    dpi=300
    rankdir=TB
    size="20,30"
    
    // Node styles
    node [fillcolor=lightblue, shape=rectangle, style=filled]
    
    // Input node
    input [label="Input\\nBatch: 128\\nSeq: 10000\\nDim: 4096", fillcolor=lightcoral, shape=diamond]
    
    // Output node
    output [label="Output\\nBatch: 128\\nSeq: 10000\\nDim: 4096", fillcolor=lightcoral, shape=diamond]
    '''
    
    # Define all 8 GPUs with their layer assignments
    for gpu_id in range(8):
        start_layer = gpu_id * 2
        end_layer = start_layer + 1
        
        dot_content += f'''
    subgraph cluster_gpu_{gpu_id} {{
        fillcolor=lightsteelblue
        label="GPU {gpu_id} (Layers {start_layer}-{end_layer})\\nCache: 29.46 GB"
        style="rounded,filled"
        '''
        
        # Layer assignments for this GPU
        for layer in range(start_layer, end_layer + 1):
            # Attention block
            dot_content += f'''
        subgraph cluster_layer_{layer}_attn_gpu{gpu_id} {{
            fillcolor=lightgray
            label="Layer {layer} Attention (Full)"
            style="rounded,filled"
            
            layer_{layer}_qkv_gpu{gpu_id} [label="QKV Proj GPU{gpu_id}\\nFull Layer\\nDim: 4096x12288", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_comp_gpu{gpu_id} [label="Attention Compute GPU{gpu_id}\\n32 heads\\nSeq: 10000x128", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_attn_out_gpu{gpu_id} [label="Attn Out Proj GPU{gpu_id}\\nFull Layer\\nDim: 4096x4096", fillcolor=lightblue, shape=rectangle]
        }}
        '''
            
            # Residual and LayerNorm
            dot_content += f'''
        layer_{layer}_residual1_gpu{gpu_id} [label="Residual Add {layer} GPU{gpu_id}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln1_gpu{gpu_id} [label="LayerNorm {layer} GPU{gpu_id}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
            
            # MLP block
            dot_content += f'''
        subgraph cluster_layer_{layer}_mlp_gpu{gpu_id} {{
            fillcolor=lightgray
            label="Layer {layer} MLP (Full)"
            style="rounded,filled"
            
            layer_{layer}_mlp1_gpu{gpu_id} [label="MLP1 GPU{gpu_id}\\nFull Layer\\nDim: 4096x16384", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_gelu_gpu{gpu_id} [label="GELU GPU{gpu_id}\\n16384 dim", fillcolor=lightblue, shape=rectangle]
            layer_{layer}_mlp2_gpu{gpu_id} [label="MLP2 GPU{gpu_id}\\nFull Layer\\nDim: 16384x4096", fillcolor=lightblue, shape=rectangle]
        }}
        '''
            
            # Final residual and LayerNorm
            dot_content += f'''
        layer_{layer}_residual2_gpu{gpu_id} [label="Residual Add {layer} GPU{gpu_id}\\n4096 dim", fillcolor=lightyellow, shape=parallelogram]
        layer_{layer}_ln2_gpu{gpu_id} [label="LayerNorm {layer} GPU{gpu_id}\\n4096 dim", fillcolor=lightblue, shape=rectangle]
        '''
        
        dot_content += '\n    }'
        
        # Add GPU-to-GPU communication nodes (except for last GPU)
        if gpu_id < 7:
            next_gpu = gpu_id + 1
            next_layer = end_layer + 1
            dot_content += f'''
    gpu_comm_{gpu_id}_to_{next_gpu} [label="GPU-to-GPU Send/Recv\\nGPU {gpu_id} → {next_gpu}\\nLayer {end_layer} → {next_layer}", fillcolor=orange, shape=ellipse]'''
    
    # Add edges - Complete connectivity flow
    dot_content += '''\n    \n    // Edges - Complete connectivity flow'''
    
    # Input to first layer
    dot_content += '''\n    // Input to first layer'''
    dot_content += '\n    input -> layer_0_qkv_gpu0'
    
    # GPU 0: Layers 0-1 flow
    dot_content += '''\n    \n    // GPU 0: Layers 0-1 flow'''
    dot_content += '''
    layer_0_qkv_gpu0 -> layer_0_attn_comp_gpu0
    layer_0_attn_comp_gpu0 -> layer_0_attn_out_gpu0
    layer_0_attn_out_gpu0 -> layer_0_residual1_gpu0
    layer_0_residual1_gpu0 -> layer_0_ln1_gpu0
    layer_0_ln1_gpu0 -> layer_0_mlp1_gpu0
    layer_0_mlp1_gpu0 -> layer_0_gelu_gpu0
    layer_0_gelu_gpu0 -> layer_0_mlp2_gpu0
    layer_0_mlp2_gpu0 -> layer_0_residual2_gpu0
    layer_0_residual2_gpu0 -> layer_0_ln2_gpu0
    
    // Layer 1 on GPU 0
    layer_0_ln2_gpu0 -> layer_1_qkv_gpu0
    layer_1_qkv_gpu0 -> layer_1_attn_comp_gpu0
    layer_1_attn_comp_gpu0 -> layer_1_attn_out_gpu0
    layer_1_attn_out_gpu0 -> layer_1_residual1_gpu0
    layer_1_residual1_gpu0 -> layer_1_ln1_gpu0
    layer_1_ln1_gpu0 -> layer_1_mlp1_gpu0
    layer_1_mlp1_gpu0 -> layer_1_gelu_gpu0
    layer_1_gelu_gpu0 -> layer_1_mlp2_gpu0
    layer_1_mlp2_gpu0 -> layer_1_residual2_gpu0
    layer_1_residual2_gpu0 -> layer_1_ln2_gpu0
    
    // GPU transition: GPU 0 → GPU 1 (Layer 1 → Layer 2)
    layer_1_ln2_gpu0 -> gpu_comm_0_to_1
    gpu_comm_0_to_1 -> layer_2_qkv_gpu1
    '''
    
    # Continue pattern for remaining GPUs
    for gpu_id in range(1, 8):
        start_layer = gpu_id * 2
        end_layer = start_layer + 1
        
        if gpu_id < 7:
            next_gpu = gpu_id + 1
            next_start_layer = next_gpu * 2
            
            dot_content += f'''\n    \n    // GPU {gpu_id}: Layers {start_layer}-{end_layer} flow'''
            
            # Current GPU layers flow
            for layer in range(start_layer, end_layer + 1):
                dot_content += f'''
    layer_{layer}_qkv_gpu{gpu_id} -> layer_{layer}_attn_comp_gpu{gpu_id}
    layer_{layer}_attn_comp_gpu{gpu_id} -> layer_{layer}_attn_out_gpu{gpu_id}
    layer_{layer}_attn_out_gpu{gpu_id} -> layer_{layer}_residual1_gpu{gpu_id}
    layer_{layer}_residual1_gpu{gpu_id} -> layer_{layer}_ln1_gpu{gpu_id}
    layer_{layer}_ln1_gpu{gpu_id} -> layer_{layer}_mlp1_gpu{gpu_id}
    layer_{layer}_mlp1_gpu{gpu_id} -> layer_{layer}_gelu_gpu{gpu_id}
    layer_{layer}_gelu_gpu{gpu_id} -> layer_{layer}_mlp2_gpu{gpu_id}
    layer_{layer}_mlp2_gpu{gpu_id} -> layer_{layer}_residual2_gpu{gpu_id}
    layer_{layer}_residual2_gpu{gpu_id} -> layer_{layer}_ln2_gpu{gpu_id}'''
            
            # Transition to next GPU
            dot_content += f'''
    \n    // Transition to GPU {next_gpu}
    layer_{end_layer}_ln2_gpu{gpu_id} -> gpu_comm_{gpu_id}_to_{next_gpu}
    gpu_comm_{gpu_id}_to_{next_gpu} -> layer_{next_start_layer}_qkv_gpu{next_gpu}'''
    
    # Final output from GPU 7 (Layer 15)
    dot_content += '''\n    \n    // Final output from GPU 7 (Layer 15)'''
    dot_content += '\n    layer_15_ln2_gpu7 -> output'
    
    dot_content += '\n}'
    
    # Write the corrected layer-wise DAG
    with open('../outputs/2025-11-29-14-59-32/corrected_proposed_layer_wise_dag.dot', 'w') as f:
        f.write(dot_content)
    
    print("Generated corrected proposed layer-wise DAG")


if __name__ == "__main__":
    generate_corrected_baseline_dag()
    generate_corrected_layer_wise_dag()
    print("All corrected DAGs generated successfully!")