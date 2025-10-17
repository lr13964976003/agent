#!/usr/bin/env python3
"""
Baseline DAG generation for TP=8, PP=2 configuration
16 GPUs across 4 nodes with tensor parallelism and pipeline parallelism
"""

import graphviz
from typing import Dict, List, Tuple

# Constants
NUM_LAYERS = 4
TOTAL_GPUS = 16
TENSOR_PARALLEL_SIZE = 8
PIPELINE_PARALLEL_SIZE = 2
HIDDEN_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128  # 4096/32
SEQ_LEN = 2048
BATCH_SIZE = 1024
NUM_EXPERTS = 16
EXPERT_DIM = 16384

# Pipeline stages - each stage has 8 GPUs for tensor parallelism
STAGE0_GPUS = list(range(8))  # GPUs 0-7
STAGE1_GPUS = list(range(8, 16))  # GPUs 8-15

# Tensor parallel groups - 8 GPUs per stage
TP_GROUPS = {
    'stage0': STAGE0_GPUS,
    'stage1': STAGE1_GPUS
}

# Model layers per stage
LAYERS_PER_STAGE = NUM_LAYERS // PIPELINE_PARALLEL_SIZE  # 2 layers per stage

# Head distribution for tensor parallelism
HEADS_PER_GPU_STAGE0 = NUM_HEADS // len(STAGE0_GPUS)  # 4 heads per GPU
HEADS_PER_GPU_STAGE1 = NUM_HEADS // len(STAGE1_GPUS)  # 4 heads per GPU

def create_baseline_dag():
    """Create complete baseline DAG for TP=8, PP=2"""
    dot = graphviz.Digraph('Baseline_TP8_PP2', 
                           comment='Baseline TP=8, PP=2 MoE Transformer DAG',
                           graph_attr={'rankdir': 'TB', 'splines': 'ortho'})
    
    # Set node attributes
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: stage0',
             shape='ellipse', fillcolor='lightblue')
    
    # Track stages
    stage0_outputs = []
    stage1_outputs = []
    
    # Stage 0: Layers 0-1 on GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='rounded', fillcolor='lightcyan')
        
        current = 'input'
        for layer in range(LAYERS_PER_STAGE):  # Layers 0 and 1
            with stage0.subgraph(name=f'cluster_stage0_layer{layer}') as layer_cluster:
                layer_cluster.attr(label=f'Stage 0 Layer {layer}', style='rounded', fillcolor='lightyellow')
                
                # ===== ATTENTION BLOCK =====
                # Layernorm 1
                ln1_node = f'stage0_layer{layer}_ln1'
                layer_cluster.node(ln1_node, 
                       f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       fillcolor='lightgreen')
                
                # QKV projection across 8 GPUs (tensor parallel)
                qkv_nodes = []
                for i in range(8):
                    qkv_node = f'stage0_layer{layer}_qkv_tp_{i}'
                    layer_cluster.node(qkv_node,
                           f'QKV Projection\\nTP Rank {i}\\nHeads={HEADS_PER_GPU_STAGE0}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: {i}',
                           fillcolor='lightcoral')
                    qkv_nodes.append(qkv_node)
                
                # Attention computation
                attn_nodes = []
                for i in range(8):
                    attn_node = f'stage0_layer{layer}_attn_tp_{i}'
                    layer_cluster.node(attn_node,
                           f'Attention Computation\\nTP Rank {i}\\nHeads={HEADS_PER_GPU_STAGE0}\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: {i}',
                           fillcolor='lightcoral')
                
                # Attention output projection
                attn_out_nodes = []
                for i in range(8):
                    out_proj_node = f'stage0_layer{layer}_attn_out_tp_{i}'
                    layer_cluster.node(out_proj_node,
                           f'Attention Output\\nTP Rank {i}\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nGPU: {i}',
                           fillcolor='lightcoral')
                    attn_out_nodes.append(out_proj_node)
                
                # All-reduce for attention output
                attn_allreduce = f'stage0_layer{layer}_attn_allreduce'
                layer_cluster.node(attn_allreduce,
                       f'Attention All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # Residual connection 1
                residual1 = f'stage0_layer{layer}_residual1'
                layer_cluster.node(residual1,
                       f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       shape='diamond', fillcolor='lightpink')
                
                # ===== MOE BLOCK =====
                # Layernorm 2
                ln2_node = f'stage0_layer{layer}_ln2'
                layer_cluster.node(ln2_node,
                       f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       fillcolor='lightgreen')
                
                # Gate computation (tensor parallel)
                gate_node = f'stage0_layer{layer}_gate'
                layer_cluster.node(gate_node,
                       f'Gate Computation\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPUs: 0-7',
                       shape='parallelogram', fillcolor='lightblue')
                
                # Expert distribution (experts distributed across 8 GPUs)
                expert_nodes = []
                for i in range(8):
                    # 2 experts per GPU in tensor parallel setup
                    for j in range(2):
                        expert_id = i * 2 + j
                        expert_node = f'stage0_layer{layer}_expert_{expert_id}_tp_{i}'
                        layer_cluster.node(expert_node,
                               f'Expert {expert_id}\\nTP Rank {i}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, expert_dim=2048]\\nGPU: {i}',
                               fillcolor='lightsteelblue')
                        expert_nodes.append(expert_node)
                
                # Expert aggregation and all-reduce
                expert_agg = f'stage0_layer{layer}_expert_agg'
                layer_cluster.node(expert_agg,
                       f'Expert Aggregation\\nInput: [batch_size=1024, seq_len=2048, expert_dim=16384]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       shape='parallelogram', fillcolor='lightyellow')
                
                expert_allreduce = f'stage0_layer{layer}_expert_allreduce'
                layer_cluster.node(expert_allreduce,
                       f'MoE All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # Residual connection 2
                residual2 = f'stage0_layer{layer}_residual2'
                layer_cluster.node(residual2,
                       f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 0-7',
                       shape='diamond', fillcolor='lightpink')
                
                stage0_outputs.append(residual2)
                current = residual2
    
    # Pipeline communication between stages
    pipeline_comm = 'pipeline_comm_stage0_to_stage1'
    dot.node(pipeline_comm,
           f'Pipeline Communication\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 7→8',
           shape='parallelogram', fillcolor='orange')
    
    # Stage 1: Layers 2-3 on GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='rounded', fillcolor='lightcyan')
        
        # Connect from pipeline communication
        current = pipeline_comm
        
        for layer in range(LAYERS_PER_STAGE):  # Layers 2 and 3
            actual_layer = layer + LAYERS_PER_STAGE
            
            with stage1.subgraph(name=f'cluster_stage1_layer{actual_layer}') as layer_cluster:
                layer_cluster.attr(label=f'Stage 1 Layer {actual_layer}', style='rounded', fillcolor='lightyellow')
                
                # ===== ATTENTION BLOCK =====
                # Layernorm 1
                ln1_node = f'stage1_layer{actual_layer}_ln1'
                layer_cluster.node(ln1_node, 
                       f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       fillcolor='lightgreen')
                
                # QKV projection across 8 GPUs (tensor parallel)
                qkv_nodes = []
                for i in range(8):
                    gpu_id = i + 8
                    qkv_node = f'stage1_layer{actual_layer}_qkv_tp_{i}'
                    layer_cluster.node(qkv_node,
                           f'QKV Projection\\nTP Rank {i}\\nHeads={HEADS_PER_GPU_STAGE1}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
                    qkv_nodes.append(qkv_node)
                
                # Attention computation
                attn_nodes = []
                for i in range(8):
                    gpu_id = i + 8
                    attn_node = f'stage1_layer{actual_layer}_attn_tp_{i}'
                    layer_cluster.node(attn_node,
                           f'Attention Computation\\nTP Rank {i}\\nHeads={HEADS_PER_GPU_STAGE1}\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
                
                # Attention output projection
                attn_out_nodes = []
                for i in range(8):
                    gpu_id = i + 8
                    out_proj_node = f'stage1_layer{actual_layer}_attn_out_tp_{i}'
                    layer_cluster.node(out_proj_node,
                           f'Attention Output\\nTP Rank {i}\\nInput: [batch_size=1024, seq_len=2048, heads=4, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
                    attn_out_nodes.append(out_proj_node)
                
                # All-reduce for attention output
                attn_allreduce = f'stage1_layer{actual_layer}_attn_allreduce'
                layer_cluster.node(attn_allreduce,
                       f'Attention All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # Residual connection 1
                residual1 = f'stage1_layer{actual_layer}_residual1'
                layer_cluster.node(residual1,
                       f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       shape='diamond', fillcolor='lightpink')
                
                # ===== MOE BLOCK =====
                # Layernorm 2
                ln2_node = f'stage1_layer{actual_layer}_ln2'
                layer_cluster.node(ln2_node,
                       f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       fillcolor='lightgreen')
                
                # Gate computation (tensor parallel)
                gate_node = f'stage1_layer{actual_layer}_gate'
                layer_cluster.node(gate_node,
                       f'Gate Computation\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPUs: 8-15',
                       shape='parallelogram', fillcolor='lightblue')
                
                # Expert distribution across 8 GPUs
                expert_nodes = []
                for i in range(8):
                    gpu_id = i + 8
                    # 2 experts per GPU
                    for j in range(2):
                        expert_id = i * 2 + j
                        expert_node = f'stage1_layer{actual_layer}_expert_{expert_id}_tp_{i}'
                        layer_cluster.node(expert_node,
                               f'Expert {expert_id}\\nTP Rank {i}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=512]\\nOutput: [batch_size=1024, seq_len=2048, expert_dim=2048]\\nGPU: {gpu_id}',
                               fillcolor='lightsteelblue')
                        expert_nodes.append(expert_node)
                
                # Expert aggregation and all-reduce
                expert_agg = f'stage1_layer{actual_layer}_expert_agg'
                layer_cluster.node(expert_agg,
                       f'Expert Aggregation\\nInput: [batch_size=1024, seq_len=2048, expert_dim=16384]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       shape='parallelogram', fillcolor='lightyellow')
                
                expert_allreduce = f'stage1_layer{actual_layer}_expert_allreduce'
                layer_cluster.node(expert_allreduce,
                       f'MoE All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # Residual connection 2
                residual2 = f'stage1_layer{actual_layer}_residual2'
                layer_cluster.node(residual2,
                       f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: 8-15',
                       shape='diamond', fillcolor='lightpink')
                
                stage1_outputs.append(residual2)
                current = residual2
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: stage1',
             shape='ellipse', fillcolor='lightgreen')
    
    # Generate edges for Stage 0
    current = 'input'
    for layer in range(LAYERS_PER_STAGE):
        dot.edge(current, f'stage0_layer{layer}_ln1')
        
        # Connect tensor parallel components
        for i in range(8):
            dot.edge(f'stage0_layer{layer}_ln1', f'stage0_layer{layer}_qkv_tp_{i}')
            dot.edge(f'stage0_layer{layer}_qkv_tp_{i}', f'stage0_layer{layer}_attn_tp_{i}')
            dot.edge(f'stage0_layer{layer}_attn_tp_{i}', f'stage0_layer{layer}_attn_out_tp_{i}')
            dot.edge(f'stage0_layer{layer}_attn_out_tp_{i}', f'stage0_layer{layer}_attn_allreduce')
        
        dot.edge(f'stage0_layer{layer}_attn_allreduce', f'stage0_layer{layer}_residual1')
        dot.edge(current, f'stage0_layer{layer}_residual1')  # Skip connection
        
        dot.edge(f'stage0_layer{layer}_residual1', f'stage0_layer{layer}_ln2')
        dot.edge(f'stage0_layer{layer}_ln2', f'stage0_layer{layer}_gate')
        
        # Connect to experts
        for i in range(8):
            dot.edge(f'stage0_layer{layer}_gate', f'stage0_layer{layer}_expert_{i*2}_tp_{i}', style='dashed')
            dot.edge(f'stage0_layer{layer}_gate', f'stage0_layer{layer}_expert_{i*2+1}_tp_{i}', style='dashed')
            dot.edge(f'stage0_layer{layer}_expert_{i*2}_tp_{i}', f'stage0_layer{layer}_expert_agg')
            dot.edge(f'stage0_layer{layer}_expert_{i*2+1}_tp_{i}', f'stage0_layer{layer}_expert_agg')
        
        dot.edge(f'stage0_layer{layer}_expert_agg', f'stage0_layer{layer}_expert_allreduce')
        dot.edge(f'stage0_layer{layer}_expert_allreduce', f'stage0_layer{layer}_residual2')
        dot.edge(f'stage0_layer{layer}_residual1', f'stage0_layer{layer}_residual2')  # Skip connection
        
        current = f'stage0_layer{layer}_residual2'
    
    # Pipeline communication
    dot.edge(current, 'pipeline_comm_stage0_to_stage1')
    
    # Generate edges for Stage 1
    current = 'pipeline_comm_stage0_to_stage1'
    for layer in range(LAYERS_PER_STAGE):
        actual_layer = layer + LAYERS_PER_STAGE
        dot.edge(current, f'stage1_layer{actual_layer}_ln1')
        
        # Connect tensor parallel components
        for i in range(8):
            gpu_id = i + 8
            dot.edge(f'stage1_layer{actual_layer}_ln1', f'stage1_layer{actual_layer}_qkv_tp_{i}')
            dot.edge(f'stage1_layer{actual_layer}_qkv_tp_{i}', f'stage1_layer{actual_layer}_attn_tp_{i}')
            dot.edge(f'stage1_layer{actual_layer}_attn_tp_{i}', f'stage1_layer{actual_layer}_attn_out_tp_{i}')
            dot.edge(f'stage1_layer{actual_layer}_attn_out_tp_{i}', f'stage1_layer{actual_layer}_attn_allreduce')
        
        dot.edge(f'stage1_layer{actual_layer}_attn_allreduce', f'stage1_layer{actual_layer}_residual1')
        dot.edge(current, f'stage1_layer{actual_layer}_residual1')  # Skip connection
        
        dot.edge(f'stage1_layer{actual_layer}_residual1', f'stage1_layer{actual_layer}_ln2')
        dot.edge(f'stage1_layer{actual_layer}_ln2', f'stage1_layer{actual_layer}_gate')
        
        # Connect to experts
        for i in range(8):
            gpu_id = i + 8
            dot.edge(f'stage1_layer{actual_layer}_gate', f'stage1_layer{actual_layer}_expert_{i*2}_tp_{i}', style='dashed')
            dot.edge(f'stage1_layer{actual_layer}_gate', f'stage1_layer{actual_layer}_expert_{i*2+1}_tp_{i}', style='dashed')
            dot.edge(f'stage1_layer{actual_layer}_expert_{i*2}_tp_{i}', f'stage1_layer{actual_layer}_expert_agg')
            dot.edge(f'stage1_layer{actual_layer}_expert_{i*2+1}_tp_{i}', f'stage1_layer{actual_layer}_expert_agg')
        
        dot.edge(f'stage1_layer{actual_layer}_expert_agg', f'stage1_layer{actual_layer}_expert_allreduce')
        dot.edge(f'stage1_layer{actual_layer}_expert_allreduce', f'stage1_layer{actual_layer}_residual2')
        dot.edge(f'stage1_layer{actual_layer}_residual1', f'stage1_layer{actual_layer}_residual2')  # Skip connection
        
        current = f'stage1_layer{actual_layer}_residual2'
    
    # Connect final output
    dot.edge(current, 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('../outputs/2025-10-17-10-16-04/baseline_tp8_pp2', format='svg', cleanup=False)
    
    # Save DOT file
    with open('../outputs/2025-10-17-10-16-04/baseline_tp8_pp2.dot', 'w') as f:
        f.write(dag.source)