#!/usr/bin/env python3
"""
MA Separation DAG generation for 4-layer MoE transformer
16 GPUs across 4 nodes with 12:4 attention:MoE allocation
"""

import graphviz
from typing import Dict, List, Tuple

# Constants
NUM_LAYERS = 4
TOTAL_GPUS = 16
ATTENTION_GPUS = 12
MOE_GPUS = 4
HIDDEN_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128  # 4096/32
SEQ_LEN = 2048
BATCH_SIZE = 1024
NUM_EXPERTS = 16
EXPERT_DIM = 16384

# GPU mapping
ATTENTION_GPU_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
MOE_GPU_IDS = [3, 7, 11, 15]

def create_ma_separation_dag():
    """Create complete MA Separation DAG"""
    dot = graphviz.Digraph('MA_Separation_MoE', 
                           comment='MA Separation MoE Transformer DAG',
                           graph_attr={'rankdir': 'TB', 'splines': 'ortho'})
    
    # Set node attributes
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
             shape='ellipse', fillcolor='lightblue')
    
    # Track previous node for connections
    prev_node = 'input'
    
    for layer in range(NUM_LAYERS):
        # Create Layer N subgraph
        with dot.subgraph(name=f'cluster_layer_{layer}') as c:
            c.attr(label=f'Layer {layer}', style='rounded', fillcolor='lightyellow')
            
            # ===== ATTENTION BLOCK =====
            # Layernorm 1
            ln1_node = f'layer{layer}_ln1'
            c.node(ln1_node, 
                   f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   fillcolor='lightgreen')
            
            # QKV projection across 12 attention GPUs
            qkv_nodes = []
            head_distribution = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]
            gpu_idx = 0
            for i, heads in enumerate(head_distribution):
                gpu_id = ATTENTION_GPU_IDS[gpu_idx]
                qkv_node = f'layer{layer}_qkv_proj_{i}'
                c.node(qkv_node,
                       f'QKV Projection\\nHeads={heads}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads={heads}, head_dim=128]\\nGPU: {gpu_id}',
                       fillcolor='lightcoral')
                qkv_nodes.append(qkv_node)
                gpu_idx += 1
            
            # Attention score computation
            attn_nodes = []
            for i, (heads, qkv_node) in enumerate(zip(head_distribution, qkv_nodes)):
                gpu_id = ATTENTION_GPU_IDS[i]
                attn_node = f'layer{layer}_attention_{i}'
                c.node(attn_node,
                       f'Attention Computation\\nHeads={heads}\\nInput: [batch_size=1024, seq_len=2048, heads={heads}, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads={heads}, head_dim=128]\\nGPU: {gpu_id}',
                       fillcolor='lightcoral')
            
            # Attention output projection
            attn_out_nodes = []
            for i, heads in enumerate(head_distribution):
                gpu_id = ATTENTION_GPU_IDS[i]
                out_proj_node = f'layer{layer}_attn_out_proj_{i}'
                c.node(out_proj_node,
                       f'Attention Output Projection\\nHeads={heads}\\nInput: [batch_size=1024, seq_len=2048, heads={heads}, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim={heads*HEAD_DIM}]\\nGPU: {gpu_id}',
                       fillcolor='lightcoral')
                attn_out_nodes.append(out_proj_node)
            
            # All-reduce for attention output
            attn_reduce = f'layer{layer}_attn_allreduce'
            c.node(attn_reduce,
                   f'Attention All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Residual connection 1
            residual1 = f'layer{layer}_residual1'
            c.node(residual1,
                   f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   shape='diamond', fillcolor='lightpink')
            
            # ===== MOE BLOCK =====
            # Layernorm 2
            ln2_node = f'layer{layer}_ln2'
            c.node(ln2_node,
                   f'LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   fillcolor='lightgreen')
            
            # Gate computation (distributed across all GPUs)
            gate_node = f'layer{layer}_gate'
            c.node(gate_node,
                   f'Gate Computation\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPUs: all GPUs',
                   shape='parallelogram', fillcolor='lightblue')
            
            # Expert routing
            expert_nodes = []
            mlp_nodes = []
            for i, moe_gpu in enumerate(MOE_GPU_IDS):
                # Each MOE GPU handles 4 experts
                expert_start = i * 4
                expert_end = (i + 1) * 4
                
                # Expert routing
                route_node = f'layer{layer}_route_{i}'
                c.node(route_node,
                       f'Expert Routing\\nExperts=[{expert_start}-{expert_end-1}]\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, expert_dim=16384]\\nGPU: {moe_gpu}',
                       shape='parallelogram', fillcolor='lightyellow', style='dashed')
                
                # Expert MLP computations
                for j in range(4):
                    expert_id = expert_start + j
                    expert_node = f'layer{layer}_expert_{expert_id}'
                    c.node(expert_node,
                           f'Expert {expert_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, expert_dim=16384]\\nGPU: {moe_gpu}',
                           fillcolor='lightsteelblue')
                    expert_nodes.append(expert_node)
                
                # Expert aggregation
                expert_agg = f'layer{layer}_expert_agg_{i}'
                c.node(expert_agg,
                       f'Expert Aggregation\\nInput: [batch_size=1024, seq_len=2048, expert_dim=16384]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: {moe_gpu}',
                       shape='parallelogram', fillcolor='lightyellow')
            
            # All-reduce for expert output
            expert_reduce = f'layer{layer}_expert_allreduce'
            c.node(expert_reduce,
                   f'MoE All-Reduce\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Residual connection 2
            residual2 = f'layer{layer}_residual2'
            c.node(residual2,
                   f'Residual Add\\nInput1: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nInput2: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
                   shape='diamond', fillcolor='lightpink')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPUs: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Generate edges
    current = 'input'
    for layer in range(NUM_LAYERS):
        # Connect to layer nodes
        dot.edge(current, f'layer{layer}_ln1')
        
        # Connect within layer
        for i in range(len(head_distribution)):
            dot.edge(f'layer{layer}_ln1', f'layer{layer}_qkv_proj_{i}')
            dot.edge(f'layer{layer}_qkv_proj_{i}', f'layer{layer}_attention_{i}')
            dot.edge(f'layer{layer}_attention_{i}', f'layer{layer}_attn_out_proj_{i}')
            dot.edge(f'layer{layer}_attn_out_proj_{i}', f'layer{layer}_attn_allreduce')
        
        dot.edge(f'layer{layer}_attn_allreduce', f'layer{layer}_residual1')
        dot.edge(current, f'layer{layer}_residual1')  # Skip connection
        
        dot.edge(f'layer{layer}_residual1', f'layer{layer}_ln2')
        dot.edge(f'layer{layer}_ln2', f'layer{layer}_gate')
        
        # Connect to experts
        for i, moe_gpu in enumerate(MOE_GPU_IDS):
            dot.edge(f'layer{layer}_gate', f'layer{layer}_route_{i}', style='dashed')
            for j in range(4):
                expert_id = i * 4 + j
                dot.edge(f'layer{layer}_route_{i}', f'layer{layer}_expert_{expert_id}')
                dot.edge(f'layer{layer}_expert_{expert_id}', f'layer{layer}_expert_agg_{i}')
        
        # Connect remainder of layer
        for i in range(len(MOE_GPU_IDS)):
            dot.edge(f'layer{layer}_expert_agg_{i}', f'layer{layer}_expert_allreduce')
        
        dot.edge(f'layer{layer}_expert_allreduce', f'layer{layer}_residual2')
        dot.edge(f'layer{layer}_residual1', f'layer{layer}_residual2')  # Skip connection
        
        current = f'layer{layer}_residual2'
    
    # Connect final output
    dot.edge(current, 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_ma_separation_dag()
    dag.render('../outputs/2025-10-17-10-16-04/ma_separation', format='svg', cleanup=False)
    
    # Save DOT file
    with open('../outputs/2025-10-17-10-16-04/ma_separation.dot', 'w') as f:
        f.write(dag.source)