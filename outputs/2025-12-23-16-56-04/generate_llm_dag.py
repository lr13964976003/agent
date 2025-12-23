#!/usr/bin/env python3
"""
LLM Parallel Strategy DAG Generator
Generates Graphviz DAG for Llama3_70B_Instruct with TP=2, PP=4 deployment
"""

import graphviz
import os

def create_llm_parallel_dag():
    """Create DAG for LLM parallel strategy deployment"""
    
    # Create directed graph
    dot = graphviz.Digraph(
        comment='Llama3_70B_Instruct Parallel Strategy DAG',
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'bgcolor': 'white',
            'fontname': 'Arial',
            'fontsize': '12',
            'ranksep': '1.0',
            'nodesep': '0.5'
        },
        node_attr={
            'fontname': 'Arial',
            'fontsize': '10',
            'shape': 'rectangle',
            'style': 'filled,rounded',
            'margin': '0.2,0.1'
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '9',
            'arrowhead': 'normal',
            'penwidth': '1.5'
        }
    )
    
    # Define colors for different node types
    colors = {
        'input': '#E8F5E8',      # Light green
        'compute': '#E6F3FF',    # Light blue
        'comm': '#FFE6E6',       # Light red
        'routing': '#FFF8E6',    # Light yellow
        'output': '#F0E6FF'      # Light purple
    }
    
    # Model dimensions
    batch_size = '?'
    seq_len = '?'
    hidden_size = 8192
    num_heads = 64
    head_dim = hidden_size // num_heads  # 128
    vocab_size = 128256
    
    # Input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor=colors['input'], style='filled')
    
    # Embedding layer (distributed across all GPUs via TP)
    dot.node('embed_0', 
             f'Embedding GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
             fillcolor=colors['compute'])
    dot.node('embed_1', 
             f'Embedding GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
             fillcolor=colors['compute'])
    
    dot.edge('input', 'embed_0')
    dot.edge('input', 'embed_1')
    
    # All-Gather for embedding outputs
    dot.node('embed_ag', 'All-Gather\\nEmbedding Output\\n[TP All-Gather]', 
             shape='parallelogram', fillcolor=colors['comm'])
    dot.edge('embed_0', 'embed_ag')
    dot.edge('embed_1', 'embed_ag')
    
    # Pipeline Stage 0: Layers 0-19 (GPUs 0,1)
    prev_node = 'embed_ag'
    for layer in range(20):
        # Attention computation for layer
        # QKV projection (split across TP)
        dot.node(f'layer{layer}_qkv_0', 
                 f'Layer {layer} QKV Proj GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_qkv_1', 
                 f'Layer {layer} QKV Proj GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(prev_node, f'layer{layer}_qkv_0')
        dot.edge(prev_node, f'layer{layer}_qkv_1')
        
        # Attention computation (attention scores and softmax)
        dot.node(f'layer{layer}_attn_0', 
                 f'Layer {layer} Attention GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_1', 
                 f'Layer {layer} Attention GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_qkv_0', f'layer{layer}_attn_0')
        dot.edge(f'layer{layer}_qkv_1', f'layer{layer}_attn_1')
        
        # Attention output projection
        dot.node(f'layer{layer}_attn_out_0', 
                 f'Layer {layer} Attn Out GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_out_1', 
                 f'Layer {layer} Attn Out GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_0', f'layer{layer}_attn_out_0')
        dot.edge(f'layer{layer}_attn_1', f'layer{layer}_attn_out_1')
        
        # All-Reduce for attention outputs
        dot.node(f'layer{layer}_attn_ar', f'Layer {layer} Attn All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_attn_out_0', f'layer{layer}_attn_ar')
        dot.edge(f'layer{layer}_attn_out_1', f'layer{layer}_attn_ar')
        
        # FFN computation
        # First linear layer (split across TP)
        dot.node(f'layer{layer}_ffn1_0', 
                 f'Layer {layer} FFN1 GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn1_1', 
                 f'Layer {layer} FFN1 GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_0')
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_1')
        
        # Activation function (SiLU)
        dot.node(f'layer{layer}_act_0', 
                 f'Layer {layer} SiLU GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_act_1', 
                 f'Layer {layer} SiLU GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_ffn1_0', f'layer{layer}_act_0')
        dot.edge(f'layer{layer}_ffn1_1', f'layer{layer}_act_1')
        
        # Second linear layer
        dot.node(f'layer{layer}_ffn2_0', 
                 f'Layer {layer} FFN2 GPU0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn2_1', 
                 f'Layer {layer} FFN2 GPU1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_act_0', f'layer{layer}_ffn2_0')
        dot.edge(f'layer{layer}_act_1', f'layer{layer}_ffn2_1')
        
        # All-Reduce for FFN outputs
        dot.node(f'layer{layer}_ffn_ar', f'Layer {layer} FFN All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_ffn2_0', f'layer{layer}_ffn_ar')
        dot.edge(f'layer{layer}_ffn2_1', f'layer{layer}_ffn_ar')
        
        prev_node = f'layer{layer}_ffn_ar'
    
    # Pipeline forwarding from Stage 0 to Stage 1
    dot.node('stage0_to_stage1', 'Pipeline Forward\\nStage 0 → Stage 1\\n[PP Forward]', 
             shape='parallelogram', fillcolor=colors['comm'])
    dot.edge(prev_node, 'stage0_to_stage1')
    
    # Pipeline Stage 1: Layers 20-39 (GPUs 2,3)
    prev_node = 'stage0_to_stage1'
    for layer in range(20, 40):
        # Similar structure as Stage 0 but on GPUs 2,3
        # QKV projection
        dot.node(f'layer{layer}_qkv_2', 
                 f'Layer {layer} QKV Proj GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_qkv_3', 
                 f'Layer {layer} QKV Proj GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(prev_node, f'layer{layer}_qkv_2')
        dot.edge(prev_node, f'layer{layer}_qkv_3')
        
        # Attention computation
        dot.node(f'layer{layer}_attn_2', 
                 f'Layer {layer} Attention GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_3', 
                 f'Layer {layer} Attention GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_qkv_2', f'layer{layer}_attn_2')
        dot.edge(f'layer{layer}_qkv_3', f'layer{layer}_attn_3')
        
        # Attention output projection
        dot.node(f'layer{layer}_attn_out_2', 
                 f'Layer {layer} Attn Out GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_out_3', 
                 f'Layer {layer} Attn Out GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_2', f'layer{layer}_attn_out_2')
        dot.edge(f'layer{layer}_attn_3', f'layer{layer}_attn_out_3')
        
        # All-Reduce for attention outputs
        dot.node(f'layer{layer}_attn_ar', f'Layer {layer} Attn All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_attn_out_2', f'layer{layer}_attn_ar')
        dot.edge(f'layer{layer}_attn_out_3', f'layer{layer}_attn_ar')
        
        # FFN computation
        dot.node(f'layer{layer}_ffn1_2', 
                 f'Layer {layer} FFN1 GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn1_3', 
                 f'Layer {layer} FFN1 GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_2')
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_3')
        
        # Activation function
        dot.node(f'layer{layer}_act_2', 
                 f'Layer {layer} SiLU GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_act_3', 
                 f'Layer {layer} SiLU GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_ffn1_2', f'layer{layer}_act_2')
        dot.edge(f'layer{layer}_ffn1_3', f'layer{layer}_act_3')
        
        # Second linear layer
        dot.node(f'layer{layer}_ffn2_2', 
                 f'Layer {layer} FFN2 GPU2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn2_3', 
                 f'Layer {layer} FFN2 GPU3\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_act_2', f'layer{layer}_ffn2_2')
        dot.edge(f'layer{layer}_act_3', f'layer{layer}_ffn2_3')
        
        # All-Reduce for FFN outputs
        dot.node(f'layer{layer}_ffn_ar', f'Layer {layer} FFN All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_ffn2_2', f'layer{layer}_ffn_ar')
        dot.edge(f'layer{layer}_ffn2_3', f'layer{layer}_ffn_ar')
        
        prev_node = f'layer{layer}_ffn_ar'
    
    # Pipeline forwarding from Stage 1 to Stage 2
    dot.node('stage1_to_stage2', 'Pipeline Forward\\nStage 1 → Stage 2\\n[PP Forward]', 
             shape='parallelogram', fillcolor=colors['comm'])
    dot.edge(prev_node, 'stage1_to_stage2')
    
    # Pipeline Stage 2: Layers 40-59 (GPUs 4,5)
    prev_node = 'stage1_to_stage2'
    for layer in range(40, 60):
        # Similar structure as previous stages but on GPUs 4,5
        # QKV projection
        dot.node(f'layer{layer}_qkv_4', 
                 f'Layer {layer} QKV Proj GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_qkv_5', 
                 f'Layer {layer} QKV Proj GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(prev_node, f'layer{layer}_qkv_4')
        dot.edge(prev_node, f'layer{layer}_qkv_5')
        
        # Attention computation
        dot.node(f'layer{layer}_attn_4', 
                 f'Layer {layer} Attention GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_5', 
                 f'Layer {layer} Attention GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_qkv_4', f'layer{layer}_attn_4')
        dot.edge(f'layer{layer}_qkv_5', f'layer{layer}_attn_5')
        
        # Attention output projection
        dot.node(f'layer{layer}_attn_out_4', 
                 f'Layer {layer} Attn Out GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_out_5', 
                 f'Layer {layer} Attn Out GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_4', f'layer{layer}_attn_out_4')
        dot.edge(f'layer{layer}_attn_5', f'layer{layer}_attn_out_5')
        
        # All-Reduce for attention outputs
        dot.node(f'layer{layer}_attn_ar', f'Layer {layer} Attn All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_attn_out_4', f'layer{layer}_attn_ar')
        dot.edge(f'layer{layer}_attn_out_5', f'layer{layer}_attn_ar')
        
        # FFN computation
        dot.node(f'layer{layer}_ffn1_4', 
                 f'Layer {layer} FFN1 GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn1_5', 
                 f'Layer {layer} FFN1 GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_4')
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_5')
        
        # Activation function
        dot.node(f'layer{layer}_act_4', 
                 f'Layer {layer} SiLU GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_act_5', 
                 f'Layer {layer} SiLU GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_ffn1_4', f'layer{layer}_act_4')
        dot.edge(f'layer{layer}_ffn1_5', f'layer{layer}_act_5')
        
        # Second linear layer
        dot.node(f'layer{layer}_ffn2_4', 
                 f'Layer {layer} FFN2 GPU4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn2_5', 
                 f'Layer {layer} FFN2 GPU5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_act_4', f'layer{layer}_ffn2_4')
        dot.edge(f'layer{layer}_act_5', f'layer{layer}_ffn2_5')
        
        # All-Reduce for FFN outputs
        dot.node(f'layer{layer}_ffn_ar', f'Layer {layer} FFN All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_ffn2_4', f'layer{layer}_ffn_ar')
        dot.edge(f'layer{layer}_ffn2_5', f'layer{layer}_ffn_ar')
        
        prev_node = f'layer{layer}_ffn_ar'
    
    # Pipeline forwarding from Stage 2 to Stage 3
    dot.node('stage2_to_stage3', 'Pipeline Forward\\nStage 2 → Stage 3\\n[PP Forward]', 
             shape='parallelogram', fillcolor=colors['comm'])
    dot.edge(prev_node, 'stage2_to_stage3')
    
    # Pipeline Stage 3: Layers 60-79 (GPUs 6,7)
    prev_node = 'stage2_to_stage3'
    for layer in range(60, 80):
        # Similar structure as previous stages but on GPUs 6,7
        # QKV projection
        dot.node(f'layer{layer}_qkv_6', 
                 f'Layer {layer} QKV Proj GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_qkv_7', 
                 f'Layer {layer} QKV Proj GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(prev_node, f'layer{layer}_qkv_6')
        dot.edge(prev_node, f'layer{layer}_qkv_7')
        
        # Attention computation
        dot.node(f'layer{layer}_attn_6', 
                 f'Layer {layer} Attention GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_7', 
                 f'Layer {layer} Attention GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_qkv_6', f'layer{layer}_attn_6')
        dot.edge(f'layer{layer}_qkv_7', f'layer{layer}_attn_7')
        
        # Attention output projection
        dot.node(f'layer{layer}_attn_out_6', 
                 f'Layer {layer} Attn Out GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_attn_out_7', 
                 f'Layer {layer} Attn Out GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//2}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_6', f'layer{layer}_attn_out_6')
        dot.edge(f'layer{layer}_attn_7', f'layer{layer}_attn_out_7')
        
        # All-Reduce for attention outputs
        dot.node(f'layer{layer}_attn_ar', f'Layer {layer} Attn All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_attn_out_6', f'layer{layer}_attn_ar')
        dot.edge(f'layer{layer}_attn_out_7', f'layer{layer}_attn_ar')
        
        # FFN computation
        dot.node(f'layer{layer}_ffn1_6', 
                 f'Layer {layer} FFN1 GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn1_7', 
                 f'Layer {layer} FFN1 GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_6')
        dot.edge(f'layer{layer}_attn_ar', f'layer{layer}_ffn1_7')
        
        # Activation function
        dot.node(f'layer{layer}_act_6', 
                 f'Layer {layer} SiLU GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_act_7', 
                 f'Layer {layer} SiLU GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_ffn1_6', f'layer{layer}_act_6')
        dot.edge(f'layer{layer}_ffn1_7', f'layer{layer}_act_7')
        
        # Second linear layer
        dot.node(f'layer{layer}_ffn2_6', 
                 f'Layer {layer} FFN2 GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        dot.node(f'layer{layer}_ffn2_7', 
                 f'Layer {layer} FFN2 GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, intermediate_size={28672//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
                 fillcolor=colors['compute'])
        
        dot.edge(f'layer{layer}_act_6', f'layer{layer}_ffn2_6')
        dot.edge(f'layer{layer}_act_7', f'layer{layer}_ffn2_7')
        
        # All-Reduce for FFN outputs
        dot.node(f'layer{layer}_ffn_ar', f'Layer {layer} FFN All-Reduce\\n[TP All-Reduce]', 
                 shape='parallelogram', fillcolor=colors['comm'])
        dot.edge(f'layer{layer}_ffn2_6', f'layer{layer}_ffn_ar')
        dot.edge(f'layer{layer}_ffn2_7', f'layer{layer}_ffn_ar')
        
        prev_node = f'layer{layer}_ffn_ar'
    
    # Final output processing
    # Output projection (head)
    dot.node('output_proj_6', 
             f'Output Projection GPU6\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
             fillcolor=colors['compute'])
    dot.node('output_proj_7', 
             f'Output Projection GPU7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size//2}]',
             fillcolor=colors['compute'])
    
    dot.edge(prev_node, 'output_proj_6')
    dot.edge(prev_node, 'output_proj_7')
    
    # All-Reduce for final output
    dot.node('final_output_ar', 'Final Output All-Reduce\\n[TP All-Reduce]', 
             shape='parallelogram', fillcolor=colors['comm'])
    dot.edge('output_proj_6', 'final_output_ar')
    dot.edge('output_proj_7', 'final_output_ar')
    
    # Final output
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]',
             shape='ellipse', fillcolor=colors['output'], style='filled')
    
    dot.edge('final_output_ar', 'output')
    
    return dot

def main():
    """Main function to generate and save DAG"""
    output_dir = "../outputs/2025-12-23-16-56-04"
    
    # Create DAG
    print("Generating LLM Parallel Strategy DAG...")
    dag = create_llm_parallel_dag()
    
    # Save as DOT file
    dot_path = os.path.join(output_dir, "llm_parallel_strategy_dag.dot")
    dag.save(dot_path)
    print(f"Saved DOT file: {dot_path}")
    
    # Render as SVG
    svg_path = os.path.join(output_dir, "llm_parallel_strategy_dag.svg")
    dag.render(os.path.join(output_dir, "llm_parallel_strategy_dag"), format='svg', cleanup=True)
    print(f"Saved SVG file: {svg_path}")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "dag_summary.json")
    summary = {
        "dag_files": {
            "dot_file": dot_path,
            "svg_file": svg_path
        },
        "parallel_strategy": {
            "tensor_parallelism": 2,
            "pipeline_parallelism": 4,
            "total_gpus": 8,
            "layers_per_stage": 20,
            "total_layers": 80
        },
        "node_types": {
            "input_nodes": 1,
            "compute_nodes": 320,  # 80 layers * 4 ops per layer * 2 TP ranks
            "communication_nodes": 160,  # 80 layers * 2 All-Reduce per layer
            "routing_nodes": 4,  # Pipeline forwarding nodes
            "output_nodes": 1
        },
        "gpu_distribution": {
            "stage_0": "GPUs 0,1 (Layers 0-19)",
            "stage_1": "GPUs 2,3 (Layers 20-39)",
            "stage_2": "GPUs 4,5 (Layers 40-59)",
            "stage_3": "GPUs 6,7 (Layers 60-79)"
        }
    }
    
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")
    
    print("DAG generation completed successfully!")
    return summary

if __name__ == "__main__":
    summary = main()