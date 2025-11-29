#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_layer_wise_dag():
    """Create a comprehensive DAG for layer-wise deployment strategy"""
    
    # Create a new directed graph
    dot = Digraph(comment='Layer-wise Deployment Strategy DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define styles for different node types
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters from deployment config
    batch_size = 128
    seq_len = 10000
    hidden_size = 16384
    num_heads = 32
    head_dim = 128
    mlp_hidden_size = 16384
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='dashed', label='Input Batch')
        c.node('input', 
               f'Input\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='ellipse', fillcolor='lightcyan')
    
    # Layer 0-1 on GPU 0
    with dot.subgraph(name='cluster_gpu0') as c:
        c.attr(style='rounded,filled', fillcolor='lightgray', label='GPU 0: Layers 0-1')
        
        # Layer Norm (Layer 0)
        c.node('gpu0_layernorm0', 
               f'LayerNorm\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Multi-Head Attention (Layer 0)
        c.node('gpu0_mha_qkv', 
               f'MHA Q/K/V Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu0_mha_attention', 
               f'MHA Attention\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu0_mha_out', 
               f'MHA Output Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 0)
        c.node('gpu0_residual0', 
               f'Residual Add\\nGPU: 0\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Layer Norm (Layer 1)
        c.node('gpu0_layernorm1', 
               f'LayerNorm\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # FFN Gate (Layer 1)
        c.node('gpu0_ffn_gate', 
               f'FFN Gate\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # FFN Experts (Layer 1)
        c.node('gpu0_ffn_experts', 
               f'FFN Experts\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu0_ffn_out', 
               f'FFN Output Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 1)
        c.node('gpu0_residual1', 
               f'Residual Add\\nGPU: 0\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
    
    # Communication from GPU 0 to GPU 1
    dot.node('comm_gpu0_gpu1', 
             f'Inter-GPU Transfer\\nGPU: 0 → 1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor='lightgreen')
    
    # Layer 8-9 on GPU 4 (representative middle layers)
    with dot.subgraph(name='cluster_gpu4') as c:
        c.attr(style='rounded,filled', fillcolor='lightgray', label='GPU 4: Layers 8-9 (Representative Middle)')
        
        # Layer Norm (Layer 8)
        c.node('gpu4_layernorm8', 
               f'LayerNorm\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Multi-Head Attention (Layer 8)
        c.node('gpu4_mha_qkv', 
               f'MHA Q/K/V Projection\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu4_mha_attention', 
               f'MHA Attention\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu4_mha_out', 
               f'MHA Output Projection\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 8)
        c.node('gpu4_residual8', 
               f'Residual Add\\nGPU: 4\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Layer Norm (Layer 9)
        c.node('gpu4_layernorm9', 
               f'LayerNorm\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # FFN Gate with expert selection (dashed line representation)
        c.node('gpu4_ffn_gate', 
               f'FFN Gate (Expert Selection)\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='rectangle', fillcolor='lightblue', style='dashed')
        
        # Tensor Split for experts
        c.node('gpu4_split_experts', 
               f'Split for Experts\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert 1
        c.node('gpu4_expert1', 
               f'Expert 1\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Expert 2
        c.node('gpu4_expert2', 
               f'Expert 2\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Aggregate experts
        c.node('gpu4_aggregate_experts', 
               f'Aggregate Experts\\nGPU: 4\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('gpu4_ffn_out', 
               f'FFN Output Projection\\nGPU: 4\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 9)
        c.node('gpu4_residual9', 
               f'Residual Add\\nGPU: 4\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
    
    # Communication from GPU 4 to GPU 5
    dot.node('comm_gpu4_gpu5', 
             f'Inter-GPU Transfer\\nGPU: 4 → 5\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor='lightgreen')
    
    # Layer 14-15 on GPU 7 (final layers)
    with dot.subgraph(name='cluster_gpu7') as c:
        c.attr(style='rounded,filled', fillcolor='lightgray', label='GPU 7: Layers 14-15 (Final)')
        
        # Layer Norm (Layer 14)
        c.node('gpu7_layernorm14', 
               f'LayerNorm\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Multi-Head Attention (Layer 14)
        c.node('gpu7_mha_qkv', 
               f'MHA Q/K/V Projection\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu7_mha_attention', 
               f'MHA Attention\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu7_mha_out', 
               f'MHA Output Projection\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 14)
        c.node('gpu7_residual14', 
               f'Residual Add\\nGPU: 7\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Layer Norm (Layer 15)
        c.node('gpu7_layernorm15', 
               f'LayerNorm\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # FFN Gate (Layer 15)
        c.node('gpu7_ffn_gate', 
               f'FFN Gate\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # FFN Experts (Layer 15)
        c.node('gpu7_ffn_experts', 
               f'FFN Experts\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        c.node('gpu7_ffn_out', 
               f'FFN Output Projection\\nGPU: 7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden_size={mlp_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='rectangle', fillcolor='lightblue')
        
        # Residual Add (Layer 15)
        c.node('gpu7_residual15', 
               f'Residual Add\\nGPU: 7\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
               shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 
             f'Output\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
             shape='ellipse', fillcolor='lightcyan')
    
    # Define edges (connections)
    # Input to GPU 0
    dot.edge('input', 'gpu0_layernorm0')
    
    # GPU 0 Layer 0 flow
    dot.edge('gpu0_layernorm0', 'gpu0_mha_qkv')
    dot.edge('gpu0_mha_qkv', 'gpu0_mha_attention')
    dot.edge('gpu0_mha_attention', 'gpu0_mha_out')
    dot.edge('gpu0_mha_out', 'gpu0_residual0')
    dot.edge('input', 'gpu0_residual0')  # Residual connection
    
    # GPU 0 Layer 1 flow
    dot.edge('gpu0_residual0', 'gpu0_layernorm1')
    dot.edge('gpu0_layernorm1', 'gpu0_ffn_gate')
    dot.edge('gpu0_ffn_gate', 'gpu0_ffn_experts')
    dot.edge('gpu0_ffn_experts', 'gpu0_ffn_out')
    dot.edge('gpu0_ffn_out', 'gpu0_residual1')
    dot.edge('gpu0_residual0', 'gpu0_residual1')  # Residual connection
    
    # GPU 0 to GPU 1 communication
    dot.edge('gpu0_residual1', 'comm_gpu0_gpu1')
    
    # Communication to GPU 4 (simplified representation)
    dot.edge('comm_gpu0_gpu1', 'gpu4_layernorm8', style='dashed', label='... through GPUs 1-3 ...')
    
    # GPU 4 Layer 8 flow
    dot.edge('gpu4_layernorm8', 'gpu4_mha_qkv')
    dot.edge('gpu4_mha_qkv', 'gpu4_mha_attention')
    dot.edge('gpu4_mha_attention', 'gpu4_mha_out')
    dot.edge('gpu4_mha_out', 'gpu4_residual8')
    dot.edge('gpu4_layernorm8', 'gpu4_residual8')  # Residual connection
    
    # GPU 4 Layer 9 flow with expert routing
    dot.edge('gpu4_residual8', 'gpu4_layernorm9')
    dot.edge('gpu4_layernorm9', 'gpu4_ffn_gate')
    dot.edge('gpu4_ffn_gate', 'gpu4_split_experts', style='dashed')  # Gate selection
    dot.edge('gpu4_split_experts', 'gpu4_expert1')
    dot.edge('gpu4_split_experts', 'gpu4_expert2')
    dot.edge('gpu4_expert1', 'gpu4_aggregate_experts')
    dot.edge('gpu4_expert2', 'gpu4_aggregate_experts')
    dot.edge('gpu4_aggregate_experts', 'gpu4_ffn_out')
    dot.edge('gpu4_ffn_out', 'gpu4_residual9')
    dot.edge('gpu4_residual8', 'gpu4_residual9')  # Residual connection
    
    # GPU 4 to GPU 5 communication
    dot.edge('gpu4_residual9', 'comm_gpu4_gpu5')
    
    # Communication to GPU 7 (simplified representation)
    dot.edge('comm_gpu4_gpu5', 'gpu7_layernorm14', style='dashed', label='... through GPUs 5-6 ...')
    
    # GPU 7 Layer 14 flow
    dot.edge('gpu7_layernorm14', 'gpu7_mha_qkv')
    dot.edge('gpu7_mha_qkv', 'gpu7_mha_attention')
    dot.edge('gpu7_mha_attention', 'gpu7_mha_out')
    dot.edge('gpu7_mha_out', 'gpu7_residual14')
    dot.edge('gpu7_layernorm14', 'gpu7_residual14')  # Residual connection
    
    # GPU 7 Layer 15 flow
    dot.edge('gpu7_residual14', 'gpu7_layernorm15')
    dot.edge('gpu7_layernorm15', 'gpu7_ffn_gate')
    dot.edge('gpu7_ffn_gate', 'gpu7_ffn_experts')
    dot.edge('gpu7_ffn_experts', 'gpu7_ffn_out')
    dot.edge('gpu7_ffn_out', 'gpu7_residual15')
    dot.edge('gpu7_residual14', 'gpu7_residual15')  # Residual connection
    
    # Final output
    dot.edge('gpu7_residual15', 'output')
    
    return dot

if __name__ == "__main__":
    # Generate the DAG
    dag = create_layer_wise_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-11-28-17-19-06/layer_wise_deployment_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-11-28-17-19-06/layer_wise_deployment_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files saved:")
    print("- ../outputs/2025-11-28-17-19-06/layer_wise_deployment_dag.dot")
    print("- ../outputs/2025-11-28-17-19-06/layer_wise_deployment_dag.svg")