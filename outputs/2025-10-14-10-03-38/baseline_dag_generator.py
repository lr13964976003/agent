import graphviz

# Create baseline DAG with Tensor Parallelism (TP=8) and Pipeline Parallelism (PP=2)
dot = graphviz.Digraph(comment='Baseline: Tensor Parallelism + Pipeline Parallelism', format='svg')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Communication/Routing

# Model parameters from paper
batch_size = 1024
seq_len = 10000
hidden_size = 8192
num_heads = 16
head_dim = 512
ffn_hidden_size = 32768

# Pipeline Stage 0: Layers 0-7 on GPUs 0-7
with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
    c0.attr(label='Pipeline Stage 0: Layers 0-7\nGPUs 0-7 (TP=8)', style='dashed')
    
    # Input to pipeline stage 0
    c0.node('input_p0', f'Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='ellipse')
    
    # Process layers 0-7
    prev_node = 'input_p0'
    for layer_idx in range(8):
        with c0.subgraph(name=f'cluster_layer_{layer_idx}') as cl:
            cl.attr(label=f'Layer {layer_idx}')
            
            # Layer Norm 1 - replicated across all TP devices
            ln1_node = f'layer{layer_idx}_ln1'
            cl.node(ln1_node, f'LayerNorm1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(prev_node, ln1_node)
            
            # QKV Linear - column parallel
            qkv_node = f'layer{layer_idx}_qkv'
            cl.node(qkv_node, f'QKV Linear (Column-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(ln1_node, qkv_node)
            
            # Attention - split across heads
            attn_node = f'layer{layer_idx}_attn'
            cl.node(attn_node, f'Multi-Head Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(qkv_node, attn_node)
            
            # All-reduce for attention output
            attn_ar_node = f'layer{layer_idx}_attn_ar'
            cl.node(attn_ar_node, f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='parallelogram')
            dot.edge(attn_node, attn_ar_node)
            
            # Residual add 1
            res1_node = f'layer{layer_idx}_res1'
            cl.node(res1_node, f'Residual Add 1\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(prev_node, res1_node)
            dot.edge(attn_ar_node, res1_node)
            
            # Layer Norm 2
            ln2_node = f'layer{layer_idx}_ln2'
            cl.node(ln2_node, f'LayerNorm2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(res1_node, ln2_node)
            
            # MLP - column then row parallel
            mlp1_node = f'layer{layer_idx}_mlp1'
            cl.node(mlp1_node, f'MLP Linear1 (Column-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(ln2_node, mlp1_node)
            
            # GELU activation
            gelu_node = f'layer{layer_idx}_gelu'
            cl.node(gelu_node, f'GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(mlp1_node, gelu_node)
            
            # MLP - row parallel
            mlp2_node = f'layer{layer_idx}_mlp2'
            cl.node(mlp2_node, f'MLP Linear2 (Row-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(gelu_node, mlp2_node)
            
            # All-reduce for MLP output
            mlp_ar_node = f'layer{layer_idx}_mlp_ar'
            cl.node(mlp_ar_node, f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='parallelogram')
            dot.edge(mlp2_node, mlp_ar_node)
            
            # Residual add 2
            res2_node = f'layer{layer_idx}_res2'
            cl.node(res2_node, f'Residual Add 2\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7', shape='rectangle')
            dot.edge(res1_node, res2_node)
            dot.edge(mlp_ar_node, res2_node)
            
            prev_node = res2_node
    
    # Pipeline send to stage 1
    send_node = 'send_to_stage1'
    c0.node(send_node, f'Pipeline Send\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 0-7 → GPUs 8-15', shape='parallelogram')
    dot.edge(prev_node, send_node)

# Pipeline Stage 1: Layers 8-15 on GPUs 8-15
with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
    c1.attr(label='Pipeline Stage 1: Layers 8-15\nGPUs 8-15 (TP=8)', style='dashed')
    
    # Pipeline receive from stage 0
    recv_node = 'recv_from_stage0'
    c1.node(recv_node, f'Pipeline Receive\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='parallelogram')
    
    # Process layers 8-15
    prev_node = recv_node
    for layer_idx in range(8, 16):
        with c1.subgraph(name=f'cluster_layer_{layer_idx}') as cl:
            cl.attr(label=f'Layer {layer_idx}')
            
            # Layer Norm 1
            ln1_node = f'layer{layer_idx}_ln1'
            cl.node(ln1_node, f'LayerNorm1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(prev_node, ln1_node)
            
            # QKV Linear - column parallel
            qkv_node = f'layer{layer_idx}_qkv'
            cl.node(qkv_node, f'QKV Linear (Column-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(ln1_node, qkv_node)
            
            # Attention
            attn_node = f'layer{layer_idx}_attn'
            cl.node(attn_node, f'Multi-Head Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(qkv_node, attn_node)
            
            # All-reduce for attention output
            attn_ar_node = f'layer{layer_idx}_attn_ar'
            cl.node(attn_ar_node, f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='parallelogram')
            dot.edge(attn_node, attn_ar_node)
            
            # Residual add 1
            res1_node = f'layer{layer_idx}_res1'
            cl.node(res1_node, f'Residual Add 1\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(prev_node, res1_node)
            dot.edge(attn_ar_node, res1_node)
            
            # Layer Norm 2
            ln2_node = f'layer{layer_idx}_ln2'
            cl.node(ln2_node, f'LayerNorm2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(res1_node, ln2_node)
            
            # MLP - column then row parallel
            mlp1_node = f'layer{layer_idx}_mlp1'
            cl.node(mlp1_node, f'MLP Linear1 (Column-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(ln2_node, mlp1_node)
            
            # GELU activation
            gelu_node = f'layer{layer_idx}_gelu'
            cl.node(gelu_node, f'GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(mlp1_node, gelu_node)
            
            # MLP - row parallel
            mlp2_node = f'layer{layer_idx}_mlp2'
            cl.node(mlp2_node, f'MLP Linear2 (Row-Parallel)\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(gelu_node, mlp2_node)
            
            # All-reduce for MLP output
            mlp_ar_node = f'layer{layer_idx}_mlp_ar'
            cl.node(mlp_ar_node, f'All-Reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='parallelogram')
            dot.edge(mlp2_node, mlp_ar_node)
            
            # Residual add 2
            res2_node = f'layer{layer_idx}_res2'
            cl.node(res2_node, f'Residual Add 2\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='rectangle')
            dot.edge(res1_node, res2_node)
            dot.edge(mlp_ar_node, res2_node)
            
            prev_node = res2_node
    
    # Final output
    output_node = 'final_output'
    c1.node(output_node, f'Final Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All GPUs 8-15', shape='ellipse')
    dot.edge(prev_node, output_node)

# Connect pipeline stages
dot.edge('send_to_stage1', 'recv_from_stage0', style='dashed', label='Pipeline Communication')

# Save files
dot.render('./outputs/2025-10-14-10-03-38/baseline_tensor_pipeline_dag', format='svg', cleanup=False)
dot.save('./outputs/2025-10-14-10-03-38/baseline_tensor_pipeline_dag.dot')

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- baseline_tensor_pipeline_dag.svg")
print("- baseline_tensor_pipeline_dag.dot")