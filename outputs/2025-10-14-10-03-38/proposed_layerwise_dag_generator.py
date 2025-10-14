import graphviz

# Create proposed DAG with Layer-wise Partitioning
# 16 layers → 16 GPUs, each layer fits in SRAM/L2 cache
dot = graphviz.Digraph(comment='Proposed: Layer-wise Partitioning Strategy', format='svg')
dot.attr(rankdir='TB', size='30,30')

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

# Create input node
dot.node('input', f'Model Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: N/A', shape='ellipse')

# Process each layer on separate GPU
prev_node = 'input'
for layer_idx in range(16):
    gpu_id = layer_idx  # GPU 0-15
    
    with dot.subgraph(name=f'cluster_layer_{layer_idx}_gpu_{gpu_id}') as cl:
        cl.attr(label=f'Layer {layer_idx} on GPU {gpu_id}\nSRAM/L2 Cache Optimized', style='dashed', color='blue')
        
        # Inter-GPU communication from previous layer
        if layer_idx > 0:
            comm_node = f'comm_layer_{layer_idx}'
            cl.node(comm_node, f'Receive from GPU {gpu_id-1}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='parallelogram')
            dot.edge(prev_node, comm_node)
            prev_node = comm_node
        
        # Layer Norm 1 - now on single GPU
        ln1_node = f'layer{layer_idx}_ln1'
        cl.node(ln1_node, f'LayerNorm1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(prev_node, ln1_node)
        
        # QKV Linear - full layer on single GPU
        qkv_node = f'layer{layer_idx}_qkv'
        cl.node(qkv_node, f'QKV Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(ln1_node, qkv_node)
        
        # Reshape QKV for attention
        reshape_qkv_node = f'layer{layer_idx}_reshape_qkv'
        cl.node(reshape_qkv_node, f'Reshape QKV\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(qkv_node, reshape_qkv_node)
        
        # Split Q, K, V
        split_q_node = f'layer{layer_idx}_split_q'
        cl.node(split_q_node, f'Split Query\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(reshape_qkv_node, split_q_node)
        
        split_k_node = f'layer{layer_idx}_split_k'
        cl.node(split_k_node, f'Split Key\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(reshape_qkv_node, split_k_node)
        
        split_v_node = f'layer{layer_idx}_split_v'
        cl.node(split_v_node, f'Split Value\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, qkv=3]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(reshape_qkv_node, split_v_node)
        
        # Attention computation
        attn_score_node = f'layer{layer_idx}_attn_score'
        cl.node(attn_score_node, f'Attention Score\nInput1: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nOutput: [batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(split_q_node, attn_score_node)
        dot.edge(split_k_node, attn_score_node)
        
        # Attention weights (softmax)
        attn_weights_node = f'layer{layer_idx}_attn_weights'
        cl.node(attn_weights_node, f'Softmax\nInput: [batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(attn_score_node, attn_weights_node)
        
        # Attention output
        attn_out_node = f'layer{layer_idx}_attn_out'
        cl.node(attn_out_node, f'Attention Output\nInput1: [batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(attn_weights_node, attn_out_node)
        dot.edge(split_v_node, attn_out_node)
        
        # Reshape attention output
        attn_reshape_node = f'layer{layer_idx}_attn_reshape'
        cl.node(attn_reshape_node, f'Reshape Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(attn_out_node, attn_reshape_node)
        
        # Output projection
        attn_proj_node = f'layer{layer_idx}_attn_proj'
        cl.node(attn_proj_node, f'Attention Output Projection\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(attn_reshape_node, attn_proj_node)
        
        # Residual add 1
        res1_node = f'layer{layer_idx}_res1'
        cl.node(res1_node, f'Residual Add 1\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(prev_node, res1_node)
        dot.edge(attn_proj_node, res1_node)
        
        # Layer Norm 2
        ln2_node = f'layer{layer_idx}_ln2'
        cl.node(ln2_node, f'LayerNorm2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(res1_node, ln2_node)
        
        # MLP Linear 1
        mlp1_node = f'layer{layer_idx}_mlp1'
        cl.node(mlp1_node, f'MLP Linear1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(ln2_node, mlp1_node)
        
        # GELU activation
        gelu_node = f'layer{layer_idx}_gelu'
        cl.node(gelu_node, f'GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(mlp1_node, gelu_node)
        
        # MLP Linear 2
        mlp2_node = f'layer{layer_idx}_mlp2'
        cl.node(mlp2_node, f'MLP Linear2\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(gelu_node, mlp2_node)
        
        # Residual add 2
        res2_node = f'layer{layer_idx}_res2'
        cl.node(res2_node, f'Residual Add 2\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}', shape='rectangle')
        dot.edge(res1_node, res2_node)
        dot.edge(mlp2_node, res2_node)
        
        # Send to next GPU
        if layer_idx < 15:
            send_node = f'send_to_gpu_{gpu_id+1}'
            cl.node(send_node, f'Send to GPU {gpu_id+1}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id} → GPU {gpu_id+1}', shape='parallelogram')
            dot.edge(res2_node, send_node)
            prev_node = send_node
        else:
            prev_node = res2_node

# Create final output node
dot.node('output', f'Model Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: 15', shape='ellipse')
dot.edge(prev_node, 'output')

# Save files
dot.render('./outputs/2025-10-14-10-03-38/proposed_layerwise_dag', format='svg', cleanup=False)
dot.save('./outputs/2025-10-14-10-03-38/proposed_layerwise_dag.dot')

print("Proposed Layer-wise DAG generated successfully!")
print("Files saved:")
print("- proposed_layerwise_dag.svg")
print("- proposed_layerwise_dag.dot")