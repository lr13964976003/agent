import graphviz
import os
import math

def calculate_attention_pool_gpus(sequence_length):
    """Calculate number of GPUs needed for attention pool based on sequence length"""
    return min(32, math.ceil(sequence_length / 1024))

def create_fa_pool_dag(sequence_length=8192):
    """Create FA Pool DAG with dynamic GPU allocation"""
    attention_pool_gpus = calculate_attention_pool_gpus(sequence_length)
    
    dot = graphviz.Digraph('fa_pool_transformer', 
                          comment=f'FA Pool 4-Layer Transformer\\nSeq Len: {sequence_length}, Pool GPUs: {attention_pool_gpus}',
                          graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'})
    
    # Set node attributes
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Embedding\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Base Layer (GPUs 0-7) - FFN operations
    with dot.subgraph(name='cluster_base') as c:
        c.attr(label='Base Layer - GPUs [0-7]\\nFFN Operations Only', style='dashed', color='red')
        
        # All FFN operations run on base layer GPUs
        # Layer 0 FFN
        c.node('layer0_ffn_start', 'Layer 0 FFN Start\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 1 FFN  
        c.node('layer1_ffn_start', 'Layer 1 FFN Start\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 2 FFN
        c.node('layer2_ffn_start', 'Layer 2 FFN Start\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Layer 3 FFN
        c.node('layer3_ffn_start', 'Layer 3 FFN Start\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]')
        
        # Final output
        c.node('output_proj', 'Output Projection\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=?, partition=?]')
    
    # Attention Pool for parallel attention computation
    with dot.subgraph(name='cluster_attention_pool') as c:
        c.attr(label=f'Attention Pool - GPUs [8-{7+attention_pool_gpus}]\\nParallel Attention ({attention_pool_gpus} GPUs)', 
               style='dashed', color='purple')
        
        # Communication nodes for data transfer
        dot.node('send_to_pool', 'Send to Attention Pool\\nGPU: [0-7] → [8-{7+attention_pool_gpus}]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node('recv_from_pool', 'Receive from Attention Pool\\nGPU: [8-{7+attention_pool_gpus}] → [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Block division for parallel attention
        block_size = math.ceil(sequence_length / attention_pool_gpus)
        
        # Layer 0 Attention in pool
        for i in range(attention_pool_gpus):
            gpu_id = 8 + i
            start_pos = i * block_size
            end_pos = min((i+1) * block_size, sequence_length)
            
            with dot.subgraph(name=f'cluster_layer0_attn_{i}') as lc:
                lc.attr(label=f'Layer 0 Attention Block {i}\\nGPU: {gpu_id}\\nSeq: {start_pos}-{end_pos}')
                
                lc.node(f'layer0_attn_qkv_{i}', f'Layer 0 QKV Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128, partition=512]')
                
                lc.node(f'layer0_attn_scores_{i}', f'Layer 0 Attention Scores Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]')
                
                lc.node(f'layer0_attn_weighted_{i}', f'Layer 0 Weighted Sum Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]')
                
                lc.node(f'layer0_attn_out_{i}', f'Layer 0 Attention Output Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, hidden_dim=4096, partition=512]')
        
        # Aggregation node for Layer 0
        dot.node('layer0_attn_aggregate', 'Layer 0 Attention Aggregation\\nGPU: All Pool GPUs\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512] from each block\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                 shape='parallelogram', fillcolor='orange')
        
        # Repeat for Layer 1, 2, 3
        for layer in [1, 2, 3]:
            # Send to pool for this layer
            dot.node(f'send_to_pool_{layer}', f'Send Layer {layer} to Pool\\nGPU: [0-7] → [8-{7+attention_pool_gpus}]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                     shape='parallelogram', fillcolor='yellow')
            
            # Parallel attention blocks for this layer
            for i in range(attention_pool_gpus):
                gpu_id = 8 + i
                start_pos = i * block_size
                end_pos = min((i+1) * block_size, sequence_length)
                
                with dot.subgraph(name=f'cluster_layer{layer}_attn_{i}') as lc:
                    lc.attr(label=f'Layer {layer} Attention Block {i}\\nGPU: {gpu_id}\\nSeq: {start_pos}-{end_pos}')
                    
                    lc.node(f'layer{layer}_attn_qkv_{i}', f'Layer {layer} QKV Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128, partition=512]')
                    
                    lc.node(f'layer{layer}_attn_scores_{i}', f'Layer {layer} Attention Scores Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]')
                    
                    lc.node(f'layer{layer}_attn_weighted_{i}', f'Layer {layer} Weighted Sum Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128], [batch_size=1024, seq_len=?, seq_len=?]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]')
                    
                    lc.node(f'layer{layer}_attn_out_{i}', f'Layer {layer} Attention Output Block {i}\\nGPU: {gpu_id}\\nInput: [batch_size=1024, seq_len={start_pos}-{end_pos}, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len={start_pos}-{end_pos}, hidden_dim=4096, partition=512]')
            
            # Aggregate for this layer
            dot.node(f'layer{layer}_attn_aggregate', f'Layer {layer} Attention Aggregation\\nGPU: All Pool GPUs\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512] from each block\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                     shape='parallelogram', fillcolor='orange')
            
            # Receive back to base layer
            dot.node(f'recv_from_pool_{layer}', f'Receive Layer {layer} from Pool\\nGPU: [8-{7+attention_pool_gpus}] → [0-7]\\nInput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]\\nOutput: [batch_size=1024, seq_len=?, hidden_dim=4096, partition=512]', 
                     shape='parallelogram', fillcolor='yellow')
    
    # Output node
    dot.node('output', 'Final Output\\nGPU: [0-7]\\nInput: [batch_size=1024, seq_len=?, vocab_size=?, partition=?]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=?]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections for FA Pool
    # Layer 0 flow
    dot.edge('input', 'send_to_pool')
    
    # Connect parallel attention blocks for layer 0
    for i in range(attention_pool_gpus):
        dot.edge('send_to_pool', f'layer0_attn_qkv_{i}')
        dot.edge(f'layer0_attn_qkv_{i}', f'layer0_attn_scores_{i}')
        dot.edge(f'layer0_attn_scores_{i}', f'layer0_attn_weighted_{i}')
        dot.edge(f'layer0_attn_weighted_{i}', f'layer0_attn_out_{i}')
        dot.edge(f'layer0_attn_out_{i}', 'layer0_attn_aggregate')
    
    dot.edge('layer0_attn_aggregate', 'recv_from_pool')
    dot.edge('recv_from_pool', 'layer0_ffn_start')
    
    # Layer 0 FFN operations
    dot.edge('layer0_ffn_start', 'layer1_ffn_start')
    
    # Layer 1 flow
    dot.edge('layer1_ffn_start', 'send_to_pool_1')
    
    for i in range(attention_pool_gpus):
        dot.edge('send_to_pool_1', f'layer1_attn_qkv_{i}')
        dot.edge(f'layer1_attn_qkv_{i}', f'layer1_attn_scores_{i}')
        dot.edge(f'layer1_attn_scores_{i}', f'layer1_attn_weighted_{i}')
        dot.edge(f'layer1_attn_weighted_{i}', f'layer1_attn_out_{i}')
        dot.edge(f'layer1_attn_out_{i}', 'layer1_attn_aggregate')
    
    dot.edge('layer1_attn_aggregate', 'recv_from_pool_1')
    dot.edge('recv_from_pool_1', 'layer2_ffn_start')
    
    # Layer 2 flow
    dot.edge('layer2_ffn_start', 'send_to_pool_2')
    
    for i in range(attention_pool_gpus):
        dot.edge('send_to_pool_2', f'layer2_attn_qkv_{i}')
        dot.edge(f'layer2_attn_qkv_{i}', f'layer2_attn_scores_{i}')
        dot.edge(f'layer2_attn_scores_{i}', f'layer2_attn_weighted_{i}')
        dot.edge(f'layer2_attn_weighted_{i}', f'layer2_attn_out_{i}')
        dot.edge(f'layer2_attn_out_{i}', 'layer2_attn_aggregate')
    
    dot.edge('layer2_attn_aggregate', 'recv_from_pool_2')
    dot.edge('recv_from_pool_2', 'layer3_ffn_start')
    
    # Layer 3 flow
    dot.edge('layer3_ffn_start', 'send_to_pool_3')
    
    for i in range(attention_pool_gpus):
        dot.edge('send_to_pool_3', f'layer3_attn_qkv_{i}')
        dot.edge(f'layer3_attn_qkv_{i}', f'layer3_attn_scores_{i}')
        dot.edge(f'layer3_attn_scores_{i}', f'layer3_attn_weighted_{i}')
        dot.edge(f'layer3_attn_weighted_{i}', f'layer3_attn_out_{i}')
        dot.edge(f'layer3_attn_out_{i}', 'layer3_attn_aggregate')
    
    dot.edge('layer3_attn_aggregate', 'recv_from_pool_3')
    dot.edge('recv_from_pool_3', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

def create_fa_pool_variants():
    """Create DAGs for different sequence lengths"""
    dag_configs = [
        {'seq_len': 4097, 'name': '4097_tokens'},
        {'seq_len': 8192, 'name': '8192_tokens'},
        {'seq_len': 16384, 'name': '16384_tokens'},
        {'seq_len': 32768, 'name': '32768_tokens'}
    ]
    
    for config in dag_configs:
        dag = create_fa_pool_dag(config['seq_len'])
        
        # Save DOT file
        dot_path = f'./generated_docs/FA/fa_pool_{config["name"]}.dot'
        svg_path = f'./generated_docs/FA/fa_pool_{config["name"]}.svg'
        
        with open(dot_path, 'w') as f:
            f.write(dag.source)
        
        # Render to SVG
        dag.format = 'svg'
        dag.render(f'./generated_docs/FA/fa_pool_{config["name"]}', cleanup=True)
        
        print(f"FA Pool DAG ({config['seq_len']} tokens) generated: {svg_path}")
        print(f"DOT file saved: {dot_path}")

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('./generated_docs/FA', exist_ok=True)
    
    # Generate all FA Pool variants
    create_fa_pool_variants()
    
    # Also generate baseline
    baseline_dag = create_baseline_dag()
    
    # Save DOT file
    with open('./generated_docs/FA/baseline_transformer.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    # Render to SVG
    baseline_dag.format = 'svg'
    baseline_dag.render('./generated_docs/FA/baseline_transformer', cleanup=True)
    
    print("All DAGs generated successfully!")