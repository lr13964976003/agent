import graphviz
import os

def create_fa_pool_dag():
    # Create main DAG
    dot = graphviz.Digraph('fa_pool_model_dag', 
                           comment='FA Pool Dynamic Parallel Strategy DAG',
                           format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='40,50', splines='ortho')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Communication
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightgray')  # Aggregation/Reduction
    dot.attr('node', shape='hexagon', style='filled', fillcolor='lightcoral')  # Dynamic allocation
    
    # Input node
    dot.node('input', '''<b>Model Input</b><br/>Input: [batch_size=1024, seq_len=?, vocab_size=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    # Resource manager for dynamic allocation
    dot.node('resource_manager', 
             '''<b>Resource Manager</b><br/>Monitors sequence length<br/>Allocates attention pool GPUs<br/>Threshold: 4096 tokens''',
             shape='hexagon', fillcolor='lightcoral')
    
    # Base layer - 8 GPUs with full model components
    with dot.subgraph(name='cluster_base_layer') as c:
        c.attr(label='Base Layer (8 GPUs - Static)', fontsize='14', style='rounded', color='blue')
        
        # Embedding layer
        with c.subgraph(name='cluster_base_embedding') as emb:
            emb.attr(label='Token Embedding', fontsize='12', style='rounded')
            for i in range(8):
                emb.node(f'base_embed_{i}', 
                        f'<b>Token Embedding</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, vocab=32000]<br/>Output: [batch=1024, seq=?, hidden=512]',
                        shape='rectangle', fillcolor='lightgreen')
        
        # Positional encoding
        with c.subgraph(name='cluster_base_pos') as pos:
            pos.attr(label='Positional Encoding', fontsize='12', style='rounded')
            for i in range(8):
                pos.node(f'base_pos_{i}', 
                        f'<b>Positional Encoding</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                        shape='rectangle', fillcolor='lightgreen')
    
    # Dynamic attention pool - up to 32 GPUs
    with dot.subgraph(name='cluster_attention_pool') as c:
        c.attr(label='Dynamic Attention Pool (0-32 GPUs)', fontsize='14', style='rounded', color='red')
        
        # Sequence length routing
        c.node('seq_router', 
               '''<b>Sequence Router</b><br/>Routes attention based on sequence length<br/>Short: base GPUs<br/>Long: attention pool''',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Create attention pool GPU nodes for different sequence length thresholds
        thresholds = [
            ("<4096", 0, "Use base GPUs"),
            ("4096-8192", 8, "8 pool GPUs"),
            ("8192-16384", 16, "16 pool GPUs"),
            (">16384", 24, "24 pool GPUs")
        ]
        
        for threshold, num_gpus, label in thresholds:
            with c.subgraph(name=f'cluster_pool_{num_gpus}') as pool:
                pool.attr(label=f'{label} for {threshold} tokens', fontsize='10', style='dashed')
                
                for layer in range(4):
                    with pool.subgraph(name=f'cluster_layer{layer}_{num_gpus}') as layer_cluster:
                        layer_cluster.attr(label=f'Layer {layer} Attention', fontsize='8')
                        
                        # Create attention computation nodes for each GPU
                        for gpu_idx in range(num_gpus if num_gpus > 0 else 8):
                            gpu_name = f"pool_gpu_{gpu_idx}" if num_gpus > 0 else f"base_gpu_{gpu_idx}"
                            
                            # Block partitioning for long sequences
                            layer_cluster.node(f'attn_block_{layer}_{gpu_idx}_{num_gpus}', 
                                             f'<b>Attention Block</b><br/>GPU: {gpu_name}<br/>Block: {gpu_idx+1}/{num_gpus if num_gpus > 0 else 8}<br/>Input: [batch=1024, block_seq=ceil(seq/{num_gpus}), hidden=4096]<br/>Output: [batch=1024, block_seq=ceil(seq/{num_gpus}), hidden=4096]',
                                             shape='rectangle', fillcolor='lightgreen')
                            
                            # KV cache sharing
                            layer_cluster.node(f'kv_cache_{layer}_{gpu_idx}_{num_gpus}', 
                                             f'<b>KV Cache Share</b><br/>GPU: {gpu_name}<br/>Replicates K,V across all GPUs<br/>[batch=1024, seq=?, heads=32, d_k=128]',
                                             shape='parallelogram', fillcolor='lightyellow', style='dashed')
    
    # Base layer FFN components (always on base GPUs)
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}_ffn') as c:
            c.attr(label=f'Layer {layer} FFN (Base GPUs)', fontsize='12', style='rounded')
            
            for i in range(8):
                # FFN up projection
                c.node(f'l{layer}_ffn_up_{i}', 
                       f'<b>FFN Up Projection</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, ffn_dim=2048]',
                       shape='rectangle', fillcolor='lightgreen')
                
                # FFN down projection
                c.node(f'l{layer}_ffn_down_{i}', 
                       f'<b>FFN Down Projection</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, ffn_dim=2048]<br/>Output: [batch=1024, seq=?, hidden=512]',
                       shape='rectangle', fillcolor='lightgreen')
                
                # FFN residual and layer norm
                c.node(f'l{layer}_ffn_res_{i}', 
                       f'<b>FFN Residual Add</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512] × 2<br/>Output: [batch=1024, seq=?, hidden=512]',
                       shape='diamond', fillcolor='lightgray')
                
                c.node(f'l{layer}_ffn_ln_{i}', 
                       f'<b>FFN Layer Norm</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, hidden=512]',
                       shape='rectangle', fillcolor='lightgreen')
    
    # Communication patterns
    # 1. Hierarchical reduction for attention results
    for layer in range(4):
        for threshold, num_gpus, _ in thresholds:
            if num_gpus > 0:
                dot.node(f'hier_reduce_{layer}_{num_gpus}', 
                         f'<b>Hierarchical Reduction</b><br/>Aggregates attention results<br/>From {num_gpus} GPUs to 8 base GPUs<br/>[batch=1024, seq=?, hidden=4096]',
                         shape='diamond', fillcolor='lightgray')
    
    # 2. Cross-GPU KV cache synchronization
    for layer in range(4):
        dot.node(f'kv_sync_{layer}', 
                 f'<b>KV Cache Sync</b><br/>Synchronizes K,V tensors<br/>Across attention pool GPUs<br/>[batch=1024, seq=?, heads=32, d_k=128]',
                 shape='parallelogram', fillcolor='lightyellow')
    
    # 3. Asynchronous communication for overlap
    dot.node('async_comm', 
             '<b>Asynchronous Communication</b><br/>Overlaps attention computation<br/>With FFN operations<br/>85% overlap efficiency',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer (Base GPUs)', fontsize='12', style='rounded')
        
        for i in range(8):
            c.node(f'final_output_{i}', 
                   f'<b>Linear Output</b><br/>GPU: base_gpu_{i}<br/>Input: [batch=1024, seq=?, hidden=512]<br/>Output: [batch=1024, seq=?, vocab=4000]',
                   shape='rectangle', fillcolor='lightgreen')
    
    # Final aggregation and output
    dot.node('final_output', 
             '''<b>Final Output</b><br/>Aggregated across 8 base GPUs<br/>Input: [batch=1024, seq=?, vocab=32000]<br/>Output: [batch=1024, seq=?, vocab=32000]''',
             shape='ellipse', fillcolor='lightblue')
    
    # Tensor parallelism all-reduce operations
    for op in ['embed', 'pos', 'ffn_up', 'ffn_down', 'final_output']:
        dot.node(f'allreduce_{op}', 
                 f'<b>All-Reduce</b><br/>TP across 8 base GPUs<br/>[batch=1024, seq=?, dim=4096]',
                 shape='parallelogram', fillcolor='lightyellow')
    
    # Create edges for the FA Pool DAG
    # Input to resource manager
    dot.edge('input', 'resource_manager')
    
    # Input to sequence router
    dot.edge('resource_manager', 'seq_router')
    
    # Embedding connections
    for i in range(8):
        dot.edge('input', f'base_embed_{i}')
        dot.edge(f'base_embed_{i}', f'base_pos_{i}')
        dot.edge(f'base_pos_{i}', f'allreduce_embed')
    
    # Layer-wise connections with dynamic routing
    for layer in range(4):
        # Route to appropriate attention computation
        dot.edge('seq_router', f'kv_sync_{layer}')
        
        # For each threshold, create connections
        for threshold, num_gpus, _ in thresholds:
            if num_gpus == 0:  # Use base GPUs for short sequences
                # Direct attention on base GPUs
                for i in range(8):
                    dot.edge(f'base_pos_{i}', f'l{layer}_ffn_ln_{i}')
                    dot.edge(f'l{layer}_ffn_ln_{i}', f'l{layer}_ffn_up_{i}')
                    dot.edge(f'l{layer}_ffn_up_{i}', f'l{layer}_ffn_down_{i}')
                    dot.edge(f'l{layer}_ffn_down_{i}', f'l{layer}_ffn_res_{i}')
                    dot.edge(f'l{layer}_ffn_ln_{i}', f'l{layer}_ffn_res_{i}')
                    dot.edge(f'l{layer}_ffn_res_{i}', f'l{layer}_ffn_ln_{i}')
            else:  # Use attention pool
                # Block-wise attention computation
                for gpu_idx in range(num_gpus):
                    dot.edge(f'kv_sync_{layer}', f'kv_cache_{layer}_{gpu_idx}_{num_gpus}')
                    dot.edge(f'kv_cache_{layer}_{gpu_idx}_{num_gpus}', f'attn_block_{layer}_{gpu_idx}_{num_gpus}')
                    
                    # Hierarchical reduction
                    dot.edge(f'attn_block_{layer}_{gpu_idx}_{num_gpus}', f'hier_reduce_{layer}_{num_gpus}')
                
                # After reduction, proceed to FFN
                for i in range(8):
                    dot.edge(f'hier_reduce_{layer}_{num_gpus}', f'l{layer}_ffn_ln_{i}')
                    dot.edge(f'l{layer}_ffn_ln_{i}', f'l{layer}_ffn_up_{i}')
                    dot.edge(f'l{layer}_ffn_up_{i}', f'l{layer}_ffn_down_{i}')
                    dot.edge(f'l{layer}_ffn_down_{i}', f'l{layer}_ffn_res_{i}')
                    dot.edge(f'l{layer}_ffn_ln_{i}', f'l{layer}_ffn_res_{i}')
                    dot.edge(f'l{layer}_ffn_res_{i}', f'l{layer}_ffn_ln_{i}')
    
    # Async communication
    dot.edge('async_comm', 'hier_reduce_0_24')
    dot.edge('async_comm', 'hier_reduce_1_24')
    dot.edge('async_comm', 'hier_reduce_2_24')
    dot.edge('async_comm', 'hier_reduce_3_24')
    
    # Output layer connections
    for i in range(8):
        dot.edge(f'l3_ffn_ln_{i}', f'final_output_{i}')
        dot.edge(f'final_output_{i}', f'allreduce_final_output')
        dot.edge(f'allreduce_final_output', 'final_output')
    
    # Save the DAG
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_model_dag', cleanup=False)
    
    # Also save as dot file
    dot.format = 'dot'
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_model_dag', cleanup=False)

if __name__ == '__main__':
    create_fa_pool_dag()