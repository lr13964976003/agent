import graphviz
import os

def create_detailed_fa_pool_dag():
    # Create detailed FA Pool DAG focusing on attention distribution
    dot = graphviz.Digraph('fa_pool_detailed_dag', 
                           comment='FA Pool Detailed Dynamic Attention Distribution DAG',
                           format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='LR', size='50,30', splines='ortho')
    dot.attr('node', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Communication
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightgray')  # Aggregation
    dot.attr('node', shape='hexagon', style='filled', fillcolor='lightcoral')  # Dynamic allocation
    
    # Input and sequence analysis
    dot.node('input', '''<b>Model Input</b><br/>[batch=1024, seq_len=?, vocab=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    dot.node('seq_analyzer', 
             '''<b>Sequence Length Analyzer</b><br/>Analyzes seq_len<br/>Determines GPU allocation<br/>
             < 4096: Base only<br/>
             4096-8192: +8 pool GPUs<br/>
             8192-16384: +16 pool GPUs<br/>
             >16384: +24 pool GPUs''',
             shape='hexagon', fillcolor='lightcoral')
    
    # Base layer components (always active)
    with dot.subgraph(name='cluster_base_components') as base:
        base.attr(label='Base Layer (8 GPUs - Always Active)', fontsize='12', style='rounded')
        
        # Embedding
        for i in range(8):
            base.node(f'base_embed_{i}', 
                     f'<b>Embedding</b><br/>base_gpu_{i}<br/>[batch=1024, seq=?, 512]',
                     shape='rectangle', fillcolor='lightgreen')
        
        # FFN components for all layers
        for layer in range(4):
            with base.subgraph(name=f'cluster_base_ffn_l{layer}') as ffn:
                ffn.attr(label=f'Layer {layer} FFN', fontsize='10', style='dashed')
                for i in range(8):
                    ffn.node(f'base_ffn_up_l{layer}_{i}', 
                            f'<b>FFN Up</b><br/>base_gpu_{i}<br/>[512→2048]',
                            shape='rectangle', fillcolor='lightgreen')
                    ffn.node(f'base_ffn_down_l{layer}_{i}', 
                            f'<b>FFN Down</b><br/>base_gpu_{i}<br/>[2048→512]',
                            shape='rectangle', fillcolor='lightgreen')
    
    # Dynamic attention computation scenarios
    scenarios = [
        ("short", "< 4096", 0, 8, "lightblue"),
        ("medium", "4096-8192", 8, 8, "lightgreen"),
        ("long", "8192-16384", 16, 16, "lightyellow"),
        ("very_long", "> 16384", 24, 24, "lightcoral")
    ]
    
    for scenario_name, seq_range, num_pool_gpus, block_size, color in scenarios:
        with dot.subgraph(name=f'cluster_{scenario_name}_scenario') as scenario:
            scenario.attr(label=f'{seq_range} tokens ({"Base" if num_pool_gpus == 0 else str(num_pool_gpus) + " pool GPUs"})', 
                         fontsize='11', style='rounded', color=color)
            
            for layer in range(4):
                with scenario.subgraph(name=f'cluster_layer{layer}_{scenario_name}') as layer_cluster:
                    layer_cluster.attr(label=f'Layer {layer}', fontsize='10')
                    
                    # Q/K/V projections
                    total_gpus = 8 if num_pool_gpus == 0 else num_pool_gpus
                    for gpu_idx in range(total_gpus):
                        gpu_name = f"base_gpu_{gpu_idx}" if num_pool_gpus == 0 else f"pool_gpu_{gpu_idx}"
                        
                        # Q projection
                        layer_cluster.node(f'q_proj_{layer}_{scenario_name}_{gpu_idx}', 
                                         f'<b>Q Proj</b><br/>{gpu_name}<br/>[512→(32×16)]',
                                         shape='rectangle', fillcolor=color)
                        
                        # K projection
                        layer_cluster.node(f'k_proj_{layer}_{scenario_name}_{gpu_idx}', 
                                         f'<b>K Proj</b><br/>{gpu_name}<br/>[512→(32×16)]',
                                         shape='rectangle', fillcolor=color)
                        
                        # V projection
                        layer_cluster.node(f'v_proj_{layer}_{scenario_name}_{gpu_idx}', 
                                         f'<b>V Proj</b><br/>{gpu_name}<br/>[512→(32×16)]',
                                         shape='rectangle', fillcolor=color)
                        
                        # Block-wise attention
                        block_seq = f"ceil(seq/{total_gpus})"
                        layer_cluster.node(f'flash_attn_{layer}_{scenario_name}_{gpu_idx}', 
                                         f'<b>Flash Attention</b><br/>{gpu_name}<br/>Block: {gpu_idx+1}/{total_gpus}<br/>[batch=1024, seq={block_seq}, 4096]',
                                         shape='rectangle', fillcolor=color)
                    
                    # Reduction and communication
                    if num_pool_gpus > 0:
                        layer_cluster.node(f'reduce_{layer}_{scenario_name}', 
                                         f'<b>Hierarchical Reduce</b><br/>{num_pool_gpus}→8 GPUs<br/>Concatenate blocks',
                                         shape='diamond', fillcolor='lightgray')
    
    # Communication patterns
    dot.node('kv_broadcast', 
             '<b>KV Cache Broadcast</b><br/>Replicates K,V tensors<br/>Across all attention GPUs',
             shape='parallelogram', fillcolor='lightyellow', style='dashed')
    
    dot.node('async_overlap', 
             '<b>Async Overlap</b><br/>Attention computation<br/>Overlaps with FFN<br/>85% efficiency',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Output processing
    dot.node('output_agg', 
             '<b>Output Aggregation</b><br/>Final result from base GPUs<br/>[batch=1024, seq=?, vocab=32000]',
             shape='ellipse', fillcolor='lightblue')
    
    # Create connections
    dot.edge('input', 'seq_analyzer')
    
    # Connect sequence analyzer to all scenarios
    for scenario_name, seq_range, num_pool_gpus, _, _ in scenarios:
        dot.edge('seq_analyzer', f'q_proj_0_{scenario_name}_0', label=seq_range)
    
    # Layer connections with dynamic routing
    for layer in range(4):
        # For each scenario
        for scenario_name, _, num_pool_gpus, _, _ in scenarios:
            total_gpus = 8 if num_pool_gpus == 0 else num_pool_gpus
            
            # QKV projections to attention
            for gpu_idx in range(total_gpus):
                dot.edge(f'q_proj_{layer}_{scenario_name}_{gpu_idx}', 
                        f'flash_attn_{layer}_{scenario_name}_{gpu_idx}')
                dot.edge(f'k_proj_{layer}_{scenario_name}_{gpu_idx}', 
                        f'flash_attn_{layer}_{scenario_name}_{gpu_idx}')
                dot.edge(f'v_proj_{layer}_{scenario_name}_{gpu_idx}', 
                        f'flash_attn_{layer}_{scenario_name}_{gpu_idx}')
                dot.edge('kv_broadcast', f'flash_attn_{layer}_{scenario_name}_{gpu_idx}')
            
            # Attention to reduction (if using pool)
            if num_pool_gpus > 0:
                for gpu_idx in range(total_gpus):
                    dot.edge(f'flash_attn_{layer}_{scenario_name}_{gpu_idx}', 
                            f'reduce_{layer}_{scenario_name}')
                
                # Reduction to FFN
                for i in range(8):
                    dot.edge(f'reduce_{layer}_{scenario_name}', f'base_ffn_up_l{layer}_{i}')
            else:
                # Direct to FFN for base-only
                for i in range(8):
                    dot.edge(f'flash_attn_{layer}_{scenario_name}_{i}', f'base_ffn_up_l{layer}_{i}')
    
    # FFN connections
    for layer in range(4):
        for i in range(8):
            dot.edge(f'base_ffn_up_l{layer}_{i}', f'base_ffn_down_l{layer}_{i}')
            dot.edge('async_overlap', f'base_ffn_up_l{layer}_{i}')
    
    # Final output
    for i in range(8):
        dot.edge(f'base_ffn_down_l3_{i}', 'output_agg')
    
    # Save the detailed DAG
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_detailed_dag', cleanup=False)
    
    # Also save as dot file
    dot.format = 'dot'
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_detailed_dag', cleanup=False)

def create_intermediate_dag():
    # Create a more practical DAG showing the flow
    dot = graphviz.Digraph('fa_pool_practical_dag', 
                           comment='FA Pool Practical Implementation DAG',
                           format='svg')
    
    dot.attr(rankdir='TB', size='25,35', splines='ortho')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Input
    dot.node('input', '''Input<br/>[batch=1024, seq_len=?, vocab=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    # Sequence threshold decision
    dot.node('decision', 
             '''Sequence Length Decision<br/>
             < 4096: Base only<br/>
             ≥ 4096: Activate pool''',
             shape='diamond', fillcolor='lightcoral')
    
    # Base layer processing
    dot.node('base_processing', 
             '''Base Layer (8 GPUs)<br/>
             Embedding + FFN<br/>
             Always active''',
             shape='rectangle', fillcolor='lightgreen')
    
    # Attention computation paths
    dot.node('base_attention', 
             '''Base Attention<br/>
             8 GPUs<br/>
             TP=8 across base''',
             shape='rectangle', fillcolor='lightgreen')
    
    # Dynamic attention pool
    dot.node('pool_attention', 
             '''Attention Pool<br/>
             8-32 GPUs<br/>
             Block-wise parallel<br/>
             Sequence length / GPUs blocks''',
             shape='rectangle', fillcolor='lightyellow')
    
    # Communication and reduction
    dot.node('communication', 
             '''Communication<br/>
             KV cache sharing<br/>
             Hierarchical reduction<br/>
             Async overlap''',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Output
    dot.node('output', '''Output<br/>[batch=1024, seq_len=?, vocab=32000]''', 
             shape='ellipse', fillcolor='lightblue')
    
    # Create edges
    dot.edge('input', 'decision')
    dot.edge('decision', 'base_processing')
    dot.edge('decision', 'base_attention', label='< 4096')
    dot.edge('decision', 'pool_attention', label='≥ 4096')
    dot.edge('base_processing', 'communication')
    dot.edge('base_attention', 'communication')
    dot.edge('pool_attention', 'communication')
    dot.edge('communication', 'output')
    
    # Save intermediate DAG
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_practical_dag', cleanup=False)
    dot.format = 'dot'
    dot.render('../outputs/2025-10-17-08-58-21/fa_pool_practical_dag', cleanup=False)

if __name__ == '__main__':
    create_detailed_fa_pool_dag()
    create_intermediate_dag()