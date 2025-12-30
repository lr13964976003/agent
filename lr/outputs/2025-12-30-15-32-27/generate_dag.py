#!/usr/bin/env python3

import graphviz

def generate_parallel_strategy_dag():
    """Generate a complete DAG for the parallel strategy deployment"""
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='filled', fillcolor='lightgray', label='Input Layer')
        c.node('input', 'Input\\nInput: [batch_size=128, seq_len=?, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=?, heads=16, d_k=32]', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Process each pipeline stage (16 stages total)
    for stage in range(16):
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(style='filled', fillcolor='lightcyan', label=f'Pipeline Stage {stage} (Layer {stage})')
            
            # Multi-Head Attention for this stage
            c.node(f'mha_{stage}', f'MHA Layer {stage}\\nGPU: All GPUs\\nInput: [batch_size=128, seq_len=?, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=?, heads=16, d_k=512]', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Gate/Router for expert selection
            c.node(f'gate_{stage}', f'Gate Layer {stage}\\nGPU: All GPUs\\nInput: [batch_size=128, seq_len=?, heads=16, d_k=512]\\nOutput: [batch_size=128, seq_len=?, top_k=2, experts=16]', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Expert nodes for this stage (16 experts per layer)
            for expert in range(16):
                gpu_id = stage * 16 + expert
                c.node(f'expert_{stage}_{expert}', f'Expert {expert} Layer {stage}\\nGPU: {gpu_id}\\nInput: [batch_size=?, seq_len=?, heads=16, d_k=512]\\nOutput: [batch_size=?, seq_len=?, heads=16, d_k=512]', 
                       shape='rectangle', fillcolor='lightgreen')
            
            # Communication nodes for expert routing
            for expert in range(16):
                c.node(f'comm_{stage}_{expert}', f'Route to Expert {expert}\\nGPU: {stage*16+expert}\\nInput: [batch_size=?, seq_len=?, top_k=2]\\nOutput: [batch_size=?, seq_len=?, tokens_selected=?]', 
                       shape='ellipse', fillcolor='lightblue')
            
            # Aggregation node
            c.node(f'agg_{stage}', f'Aggregate Experts Layer {stage}\\nGPU: All GPUs\\nInput: [batch_size=128, seq_len=?, experts=16, d_k=512]\\nOutput: [batch_size=128, seq_len=?, heads=16, d_k=512]', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    with dot.subgraph(name='cluster_output') as c:
        c.attr(style='filled', fillcolor='lightgray', label='Output Layer')
        c.node('output', 'Output\\nInput: [batch_size=128, seq_len=?, heads=16, d_k=512]\\nOutput: [batch_size=128, seq_len=?, vocab_size=?]', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to first stage
    dot.edge('input', 'mha_0')
    
    # Connect within each stage
    for stage in range(16):
        # MHA -> Gate
        dot.edge(f'mha_{stage}', f'gate_{stage}')
        
        # Gate -> Communication (dashed for routing decisions)
        for expert in range(16):
            dot.edge(f'gate_{stage}', f'comm_{stage}_{expert}', style='dashed', color='red')
        
        # Communication -> Expert computation
        for expert in range(16):
            dot.edge(f'comm_{stage}_{expert}', f'expert_{stage}_{expert}')
        
        # Expert computation -> Aggregation
        for expert in range(16):
            dot.edge(f'expert_{stage}_{expert}', f'agg_{stage}')
    
    # Connect between stages (pipeline communication)
    for stage in range(15):
        dot.edge(f'agg_{stage}', f'mha_{stage+1}', color='blue', penwidth='2')
    
    # Last stage to output
    dot.edge('agg_15', 'output')
    
    return dot

def generate_simplified_dag():
    """Generate a more readable simplified DAG"""
    
    dot = graphviz.Digraph(comment='Parallel Strategy Deployment DAG - Simplified')
    dot.attr(rankdir='TB', size='50,50', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='8')
    
    # Input
    dot.node('input', 'Input\\n[128, ?, 16, 32]', shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Process each layer
    for layer in range(16):
        # MHA
        dot.node(f'mha_{layer}', f'MHA L{layer}\\n[128, ?, 16, 512]', shape='rectangle', fillcolor='lightgreen', style='filled')
        
        # Gate
        dot.node(f'gate_{layer}', f'Gate L{layer}\\n[128, ?, 16, 512]', shape='parallelogram', fillcolor='lightyellow', style='filled')
        
        # Expert computation (grouped)
        dot.node(f'experts_{layer}', f'Experts L{layer}\\nGPUs {layer*16}-{layer*16+15}\\n[?, ?, 16, 512]', shape='rectangle', fillcolor='lightgreen', style='filled')
        
        # Communication
        dot.node(f'comm_{layer}', f'Expert Routing L{layer}\\n[?, ?, 2, 16]', shape='ellipse', fillcolor='lightblue', style='filled')
        
        # Aggregation
        dot.node(f'agg_{layer}', f'Aggregate L{layer}\\n[128, ?, 16, 512]', shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Output
    dot.node('output', 'Output\\n[128, ?, vocab]', shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Connections
    dot.edge('input', 'mha_0')
    
    for layer in range(16):
        dot.edge(f'mha_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'comm_{layer}', style='dashed', color='red')
        dot.edge(f'comm_{layer}', f'experts_{layer}')
        dot.edge(f'experts_{layer}', f'agg_{layer}')
        
        if layer < 15:
            dot.edge(f'agg_{layer}', f'mha_{layer+1}', color='blue', penwidth='2')
    
    dot.edge('agg_15', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate both versions
    print("Generating complete DAG...")
    complete_dag = generate_parallel_strategy_dag()
    complete_dag.render('./outputs/2025-12-30-15-32-27/parallel_strategy_complete', format='dot', cleanup=False)
    complete_dag.render('./outputs/2025-12-30-15-32-27/parallel_strategy_complete', format='svg', cleanup=False)
    
    print("Generating simplified DAG...")
    simplified_dag = generate_simplified_dag()
    simplified_dag.render('./outputs/2025-12-30-15-32-27/parallel_strategy_simplified', format='dot', cleanup=False)
    simplified_dag.render('./outputs/2025-12-30-15-32-27/parallel_strategy_simplified', format='svg', cleanup=False)
    
    print("DAG generation complete!")
    
    # Also save the DOT code as text files
    with open('./outputs/2025-12-30-15-32-27/parallel_strategy_complete.dot', 'w') as f:
        f.write(complete_dag.source)
    
    with open('./outputs/2025-12-30-15-32-27/parallel_strategy_simplified.dot', 'w') as f:
        f.write(simplified_dag.source)
    
    print("DOT files saved!")