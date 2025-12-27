#!/usr/bin/env python3

import graphviz

def create_moe_dag():
    """
    Create a detailed DAG for the MoE model with hybrid parallelism strategy.
    Shows complete flow with EP (16 experts) + TP (degree 2) + PP (8 stages).
    """
    
    # Create directed graph
    dot = graphviz.Digraph(comment='MoE Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='30,30', dpi='300')
    dot.attr('node', fontsize='9', margin='0.03')
    
    # Define input specifications
    input_spec = "[batch_size=128, seq_len=10240, heads=16, d_k=32]"
    hidden_spec = "[batch_size=128, seq_len=10240, hidden=1024]"
    expert_spec = "[batch_size=128, seq_len=640, heads=16, d_k=32]"
    
    # Create a simplified view focusing on key components
    # Input node
    dot.node('input', f'Input\\nInput: {input_spec}\\nOutput: {input_spec}',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Process each pipeline stage
    for stage in range(8):
        layer_start = stage * 2
        
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(label=f'Pipeline Stage {stage} (Layers {layer_start}-{layer_start+1})',
                   style='rounded,filled', fillcolor='lightgray')
            
            # Show layer processing
            for layer in [layer_start, layer_start+1]:
                # MHA computation
                c.node(f'mha_{layer}', f'MHA L{layer}\\nGPU: 0-511\\nInput: {input_spec}\\nOutput: {input_spec}',
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Gate routing
                c.node(f'gate_{layer}', f'Gate L{layer}\\nGPU: 0-511\\nInput: {input_spec}\\nOutput: routing_decisions',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # Show expert processing (simplified view)
                for expert_id in range(16):
                    expert_gpu_base = (layer * 16 + expert_id) * 2
                    tp_gpu_0 = expert_gpu_base % 512
                    tp_gpu_1 = (expert_gpu_base + 1) % 512
                    
                    # Expert routing (dashed)
                    c.node(f'route_{layer}_{expert_id}',
                           f'Route to Expert {expert_id}\\nGPU: {tp_gpu_0},{tp_gpu_1}\\nInput: {expert_spec}\\nOutput: {expert_spec}',
                           shape='ellipse', style='dashed,filled', fillcolor='lightblue')
                    
                    # Expert computation TP-0
                    c.node(f'expert_{layer}_{expert_id}_0',
                           f'Expert {expert_id} TP-0\\nGPU: {tp_gpu_0}\\nInput: {expert_spec}\\nOutput: expert_out[hidden=512]',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    # Expert computation TP-1
                    c.node(f'expert_{layer}_{expert_id}_1',
                           f'Expert {expert_id} TP-1\\nGPU: {tp_gpu_1}\\nInput: {expert_spec}\\nOutput: expert_out[hidden=512]',
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    # TP communication
                    c.node(f'tp_{layer}_{expert_id}',
                           f'TP All-Reduce\\nGPU: {tp_gpu_0}↔{tp_gpu_1}\\nInput: expert_out[hidden=512]\\nOutput: expert_full[hidden=1024]',
                           shape='ellipse', style='filled', fillcolor='lightblue')
                
                # Expert aggregation
                c.node(f'agg_{layer}', f'Aggregate Experts L{layer}\\nGPU: 0-511\\nInput: 16 expert outputs\\nOutput: {hidden_spec}',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # Layer output
                c.node(f'output_{layer}', f'Layer {layer} Output\\nGPU: 0-511\\nInput: {hidden_spec}\\nOutput: {input_spec}',
                       shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'mha_0')
    
    for layer in range(16):
        # MHA to gate
        dot.edge(f'mha_{layer}', f'gate_{layer}')
        
        # Gate to expert routing (dashed)
        for expert_id in range(16):
            dot.edge(f'gate_{layer}', f'route_{layer}_{expert_id}', style='dashed')
            dot.edge(f'route_{layer}_{expert_id}', f'expert_{layer}_{expert_id}_0')
            dot.edge(f'route_{layer}_{expert_id}', f'expert_{layer}_{expert_id}_1')
            
            # Expert to TP communication
            dot.edge(f'expert_{layer}_{expert_id}_0', f'tp_{layer}_{expert_id}')
            dot.edge(f'expert_{layer}_{expert_id}_1', f'tp_{layer}_{expert_id}')
            
            # TP to aggregation
            dot.edge(f'tp_{layer}_{expert_id}', f'agg_{layer}')
        
        # Aggregation to output
        dot.edge(f'agg_{layer}', f'output_{layer}')
        
        # Connect to next layer
        if layer < 15:
            # Communication between layers
            dot.edge(f'output_{layer}', f'mha_{layer+1}')
    
    # Final output
    dot.node('final_output', f'Final Output\\nInput: {input_spec}\\nOutput: {input_spec}',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    dot.edge('output_15', 'final_output')
    
    return dot

if __name__ == '__main__':
    dag = create_moe_dag()
    
    # Save DOT file
    dag.save('./outputs/2025-12-26-18-39-29/moe_parallel_dag.dot')
    
    # Render to SVG
    dag.render('./outputs/2025-12-26-18-39-29/moe_parallel_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files saved:")
    print("- moe_parallel_dag.dot")
    print("- moe_parallel_dag.svg")