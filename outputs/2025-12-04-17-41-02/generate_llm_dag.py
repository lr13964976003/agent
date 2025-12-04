#!/usr/bin/env python3

import graphviz

def create_llm_deployment_dag():
    """Create a complete DAG for LLM deployment with EP64_TP2_PP1 strategy"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP64_TP2_PP1 Strategy')
    dot.attr(dpi='300', rankdir='TB', size='40,60')
    dot.attr('node', fontname='Arial', fontsize='9')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', fillcolor='lightblue', shape='rectangle', style='filled')  # Computation
    dot.attr('node', fillcolor='lightgreen', shape='ellipse', style='filled')   # Communication
    dot.attr('node', fillcolor='yellow', shape='parallelogram', style='filled') # Routing/Aggregation
    
    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(bgcolor='lightgray', label='Input Layer', style='rounded')
        c.node('input', 
               label='Input Tokens\\nGPU: All 128 GPUs\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]',
               fillcolor='lightcoral', shape='rectangle')
    
    # Process each layer
    for layer in range(1, 17):
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(bgcolor='lightblue', label=f'Layer {layer} - Attention + MoE', style='rounded')
            
            # Attention components
            attn_norm = f'attn_norm_{layer}'
            c.node(attn_norm, 
                   label=f'Layer Norm (Attention)\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            
            # Q, K V projections
            q_proj = f'attn_q_{layer}'
            k_proj = f'attn_k_{layer}'
            v_proj = f'attn_v_{layer}'
            
            c.node(q_proj, 
                   label=f'Q Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            c.node(k_proj, 
                   label=f'K Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            c.node(v_proj, 
                   label=f'V Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            
            # Attention computation
            attn_score = f'attn_score_{layer}'
            attn_out = f'attn_out_{layer}'
            
            c.node(attn_score, 
                   label=f'Attention Scores\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            c.node(attn_out, 
                   label=f'Attention Output\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
            
            # MoE Gate
            moe_gate = f'moe_gate_{layer}'
            c.node(moe_gate, 
                   label=f'MoE Gate\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 64]',
                   fillcolor='yellow', shape='parallelogram')
            
            # Create all 64 experts with proper GPU assignments
            expert_outputs = []
            
            for expert_id in range(64):
                # Each expert gets 2 GPUs (TP2)
                gpu_start = expert_id * 2
                gpu_end = gpu_start + 1
                
                # Expert split
                tp_split = f'tp_split_{expert_id}_{layer}'
                c.node(tp_split, 
                       label=f'TP Split Expert {expert_id}\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 16]\\nOutput: [128, 1024, 8]',
                       fillcolor='lightgreen', shape='ellipse')
                
                # Expert compute parts (TP2)
                expert_compute_0 = f'expert_compute_{expert_id}_0_{layer}'
                expert_compute_1 = f'expert_compute_{expert_id}_1_{layer}'
                
                c.node(expert_compute_0, 
                       label=f'Expert {expert_id} Compute Part 0\\nGPU: {gpu_start}\\nInput: [128, 1024, 8]\\nOutput: [128, 1024, 1024]',
                       fillcolor='lightblue', shape='rectangle')
                c.node(expert_compute_1, 
                       label=f'Expert {expert_id} Compute Part 1\\nGPU: {gpu_end}\\nInput: [128, 1024, 8]\\nOutput: [128, 1024, 1024]',
                       fillcolor='lightblue', shape='rectangle')
                
                # TP All-reduce
                tp_allreduce = f'tp_allreduce_{expert_id}_{layer}'
                c.node(tp_allreduce, 
                       label=f'TP All-Reduce Expert {expert_id}\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 2048]',
                       fillcolor='lightgreen', shape='ellipse')
                
                # Expert output
                expert_out = f'expert_{expert_id}_{layer}'
                c.node(expert_out, 
                       label=f'Expert {expert_id} Output\\nGPU: {gpu_start}-{gpu_end}\\nInput: [128, 1024, 2048]\\nOutput: [128, 1024, 2048]',
                       fillcolor='lightblue', shape='rectangle')
                
                expert_outputs.append(expert_out)
            
            # Expert aggregation - collects from all 64 experts
            expert_agg = f'expert_agg_{layer}'
            agg_label = f'Expert Aggregation\\nGPU: All 128 GPUs\\nInput: [128, 1024, 2048] x 64\\nOutput: [128, 1024, 1024]'
            c.node(expert_agg, label=agg_label, fillcolor='yellow', shape='parallelogram')
            
            # Layer normalization
            layer_norm = f'layer_norm_{layer}'
            c.node(layer_norm, 
                   label=f'Layer Norm {layer}\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
                   fillcolor='lightblue', shape='rectangle')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(bgcolor='lightgray', label='Output Layer', style='rounded')
        c.node('output_norm', 
               label='Final Layer Norm\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]',
               fillcolor='lightblue', shape='rectangle')
        c.node('output_proj', 
               label='Output Projection\\nGPU: All 128 GPUs\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, vocab_size]',
               fillcolor='lightblue', shape='rectangle')
        c.node('output', 
               label='Output Tokens\\nGPU: All 128 GPUs\\nInput: [128, 1024, vocab_size]\\nOutput: [128, 1024]',
               fillcolor='lightcoral', shape='rectangle')
    
    # Create edges for layer 1 (detailed)
    dot.edge('input', 'attn_norm_1', label='Token embeddings')
    dot.edge('attn_norm_1', 'attn_q_1')
    dot.edge('attn_norm_1', 'attn_k_1')
    dot.edge('attn_norm_1', 'attn_v_1')
    dot.edge('attn_q_1', 'attn_score_1', label='Q matrix')
    dot.edge('attn_k_1', 'attn_score_1', label='K matrix')
    dot.edge('attn_v_1', 'attn_out_1', label='V matrix')
    dot.edge('attn_score_1', 'attn_out_1', label='Attention weights')
    dot.edge('attn_out_1', 'moe_gate_1', label='Attention output')
    
    # Connect all 64 experts in layer 1
    for expert_id in range(64):
        gpu_start = expert_id * 2
        # Gate selection (dashed line)
        dot.edge('moe_gate_1', f'tp_split_{expert_id}_1', 
                label=f'Gate selection {expert_id}', style='dashed')
        
        # Expert computation flow
        dot.edge(f'tp_split_{expert_id}_1', f'expert_compute_{expert_id}_0_1')
        dot.edge(f'tp_split_{expert_id}_1', f'expert_compute_{expert_id}_1_1')
        dot.edge(f'expert_compute_{expert_id}_0_1', f'tp_allreduce_{expert_id}_1')
        dot.edge(f'expert_compute_{expert_id}_1_1', f'tp_allreduce_{expert_id}_1')
        dot.edge(f'tp_allreduce_{expert_id}_1', f'expert_{expert_id}_1')
        dot.edge(f'expert_{expert_id}_1', 'expert_agg_1', label=f'Expert {expert_id} output')
    
    dot.edge('expert_agg_1', 'layer_norm_1')
    
    # Connect remaining layers (2-16) with full detail
    for layer in range(2, 17):
        prev_layer = layer - 1
        
        # Attention components
        dot.edge(f'layer_norm_{prev_layer}', f'attn_norm_{layer}')
        dot.edge(f'attn_norm_{layer}', f'attn_q_{layer}')
        dot.edge(f'attn_norm_{layer}', f'attn_k_{layer}')
        dot.edge(f'attn_norm_{layer}', f'attn_v_{layer}')
        dot.edge(f'attn_q_{layer}', f'attn_score_{layer}')
        dot.edge(f'attn_k_{layer}', f'attn_score_{layer}')
        dot.edge(f'attn_v_{layer}', f'attn_out_{layer}')
        dot.edge(f'attn_score_{layer}', f'attn_out_{layer}')
        dot.edge(f'attn_out_{layer}', f'moe_gate_{layer}')
        
        # Connect all 64 experts for this layer
        for expert_id in range(64):
            # Gate selection (dashed line)
            dot.edge(f'moe_gate_{layer}', f'tp_split_{expert_id}_{layer}', 
                    label=f'Gate selection {expert_id}', style='dashed')
            
            # Expert computation flow
            dot.edge(f'tp_split_{expert_id}_{layer}', f'expert_compute_{expert_id}_0_{layer}')
            dot.edge(f'tp_split_{expert_id}_{layer}', f'expert_compute_{expert_id}_1_{layer}')
            dot.edge(f'expert_compute_{expert_id}_0_{layer}', f'tp_allreduce_{expert_id}_{layer}')
            dot.edge(f'expert_compute_{expert_id}_1_{layer}', f'tp_allreduce_{expert_id}_{layer}')
            dot.edge(f'tp_allreduce_{expert_id}_{layer}', f'expert_{expert_id}_{layer}')
            dot.edge(f'expert_{expert_id}_{layer}', f'expert_agg_{layer}', label=f'Expert {expert_id} output')
        
        dot.edge(f'expert_agg_{layer}', f'layer_norm_{layer}')
    
    # Final output connections
    dot.edge('layer_norm_16', 'output_norm')
    dot.edge('output_norm', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

if __name__ == "__main__":
    # Generate the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file_path = "../outputs/2025-12-04-17-41-02/llm_deployment_complete.dot"
    dag.save(dot_file_path)
    
    # Save as SVG image
    svg_file_path = "../outputs/2025-12-04-17-41-02/llm_deployment_complete.svg"
    dag.render(svg_file_path, format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_file_path}")
    print(f"SVG saved to: {svg_file_path}")