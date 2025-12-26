#!/usr/bin/env python3

import graphviz

def create_detailed_parallel_dag():
    """Create a detailed DAG showing attention submodules and parallel strategy"""
    
    dot = graphviz.Digraph(comment='Detailed Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing
    
    # Input node
    dot.node('input', 'Input Embedding\nGPU: 0-3\nInput: [batch=128, seq=10240, hidden=512]\nOutput: [batch=128, seq=10240, hidden=128]', 
             shape='rectangle', fillcolor='lightcoral')
    
    # Pipeline Stage 0 (Layers 0-7)
    for layer in range(8):
        # GPU assignments for this layer
        tp_gpus = [f"{i}" for i in range(4)]  # TP group 0-3
        ep_gpus = [f"{i}" for i in range(8)]  # EP group 0-7
        
        # Layer input from previous layer or input
        if layer == 0:
            prev_node = 'input'
        else:
            prev_node = f'layer_{layer-1}_output'
        
        # Attention Layer - broken down into submodules
        # 1. Q/K/V Projections (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_qkv_gpu{gpu}', 
                    f'Q/K/V Projection\nGPU: {gpu}\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, qkv=128]', 
                    shape='rectangle', fillcolor='lightblue')
            
            if layer == 0:
                dot.edge('input', f'layer_{layer}_qkv_gpu{gpu}')
            else:
                # TP all-gather from previous layer
                dot.node(f'layer_{layer}_tp_gather', 
                        f'TP All-Gather\nGPU: 0-3\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=512]', 
                        shape='ellipse', fillcolor='lightgreen')
                dot.edge(prev_node, f'layer_{layer}_tp_gather')
                dot.edge(f'layer_{layer}_tp_gather', f'layer_{layer}_qkv_gpu{gpu}')
        
        # 2. Attention Computation (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_attn_gpu{gpu}', 
                    f'Attention Computation\nGPU: {gpu}\nInput: [batch=128, seq=10240, qkv=128]\nOutput: [batch=128, seq=10240, attn=128]', 
                    shape='rectangle', fillcolor='lightblue')
            dot.edge(f'layer_{layer}_qkv_gpu{gpu}', f'layer_{layer}_attn_gpu{gpu}')
        
        # 3. Attention Output Projection (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_attn_out_gpu{gpu}', 
                    f'Attention Output Projection\nGPU: {gpu}\nInput: [batch=128, seq=10240, attn=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                    shape='rectangle', fillcolor='lightblue')
            dot.edge(f'layer_{layer}_attn_gpu{gpu}', f'layer_{layer}_attn_out_gpu{gpu}')
        
        # 4. TP All-Reduce for attention output
        dot.node(f'layer_{layer}_attn_allreduce', 
                f'Attention TP All-Reduce\nGPU: 0-3\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                shape='ellipse', fillcolor='lightgreen')
        
        for gpu in tp_gpus:
            dot.edge(f'layer_{layer}_attn_out_gpu{gpu}', f'layer_{layer}_attn_allreduce')
        
        # 5. Expert Routing (Gate)
        dot.node(f'layer_{layer}_gate', 
                f'Expert Gate Routing\nGPU: 0-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, expert_ids]', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Connect gate to all TP GPUs for input
        for gpu in tp_gpus:
            dot.edge(f'layer_{layer}_attn_allreduce', f'layer_{layer}_gate', style='dashed')
        
        # 6. Expert Parallelism - 16 experts across 8 GPUs (2 per GPU)
        for expert_gpu in range(8):
            for expert in range(2):
                expert_id = expert_gpu * 2 + expert
                dot.node(f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}', 
                        f'Expert {expert_id}\nGPU: {expert_gpu}\nInput: [batch=16, seq=10240, hidden=128]\nOutput: [batch=16, seq=10240, hidden=128]', 
                        shape='rectangle', fillcolor='lightblue')
                
                # All-to-all communication for expert routing
                dot.node(f'layer_{layer}_all2all_expert_{expert_id}', 
                        f'All-to-All Expert {expert_id}\nGPU: 0-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=16, seq=10240, hidden=128]', 
                        shape='ellipse', fillcolor='lightgreen', style='dashed')
                
                dot.edge(f'layer_{layer}_gate', f'layer_{layer}_all2all_expert_{expert_id}', style='dashed')
                dot.edge(f'layer_{layer}_all2all_expert_{expert_id}', f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}')
        
        # 7. Expert output all-to-all
        for expert_gpu in range(8):
            for expert in range(2):
                expert_id = expert_gpu * 2 + expert
                dot.node(f'layer_{layer}_all2all_back_{expert_id}', 
                        f'All-to-All Back {expert_id}\nGPU: 0-7\nInput: [batch=16, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                        shape='ellipse', fillcolor='lightgreen', style='dashed')
                
                dot.edge(f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}', f'layer_{layer}_all2all_back_{expert_id}')
        
        # 8. Final layer output (combine expert outputs)
        dot.node(f'layer_{layer}_output', 
                f'Layer {layer} Output\nGPU: 0-3\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                shape='rectangle', fillcolor='lightblue')
        
        # Connect all expert outputs to final layer output
        for expert_gpu in range(8):
            for expert in range(2):
                expert_id = expert_gpu * 2 + expert
                dot.edge(f'layer_{layer}_all2all_back_{expert_id}', f'layer_{layer}_output')
    
    # Pipeline communication between stages
    dot.node('pipeline_stage0_to_stage1', 
            f'Pipeline Stage 0→1\nGPU: 0-3 → 4-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
            shape='ellipse', fillcolor='lightgreen')
    
    dot.edge('layer_7_output', 'pipeline_stage0_to_stage1')
    
    # Pipeline Stage 1 (Layers 8-15) - Similar structure but different GPU assignments
    for layer in range(8, 16):
        actual_layer = layer - 8
        # GPU assignments for pipeline stage 1
        tp_gpus = [f"{i+4}" for i in range(4)]  # TP group 4-7
        ep_gpus = [f"{i+8}" for i in range(8)]  # EP group 8-15
        
        # Layer input
        if layer == 8:
            prev_node = 'pipeline_stage0_to_stage1'
        else:
            prev_node = f'layer_{layer-1}_output'
        
        # Attention Layer - broken down into submodules
        # 1. Q/K/V Projections (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_qkv_gpu{gpu}', 
                    f'Q/K/V Projection\nGPU: {gpu}\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, qkv=128]', 
                    shape='rectangle', fillcolor='lightblue')
            
            if layer == 8:
                dot.edge('pipeline_stage0_to_stage1', f'layer_{layer}_qkv_gpu{gpu}')
            else:
                # TP all-gather from previous layer
                dot.node(f'layer_{layer}_tp_gather', 
                        f'TP All-Gather\nGPU: 4-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=512]', 
                        shape='ellipse', fillcolor='lightgreen')
                dot.edge(prev_node, f'layer_{layer}_tp_gather')
                dot.edge(f'layer_{layer}_tp_gather', f'layer_{layer}_qkv_gpu{gpu}')
        
        # 2. Attention Computation (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_attn_gpu{gpu}', 
                    f'Attention Computation\nGPU: {gpu}\nInput: [batch=128, seq=10240, qkv=128]\nOutput: [batch=128, seq=10240, attn=128]', 
                    shape='rectangle', fillcolor='lightblue')
            dot.edge(f'layer_{layer}_qkv_gpu{gpu}', f'layer_{layer}_attn_gpu{gpu}')
        
        # 3. Attention Output Projection (TP across 4 GPUs)
        for i, gpu in enumerate(tp_gpus):
            dot.node(f'layer_{layer}_attn_out_gpu{gpu}', 
                    f'Attention Output Projection\nGPU: {gpu}\nInput: [batch=128, seq=10240, attn=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                    shape='rectangle', fillcolor='lightblue')
            dot.edge(f'layer_{layer}_attn_gpu{gpu}', f'layer_{layer}_attn_out_gpu{gpu}')
        
        # 4. TP All-Reduce for attention output
        dot.node(f'layer_{layer}_attn_allreduce', 
                f'Attention TP All-Reduce\nGPU: 4-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                shape='ellipse', fillcolor='lightgreen')
        
        for gpu in tp_gpus:
            dot.edge(f'layer_{layer}_attn_out_gpu{gpu}', f'layer_{layer}_attn_allreduce')
        
        # 5. Expert Routing (Gate)
        dot.node(f'layer_{layer}_gate', 
                f'Expert Gate Routing\nGPU: 8-15\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, expert_ids]', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Connect gate to all TP GPUs for input
        for gpu in tp_gpus:
            dot.edge(f'layer_{layer}_attn_allreduce', f'layer_{layer}_gate', style='dashed')
        
        # 6. Expert Parallelism - 16 experts across 8 GPUs (2 per GPU)
        for expert_gpu in range(8, 16):
            for expert in range(2):
                expert_id = (expert_gpu - 8) * 2 + expert
                dot.node(f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}', 
                        f'Expert {expert_id}\nGPU: {expert_gpu}\nInput: [batch=16, seq=10240, hidden=128]\nOutput: [batch=16, seq=10240, hidden=128]', 
                        shape='rectangle', fillcolor='lightblue')
                
                # All-to-all communication for expert routing
                dot.node(f'layer_{layer}_all2all_expert_{expert_id}', 
                        f'All-to-All Expert {expert_id}\nGPU: 8-15\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=16, seq=10240, hidden=128]', 
                        shape='ellipse', fillcolor='lightgreen', style='dashed')
                
                dot.edge(f'layer_{layer}_gate', f'layer_{layer}_all2all_expert_{expert_id}', style='dashed')
                dot.edge(f'layer_{layer}_all2all_expert_{expert_id}', f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}')
        
        # 7. Expert output all-to-all
        for expert_gpu in range(8, 16):
            for expert in range(2):
                expert_id = (expert_gpu - 8) * 2 + expert
                dot.node(f'layer_{layer}_all2all_back_{expert_id}', 
                        f'All-to-All Back {expert_id}\nGPU: 8-15\nInput: [batch=16, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                        shape='ellipse', fillcolor='lightgreen', style='dashed')
                
                dot.edge(f'layer_{layer}_expert_{expert_id}_gpu{expert_gpu}', f'layer_{layer}_all2all_back_{expert_id}')
        
        # 8. Final layer output (combine expert outputs)
        dot.node(f'layer_{layer}_output', 
                f'Layer {layer} Output\nGPU: 4-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, hidden=128]', 
                shape='rectangle', fillcolor='lightblue')
        
        # Connect all expert outputs to final layer output
        for expert_gpu in range(8, 16):
            for expert in range(2):
                expert_id = (expert_gpu - 8) * 2 + expert
                dot.edge(f'layer_{layer}_all2all_back_{expert_id}', f'layer_{layer}_output')
    
    # Final output
    dot.node('output', 'Final Output\nGPU: 4-7\nInput: [batch=128, seq=10240, hidden=128]\nOutput: [batch=128, seq=10240, vocab=50000]', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.edge('layer_15_output', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_detailed_parallel_dag()
    
    # Save DOT file
    dot_file_path = "./outputs/2025-12-26-17-44-50/parallel_strategy_detailed_dag.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG image
    svg_file_path = "./outputs/2025-12-26-17-44-50/parallel_strategy_detailed_dag.svg"
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Detailed DAG generated and saved to:")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG image: {svg_file_path}")
    
    # Also print the DOT content for verification
    print("\nGenerated DOT content:")
    print(dag.source)