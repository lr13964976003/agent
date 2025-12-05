#!/usr/bin/env python3

def generate_complete_moe_dag():
    """Generate a complete DAG for the 30B MoE model deployment with ALL layers properly connected"""
    
    # Helper function to create attention layer nodes
    def create_attention_layer(layer_num, gpu_ids, gpu_colors):
        nodes = []
        for i, (gpu_id, color) in enumerate(zip(gpu_ids, gpu_colors)):
            # QKV Projection
            qkv_node = f'layer{layer_num}_attn_qkv_gpu{gpu_id}'
            nodes.append(f'{qkv_node} [fillcolor={color}, label="Layer{layer_num} Attention QKV Proj\\n(Column Parallel)\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 256]\\nGPU: {gpu_id}"];')
            
            # Attention Scores
            score_node = f'layer{layer_num}_attn_score_gpu{gpu_id}'
            nodes.append(f'{score_node} [fillcolor={color}, label="Layer{layer_num} Attention Scores\\nInput: [64, 4, 1024, 1024]\\nOutput: [64, 4, 1024, 1024]\\nGPU: {gpu_id}"];')
            
            # Attention Output
            out_node = f'layer{layer_num}_attn_out_gpu{gpu_id}'
            nodes.append(f'{out_node} [fillcolor={color}, label="Layer{layer_num} Attention Output\\n(Row Parallel)\\nInput: [64, 1024, 256]\\nOutput: [64, 1024, 256]\\nGPU: {gpu_id}"];')
            
            # Connections
            nodes.append(f'{qkv_node} -> {score_node};')
            nodes.append(f'{score_node} -> {out_node};')
        
        # All-Reduce
        allreduce_node = f'layer{layer_num}_attn_allreduce'
        nodes.append(f'{allreduce_node} [shape=ellipse, fillcolor=lightgray, label="Layer{layer_num} Attention\\nAll-Reduce Sum\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: {",".join(map(str, gpu_ids))}"];')
        
        for gpu_id in gpu_ids:
            nodes.append(f'layer{layer_num}_attn_out_gpu{gpu_id} -> {allreduce_node};')
        
        return nodes, allreduce_node
    
    # Helper function to create MoE layer nodes
    def create_moe_layer(layer_num, gpu_ids, gpu_colors, prev_allreduce_node):
        nodes = []
        
        # MoE Routing
        route_node = f'layer{layer_num}_moe_route'
        nodes.append(f'{route_node} [shape=parallelogram, fillcolor={gpu_colors[0]}, label="Layer{layer_num} MoE Routing\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1]\\nGPU: {",".join(map(str, gpu_ids))}"];')
        nodes.append(f'{prev_allreduce_node} -> {route_node};')
        
        # Route to individual GPUs
        for gpu_id in gpu_ids:
            route_gpu_node = f'layer{layer_num}_moe_route_gpu{gpu_id}'
            nodes.append(f'{route_gpu_node} [shape=parallelogram, fillcolor={gpu_colors[0]}, label="Layer{layer_num} MoE Route\\nGPU: {gpu_id}"];')
            nodes.append(f'{route_node} -> {route_gpu_node};')
        
        # All-to-All Communication
        all2all_node = f'layer{layer_num}_moe_all2all'
        nodes.append(f'{all2all_node} [shape=ellipse, fillcolor=lightgray, label="Layer{layer_num} MoE\\nAll-to-All Communication\\nGPU: 0-15"];')
        
        for gpu_id in gpu_ids:
            nodes.append(f'layer{layer_num}_moe_route_gpu{gpu_id} -> {all2all_node};')
        
        # Expert computations - 16 experts across all GPUs
        expert_nodes = []
        all_colors = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 
                     'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
                     'lightyellow', 'lightyellow', 'lightyellow', 'lightyellow', 
                     'lightcoral', 'lightcoral', 'lightcoral', 'lightcoral']
        
        for expert_id in range(16):
            gpu_id = expert_id
            expert_node = f'layer{layer_num}_expert{expert_id}'
            color = all_colors[expert_id]
            nodes.append(f'{expert_node} [fillcolor={color}, label="Layer{layer_num} Expert {expert_id//4}_{expert_id%4}\\nInput: [~70, 1024, 1024]\\nOutput: [~70, 1024, 2048]\\nGPU: {gpu_id}"];')
            nodes.append(f'{all2all_node} -> {expert_node};')
            expert_nodes.append(expert_node)
        
        # Aggregation node
        agg_node = f'layer{layer_num}_moe_agg'
        nodes.append(f'{agg_node} [shape=parallelogram, fillcolor={gpu_colors[0]}, label="Layer{layer_num} MoE\\nOutput Aggregation\\nInput: [64, 1024, 1024]\\nOutput: [64, 1024, 1024]\\nGPU: {",".join(map(str, gpu_ids))}"];')
        
        # FIX: Connect all experts to aggregation
        for expert_node in expert_nodes:
            nodes.append(f'{expert_node} -> {agg_node};')
        
        return nodes, agg_node
    
    all_nodes = []
    prev_agg_node = None
    
    # Stage 0: Layers 0-3 on GPUs 0-3 (blue)
    for layer_num in range(4):
        gpu_ids = [0, 1, 2, 3]
        gpu_colors = ['lightblue', 'lightblue', 'lightblue', 'lightblue']
        
        # Attention layer
        attn_nodes, attn_allreduce = create_attention_layer(layer_num, gpu_ids, gpu_colors)
        all_nodes.extend(attn_nodes)
        
        if prev_agg_node:
            # Connect previous layer to current layer
            for gpu_id in gpu_ids:
                all_nodes.append(f'{prev_agg_node} -> layer{layer_num}_attn_qkv_gpu{gpu_id};')
        else:
            # Connect input to first layer
            for gpu_id in gpu_ids:
                all_nodes.append(f'dp_split -> layer{layer_num}_attn_qkv_gpu{gpu_id};')
        
        # MoE layer
        moe_nodes, moe_agg = create_moe_layer(layer_num, gpu_ids, gpu_colors, attn_allreduce)
        all_nodes.extend(moe_nodes)
        prev_agg_node = moe_agg
    
    # Stage 1: Layers 4-7 on GPUs 4-7 (green)
    for layer_num in range(4, 8):
        gpu_ids = [4, 5, 6, 7]
        gpu_colors = ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']
        
        # Attention layer
        attn_nodes, attn_allreduce = create_attention_layer(layer_num, gpu_ids, gpu_colors)
        all_nodes.extend(attn_nodes)
        
        # Connect previous layer to current layer (pipeline stage transition)
        for gpu_id in gpu_ids:
            all_nodes.append(f'{prev_agg_node} -> layer{layer_num}_attn_qkv_gpu{gpu_id};')
        
        # MoE layer
        moe_nodes, moe_agg = create_moe_layer(layer_num, gpu_ids, gpu_colors, attn_allreduce)
        all_nodes.extend(moe_nodes)
        prev_agg_node = moe_agg
    
    # Stage 2: Layers 8-11 on GPUs 8-11 (yellow)
    for layer_num in range(8, 12):
        gpu_ids = [8, 9, 10, 11]
        gpu_colors = ['lightyellow', 'lightyellow', 'lightyellow', 'lightyellow']
        
        # Attention layer
        attn_nodes, attn_allreduce = create_attention_layer(layer_num, gpu_ids, gpu_colors)
        all_nodes.extend(attn_nodes)
        
        # Connect previous layer to current layer
        for gpu_id in gpu_ids:
            all_nodes.append(f'{prev_agg_node} -> layer{layer_num}_attn_qkv_gpu{gpu_id};')
        
        # MoE layer
        moe_nodes, moe_agg = create_moe_layer(layer_num, gpu_ids, gpu_colors, attn_allreduce)
        all_nodes.extend(moe_nodes)
        prev_agg_node = moe_agg
    
    # Stage 3: Layers 12-15 on GPUs 12-15 (coral)
    for layer_num in range(12, 16):
        gpu_ids = [12, 13, 14, 15]
        gpu_colors = ['lightcoral', 'lightcoral', 'lightcoral', 'lightcoral']
        
        # Attention layer
        attn_nodes, attn_allreduce = create_attention_layer(layer_num, gpu_ids, gpu_colors)
        all_nodes.extend(attn_nodes)
        
        # Connect previous layer to current layer
        for gpu_id in gpu_ids:
            all_nodes.append(f'{prev_agg_node} -> layer{layer_num}_attn_qkv_gpu{gpu_id};')
        
        # MoE layer
        moe_nodes, moe_agg = create_moe_layer(layer_num, gpu_ids, gpu_colors, attn_allreduce)
        all_nodes.extend(moe_nodes)
        prev_agg_node = moe_agg
    
    # Final output aggregation and output
    output_agg = 'output_agg'
    all_nodes.append(f'{output_agg} [shape=parallelogram, fillcolor=lightpink, label="Output Aggregation\\nAll-Reduce Sum\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nGPU: 12,13,14,15"];')
    
    output = 'output'
    all_nodes.append(f'{output} [shape=ellipse, fillcolor=white, label="Final Output\\n[batch_size=128, seq_len=1024, hidden=1024]\\nGPU: ALL"];')
    
    all_nodes.append(f'{prev_agg_node} -> {output_agg};')
    all_nodes.append(f'{output_agg} -> {output};')
    
    # Build final DOT content
    dot_content = '''digraph MoE_Deployment_Complete {
    graph [bgcolor=white, rankdir=TB, splines=ortho, ranksep=1.5, nodesep=0.8];
    node [shape=rectangle, style=filled, fillcolor=lightblue, fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    # Input node - ellipse
    input [shape=ellipse, fillcolor=white, label="Input\\n[batch_size=128, seq_len=1024, hidden=1024]\\nGPU: ALL"];
    
    # Data parallel split - parallelogram  
    dp_split [shape=parallelogram, fillcolor=lightpink, label="Data Parallel Split\\n[batch_size=64, seq_len=1024, hidden=1024]\\nGPU: Routing"];
    input -> dp_split;
    '''
    
    # Add all the generated nodes
    for node in all_nodes:
        dot_content += f'    {node}\\n'
    
    dot_content += '}'
    
    return dot_content

def main():
    # Generate the complete DAG
    dag_content = generate_complete_moe_dag()
    
    # Save to file
    output_path = "../outputs/2025-12-05-10-26-38/moe_deployment_complete.dot"
    with open(output_path, 'w') as f:
        f.write(dag_content)
    
    print(f"Generated complete DAG: {output_path}")
    
    # Also generate SVG using Graphviz
    try:
        import subprocess
        svg_path = output_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', output_path, '-o', svg_path], check=True)
        print(f"Generated SVG: {svg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Graphviz not available, skipping SVG generation")
    
    return output_path

if __name__ == "__main__":
    main()