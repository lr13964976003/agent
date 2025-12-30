#!/usr/bin/env python3
"""
Generate DAG for 10B MoE model with EP=16, PP=4, TP=2, DP=2
Total: 128 GPUs
"""

import graphviz

def create_moe_dag():
    # Create directed graph
    dot = graphviz.Digraph(comment='10B MoE Model Parallel Strategy DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rect', style='filled', fillcolor='lightgreen')    # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Global attributes for all nodes
    dot.attr('node', fontsize='10', height='0.8', width='2.0')
    
    # Create DP level (2 replicas)
    with dot.subgraph(name='cluster_dp') as dp:
        dp.attr(label='DP=2 (Request Parallelism)', style='rounded,filled', fillcolor='lightgray', fontname='bold')
        
        # Replica 1
        with dp.subgraph(name='cluster_replica1') as replica1:
            replica1.attr(label='Pipeline Replica 1 (64 GPUs)', style='rounded,filled', fillcolor='lightblue')
            create_pipeline(dot, replica1, "R1", 0)        
        # Replica 2
        with dp.subgraph(name='cluster_replica2') as replica2:
            replica2.attr(label='Pipeline Replica 2 (64 GPUs)', style='rounded,filled', fillcolor='lightblue')
            create_pipeline(dot, replica2, "R2", 64)
    
    return dot

def create_pipeline(dot, parent, prefix, gpu_offset):
    """Create 4-stage pipeline with PP=4"""
    
    # Input node
    input_node = f"{prefix}_Input"
    parent.node(input_node, 
                label=f"Input\\nInput: [batch_size=128, seq_len=128-10240]\\nOutput: [batch_size=128, seq_len=128-10240]",
                shape='ellipse', fillcolor='lightblue')
    
    prev_node = input_node
    
    # 4 Pipeline stages
    for stage_idx in range(4):
        stage_name = f"Stage{stage_idx+1}"
        start_layer = stage_idx * 4 + 1
        end_layer = (stage_idx + 1) * 4
        
        with parent.subgraph(name=f'cluster_{prefix}_{stage_name}') as stage:
            stage.attr(label=f'{stage_name}: Layers {start_layer}-{end_layer} (16 GPUs)', 
                      style='rounded,filled', fillcolor='lightgreen')
            
            # Create stage input
            stage_input = f"{prefix}_{stage_name}_Input"
            stage.node(stage_input,
                      label=f"{stage_name} Input\\nInput: [batch_size=128, seq_len=128-10240]\\nOutput: [batch_size=128, seq_len=128-10240]",
                      shape='ellipse', fillcolor='lightblue')
            
            # Connect from previous stage
            if stage_idx > 0:
                dot.edge(prev_node, stage_input, label="PP Communication")
            else:
                dot.edge(input_node, stage_input)
            
            # Create 4 layers in this stage
            current_node = stage_input
            for layer_idx in range(4):
                layer_num = start_layer + layer_idx
                layer_node = create_layer(stage, prefix, stage_name, layer_num, gpu_offset + stage_idx * 16)
                dot.edge(current_node, layer_node)
                current_node = layer_node
            
            # Stage output
            stage_output = f"{prefix}_{stage_name}_Output"
            stage.node(stage_output,
                      label=f"{stage_name} Output\\nInput: [batch_size=128, seq_len=128-10240]\\nOutput: [batch_size=128, seq_len=128-10240]",
                      shape='ellipse', fillcolor='lightblue')
            dot.edge(current_node, stage_output)
            
            prev_node = stage_output
    
    # Final output
    output_node = f"{prefix}_Output"
    parent.node(output_node,
               label=f"Output\\nInput: [batch_size=128, seq_len=128-10240]\\nOutput: [batch_size=128, seq_len=128-10240]",
               shape='ellipse', fillcolor='lightblue')
    dot.edge(prev_node, output_node, label="Final Output")

def create_layer(parent, prefix, stage_name, layer_num, gpu_offset):
    """Create a single layer with attention and MoE components"""
    
    layer_id = f"{prefix}_{stage_name}_L{layer_num}"
    
    # Layer input
    layer_input = f"{layer_id}_Input"
    parent.node(layer_input,
               label=f"Layer {layer_num} Input\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='ellipse', fillcolor='lightblue')
    
    # Attention with TP=2
    attention_input = f"{layer_id}_Attn_Input"
    parent.node(attention_input,
               label=f"Attention Input\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='ellipse', fillcolor='lightblue')
    parent.edge(layer_input, attention_input)
    
    # TP Attention shards
    attn_shard1 = f"{layer_id}_Attn_TP1"
    attn_shard2 = f"{layer_id}_Attn_TP2"
    
    parent.node(attn_shard1,
               label=f"Attention TP GPU {gpu_offset}\\n8 heads\\nInput: [batch_size=128, seq_len=128-10240, heads=8, d_k=32]\\nOutput: [batch_size=128, seq_len=128-10240, heads=8, d_k=32]",
               shape='rect', fillcolor='lightgreen')
    
    parent.node(attn_shard2,
               label=f"Attention TP GPU {gpu_offset+1}\\n8 heads\\nInput: [batch_size=128, seq_len=128-10240, heads=8, d_k=32]\\nOutput: [batch_size=128, seq_len=128-10240, heads=8, d_k=32]",
               shape='rect', fillcolor='lightgreen')
    
    parent.edge(attention_input, attn_shard1)
    parent.edge(attention_input, attn_shard2)
    
    # Attention AllReduce
    attn_output = f"{layer_id}_Attn_Output"
    parent.node(attn_output,
               label=f"Attention Output\\nAllReduce GPUs {gpu_offset}-{gpu_offset+1}\\nInput: [batch_size=128, seq_len=128-10240, heads=16, d_k=32]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='ellipse', fillcolor='lightblue')
    
    parent.edge(attn_shard1, attn_output)
    parent.edge(attn_shard2, attn_output)
    
    # MoE with EP=16
    moe_input = f"{layer_id}_MoE_Input"
    parent.node(moe_input,
               label=f"MoE Input\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='ellipse', fillcolor='lightblue')
    parent.edge(attn_output, moe_input)
    
    # Router
    router = f"{layer_id}_Router"
    parent.node(router,
               label=f"Router GPU {gpu_offset}\\nExpert Selection\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: routing decisions",
               shape='parallelogram', fillcolor='lightyellow')
    parent.edge(moe_input, router)
    
    # Expert selection (dashed lines)
    experts = []
    for expert_id in range(16):
        expert_gpu = gpu_offset + expert_id
        expert = f"{layer_id}_Expert{expert_id}"
        parent.node(expert,
                   label=f"Expert {expert_id} GPU {expert_gpu}\\nInput: [batch_size=?, seq_len=?, hidden=512]\\nOutput: [batch_size=?, seq_len=?, hidden=512]",
                   shape='rect', fillcolor='lightgreen')
        
        # Dashed line for expert selection
        parent.edge(router, expert, style='dashed', label=f"select")
        experts.append(expert)
    
    # Expert aggregation
    moe_output = f"{layer_id}_MoE_Output"
    parent.node(moe_output,
               label=f"MoE Output\\nExpert Aggregation\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='parallelogram', fillcolor='lightyellow')
    
    for expert in experts:
        parent.edge(expert, moe_output)
    
    # Layer output
    layer_output = f"{layer_id}_Output"
    parent.node(layer_output,
               label=f"Layer {layer_num} Output\\nInput: [batch_size=128, seq_len=128-10240, hidden=512]\\nOutput: [batch_size=128, seq_len=128-10240, hidden=512]",
               shape='ellipse', fillcolor='lightblue')
    parent.edge(moe_output, layer_output)
    
    return layer_output

if __name__ == "__main__":
    # Generate the DAG
    dag = create_moe_dag()
    
    # Save DOT file
    dot_file = "./outputs/2025-12-30-10-42-40/moe_parallel_dag.dot"
    dag.save(dot_file)
    
    # Render to SVG
    svg_file = "./outputs/2025-12-30-10-42-40/moe_parallel_dag.svg"
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    # Also print the DOT source
    print("\nDOT source code:")
    print(dag.source)