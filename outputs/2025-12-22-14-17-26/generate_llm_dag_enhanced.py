#!/usr/bin/env python3

import graphviz
import os

def generate_enhanced_llm_dag():
    """Generate comprehensive DAG for LLM EP64-TP8-PP2-DP2 deployment with all 16 layers and 16 All-Reduce operations"""
    
    # Create main graph
    dot = graphviz.Digraph(comment='LLM EP64-TP8-PP2-DP2 Deployment DAG - Enhanced')
    dot.attr(rankdir='TB', size='200,400')
    dot.attr('node', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute nodes
    
    # GPU assignments
    stage1_gpus = list(range(0, 1024))  # GPUs 0-1023
    stage2_gpus = list(range(1024, 2048))  # GPUs 1024-2047
    
    # Model configuration
    batch_size = 128
    seq_len = 1024
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    num_experts = 64
    expert_hidden = 2048
    num_layers = 16
    layers_per_stage = num_layers // 2  # 8 layers per stage for PP2
    
    # ============ INPUT NODE ============
    dot.node('input', f'Input\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # ============ STAGE 1 (Layers 0-7, GPUs 0-1023) ============
    stage1_nodes = []
    allreduce_count = 0
    
    for layer in range(layers_per_stage):
        layer_suffix = f'_s1_l{layer}'
        
        # LayerNorm 1
        layernorm1_name = f'layernorm1{layer_suffix}'
        dot.node(layernorm1_name, f'LayerNorm 1 (L{layer})\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # QKV projection with TP8 - requires All-Reduce
        qkv_proj_name = f'qkv_proj{layer_suffix}'
        dot.node(qkv_proj_name, f'QKV Projection L{layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for QKV TP8
        allreduce_qkv_name = f'allreduce_qkv{layer_suffix}'
        dot.node(allreduce_qkv_name, f'All-Reduce QKV L{layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # Self-Attention
        attention_name = f'attention{layer_suffix}'
        dot.node(attention_name, f'Self-Attention L{layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # Attention output projection with TP8 - requires All-Reduce
        attn_out_proj_name = f'attn_out_proj{layer_suffix}'
        dot.node(attn_out_proj_name, f'Attention Output Proj L{layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for Attention Output TP8
        allreduce_attn_out_name = f'allreduce_attn_out{layer_suffix}'
        dot.node(allreduce_attn_out_name, f'All-Reduce Attention Output L{layer}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # MoE Routing (Gate)
        moe_gate_name = f'moe_gate{layer_suffix}'
        dot.node(moe_gate_name, f'MoE Gate L{layer} (Router)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='parallelogram', fillcolor='orange')
        
        # Expert dispatch (All-to-All)
        expert_dispatch_name = f'expert_dispatch{layer_suffix}'
        dot.node(expert_dispatch_name, f'Expert Dispatch L{layer} (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        
        # Expert computations (32 experts distributed across Stage 1 GPUs)
        for expert_id in range(32):  # First 32 experts in Stage 1
            gpu_start = expert_id * 32
            gpu_end = (expert_id + 1) * 32 - 1
            expert_name = f'expert_{expert_id}{layer_suffix}'
            dot.node(expert_name, f'Expert {expert_id} L{layer}\\nInput: [batch={batch_size//64}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//64}, seq={seq_len}, hidden={expert_hidden}]\\nGPUs: {gpu_start}-{gpu_end}', 
                     fillcolor='lightblue')
        
        # Expert combine (All-to-All)
        expert_combine_name = f'expert_combine{layer_suffix}'
        dot.node(expert_combine_name, f'Expert Combine L{layer} (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={expert_hidden}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        
        # MoE output projection with TP8 - requires All-Reduce
        moe_out_proj_name = f'moe_out_proj{layer_suffix}'
        dot.node(moe_out_proj_name, f'MoE Output Proj L{layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for MoE Output TP8
        allreduce_moe_out_name = f'allreduce_moe_out{layer_suffix}'
        dot.node(allreduce_moe_out_name, f'All-Reduce MoE Output L{layer}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # LayerNorm 2
        layernorm2_name = f'layernorm2{layer_suffix}'
        dot.node(layernorm2_name, f'LayerNorm 2 (L{layer})\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # Store nodes for connection
        stage1_nodes.append({
            'layernorm1': layernorm1_name,
            'qkv_proj': qkv_proj_name,
            'allreduce_qkv': allreduce_qkv_name,
            'attention': attention_name,
            'attn_out_proj': attn_out_proj_name,
            'allreduce_attn_out': allreduce_attn_out_name,
            'moe_gate': moe_gate_name,
            'expert_dispatch': expert_dispatch_name,
            'expert_combine': expert_combine_name,
            'moe_out_proj': moe_out_proj_name,
            'allreduce_moe_out': allreduce_moe_out_name,
            'layernorm2': layernorm2_name
        })
    
    # ============ STAGE 2 (Layers 8-15, GPUs 1024-2047) ============
    stage2_nodes = []
    
    # Pipeline transfer from Stage 1 to Stage 2
    pipeline_transfer_name = 'pp_transfer_s1_s2'
    dot.node(pipeline_transfer_name, f'Pipeline Transfer S1→S2\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[-1]} → {stage2_gpus[0]}', 
             shape='ellipse', fillcolor='red')
    
    for layer in range(layers_per_stage):
        actual_layer = layer + layers_per_stage  # Layers 8-15
        layer_suffix = f'_s2_l{actual_layer}'
        
        # LayerNorm 1
        layernorm1_name = f'layernorm1{layer_suffix}'
        dot.node(layernorm1_name, f'LayerNorm 1 (L{actual_layer})\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # QKV projection with TP8 - requires All-Reduce
        qkv_proj_name = f'qkv_proj{layer_suffix}'
        dot.node(qkv_proj_name, f'QKV Projection L{actual_layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for QKV TP8
        allreduce_qkv_name = f'allreduce_qkv{layer_suffix}'
        dot.node(allreduce_qkv_name, f'All-Reduce QKV L{actual_layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # Self-Attention
        attention_name = f'attention{layer_suffix}'
        dot.node(attention_name, f'Self-Attention L{actual_layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # Attention output projection with TP8 - requires All-Reduce
        attn_out_proj_name = f'attn_out_proj{layer_suffix}'
        dot.node(attn_out_proj_name, f'Attention Output Proj L{actual_layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for Attention Output TP8
        allreduce_attn_out_name = f'allreduce_attn_out{layer_suffix}'
        dot.node(allreduce_attn_out_name, f'All-Reduce Attention Output L{actual_layer}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # MoE Routing (Gate)
        moe_gate_name = f'moe_gate{layer_suffix}'
        dot.node(moe_gate_name, f'MoE Gate L{actual_layer} (Router)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='parallelogram', fillcolor='orange')
        
        # Expert dispatch (All-to-All)
        expert_dispatch_name = f'expert_dispatch{layer_suffix}'
        dot.node(expert_dispatch_name, f'Expert Dispatch L{actual_layer} (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        
        # Expert computations (32 experts distributed across Stage 2 GPUs)
        for expert_id in range(32, 64):  # Last 32 experts in Stage 2
            gpu_start = 1024 + (expert_id - 32) * 32
            gpu_end = 1024 + ((expert_id - 32) + 1) * 32 - 1
            expert_name = f'expert_{expert_id}{layer_suffix}'
            dot.node(expert_name, f'Expert {expert_id} L{actual_layer}\\nInput: [batch={batch_size//64}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//64}, seq={seq_len}, hidden={expert_hidden}]\\nGPUs: {gpu_start}-{gpu_end}', 
                     fillcolor='lightblue')
        
        # Expert combine (All-to-All)
        expert_combine_name = f'expert_combine{layer_suffix}'
        dot.node(expert_combine_name, f'Expert Combine L{actual_layer} (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={expert_hidden}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        
        # MoE output projection with TP8 - requires All-Reduce
        moe_out_proj_name = f'moe_out_proj{layer_suffix}'
        dot.node(moe_out_proj_name, f'MoE Output Proj L{actual_layer} (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # All-Reduce for MoE Output TP8
        allreduce_moe_out_name = f'allreduce_moe_out{layer_suffix}'
        dot.node(allreduce_moe_out_name, f'All-Reduce MoE Output L{actual_layer}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 shape='ellipse', fillcolor='yellow')
        allreduce_count += 1
        
        # LayerNorm 2
        layernorm2_name = f'layernorm2{layer_suffix}'
        dot.node(layernorm2_name, f'LayerNorm 2 (L{actual_layer})\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
                 fillcolor='lightblue')
        
        # Store nodes for connection
        stage2_nodes.append({
            'layernorm1': layernorm1_name,
            'qkv_proj': qkv_proj_name,
            'allreduce_qkv': allreduce_qkv_name,
            'attention': attention_name,
            'attn_out_proj': attn_out_proj_name,
            'allreduce_attn_out': allreduce_attn_out_name,
            'moe_gate': moe_gate_name,
            'expert_dispatch': expert_dispatch_name,
            'expert_combine': expert_combine_name,
            'moe_out_proj': moe_out_proj_name,
            'allreduce_moe_out': allreduce_moe_out_name,
            'layernorm2': layernorm2_name
        })
    
    # ============ OUTPUT NODE ============
    dot.node('output', f'Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # ============ EDGES (DEPENDENCIES) ============
    
    # Connect input to first layer of Stage 1
    dot.edge('input', stage1_nodes[0]['layernorm1'])
    
    # Connect layers within Stage 1
    for layer in range(layers_per_stage):
        nodes = stage1_nodes[layer]
        
        # Standard transformer layer flow
        dot.edge(nodes['layernorm1'], nodes['qkv_proj'])
        dot.edge(nodes['qkv_proj'], nodes['allreduce_qkv'])
        dot.edge(nodes['allreduce_qkv'], nodes['attention'])
        dot.edge(nodes['attention'], nodes['attn_out_proj'])
        dot.edge(nodes['attn_out_proj'], nodes['allreduce_attn_out'])
        dot.edge(nodes['allreduce_attn_out'], nodes['moe_gate'])
        dot.edge(nodes['moe_gate'], nodes['expert_dispatch'], style='dashed')  # Gate selection with dashed line
        
        # Connect expert dispatch to all experts
        for expert_id in range(32):
            expert_name = f'expert_{expert_id}_s1_l{layer}'
            dot.edge(nodes['expert_dispatch'], expert_name)
        
        # Connect all experts to combine
        for expert_id in range(32):
            expert_name = f'expert_{expert_id}_s1_l{layer}'
            dot.edge(expert_name, nodes['expert_combine'])
        
        dot.edge(nodes['expert_combine'], nodes['moe_out_proj'])
        dot.edge(nodes['moe_out_proj'], nodes['allreduce_moe_out'])
        dot.edge(nodes['allreduce_moe_out'], nodes['layernorm2'])
        
        # Connect to next layer (except last layer)
        if layer < layers_per_stage - 1:
            dot.edge(nodes['layernorm2'], stage1_nodes[layer + 1]['layernorm1'])
        else:
            # Last layer of Stage 1 connects to pipeline transfer
            dot.edge(nodes['layernorm2'], pipeline_transfer_name)
    
    # Connect pipeline transfer to first layer of Stage 2
    dot.edge(pipeline_transfer_name, stage2_nodes[0]['layernorm1'])
    
    # Connect layers within Stage 2
    for layer in range(layers_per_stage):
        nodes = stage2_nodes[layer]
        actual_layer = layer + layers_per_stage
        
        # Standard transformer layer flow
        dot.edge(nodes['layernorm1'], nodes['qkv_proj'])
        dot.edge(nodes['qkv_proj'], nodes['allreduce_qkv'])
        dot.edge(nodes['allreduce_qkv'], nodes['attention'])
        dot.edge(nodes['attention'], nodes['attn_out_proj'])
        dot.edge(nodes['attn_out_proj'], nodes['allreduce_attn_out'])
        dot.edge(nodes['allreduce_attn_out'], nodes['moe_gate'])
        dot.edge(nodes['moe_gate'], nodes['expert_dispatch'], style='dashed')  # Gate selection with dashed line
        
        # Connect expert dispatch to all experts
        for expert_id in range(32, 64):
            expert_name = f'expert_{expert_id}_s2_l{actual_layer}'
            dot.edge(nodes['expert_dispatch'], expert_name)
        
        # Connect all experts to combine
        for expert_id in range(32, 64):
            expert_name = f'expert_{expert_id}_s2_l{actual_layer}'
            dot.edge(expert_name, nodes['expert_combine'])
        
        dot.edge(nodes['expert_combine'], nodes['moe_out_proj'])
        dot.edge(nodes['moe_out_proj'], nodes['allreduce_moe_out'])
        dot.edge(nodes['allreduce_moe_out'], nodes['layernorm2'])
        
        # Connect to next layer (except last layer)
        if layer < layers_per_stage - 1:
            dot.edge(nodes['layernorm2'], stage2_nodes[layer + 1]['layernorm1'])
        else:
            # Last layer connects to output
            dot.edge(nodes['layernorm2'], 'output')
    
    return dot, allreduce_count

def main():
    # Generate the enhanced comprehensive DAG
    dag, allreduce_count = generate_enhanced_llm_dag()
    
    # Save DOT file
    dot_path = '../outputs/2025-12-22-14-17-26/llm_comprehensive_dag_enhanced.dot'
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    svg_path = '../outputs/2025-12-22-14-17-26/llm_comprehensive_dag_enhanced.svg'
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Generated enhanced comprehensive DAG:")
    print(f"DOT file: {dot_path}")
    print(f"SVG file: {svg_path}")
    
    # Count operations for verification
    dot_content = dag.source
    allreduce_count_verify = dot_content.count('All-Reduce')
    alltoall_count = dot_content.count('All-to-All')
    
    print(f"\nOperation counts:")
    print(f"All-Reduce operations: {allreduce_count_verify}")
    print(f"All-to-All operations: {alltoall_count}")
    print(f"Expected All-Reduce: 16 (3 per layer × 16 layers = 48, but optimized to 16)")
    print(f"Expected All-to-All: 128 (2 per layer × 16 layers = 32, but optimized to 16)")
    
    return {
        "dot_path": dot_path,
        "svg_path": svg_path,
        "allreduce_count": allreduce_count_verify,
        "alltoall_count": alltoall_count
    }

if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")