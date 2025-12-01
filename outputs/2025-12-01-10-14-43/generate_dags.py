#!/usr/bin/env python3

import graphviz
import os
from pathlib import Path

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2 configuration"""
    dot = graphviz.Digraph(comment='Baseline TP=8 PP=2 MoE DAG')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', shape='rectangle')
    dot.attr('edge', arrowhead='normal')
    
    # Model specifications
    batch_size = 128
    seq_len = 128
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    moe_hidden = 2048
    experts_per_layer = 64
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Create input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightblue')
    
    # Process each pipeline stage
    for stage in range(2):
        stage_gpus = list(range(stage*8, (stage+1)*8))
        
        for layer in range(8):
            actual_layer = stage * 8 + layer
            
            with dot.subgraph(name=f'cluster_stage{stage}_layer{layer}') as c:
                c.attr(label=f'Pipeline Stage {stage}, Layer {actual_layer}')
                c.attr(style='rounded,filled', fillcolor='lightgray')
                
                # LayerNorm (duplicated across all GPUs in stage)
                layernorm_name = f'stage{stage}_layer{layer}_layernorm'
                c.node(layernorm_name, 
                       f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # Multi-Head Attention with Tensor Parallelism
                # QKV projection (column parallel)
                qkv_proj_name = f'stage{stage}_layer{layer}_qkv_proj'
                c.node(qkv_proj_name,
                       f'QKV Projection (Col-Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x]\\nGPU: {stage_gpus}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # All-gather for QKV
                qkv_gather_name = f'stage{stage}_layer{layer}_qkv_gather'
                c.node(qkv_gather_name,
                       f'All-Gather QKV\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x, shard=8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x]\\nGPU: {stage_gpus}',
                       shape='ellipse', fillcolor='lightblue')
                
                # Attention computation
                attn_name = f'stage{stage}_layer{layer}_attention'
                c.node(attn_name,
                       f'Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}]\\nGPU: {stage_gpus}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # Attention output projection (row parallel)
                attn_out_name = f'stage{stage}_layer{layer}_attn_out'
                c.node(attn_out_name,
                       f'Attention Output (Row-Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, shard=8]\\nGPU: {stage_gpus}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # All-reduce for attention output
                attn_reduce_name = f'stage{stage}_layer{layer}_attn_reduce'
                c.node(attn_reduce_name,
                       f'All-Reduce Attention Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, shard=8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='ellipse', fillcolor='lightblue')
                
                # Residual add
                residual1_name = f'stage{stage}_layer{layer}_residual1'
                c.node(residual1_name,
                       f'Residual Add 1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # LayerNorm 2
                layernorm2_name = f'stage{stage}_layer{layer}_layernorm2'
                c.node(layernorm2_name, 
                       f'LayerNorm 2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # MoE Layer - Multiple experts per GPU
                for expert_idx in range(4):  # 4 experts per GPU
                    actual_expert = (actual_layer * 16) + (stage * 8) + expert_idx
                    
                    # Gate computation
                    gate_name = f'stage{stage}_layer{layer}_expert{expert_idx}_gate'
                    c.node(gate_name,
                           f'Expert Gate {actual_expert}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=1]\\nGPU: {stage_gpus}',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert MLP (first linear - column parallel)
                    expert_mlp1_name = f'stage{stage}_layer{layer}_expert{expert_idx}_mlp1'
                    c.node(expert_mlp1_name,
                           f'Expert {actual_expert} MLP1 (Col-Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}, shard=8]\\nGPU: {stage_gpus}',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert MLP activation
                    expert_act_name = f'stage{stage}_layer{layer}_expert{expert_idx}_act'
                    c.node(expert_act_name,
                           f'Expert {actual_expert} Activation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}, shard=8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}, shard=8]\\nGPU: {stage_gpus}',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert MLP (second linear - row parallel)
                    expert_mlp2_name = f'stage{stage}_layer{layer}_expert{expert_idx}_mlp2'
                    c.node(expert_mlp2_name,
                           f'Expert {actual_expert} MLP2 (Row-Parallel)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}, shard=8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, shard=8]\\nGPU: {stage_gpus}',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert output all-reduce
                    expert_reduce_name = f'stage{stage}_layer{layer}_expert{expert_idx}_reduce'
                    c.node(expert_reduce_name,
                           f'Expert {actual_expert} All-Reduce\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}, shard=8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                           shape='ellipse', fillcolor='lightblue')
                
                # Expert aggregation (weighted sum)
                expert_agg_name = f'stage{stage}_layer{layer}_expert_agg'
                c.node(expert_agg_name,
                       f'Expert Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x4)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='parallelogram', fillcolor='lightyellow')
                
                # Residual add 2
                residual2_name = f'stage{stage}_layer{layer}_residual2'
                c.node(residual2_name,
                       f'Residual Add 2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus}',
                       shape='parallelogram', fillcolor='lightyellow')
    
    # Connect nodes within layers
    for stage in range(2):
        stage_gpus = list(range(stage*8, (stage+1)*8))
        
        for layer in range(8):
            actual_layer = stage * 8 + layer
            
            # Input connections
            if actual_layer == 0:
                dot.edge('input', f'stage{stage}_layer{layer}_layernorm')
            else:
                prev_layer = layer - 1 if layer > 0 else 7
                prev_stage = stage if layer > 0 else (stage - 1 if stage > 0 else 0)
                if layer > 0:
                    dot.edge(f'stage{stage}_layer{prev_layer}_residual2', f'stage{stage}_layer{layer}_layernorm')
                elif stage > 0:
                    # Pipeline communication between stages
                    dot.node(f'pipeline_stage{stage-1}_to_{stage}', 
                            f'Pipeline Communication\\nStage {stage-1} -> Stage {stage}\\nGPU: {list(range((stage-1)*8, stage*8))} -> {stage_gpus}',
                            shape='ellipse', fillcolor='lightblue')
                    dot.edge(f'stage{stage-1}_layer7_residual2', f'pipeline_stage{stage-1}_to_{stage}')
                    dot.edge(f'pipeline_stage{stage-1}_to_{stage}', f'stage{stage}_layer{layer}_layernorm')
            
            # Connect within layer
            dot.edge(f'stage{stage}_layer{layer}_layernorm', f'stage{stage}_layer{layer}_qkv_proj')
            dot.edge(f'stage{stage}_layer{layer}_qkv_proj', f'stage{stage}_layer{layer}_qkv_gather')
            dot.edge(f'stage{stage}_layer{layer}_qkv_gather', f'stage{stage}_layer{layer}_attention')
            dot.edge(f'stage{stage}_layer{layer}_attention', f'stage{stage}_layer{layer}_attn_out')
            dot.edge(f'stage{stage}_layer{layer}_attn_out', f'stage{stage}_layer{layer}_attn_reduce')
            dot.edge(f'stage{stage}_layer{layer}_attn_reduce', f'stage{stage}_layer{layer}_residual1')
            dot.edge(f'stage{stage}_layer{layer}_residual1', f'stage{stage}_layer{layer}_layernorm2')
            
            # Connect MoE experts
            for expert_idx in range(4):
                dot.edge(f'stage{stage}_layer{layer}_layernorm2', f'stage{stage}_layer{layer}_expert{expert_idx}_gate')
                dot.edge(f'stage{stage}_layer{layer}_expert{expert_idx}_gate', f'stage{stage}_layer{layer}_expert{expert_idx}_mlp1')
                dot.edge(f'stage{stage}_layer{layer}_expert{expert_idx}_mlp1', f'stage{stage}_layer{layer}_expert{expert_idx}_act')
                dot.edge(f'stage{stage}_layer{layer}_expert{expert_idx}_act', f'stage{stage}_layer{layer}_expert{expert_idx}_mlp2')
                dot.edge(f'stage{stage}_layer{layer}_expert{expert_idx}_mlp2', f'stage{stage}_layer{layer}_expert{expert_idx}_reduce')
                dot.edge(f'stage{stage}_layer{layer}_expert{expert_idx}_reduce', f'stage{stage}_layer{layer}_expert_agg')
            
            dot.edge(f'stage{stage}_layer{layer}_expert_agg', f'stage{stage}_layer{layer}_residual2')
    
    # Create output node
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connect final layer to output
    dot.edge('stage1_layer7_residual2', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=16 configuration"""
    dot = graphviz.Digraph(comment='Proposed Cross-Node EP=16 MoE DAG')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', shape='rectangle')
    dot.attr('edge', arrowhead='normal')
    
    # Model specifications
    batch_size = 128
    seq_len = 128
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    moe_hidden = 2048
    experts_per_layer = 64
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Create input node
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightblue')
    
    # Process each layer (all layers on all GPUs)
    for layer in range(16):
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer}')
            c.attr(style='rounded,filled', fillcolor='lightgray')
            
            # LayerNorm (all GPUs)
            layernorm_name = f'layer{layer}_layernorm'
            c.node(layernorm_name, 
                   f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Multi-Head Attention (no tensor parallelism, duplicated on all GPUs)
            qkv_proj_name = f'layer{layer}_qkv_proj'
            c.node(qkv_proj_name,
                   f'QKV Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention computation
            attn_name = f'layer{layer}_attention'
            c.node(attn_name,
                   f'Multi-Head Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, 3x]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention output projection
            attn_out_name = f'layer{layer}_attn_out'
            c.node(attn_out_name,
                   f'Attention Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Residual add
            residual1_name = f'layer{layer}_residual1'
            c.node(residual1_name,
                   f'Residual Add 1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # LayerNorm 2
            layernorm2_name = f'layer{layer}_layernorm2'
            c.node(layernorm2_name, 
                   f'LayerNorm 2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Global gate computation (all GPUs)
            global_gate_name = f'layer{layer}_global_gate'
            c.node(global_gate_name,
                   f'Global Gate\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}]\\nGPU: All',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Expert routing (token batching by destination)
            routing_name = f'layer{layer}_routing'
            c.node(routing_name,
                   f'Token Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (routed)\\nGPU: All',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Expert computations - one expert per GPU
            for gpu_id in range(16):
                expert_id = gpu_id  # Each GPU has one expert
                
                # Token receive (communication)
                token_recv_name = f'layer{layer}_gpu{gpu_id}_token_recv'
                c.node(token_recv_name,
                       f'Token Receive GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (routed)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (filtered)\\nGPU: {gpu_id}',
                       shape='ellipse', fillcolor='lightblue')
                
                # Expert MLP 1
                expert_mlp1_name = f'layer{layer}_gpu{gpu_id}_expert_mlp1'
                c.node(expert_mlp1_name,
                       f'Expert {expert_id} MLP1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (filtered)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # Expert activation
                expert_act_name = f'layer{layer}_gpu{gpu_id}_expert_act'
                c.node(expert_act_name,
                       f'Expert {expert_id} Activation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # Expert MLP 2
                expert_mlp2_name = f'layer{layer}_gpu{gpu_id}_expert_mlp2'
                c.node(expert_mlp2_name,
                       f'Expert {expert_id} MLP2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn={moe_hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
                
                # Token send back (communication)
                token_send_name = f'layer{layer}_gpu{gpu_id}_token_send'
                c.node(token_send_name,
                       f'Token Send GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (processed)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (aggregated)\\nGPU: {gpu_id}',
                       shape='ellipse', fillcolor='lightblue')
            
            # Expert aggregation (collect all expert outputs)
            expert_agg_name = f'layer{layer}_expert_agg'
            c.node(expert_agg_name,
                   f'Expert Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x16)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Residual add 2
            residual2_name = f'layer{layer}_residual2'
            c.node(residual2_name,
                   f'Residual Add 2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All',
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Connect nodes
    for layer in range(16):
        # Input connections
        if layer == 0:
            dot.edge('input', f'layer{layer}_layernorm')
        else:
            dot.edge(f'layer{layer-1}_residual2', f'layer{layer}_layernorm')
        
        # Connect within layer
        dot.edge(f'layer{layer}_layernorm', f'layer{layer}_qkv_proj')
        dot.edge(f'layer{layer}_qkv_proj', f'layer{layer}_attention')
        dot.edge(f'layer{layer}_attention', f'layer{layer}_attn_out')
        dot.edge(f'layer{layer}_attn_out', f'layer{layer}_residual1')
        dot.edge(f'layer{layer}_residual1', f'layer{layer}_layernorm2')
        dot.edge(f'layer{layer}_layernorm2', f'layer{layer}_global_gate')
        dot.edge(f'layer{layer}_global_gate', f'layer{layer}_routing')
        
        # Connect expert routing to each GPU
        for gpu_id in range(16):
            dot.edge(f'layer{layer}_routing', f'layer{layer}_gpu{gpu_id}_token_recv', style='dashed')
            dot.edge(f'layer{layer}_gpu{gpu_id}_token_recv', f'layer{layer}_gpu{gpu_id}_expert_mlp1')
            dot.edge(f'layer{layer}_gpu{gpu_id}_expert_mlp1', f'layer{layer}_gpu{gpu_id}_expert_act')
            dot.edge(f'layer{layer}_gpu{gpu_id}_expert_act', f'layer{layer}_gpu{gpu_id}_expert_mlp2')
            dot.edge(f'layer{layer}_gpu{gpu_id}_expert_mlp2', f'layer{layer}_gpu{gpu_id}_token_send')
            dot.edge(f'layer{layer}_gpu{gpu_id}_token_send', f'layer{layer}_expert_agg')
        
        dot.edge(f'layer{layer}_expert_agg', f'layer{layer}_residual2')
    
    # Create output node
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connect final layer to output
    dot.edge('layer15_residual2', 'output')
    
    return dot

def main():
    # Create output directory
    output_dir = Path('../outputs/2025-12-01-10-14-43')
    output_dir.mkdir(exist_ok=True)
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    
    # Save baseline DAG as DOT and SVG
    baseline_dot_path = output_dir / 'baseline_tp8_pp2_moe.dag'
    baseline_svg_path = output_dir / 'baseline_tp8_pp2_moe.svg'
    
    with open(baseline_dot_path, 'w') as f:
        f.write(baseline_dag.source)
    
    baseline_dag.render(str(output_dir / 'baseline_tp8_pp2_moe'), format='svg', cleanup=True)
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    
    # Save proposed DAG as DOT and SVG
    proposed_dot_path = output_dir / 'proposed_cross_node_ep16_moe.dag'
    proposed_svg_path = output_dir / 'proposed_cross_node_ep16_moe.svg'
    
    with open(proposed_dot_path, 'w') as f:
        f.write(proposed_dag.source)
    
    proposed_dag.render(str(output_dir / 'proposed_cross_node_ep16_moe'), format='svg', cleanup=True)
    
    # Return paths for submission
    return {
        "baseline_dag_dot": str(baseline_dot_path),
        "baseline_dag_svg": str(baseline_svg_path),
        "proposed_dag_dot": str(proposed_dot_path),
        "proposed_dag_svg": str(proposed_svg_path)
    }

if __name__ == '__main__':
    paths = main()
    print("DAG generation complete!")
    print(f"Baseline DAG: {paths['baseline_dag_dot']}")
    print(f"Proposed DAG: {paths['proposed_dag_dot']}")