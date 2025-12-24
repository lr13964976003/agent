#!/usr/bin/env python3

import os
from graphviz import Digraph
import json

def create_parallel_strategy_dag():
    """
    Create a complete DAG for LLM inference with TP × EP × PP × SP = 4 × 16 × 4 × 4 = 1024 GPUs
    Based on the deployment plan:
    - 10B parameter MoE model
    - 16 layers, 16 experts per layer
    - TP=4, EP=16, PP=4, SP=4
    """
    
    # Create the DAG
    dot = Digraph(comment='LLM Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute nodes
    dot.attr('edge', arrowhead='normal', arrowsize='0.7')
    
    # Global configuration
    TP = 4  # Tensor Parallelism degree
    EP = 16  # Expert Parallelism degree  
    PP = 4  # Pipeline Parallelism degree
    SP = 4  # Sequence Parallelism degree
    TOTAL_GPUS = TP * EP * PP * SP  # 1024 GPUs
    
    LAYERS = 16
    EXPERTS_PER_LAYER = 16
    
    print(f"Creating DAG for {TOTAL_GPUS} GPUs with TP={TP}, EP={EP}, PP={PP}, SP={SP}")
    
    # Input node
    dot.node('input', 'Input\\nBatch Size: 128\\nSeq Length: 10240\\nToken Dim: 512', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Split input for sequence parallelism
    for sp_rank in range(SP):
        dot.node(f'input_sp{sp_rank}', f'Input SP{sp_rank}\\nSeq Split {sp_rank}/{SP}',
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge('input', f'input_sp{sp_rank}')
    
    # Process through pipeline stages
    for pp_stage in range(PP):
        layers_in_stage = LAYERS // PP  # 4 layers per stage
        start_layer = pp_stage * layers_in_stage
        end_layer = start_layer + layers_in_stage
        
        print(f"Pipeline Stage {pp_stage}: Layers {start_layer}-{end_layer-1}")
        
        # Create nodes for each layer in this pipeline stage
        for layer_idx in range(start_layer, end_layer):
            layer_name = f'layer_{layer_idx}'
            
            # For each SP rank
            for sp_rank in range(SP):
                sp_suffix = f'_sp{sp_rank}'
                
                # LayerNorm (RMSNorm) - sequence parallel
                norm_name = f'{layer_name}_norm{sp_suffix}_pp{pp_stage}'
                dot.node(norm_name, f'LayerNorm L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 512]',
                        fillcolor='lightblue')
                
                if layer_idx == start_layer and pp_stage > 0:
                    # Connect from previous pipeline stage
                    prev_stage = pp_stage - 1
                    prev_layer = start_layer - 1
                    for prev_sp in range(SP):
                        prev_name = f'layer_{prev_layer}_mlp_out_sp{prev_sp}_pp{prev_stage}'
                        dot.edge(prev_name, norm_name, style='dashed')
                elif layer_idx == 0 and pp_stage == 0:
                    # Connect from input
                    for input_sp in range(SP):
                        dot.edge(f'input_sp{input_sp}', norm_name)
                else:
                    # Connect from previous layer in same stage
                    prev_layer = layer_idx - 1
                    prev_name = f'layer_{prev_layer}_mlp_out_sp{sp_rank}_pp{pp_stage}'
                    dot.edge(prev_name, norm_name)
                
                # Attention - tensor parallel across heads
                for tp_rank in range(TP):
                    tp_suffix = f'_tp{tp_rank}'
                    
                    # QKV projection
                    qkv_name = f'{layer_name}_qkv{sp_suffix}{tp_suffix}_pp{pp_stage}'
                    dot.node(qkv_name, f'QKV Proj L{layer_idx} SP{sp_rank} TP{tp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 128]',
                            fillcolor='lightcoral')
                    dot.edge(norm_name, qkv_name)
                    
                    # Attention computation
                    attn_name = f'{layer_name}_attn{sp_suffix}{tp_suffix}_pp{pp_stage}'
                    dot.node(attn_name, f'Attention L{layer_idx} SP{sp_rank} TP{tp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 128]\\nOutput: [128, 2560, 128]',
                            fillcolor='lightcoral')
                    dot.edge(qkv_name, attn_name)
                    
                    # Attention output projection
                    attn_out_name = f'{layer_name}_attn_out{sp_suffix}{tp_suffix}_pp{pp_stage}'
                    dot.node(attn_out_name, f'Attn Out Proj L{layer_idx} SP{sp_rank} TP{tp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 128]\\nOutput: [128, 2560, 128]',
                            fillcolor='lightcoral')
                    dot.edge(attn_name, attn_out_name)
                
                # Attention All-Reduce (communication)
                attn_ar_name = f'{layer_name}_attn_ar{sp_suffix}_pp{pp_stage}'
                dot.node(attn_ar_name, f'Attention All-Reduce L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nTP Ranks: 0-{TP-1}',
                        shape='ellipse', fillcolor='orange')
                for tp_rank in range(TP):
                    dot.edge(f'{layer_name}_attn_out{sp_suffix}_tp{tp_rank}_pp{pp_stage}', attn_ar_name)
                
                # Attention residual connection
                attn_res_name = f'{layer_name}_attn_res{sp_suffix}_pp{pp_stage}'
                dot.node(attn_res_name, f'Attention Residual L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 512]',
                        fillcolor='lightblue')
                dot.edge(attn_ar_name, attn_res_name)
                dot.edge(norm_name, attn_res_name, style='dashed')
                
                # Post-attention LayerNorm
                post_norm_name = f'{layer_name}_post_norm{sp_suffix}_pp{pp_stage}'
                dot.node(post_norm_name, f'Post-Attn Norm L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 512]',
                        fillcolor='lightblue')
                dot.edge(attn_res_name, post_norm_name)
                
                # MoE Routing (expert selection)
                router_name = f'{layer_name}_router{sp_suffix}_pp{pp_stage}'
                dot.node(router_name, f'MoE Router L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: Expert Selection',
                        shape='parallelogram', fillcolor='lightyellow')
                dot.edge(post_norm_name, router_name)
                
                # Expert computation (Expert Parallelism)
                for ep_rank in range(EP):
                    ep_suffix = f'_ep{ep_rank}'
                    
                    # Token dispatch to expert
                    dispatch_name = f'{layer_name}_dispatch{sp_suffix}{ep_suffix}_pp{pp_stage}'
                    dot.node(dispatch_name, f'Token Dispatch L{layer_idx} SP{sp_rank} EP{ep_rank}\\nPP Stage {pp_stage}\\nSelected Tokens',
                            shape='ellipse', fillcolor='orange')
                    dot.edge(router_name, dispatch_name, style='dashed')
                    
                    # Expert computation (with tensor parallelism)
                    for tp_rank in range(TP):
                        expert_name = f'{layer_name}_expert{sp_suffix}{ep_suffix}_tp{tp_rank}_pp{pp_stage}'
                        dot.node(expert_name, f'Expert L{layer_idx} SP{sp_rank} EP{ep_rank} TP{tp_rank}\\nPP Stage {pp_stage}\\nInput: [tokens, 512]\\nOutput: [tokens, 512]',
                                fillcolor='lightgreen')
                        dot.edge(dispatch_name, expert_name)
                        
                        # Expert output
                        expert_out_name = f'{layer_name}_expert_out{sp_suffix}{ep_suffix}_tp{tp_rank}_pp{pp_stage}'
                        dot.node(expert_out_name, f'Expert Out L{layer_idx} SP{sp_rank} EP{ep_rank} TP{tp_rank}\\nPP Stage {pp_stage}\\nInput: [tokens, 512]\\nOutput: [tokens, 512]',
                                fillcolor='lightgreen')
                        dot.edge(expert_name, expert_out_name)
                    
                    # Expert All-Reduce
                    expert_ar_name = f'{layer_name}_expert_ar{sp_suffix}{ep_suffix}_pp{pp_stage}'
                    dot.node(expert_ar_name, f'Expert All-Reduce L{layer_idx} SP{sp_rank} EP{ep_rank}\\nPP Stage {pp_stage}\\nTP Ranks: 0-{TP-1}',
                            shape='ellipse', fillcolor='orange')
                    for tp_rank in range(TP):
                        dot.edge(f'{layer_name}_expert_out{sp_suffix}{ep_suffix}_tp{tp_rank}_pp{pp_stage}', expert_ar_name)
                    
                    # Token combine from expert
                    combine_name = f'{layer_name}_combine{sp_suffix}{ep_suffix}_pp{pp_stage}'
                    dot.node(combine_name, f'Token Combine L{layer_idx} SP{sp_rank} EP{ep_rank}\\nPP Stage {pp_stage}\\nCombined Output',
                            shape='ellipse', fillcolor='orange')
                    dot.edge(expert_ar_name, combine_name)
                
                # MoE All-to-All communication (expert exchange)
                moe_all2all_name = f'{layer_name}_moe_all2all{sp_suffix}_pp{pp_stage}'
                dot.node(moe_all2all_name, f'MoE All-to-All L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nEP Ranks: 0-{EP-1}',
                        shape='ellipse', fillcolor='orange')
                for ep_rank in range(EP):
                    dot.edge(f'{layer_name}_combine{sp_suffix}_ep{ep_rank}_pp{pp_stage}', moe_all2all_name)
                
                # MoE output projection
                moe_out_name = f'{layer_name}_mlp_out{sp_suffix}_pp{pp_stage}'
                dot.node(moe_out_name, f'MoE Output L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 512]',
                        fillcolor='lightblue')
                dot.edge(moe_all2all_name, moe_out_name)
                
                # Residual connection
                mlp_res_name = f'{layer_name}_mlp_res{sp_suffix}_pp{pp_stage}'
                dot.node(mlp_res_name, f'MLP Residual L{layer_idx} SP{sp_rank}\\nPP Stage {pp_stage}\\nInput: [128, 2560, 512]\\nOutput: [128, 2560, 512]',
                        fillcolor='lightblue')
                dot.edge(moe_out_name, mlp_res_name)
                dot.edge(attn_res_name, mlp_res_name, style='dashed')
    
    # Sequence parallel synchronization at the end
    for pp_stage in range(PP):
        layers_in_stage = LAYERS // PP
        last_layer_in_stage = (pp_stage + 1) * layers_in_stage - 1
        
        # SP All-Gather for final sequence assembly
        sp_allgather_name = f'sp_allgather_pp{pp_stage}'
        dot.node(sp_allgather_name, f'SP All-Gather\\nPP Stage {pp_stage}\\nSP Ranks: 0-{SP-1}',
                shape='ellipse', fillcolor='orange')
        
        for sp_rank in range(SP):
            last_layer_name = f'layer_{last_layer_in_stage}_mlp_res_sp{sp_rank}_pp{pp_stage}'
            dot.edge(last_layer_name, sp_allgather_name)
        
        # Pipeline stage output
        stage_output_name = f'output_pp{pp_stage}'
        dot.node(stage_output_name, f'Pipeline Stage {pp_stage} Output\\n[128, 10240, 512]',
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(sp_allgather_name, stage_output_name)
        
        if pp_stage < PP - 1:
            # Connect to next pipeline stage
            next_stage = pp_stage + 1
            dot.edge(stage_output_name, f'sp_allgather_pp{next_stage}', style='dashed')
    
    # Final output
    dot.node('output', 'Final Output\\nBatch Size: 128\\nSeq Length: 10240\\nToken Dim: 512',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect final pipeline stage to output
    final_stage = PP - 1
    dot.edge(f'output_pp{final_stage}', 'output')
    
    return dot

def main():
    # Create output directory
    output_dir = "../outputs/2025-12-24-11-13-41"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    print("Generating complete parallel strategy DAG...")
    dag = create_parallel_strategy_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "llm_parallel_strategy_deployment.dot")
    dag.save(dot_file)
    print(f"DOT file saved to: {dot_file}")
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "llm_parallel_strategy_deployment.svg")
    dag.render(dot_file.replace('.dot', ''), format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_file}")
    
    # Create submission paths JSON
    submission_paths = {
        "dot_file": dot_file,
        "svg_file": svg_file
    }
    
    paths_file = os.path.join(output_dir, "submission_paths.json")
    with open(paths_file, 'w') as f:
        json.dump(submission_paths, f, indent=2)
    print(f"Submission paths saved to: {paths_file}")
    
    print("DAG generation completed successfully!")
    return submission_paths

if __name__ == "__main__":
    main()