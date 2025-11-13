#!/usr/bin/env python3
"""
Simplified Megatron-LM DAG Generator
Shows tensor + pipeline parallelism combination
"""

import graphviz

def create_megatron_lm_dag():
    """Create simplified Megatron-LM DAG"""
    
    dot = graphviz.Digraph('Megatron_LM', 
                           comment='Megatron-LM Tensor + Pipeline Parallelism',
                           format='svg',
                           graph_attr={
                               'rankdir': 'TB',
                               'compound': 'true',
                               'ranksep': '1.0',
                               'nodesep': '0.3'
                           })
    
    # Model configurations
    configs = {
        'LLaMA_7B': {
            'layers': 32,
            'tensor_parallel': 2,  # Split across 2 GPUs per stage
            'pipeline_parallel': 3,  # 3 pipeline stages
            'hidden_size': 4096,
            'per_rank_hidden': 2048,
            'heads': 32,
            'per_rank_heads': 16,
            'ffn_size': 11008,
            'per_rank_ffn': 5504,
            'batch': 6,
            'seq': 2048,
            'vocab': 32000
        },
        'GPT3_2B': {
            'layers': 24,
            'tensor_parallel': 2,
            'pipeline_parallel': 3,
            'hidden_size': 2048,
            'per_rank_hidden': 1024,
            'heads': 16,
            'per_rank_heads': 8,
            'ffn_size': 8192,
            'per_rank_ffn': 4096,
            'batch': 12,
            'seq': 2048,
            'vocab': 50257
        }
    }
    
    for model_name, cfg in configs.items():
        with dot.subgraph(name=f'cluster_{model_name}') as c:
            c.attr(label=f'{model_name} Megatron-LM\\nTensor×{cfg["tensor_parallel"]} Pipeline×{cfg["pipeline_parallel"]}',
                   style='rounded,filled', fillcolor='lightyellow')
            
            # Pipeline stages
            for stage_idx in range(cfg['pipeline_parallel']):
                stage_start = stage_idx * (cfg['layers'] // cfg['pipeline_parallel']) + 1
                stage_end = min((stage_idx + 1) * (cfg['layers'] // cfg['pipeline_parallel']), cfg['layers'])
                
                with c.subgraph(name=f'cluster_{model_name}_stage_{stage_idx + 1}') as stage:
                    stage.attr(label=f'Stage {stage_idx + 1}\\nLayers {stage_start}-{stage_end}',
                               style='rounded,filled', fillcolor='lightblue')
                    
                    # Tensor parallel ranks
                    for rank_idx in range(cfg['tensor_parallel']):
                        device = f'GPU_{stage_idx * cfg["tensor_parallel"] + rank_idx + 1}'
                        
                        with stage.subgraph(name=f'cluster_{model_name}_stage_{stage_idx + 1}_rank_{rank_idx + 1}') as rank:
                            rank.attr(label=f'Rank {rank_idx + 1}\\n{device}',
                                      style='rounded,filled', fillcolor='lightgreen')
                            
                            # Create layers for this rank
                            for layer_idx in range(stage_start, stage_end + 1):
                                layer_prefix = f'{model_name}_S{stage_idx + 1}_R{rank_idx + 1}_L{layer_idx}'
                                
                                # Input split
                                split = f'{layer_prefix}_split'
                                rank.node(split, f'Input Split\\n[batch={cfg["batch"]}, seq={cfg["seq"]}, hidden={cfg["hidden_size"]}]→[hidden={cfg["per_rank_hidden"]}]',
                                          shape='parallelogram', style='filled', fillcolor='orange')
                                
                                # LayerNorm
                                norm1 = f'{layer_prefix}_norm1'
                                rank.node(norm1, f'LayerNorm\\nDevice: {device}', shape='ellipse', style='filled', fillcolor='lightgray')
                                
                                # QKV
                                qkv = f'{layer_prefix}_qkv'
                                rank.node(qkv, f'QKV Proj\\n[hidden={cfg["per_rank_hidden"]}]→[heads={cfg["per_rank_heads"]}, dim=128]', shape='rectangle', style='filled', fillcolor='lightcoral')
                                
                                # Attention
                                attn = f'{layer_prefix}_attn'
                                rank.node(attn, f'MHA\\n[heads={cfg["per_rank_heads"]}, dim=128]→[hidden={cfg["per_rank_hidden"]}]', shape='rectangle', style='filled', fillcolor='lightcoral')
                                
                                # All-reduce
                                ar1 = f'{layer_prefix}_ar1'
                                rank.node(ar1, f'All-Reduce\\n[hidden={cfg["per_rank_hidden"]}]', shape='ellipse', style='dashed,filled', fillcolor='yellow')
                                
                                # Residual 1
                                res1 = f'{layer_prefix}_res1'
                                rank.node(res1, f'Residual 1', shape='parallelogram', style='filled', fillcolor='lightyellow')
                                
                                # LayerNorm 2
                                norm2 = f'{layer_prefix}_norm2'
                                rank.node(norm2, f'LayerNorm 2\\nDevice: {device}', shape='ellipse', style='filled', fillcolor='lightgray')
                                
                                # FFN
                                ffn_up = f'{layer_prefix}_ffn_up'
                                ffn_gate = f'{layer_prefix}_ffn_gate'
                                ffn_down = f'{layer_prefix}_ffn_down'
                                
                                rank.node(ffn_up, f'FFN Up\\n[hidden={cfg["per_rank_hidden"]}]→[ffn={cfg["per_rank_ffn"]}]', shape='rectangle', style='filled', fillcolor='lightcyan')
                                rank.node(ffn_gate, f'FFN Gate\\n[hidden={cfg["per_rank_hidden"]}]→[ffn={cfg["per_rank_ffn"]}]', shape='rectangle', style='filled', fillcolor='lightcyan')
                                rank.node(ffn_down, f'FFN Down\\n[ffn={cfg["per_rank_ffn"]}]→[hidden={cfg["per_rank_hidden"]}]', shape='rectangle', style='filled', fillcolor='lightcyan')
                                
                                # All-reduce 2
                                ar2 = f'{layer_prefix}_ar2'
                                rank.node(ar2, f'All-Reduce 2\\n[hidden={cfg["per_rank_hidden"]}]', shape='ellipse', style='dashed,filled', fillcolor='yellow')
                                
                                # Residual 2
                                res2 = f'{layer_prefix}_res2'
                                rank.node(res2, f'Residual 2', shape='parallelogram', style='filled', fillcolor='lightyellow')
                                
                                # Connections
                                rank.edge(split, norm1)
                                rank.edge(norm1, qkv)
                                rank.edge(qkv, attn)
                                rank.edge(attn, ar1)
                                rank.edge(ar1, res1)
                                rank.edge(res1, norm2)
                                rank.edge(norm2, ffn_up)
                                rank.edge(ffn_up, ffn_gate)
                                rank.edge(ffn_gate, ffn_down)
                                rank.edge(ffn_down, ar2)
                                rank.edge(ar2, res2)
            
            # Pipeline communication
            for stage_idx in range(cfg['pipeline_parallel'] - 1):
                comm_node = f'{model_name}_comm_{stage_idx + 1}_{stage_idx + 2}'
                c.node(comm_node, f'Pipeline Comm\\nStage {stage_idx + 1} → {stage_idx + 2}\\n[batch={cfg["batch"]}, seq={cfg["seq"]}, hidden={cfg["hidden_size"]}]', 
                       shape='ellipse', style='dashed,filled', fillcolor='lightyellow')
                
                # Connect stages
                last_layer = (stage_idx + 1) * (cfg['layers'] // cfg['pipeline_parallel'])
                next_stage = stage_idx + 2
                if next_stage <= cfg['pipeline_parallel']:
                    next_layer = (stage_idx + 1) * (cfg['layers'] // cfg['pipeline_parallel']) + 1
                    for rank_idx in range(cfg['tensor_parallel']):
                        c.edge(f'{model_name}_S{stage_idx + 1}_R{rank_idx + 1}_L{last_layer}_res2', comm_node)
                        c.edge(comm_node, f'{model_name}_S{next_stage}_R{rank_idx + 1}_L{next_layer}_split')
            
            # Input/Output
            input_node = f'{model_name}_input'
            output_node = f'{model_name}_output'
            
            c.node(input_node, f'Input\\n[batch={cfg["batch"]}, seq={cfg["seq"]}, vocab={cfg["vocab"]}]',
                   shape='ellipse', style='filled', fillcolor='lightgreen')
            c.node(output_node, f'Output\\n[batch={cfg["batch"]}, seq={cfg["seq"]}, vocab={cfg["vocab"]}]',
                   shape='doubleoctagon', style='filled', fillcolor='lightgreen')
            
            # Connect input to first stage
            for rank_idx in range(cfg['tensor_parallel']):
                c.edge(input_node, f'{model_name}_S1_R{rank_idx + 1}_L1_split')
            
            # Connect last stage to output
            for rank_idx in range(cfg['tensor_parallel']):
                last_layer = cfg['layers']
                c.edge(f'{model_name}_S{cfg["pipeline_parallel"]}_R{rank_idx + 1}_L{last_layer}_res2', output_node)
    
    return dot

if __name__ == "__main__":
    dag = create_megatron_lm_dag()
    dag.render('../outputs/2025-10-30-09-17-39/megatron_lm', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/megatron_lm.dot')
    print("Megatron-LM DAG generated successfully")