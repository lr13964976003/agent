#!/usr/bin/env python3
"""
Megatron-LM Baseline DAG Generator
Shows tensor parallelism + GPipe combination
"""

import graphviz

def create_baseline_megatron_lm_dag():
    """Create Megatron-LM baseline DAG with tensor + pipeline parallelism"""
    
    dot = graphviz.Digraph('Megatron_LM_Baseline', 
                           comment='Megatron-LM Tensor + Pipeline Parallelism',
                           format='svg',
                           graph_attr={
                               'rankdir': 'TB',
                               'compound': 'true',
                               'ranksep': '1.5',
                               'nodesep': '0.2'
                           })
    
    # Configuration for both models
    configs = {
        'llama_7b': {
            'tensor_parallel': 2,
            'pipeline_parallel': 3,
            'total_layers': 32,
            'layers_per_pipeline': 11,
            'hidden_size': 4096,
            'hidden_per_tensor': 2048,
            'num_heads': 32,
            'heads_per_tensor': 16,
            'head_dim': 128,
            'ffn_hidden_size': 11008,
            'ffn_per_tensor': 5504,
            'vocab_size': 32000,
            'batch_size': 6,
            'seq_len': 2048,
            'devices': ['GPU_1', 'GPU_2', 'GPU_3', 'GPU_4', 'GPU_5', 'GPU_6']
        },
        'gpt3_2b': {
            'tensor_parallel': 2,
            'pipeline_parallel': 3,
            'total_layers': 24,
            'layers_per_pipeline': 8,
            'hidden_size': 2048,
            'hidden_per_tensor': 1024,
            'num_heads': 16,
            'heads_per_tensor': 8,
            'head_dim': 128,
            'ffn_hidden_size': 8192,
            'ffn_per_tensor': 4096,
            'vocab_size': 50257,
            'batch_size': 12,
            'seq_len': 2048,
            'devices': ['GPU_1', 'GPU_2', 'GPU_3', 'GPU_4', 'GPU_5', 'GPU_6']
        }
    }
    
    for model_name, cfg in configs.items():
        with dot.subgraph(name=f'cluster_{model_name}_megatron') as c:
            c.attr(label=f'{model_name.upper()} Megatron-LM\\nTensor ×{cfg["tensor_parallel"]} Pipeline ×{cfg["pipeline_parallel"]}', 
                   style='rounded,filled', 
                   fillcolor='lightyellow',
                   color='black')
            
            # Create tensor parallel groups
            for pipeline_stage in range(cfg['pipeline_parallel']):
                stage_device = cfg['devices'][pipeline_stage]
                
                with c.subgraph(name=f'cluster_{model_name}_pipeline_{pipeline_stage + 1}') as pipeline:
                    pipeline.attr(label=f'Pipeline Stage {pipeline_stage + 1}\\n{stage_device}\\nLayers {pipeline_stage * cfg["layers_per_pipeline"] + 1}-{min((pipeline_stage + 1) * cfg["layers_per_pipeline"], cfg["total_layers"])}', 
                                  style='rounded,filled', 
                                  fillcolor='lightblue' if pipeline_stage % 2 == 0 else 'lightgreen',
                                  color='black')
                    
                    # Create tensor parallel ranks
                    for tensor_rank in range(cfg['tensor_parallel']):
                        device_name = cfg['devices'][pipeline_stage * cfg['tensor_parallel'] + tensor_rank]
                        
                        with pipeline.subgraph(name=f'cluster_{model_name}_stage_{pipeline_stage + 1}_rank_{tensor_rank + 1}') as rank:
                            rank.attr(label=f'Tensor Rank {tensor_rank + 1}\\n{device_name}', 
                                      style='rounded,filled', 
                                      fillcolor='lightcoral' if tensor_rank % 2 == 0 else 'lightcyan',
                                      color='black')
                            
                            # Create layers with tensor parallel operations
                            start_layer = pipeline_stage * cfg['layers_per_pipeline'] + 1
                            end_layer = min((pipeline_stage + 1) * cfg['layers_per_pipeline'], cfg['total_layers'])
                            
                            for layer_idx in range(start_layer, end_layer + 1):
                                layer_prefix = f'{model_name}_p{pipeline_stage + 1}_t{tensor_rank + 1}_l{layer_idx}'
                                
                                # Tensor-parallel operations
                                input_split = f'{layer_prefix}_input_split'
                                qkv_split = f'{layer_prefix}_qkv_split'
                                attn_allreduce = f'{layer_prefix}_attn_allreduce'
                                ffn_split = f'{layer_prefix}_ffn_split'
                                ffn_allreduce = f'{layer_prefix}_ffn_allreduce'
                                
                                # QKV for tensor parallel
                                qkv_rank = f'{layer_prefix}_qkv'
                                attn_calc = f'{layer_prefix}_attn'
                                attn_out = f'{layer_prefix}_attn_out'
                                attn_res = f'{layer_prefix}_attn_res'
                                
                                # FFN for tensor parallel
                                ffn_up = f'{layer_prefix}_ffn_up'
                                ffn_gate = f'{layer_prefix}_ffn_gate'
                                ffn_down = f'{layer_prefix}_ffn_down'
                                ffn_res = f'{layer_prefix}_ffn_res'
                                
                                # LayerNorms
                                norm1 = f'{layer_prefix}_norm1'
                                norm2 = f'{layer_prefix}_norm2'
                                
                                # Communication nodes
                                rank.node(input_split, 
                                          f'Input Split\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='parallelogram', style='filled', fillcolor='orange')
                                
                                rank.node(qkv_split, 
                                          f'QKV Split\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, heads={cfg["heads_per_tensor"]}, head_dim={cfg["head_dim"]}]',
                                          shape='parallelogram', style='filled', fillcolor='orange')
                                
                                rank.node(attn_allreduce, 
                                          f'Attention All-Reduce\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='ellipse', style='dashed,filled', fillcolor='yellow')
                                
                                rank.node(ffn_allreduce, 
                                          f'FFN All-Reduce\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='ellipse', style='dashed,filled', fillcolor='yellow')
                                
                                # Computation nodes
                                rank.node(norm1, f'LayerNorm 1\\nDevice: {device_name}',
                                          shape='ellipse', style='filled', fillcolor='lightgray')
                                
                                rank.node(qkv_rank, f'QKV Projection\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, heads={cfg["heads_per_tensor"]}, head_dim={cfg["head_dim"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcoral')
                                
                                rank.node(attn_calc, f'Multi-Head Attention\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, heads={cfg["heads_per_tensor"]}, head_dim={cfg["head_dim"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcoral')
                                
                                rank.node(attn_out, f'Attention Output\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcoral')
                                
                                rank.node(attn_res, f'Attention+Residual\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='parallelogram', style='filled', fillcolor='lightyellow')
                                
                                rank.node(norm2, f'LayerNorm 2\\nDevice: {device_name}',
                                          shape='ellipse', style='filled', fillcolor='lightgray')
                                
                                rank.node(ffn_up, f'FFN Up\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_per_tensor"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcyan')
                                
                                rank.node(ffn_gate, f'FFN Gate\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_per_tensor"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcyan')
                                
                                rank.node(ffn_down, f'FFN Down\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_per_tensor"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='rectangle', style='filled', fillcolor='lightcyan')
                                
                                rank.node(ffn_res, f'FFN+Residual\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_per_tensor"]}]',
                                          shape='parallelogram', style='filled', fillcolor='lightyellow')
                                
                                # Connections within layer
                                rank.edge(input_split, norm1)
                                rank.edge(norm1, qkv_split)
                                rank.edge(qkv_split, qkv_rank)
                                rank.edge(qkv_rank, attn_calc)
                                rank.edge(attn_calc, attn_out)
                                rank.edge(attn_out, attn_allreduce)
                                rank.edge(attn_allreduce, attn_res)
                                rank.edge(attn_res, norm2)
                                rank.edge(norm2, ffn_split)
                                rank.edge(ffn_split, ffn_up)
                                rank.edge(ffn_up, ffn_gate)
                                rank.edge(ffn_gate, ffn_down)
                                rank.edge(ffn_down, ffn_allreduce)
                                rank.edge(ffn_allreduce, ffn_res)
            
            # Pipeline communication
            for pipeline_stage in range(cfg['pipeline_parallel'] - 1):
                comm_node = f'{model_name}_pipeline_comm_{pipeline_stage + 1}_to_{pipeline_stage + 2}'
                c.node(comm_node, 
                       f'Pipeline Communication\\nStage {pipeline_stage + 1} → {pipeline_stage + 2}\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='ellipse', style='dashed,filled', fillcolor='lightyellow')
                
                # Connect tensor-parallel outputs to pipeline communication
                for tensor_rank in range(cfg['tensor_parallel']):
                    from_stage = pipeline_stage + 1
                    to_stage = pipeline_stage + 2
                    start_layer_next = to_stage * cfg['layers_per_pipeline'] + 1
                    
                    if start_layer_next <= cfg['total_layers']:
                        c.edge(f'{model_name}_p{from_stage}_t{tensor_rank + 1}_l{from_stage * cfg["layers_per_pipeline"]}_ffn_res', 
                               comm_node)
                        c.edge(comm_node, 
                               f'{model_name}_p{to_stage}_t{tensor_rank + 1}_l{start_layer_next}_input_split')
            
            # Model input/output
            input_model = f'{model_name}_megatron_input'
            output_model = f'{model_name}_megatron_output'
            
            c.node(input_model, 
                   f'Model Input\\n{model_name.upper()}\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='ellipse', style='filled', fillcolor='orange')
            
            c.node(output_model, 
                   f'Model Output\\n{model_name.upper()}\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='doubleoctagon', style='filled', fillcolor='orange')
            
            # Connect input to first stage
            for tensor_rank in range(cfg['tensor_parallel']):
                c.edge(input_model, 
                       f'{model_name}_p1_t{tensor_rank + 1}_l1_input_split')
            
            # Connect last stage to output
            last_stage = cfg['pipeline_parallel']
            for tensor_rank in range(cfg['tensor_parallel']):
                c.edge(f'{model_name}_p{last_stage}_t{tensor_rank + 1}_l{cfg["total_layers"]}_ffn_res', 
                       output_model)
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_megatron_lm_dag()
    dag.render('../outputs/2025-10-30-09-17-39/baseline_megatron_lm', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/baseline_megatron_lm.dot')
    print("Megatron-LM Baseline DAG generated successfully")