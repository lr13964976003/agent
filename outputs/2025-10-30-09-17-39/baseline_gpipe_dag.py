#!/usr/bin/env python3
"""
GPipe Baseline DAG Generator
Shows micro-batch pipeline parallelism with uniform layer distribution
"""

import graphviz

def create_baseline_gpipe_dag():
    """Create GPipe baseline DAG with micro-batch pipeline"""
    
    dot = graphviz.Digraph('GPipe_Baseline', 
                           comment='GPipe Micro-Batch Pipeline',
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
            'total_layers': 32,
            'pipeline_stages': 4,
            'layers_per_stage': 8,
            'micro_batches': 4,
            'hidden_size': 4096,
            'num_heads': 32,
            'head_dim': 128,
            'ffn_hidden_size': 11008,
            'vocab_size': 32000,
            'batch_size': 6,
            'seq_len': 2048
        },
        'gpt3_2b': {
            'total_layers': 24,
            'pipeline_stages': 3,
            'layers_per_stage': 8,
            'micro_batches': 3,
            'hidden_size': 2048,
            'num_heads': 16,
            'head_dim': 128,
            'ffn_hidden_size': 8192,
            'vocab_size': 50257,
            'batch_size': 12,
            'seq_len': 2048
        }
    }
    
    # Define devices for GPipe (assuming 4/3 devices respectively)
    devices_llama = [f'GPU_{i+1}' for i in range(4)]
    devices_gpt3 = [f'GPU_{i+1}' for i in range(3)]
    
    for model_name, cfg in configs.items():
        devices = devices_llama if model_name == 'llama_7b' else devices_gpt3
        
        with dot.subgraph(name=f'cluster_{model_name}_gpipe') as c:
            c.attr(label=f'{model_name.upper()} GPipe Pipeline ({cfg["pipeline_stages"]} stages)', 
                   style='rounded,filled', 
                   fillcolor='lightyellow',
                   color='black')
            
            # Create pipeline stages
            for stage_idx, device in enumerate(devices):
                with c.subgraph(name=f'cluster_{model_name}_stage_{stage_idx + 1}') as stage:
                    stage.attr(label=f'Stage {stage_idx + 1}\\n{device}\\nLayers {stage_idx * cfg["layers_per_stage"] + 1}-{min((stage_idx + 1) * cfg["layers_per_stage"], cfg["total_layers"])}', 
                               style='rounded,filled', 
                               fillcolor='lightblue' if stage_idx % 2 == 0 else 'lightgreen',
                               color='black')
                    
                    # Create micro-batch processing
                    for batch_idx in range(cfg['micro_batches']):
                        batch_name = f'{model_name}_stage_{stage_idx + 1}_batch_{batch_idx + 1}'
                        
                        # Input to stage
                        input_node = f'{batch_name}_input'
                        stage.node(input_node, 
                                   f'Micro-batch {batch_idx + 1}\\n[batch={cfg["batch_size"]//cfg["micro_batches"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                                   shape='ellipse', style='filled', fillcolor='lightgreen')
                        
                        # Create layers for this stage
                        start_layer = stage_idx * cfg['layers_per_stage'] + 1
                        end_layer = min((stage_idx + 1) * cfg['layers_per_stage'], cfg['total_layers'])
                        
                        prev_node = input_node
                        for layer_idx in range(start_layer, end_layer + 1):
                            layer_nodes = []
                            
                            # QKV + Attention
                            qkv = f'{batch_name}_layer_{layer_idx}_qkv'
                            attn = f'{batch_name}_layer_{layer_idx}_attn'
                            attn_out = f'{batch_name}_layer_{layer_idx}_attn_out'
                            attn_res = f'{batch_name}_layer_{layer_idx}_attn_res'
                            
                            # FFN
                            ffn_up = f'{batch_name}_layer_{layer_idx}_ffn_up'
                            ffn_gate = f'{batch_name}_layer_{layer_idx}_ffn_gate'
                            ffn_down = f'{batch_name}_layer_{layer_idx}_ffn_down'
                            ffn_res = f'{batch_name}_layer_{layer_idx}_ffn_res'
                            
                            # LayerNorms
                            norm1 = f'{batch_name}_layer_{layer_idx}_norm1'
                            norm2 = f'{batch_name}_layer_{layer_idx}_norm2'
                            
                            # Create nodes
                            stage.node(norm1, f'L{layer_idx} Norm1\\nDevice: {device}',
                                       shape='ellipse', style='filled', fillcolor='lightgray')
                            
                            stage.node(qkv, f'L{layer_idx} QKV\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcoral')
                            
                            stage.node(attn, f'L{layer_idx} Attention\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcoral')
                            
                            stage.node(attn_out, f'L{layer_idx} Attn Out\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcoral')
                            
                            stage.node(attn_res, f'L{layer_idx} Attn+Res\\nDevice: {device}',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                            
                            stage.node(norm2, f'L{layer_idx} Norm2\\nDevice: {device}',
                                       shape='ellipse', style='filled', fillcolor='lightgray')
                            
                            stage.node(ffn_up, f'L{layer_idx} FFN Up\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcyan')
                            
                            stage.node(ffn_gate, f'L{layer_idx} FFN Gate\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcyan')
                            
                            stage.node(ffn_down, f'L{layer_idx} FFN Down\\nDevice: {device}',
                                       shape='rectangle', style='filled', fillcolor='lightcyan')
                            
                            stage.node(ffn_res, f'L{layer_idx} FFN+Res\\nDevice: {device}',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                            
                            # Connections within layer
                            stage.edge(prev_node, norm1)
                            stage.edge(norm1, qkv)
                            stage.edge(qkv, attn)
                            stage.edge(attn, attn_out)
                            stage.edge(attn_out, attn_res)
                            stage.edge(attn_res, norm2)
                            stage.edge(norm2, ffn_up)
                            stage.edge(ffn_up, ffn_gate)
                            stage.edge(ffn_gate, ffn_down)
                            stage.edge(ffn_down, ffn_res)
                            
                            prev_node = ffn_res
                        
                        # Output from stage
                        output_node = f'{batch_name}_output'
                        stage.node(output_node, 
                                   f'Stage {stage_idx + 1} Output\\n[batch={cfg["batch_size"]//cfg["micro_batches"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                                   shape='ellipse', style='filled', fillcolor='lightgreen')
                        stage.edge(prev_node, output_node)
            
            # Inter-stage communication
            for stage_idx in range(len(devices) - 1):
                for batch_idx in range(cfg['micro_batches']):
                    from_stage = stage_idx + 1
                    to_stage = stage_idx + 2
                    comm_node = f'{model_name}_comm_stage_{from_stage}_to_{to_stage}_batch_{batch_idx + 1}'
                    
                    c.node(comm_node, 
                           f'Communication\\nStage {from_stage} → {to_stage}\\nBatch {batch_idx + 1}\\n[batch={cfg["batch_size"]//cfg["micro_batches"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                           shape='ellipse', style='dashed,filled', fillcolor='lightyellow')
                    
                    c.edge(f'{model_name}_stage_{from_stage}_batch_{batch_idx + 1}_output', comm_node)
                    c.edge(comm_node, f'{model_name}_stage_{to_stage}_batch_{batch_idx + 1}_input')
            
            # Input/Output for model
            input_model = f'{model_name}_gpipe_input'
            output_model = f'{model_name}_gpipe_output'
            
            c.node(input_model, 
                   f'Model Input\\n{model_name.upper()}\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='ellipse', style='filled', fillcolor='orange')
            
            c.node(output_model, 
                   f'Model Output\\n{model_name.upper()}\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='doubleoctagon', style='filled', fillcolor='orange')
            
            # Connect first/last stages
            for batch_idx in range(cfg['micro_batches']):
                c.edge(input_model, f'{model_name}_stage_1_batch_{batch_idx + 1}_input')
                last_stage = cfg['pipeline_stages']
                c.edge(f'{model_name}_stage_{last_stage}_batch_{batch_idx + 1}_output', output_model)
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_gpipe_dag()
    dag.render('../outputs/2025-10-30-09-17-39/baseline_gpipe', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/baseline_gpipe.dot')
    print("GPipe Baseline DAG generated successfully")