#!/usr/bin/env python3
"""
Baseline Sequential DAG Generator
Shows uniform layer distribution without parallelism
"""

import graphviz

def create_baseline_sequential_dag():
    """Create baseline DAG showing sequential execution"""
    
    dot = graphviz.Digraph('Baseline_Sequential', 
                           comment='Baseline Sequential Execution',
                           format='svg',
                           graph_attr={
                               'rankdir': 'TB',
                               'compound': 'true',
                               'ranksep': '0.8',
                               'nodesep': '0.1'
                           })
    
    # LLaMA-7B and GPT3-2B configurations
    configs = {
        'llama_7b': {
            'layers': 32,
            'hidden_size': 4096,
            'num_heads': 32,
            'head_dim': 128,
            'ffn_hidden_size': 11008,
            'vocab_size': 32000,
            'batch_size': 6,
            'seq_len': 2048,
            'device': 'Single_GPU'
        },
        'gpt3_2b': {
            'layers': 24,
            'hidden_size': 2048,
            'num_heads': 16,
            'head_dim': 128,
            'ffn_hidden_size': 8192,
            'vocab_size': 50257,
            'batch_size': 12,
            'seq_len': 2048,
            'device': 'Single_GPU'
        }
    }
    
    for model_name, cfg in configs.items():
        with dot.subgraph(name=f'cluster_{model_name}') as c:
            c.attr(label=f'{model_name.upper()} Sequential Execution', 
                   style='rounded,filled', 
                   fillcolor='lightyellow',
                   color='black')
            
            # Input and output nodes
            input_node = f'{model_name}_input'
            embedding_node = f'{model_name}_embedding'
            output_proj = f'{model_name}_output_proj'
            final_output = f'{model_name}_final_output'
            
            c.node(input_node, 
                   f'Input\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='ellipse', style='filled', fillcolor='lightgreen')
            
            c.node(embedding_node, 
                   f'Embedding\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            c.node(output_proj, 
                   f'Output Projection\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='rectangle', style='filled', fillcolor='lightblue')
            
            c.node(final_output, 
                   f'Final Output\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, vocab={cfg["vocab_size"]}]',
                   shape='doubleoctagon', style='filled', fillcolor='lightgreen')
            
            # Create all layers
            for layer_idx in range(1, cfg['layers'] + 1):
                layer_id = f'{model_name}_layer_{layer_idx}'
                
                # Multi-Head Attention
                qkv_proj = f'{layer_id}_qkv_proj'
                attn_calc = f'{layer_id}_attn_calc'
                attn_out = f'{layer_id}_attn_out'
                attn_res = f'{layer_id}_attn_res'
                
                # FFN
                ffn_up = f'{layer_id}_ffn_up'
                ffn_gate = f'{layer_id}_ffn_gate'
                ffn_down = f'{layer_id}_ffn_down'
                ffn_res = f'{layer_id}_ffn_res'
                
                # LayerNorm
                norm1 = f'{layer_id}_norm1'
                norm2 = f'{layer_id}_norm2'
                
                # Create nodes
                c.node(qkv_proj, 
                       f'L{layer_idx} QKV Proj\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, heads={cfg["num_heads"]}, head_dim={cfg["head_dim"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                c.node(attn_calc, 
                       f'L{layer_idx} Attention\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, heads={cfg["num_heads"]}, head_dim={cfg["head_dim"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                c.node(attn_out, 
                       f'L{layer_idx} Attn Output\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                c.node(attn_res, 
                       f'L{layer_idx} Attn+Residual\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                c.node(ffn_up, 
                       f'L{layer_idx} FFN Up\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_hidden_size"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_gate, 
                       f'L{layer_idx} FFN Gate\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_hidden_size"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_down, 
                       f'L{layer_idx} FFN Down\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, ffn={cfg["ffn_hidden_size"]}] → [batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_res, 
                       f'L{layer_idx} FFN+Residual\\n[batch={cfg["batch_size"]}, seq={cfg["seq_len"]}, hidden={cfg["hidden_size"]}]',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                c.node(norm1, f'L{layer_idx} LayerNorm 1',
                       shape='ellipse', style='filled', fillcolor='lightgray')
                
                c.node(norm2, f'L{layer_idx} LayerNorm 2',
                       shape='ellipse', style='filled', fillcolor='lightgray')
                
                # Connections
                if layer_idx == 1:
                    c.edge(embedding_node, norm1)
                else:
                    c.edge(f'{model_name}_layer_{layer_idx-1}_ffn_res', norm1)
                
                c.edge(norm1, qkv_proj)
                c.edge(qkv_proj, attn_calc)
                c.edge(attn_calc, attn_out)
                c.edge(attn_out, attn_res)
                c.edge(attn_res, norm2)
                c.edge(norm2, ffn_up)
                c.edge(ffn_up, ffn_gate)
                c.edge(ffn_gate, ffn_down)
                c.edge(ffn_down, ffn_res)
                
                if layer_idx == cfg['layers']:
                    c.edge(ffn_res, output_proj)
    
    # Connections for input/output
    dot.edge('llama_7b_input', 'llama_7b_embedding')
    dot.edge('llama_7b_output_proj', 'llama_7b_final_output')
    
    dot.edge('gpt3_2b_input', 'gpt3_2b_embedding')
    dot.edge('gpt3_2b_output_proj', 'gpt3_2b_final_output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_sequential_dag()
    dag.render('../outputs/2025-10-30-09-17-39/baseline_sequential', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/baseline_sequential.dot')
    print("Baseline Sequential DAG generated successfully")