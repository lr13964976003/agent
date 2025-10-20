import graphviz
import os

def create_baseline_dag():
    """Create baseline TP8 PP2 DAG"""
    dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', 
                          filename='moe_baseline_tp8_pp2.dot',
                          node_attr={'shape': 'rectangle'})
    dot.attr(rankdir='TB', bgcolor='white')
    
    # Input
    dot.node('input', 'Input\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
             shape='ellipse', fillcolor='lightgreen', style='filled')
    
    # Process each layer
    for layer in [1, 2, 3, 4]:
        stage = 0 if layer <= 2 else 1
        gpu_start = 0 if stage == 0 else 8
        gpu_range = f"{gpu_start}-{gpu_start+7}"
        
        # Layer input
        layer_input = f'layer_{layer}_input'
        dot.node(layer_input,
                f'Layer {layer} Input\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                shape='parallelogram', fillcolor='lightyellow', style='filled')
        
        # MHA components
        mha_norm = f'layer_{layer}_mha_norm'
        dot.node(mha_norm,
                f'MHA LayerNorm {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                fillcolor='lightcoral', style='filled')
        
        # TP split for MHA
        for tp_rank in range(8):
            gpu_id = gpu_start + tp_rank
            
            # QKV projections (column parallel)
            q_proj = f'layer_{layer}_q_{tp_rank}'
            k_proj = f'layer_{layer}_k_{tp_rank}' 
            v_proj = f'layer_{layer}_v_{tp_rank}'
            
            dot.node(q_proj,
                    f'Q Proj {layer}.{tp_rank}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                    fillcolor='lightblue', style='filled')
            dot.node(k_proj,
                    f'K Proj {layer}.{tp_rank}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                    fillcolor='lightblue', style='filled')
            dot.node(v_proj,
                    f'V Proj {layer}.{tp_rank}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                    fillcolor='lightblue', style='filled')
            
            # Attention
            attn = f'layer_{layer}_attn_{tp_rank}'
            dot.node(attn,
                    f'Attention {layer}.{tp_rank}\n[batch=1024, seq=10000, head=2, dim=512]\nGPU: {gpu_id}',
                    fillcolor='lightgreen', style='filled')
            
            # Output projection
            o_proj = f'layer_{layer}_o_{tp_rank}'
            dot.node(o_proj,
                    f'O Proj {layer}.{tp_rank}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                    fillcolor='lightblue', style='filled')
            
            # Connect MHA flow
            dot.edge(mha_norm, q_proj)
            dot.edge(mha_norm, k_proj)
            dot.edge(mha_norm, v_proj)
            dot.edge(q_proj, attn)
            dot.edge(k_proj, attn)
            dot.edge(v_proj, attn)
            dot.edge(attn, o_proj)
        
        # All-reduce
        attn_reduce = f'layer_{layer}_attn_reduce'
        dot.node(attn_reduce,
                f'All-Reduce {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                shape='parallelogram', fillcolor='orange', style='filled')
        
        # Residual
        attn_res = f'layer_{layer}_attn_res'
        dot.node(attn_res,
                f'Residual Add {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                fillcolor='lightyellow', style='filled')
        
        # Expert components
        exp_norm = f'layer_{layer}_exp_norm'
        dot.node(exp_norm,
                f'Expert Norm {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                fillcolor='lightcoral', style='filled')
        
        # 8 experts per GPU * 8 GPUs = 64 total experts per layer
        expert_start = 0 if stage == 0 else 8
        for gpu_offset in range(8):
            gpu_id = gpu_start + gpu_offset
            for expert_id in range(8):
                actual_expert = expert_start * 8 + expert_id + gpu_offset * 8
                
                gate = f'layer_{layer}_gate_{actual_expert}'
                expert = f'layer_{layer}_expert_{actual_expert}'
                
                dot.node(gate,
                        f'Gate {actual_expert}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                        shape='diamond', fillcolor='yellow', style='filled')
                
                dot.node(expert,
                        f'Expert {actual_expert}\n[batch=1024, seq=10000, hidden=1024]\nGPU: {gpu_id}',
                        fillcolor='lightpink', style='filled')
                
                dot.edge(exp_norm, gate)
                dot.edge(gate, expert, style='dashed')
        
        # Expert aggregation
        exp_agg = f'layer_{layer}_exp_agg'
        dot.node(exp_agg,
                f'Expert Agg {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                shape='parallelogram', fillcolor='orange', style='filled')
        
        exp_res = f'layer_{layer}_exp_res'
        dot.node(exp_res,
                f'Expert Residual {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: {gpu_range}',
                fillcolor='lightyellow', style='filled')
        
        # Connect all components
        if layer == 1:
            dot.edge('input', layer_input)
        else:
            if layer == 3:
                # Pipeline communication
                pipeline_comm = 'pipeline_comm_2_to_3'
                dot.node(pipeline_comm,
                        'Pipeline Comm\n[batch=1024, seq=10000, hidden=8192]\nGPU: 7 to 8',
                        shape='ellipse', fillcolor='lightgreen', style='filled')
                dot.edge(f'layer_2_exp_res', pipeline_comm)
                dot.edge(pipeline_comm, layer_input)
            else:
                dot.edge(f'layer_{layer-1}_exp_res', layer_input)
        
        # Connect MHA chain
        for tp_rank in range(8):
            dot.edge(f'layer_{layer}_o_{tp_rank}', attn_reduce)
        dot.edge(attn_reduce, attn_res)
        dot.edge(layer_input, attn_res)
        
        # Connect expert chain
        expert_start = 0 if stage == 0 else 8
        for gpu_offset in range(8):
            gpu_id = gpu_start + gpu_offset
            for expert_id in range(8):
                actual_expert = expert_start * 8 + expert_id + gpu_offset * 8
                dot.edge(f'layer_{layer}_expert_{actual_expert}', exp_agg)
        
        dot.edge(attn_res, exp_norm)
        dot.edge(exp_agg, exp_res)
        dot.edge(attn_res, exp_res)
    
    # Output
    dot.edge('layer_4_exp_res', 'output')
    
    return dot

def create_ep16_dag():
    """Create EP16 proposed DAG"""
    dot = graphviz.Digraph('MoE_EP16_Proposed', 
                          filename='moe_ep16_proposed.dot',
                          node_attr={'shape': 'rectangle'})
    dot.attr(rankdir='TB', bgcolor='white')
    
    # Input
    dot.node('input', 'Input\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
             shape='ellipse', fillcolor='lightgreen', style='filled')
    
    # Process each layer
    for layer in [1, 2, 3, 4]:
        # Layer input
        layer_input = f'layer_{layer}_input'
        dot.node(layer_input,
                f'Layer {layer} Input\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                shape='ellipse', fillcolor='lightgreen', style='filled')
        
        # MHA components (replicated)
        mha_norm = f'layer_{layer}_mha_norm'
        dot.node(mha_norm,
                f'MHA LayerNorm {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                fillcolor='lightcoral', style='filled')
        
        # Full MHA (not split)
        mha_q = f'layer_{layer}_mha_q'
        mha_k = f'layer_{layer}_mha_k'
        mha_v = f'layer_{layer}_mha_v'
        mha_attn = f'layer_{layer}_mha_attn'
        mha_o = f'layer_{layer}_mha_o'
        
        dot.node(mha_q, f'MHA Q Proj {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all', fillcolor='lightblue', style='filled')
        dot.node(mha_k, f'MHA K Proj {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all', fillcolor='lightblue', style='filled')
        dot.node(mha_v, f'MHA V Proj {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all', fillcolor='lightblue', style='filled')
        dot.node(mha_attn, f'MHA {layer}\n[batch=1024, seq=10000, heads=16, dim=512]\nGPU: all', fillcolor='lightgreen', style='filled')
        dot.node(mha_o, f'MHA O Proj {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all', fillcolor='lightblue', style='filled')
        
        # MHA residual
        mha_res = f'layer_{layer}_mha_res'
        dot.node(mha_res,
                f'MHA Residual {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                fillcolor='lightyellow', style='filled')
        
        # Expert components
        exp_norm = f'layer_{layer}_exp_norm'
        dot.node(exp_norm,
                f'Expert Norm {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                fillcolor='lightcoral', style='filled')
        
        # Global gating
        global_gate = f'layer_{layer}_global_gate'
        dot.node(global_gate,
                f'Global Gate {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                shape='diamond', fillcolor='yellow', style='filled')
        
        # Token distribution
        token_dist = f'layer_{layer}_token_dist'
        dot.node(token_dist,
                f'Token Distribution {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all to 0-15',
                shape='parallelogram', fillcolor='orange', style='filled')
        
        # 16 Experts - one per GPU
        for expert_id in range(16):
            gpu_id = expert_id
            
            # Token receive
            token_recv = f'layer_{layer}_expert{expert_id}_recv'
            dot.node(token_recv,
                    f'Expert {expert_id} Recv\n[batch=variable, seq=variable, hidden=8192]\nGPU: {gpu_id}',
                    shape='parallelogram', fillcolor='lightgreen', style='filled')
            
            # Expert gate
            expert_gate = f'layer_{layer}_expert{expert_id}_gate'
            dot.node(expert_gate,
                    f'Expert {expert_id} Gate\n[batch=variable, seq=variable, hidden=8192]\nGPU: {gpu_id}',
                    shape='diamond', fillcolor='yellow', style='filled')
            
            # Expert MLP
            expert = f'layer_{layer}_expert{expert_id}'
            dot.node(expert,
                    f'Expert {expert_id}\n[batch=variable, seq=variable, hidden=8192]\nGPU: {gpu_id}',
                    fillcolor='lightpink', style='filled')
            
            # Token send
            token_send = f'layer_{layer}_expert{expert_id}_send'
            dot.node(token_send,
                    f'Expert {expert_id} Send\n[batch=variable, seq=variable, hidden=8192]\nGPU: {gpu_id}',
                    shape='parallelogram', fillcolor='lightgreen', style='filled')
            
            # Connect expert flow
            dot.edge(token_dist, token_recv)
            dot.edge(token_recv, expert_gate)
            dot.edge(expert_gate, expert, style='dashed')
            dot.edge(expert, token_send)
        
        # Expert aggregation
        exp_agg = f'layer_{layer}_exp_agg'
        dot.node(exp_agg,
                f'Expert Aggregation {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                shape='parallelogram', fillcolor='orange', style='filled')
        
        # Expert residual
        exp_res = f'layer_{layer}_exp_res'
        dot.node(exp_res,
                f'Expert Residual {layer}\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                fillcolor='lightyellow', style='filled')
        
        # Layer output
        layer_output = f'layer_{layer}_output'
        dot.node(layer_output,
                f'Layer {layer} Output\n[batch=1024, seq=10000, hidden=8192]\nGPU: all',
                shape='ellipse', fillcolor='lightgreen', style='filled')
        
        # Connect all components
        if layer == 1:
            dot.edge('input', layer_input)
        else:
            dot.edge(f'layer_{layer-1}_output', layer_input)
        
        # MHA chain
        dot.edge(layer_input, mha_norm)
        dot.edge(mha_norm, mha_q)
        dot.edge(mha_norm, mha_k)
        dot.edge(mha_norm, mha_v)
        dot.edge(mha_q, mha_attn)
        dot.edge(mha_k, mha_attn)
        dot.edge(mha_v, mha_attn)
        dot.edge(mha_attn, mha_o)
        dot.edge(mha_o, mha_res)
        dot.edge(layer_input, mha_res)
        
        # Expert chain
        dot.edge(mha_res, exp_norm)
        dot.edge(exp_norm, global_gate)
        dot.edge(global_gate, token_dist)
        dot.edge(exp_norm, token_dist)
        
        for expert_id in range(16):
            dot.edge(f'layer_{layer}_expert{expert_id}_send', exp_agg)
        
        dot.edge(exp_agg, exp_res)
        dot.edge(mha_res, exp_res)
        dot.edge(exp_res, layer_output)
    
    # Final output
    dot.edge('layer_4_output', 'output')
    
    return dot

if __name__ == '__main__':
    # Create baseline DAG
    baseline_dot = create_baseline_dag()
    baseline_dot.render('../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2', format='svg', cleanup=False)
    baseline_dot.save('../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2.dot')
    
    # Create EP16 DAG
    ep16_dot = create_ep16_dag()
    ep16_dot.render('../outputs/2025-10-19-21-59-45/moe_ep16_proposed', format='svg', cleanup=False)
    ep16_dot.save('../outputs/2025-10-19-21-59-45/moe_ep16_proposed.dot')
    
    print("Generated DAGs:")
    print("- Baseline: ../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2.dot")
    print("- EP16: ../outputs/2025-10-19-21-59-45/moe_ep16_proposed.dot")