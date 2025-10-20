import graphviz
from typing import Dict, List, Tuple

class MoEEP16DAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph('MoE_EP16_Proposed', 
                                  filename='moe_ep16_proposed.dag',
                                  node_attr={'shape': 'rectangle'})
        self.dot.attr(rankdir='TB', bgcolor='white')
        self.setup_styles()
    
    def setup_styles(self):
        self.dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        self.dot.attr('edge', arrowhead='normal')
    
    def add_input_layer(self):
        # Input node
        self.dot.node('input', 
                     'Input\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     shape='ellipse', fillcolor='lightgreen')
    
    def add_expert_layer(self, layer_num: int):
        
        # Layer input
        layer_input = f'layer_{layer_num}_input'
        self.dot.node(layer_input,
                     f'Layer {layer_num} Input\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     shape='ellipse', fillcolor='lightgreen')
        
        # Multi-Head Attention for the layer
        mha_norm = f'layer_{layer_num}_mha_norm'
        self.dot.node(mha_norm,
                     f'Layer {layer_num} MHA LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     fillcolor='lightcoral')
        
        # MHA components (replicated across all GPUs)
        mha_q_proj = f'layer_{layer_num}_q_proj'
        self.dot.node(mha_q_proj,
                     f'MHA Q Projection\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nGPU: all GPUs',
                     fillcolor='lightblue')
        
        mha_k_proj = f'layer_{layer_num}_k_proj'
        self.dot.node(mha_k_proj,
                     f'MHA K Projection\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nGPU: all GPUs',
                     fillcolor='lightblue')
        
        mha_v_proj = f'layer_{layer_num}_v_proj'
        self.dot.node(mha_v_proj,
                     f'MHA V Projection\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nGPU: all GPUs',
                     fillcolor='lightblue')
        
        mha_attention = f'layer_{layer_num}_attention'
        self.dot.node(mha_attention,
                     f'MHA Computation\nInput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nOutput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nGPU: all GPUs',
                     fillcolor='lightgreen')
        
        mha_o_proj = f'layer_{layer_num}_o_proj'
        self.dot.node(mha_o_proj,
                     f'MHA O Projection\nInput: [batch_size=1024, seq_len=10000, heads=16, head_dim=512]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     fillcolor='lightblue')
        
        # Attention residual
        attn_residual = f'layer_{layer_num}_attn_residual'
        self.dot.node(attn_residual,
                     f'Attention Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192] x2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     fillcolor='lightyellow')
        
        # Expert Layer Norm (replicated)
        expert_norm = f'layer_{layer_num}_expert_norm'
        self.dot.node(expert_norm,
                     f'Expert LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     fillcolor='lightcoral')
        
        # Global gating mechanism
        global_gate = f'layer_{layer_num}_global_gate'
        self.dot.node(global_gate,
                     f'Global Gating\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, expert_scores=16]\nGPU: all GPUs',
                     shape='diamond', fillcolor='yellow')
        
        # Token distribution across 16 experts
        token_distribute = f'layer_{layer_num}_token_distribute'
        self.dot.node(token_distribute,
                     f'Token Distribution\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=variable, seq_len=variable, hidden=8192] x16\nGPU: 0-15',
                     shape='parallelogram', fillcolor='orange')
        
        # 16 Experts - one per GPU
        for expert_id in range(16):
            gpu_id = expert_id
            
            # Token receive from distribution
            token_recv = f'layer_{layer_num}_expert{expert_id}_recv'
            self.dot.node(token_recv,
                         f'Expert {expert_id} Token Receive\nInput: [batch_size=variable, seq_len=variable, hidden=8192]\nOutput: [batch_size=variable, seq_len=variable, hidden=8192]\nGPU: {gpu_id}',
                         shape='parallelogram', fillcolor='lightgreen')
            
            # Expert gate
            expert_gate = f'layer_{layer_num}_expert{expert_id}_gate'
            self.dot.node(expert_gate,
                         f'Expert {expert_id} Gate\nInput: [batch_size=variable, seq_len=variable, hidden=8192]\nOutput: [batch_size=variable, seq_len=variable, gate=1]\nGPU: {gpu_id}',
                         shape='diamond', fillcolor='yellow')
            
            # Expert MLP
            expert_mlp = f'layer_{layer_num}_expert{expert_id}_mlp'
            self.dot.node(expert_mlp,
                         f'Expert {expert_id} MLP\nInput: [batch_size=variable, seq_len=variable, hidden=8192]\nOutput: [batch_size=variable, seq_len=variable, hidden=8192]\nGPU: {gpu_id}',
                         fillcolor='lightpink')
            
            # Token send back to aggregation
            token_send = f'layer_{layer_num}_expert{expert_id}_send'
            self.dot.node(token_send,
                         f'Expert {expert_id} Token Send\nInput: [batch_size=variable, seq_len=variable, hidden=8192]\nOutput: [batch_size=variable, seq_len=variable, hidden=8192]\nGPU: {gpu_id}',
                         shape='parallelogram', fillcolor='lightgreen')
            
            # Gate connection (dashed)
            self.dot.edge(expert_gate, expert_mlp, style='dashed', label='route tokens')
            self.dot.edge(token_recv, expert_gate)
            self.dot.edge(expert_mlp, token_send)
        
        # Expert aggregation
        expert_agg = f'layer_{layer_num}_expert_agg'
        self.dot.node(expert_agg,
                     f'Expert Output Aggregation\nInput: [batch_size=variable, seq_len=variable, hidden=8192] x16\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     shape='parallelogram', fillcolor='orange')
        
        # Expert residual
        expert_residual = f'layer_{layer_num}_expert_residual'
        self.dot.node(expert_residual,
                     f'Expert Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192] x2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     fillcolor='lightyellow')
        
        # Layer output
        layer_output = f'layer_{layer_num}_output'
        self.dot.node(layer_output,
                     f'Layer {layer_num} Output\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs',
                     shape='ellipse', fillcolor='lightgreen')
        
        # Connect MHA components
        self.dot.edge(layer_input, mha_norm)
        self.dot.edge(mha_norm, mha_q_proj)
        self.dot.edge(mha_norm, mha_k_proj)
        self.dot.edge(mha_norm, mha_v_proj)
        self.dot.edge(mha_q_proj, mha_attention)
        self.dot.edge(mha_k_proj, mha_attention)
        self.dot.edge(mha_v_proj, mha_attention)
        self.dot.edge(mha_attention, mha_o_proj)
        self.dot.edge(mha_o_proj, attn_allreduce)
        self.dot.edge(attn_allreduce, attn_residual)
        self.dot.edge(layer_input, attn_residual)  # residual connection
        
        # Connect Expert components
        self.dot.edge(attn_residual, expert_norm)
        self.dot.edge(expert_norm, global_gate)
        self.dot.edge(global_gate, token_distribute)
        self.dot.edge(expert_norm, token_distribute)  # tokens to distribute
        
        # Connect expert pipeline
        for expert_id in range(16):
            self.dot.edge(token_distribute, f'layer_{layer_num}_expert{expert_id}_recv')
            self.dot.edge(f'layer_{layer_num}_expert{expert_id}_send', expert_agg)
        
        self.dot.edge(expert_agg, expert_residual)
        self.dot.edge(attn_residual, expert_residual)  # residual connection
        self.dot.edge(expert_residual, layer_output)
        
        return layer_output
    
    def add_output_layer(self):
        self.dot.node('output',
                     'Final Output\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, vocab_size]\nGPU: all GPUs',
                     shape='ellipse', fillcolor='lightgreen')
    
    def generate_dag(self):
        self.add_input_layer()
        
        # Add 4 layers
        prev_output = 'input'
        for layer_num in range(1, 5):
            layer_output = self.add_expert_layer(layer_num)
            self.dot.edge(prev_output, f'layer_{layer_num}_input')
            prev_output = layer_output
        
        self.add_output_layer()
        self.dot.edge(prev_output, 'output')
        
        # Save files
        self.dot.render('../outputs/2025-10-19-21-59-45/moe_ep16_proposed', format='svg', cleanup=False)
        self.dot.save('../outputs/2025-10-19-21-59-45/moe_ep16_proposed.dot')
        
        return '../outputs/2025-10-19-21-59-45/moe_ep16_proposed.dot'

if __name__ == '__main__':
    generator = MoEEP16DAGGenerator()
    dag_path = generator.generate_dag()
    print(f"Generated EP16 DAG: {dag_path}")