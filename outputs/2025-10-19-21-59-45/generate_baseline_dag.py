import graphviz
from typing import Dict, List, Tuple

class MoEBaselineDAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', 
                                  filename='moe_baseline_tp8_pp2.dag',
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
    
    def add_layer_structure(self, layer_num: int, stage: int, gpu_start: int):
        
        # Layer input aggregation
        layer_input = f'layer_{layer_num}_input'
        self.dot.node(layer_input,
                     f'Layer {layer_num} Input Aggregation\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     shape='parallelogram', fillcolor='lightyellow')
        
        # Multi-Head ANNNention
        mha_norm = f'layer_{layer_num}_mha_norm'
        self.dot.node(mha_norm,
                     f'Layer {layer_num} MHA LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     fillcolor='lightcoral')
        
        # MHA split across 8 GPUs with tensor parallelism
        for tp_rank in range(8):
            gpu_id = gpu_start + tp_rank
            
            # Query projection (column parallel)
            q_proj = f'layer_{layer_num}_q_proj_{tp_rank}'
            self.dot.node(q_proj,
                         f'Q Projection {tp_rank}\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nOutput: [batch_size=1024, seq_len=10000, head_dim*heads=8192/8=1024]\nGPU: {gpu_id}',
                         fillcolor='lightblue')
            
            # Key projection (column parallel)
            k_proj = f'layer_{layer_num}_k_proj_{tp_rank}'
            self.dot.node(k_proj,
                         f'K Projection {tp_rank}\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nOutput: [batch_size=1024, seq_len=10000, head_dim*heads=8192/8=1024]\nGPU: {gpu_id}',
                         fillcolor='lightblue')
            
            # Value projection (column parallel)
            v_proj = f'layer_{layer_num}_v_proj_{tp_rank}'
            self.dot.node(v_proj,
                         f'V Projection {tp_rank}\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nOutput: [batch_size=1024, seq_len=10000, head_dim*heads=8192/8=1024]\nGPU: {gpu_id}',
                         fillcolor='lightblue')
            
            # Attention computation
            attention = f'layer_{layer_num}_attention_{tp_rank}'
            self.dot.node(attention,
                         f'TP Attention {tp_rank}\nInput: [batch_size=1024, seq_len=10000, head_dim=512*heads=16/8=2]\nOutput: [batch_size=1024, seq_len=10000, head_dim=512*heads=16/8=2]\nGPU: {gpu_id}',
                         fillcolor='lightgreen')
            
            # Output projection (row parallel)
            o_proj = f'layer_{layer_num}_o_proj_{tp_rank}'
            self.dot.node(o_proj,
                         f'O Projection {tp_rank}\nInput: [batch_size=1024, seq_len=10000, head_dim=512*heads=16/8=2]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nGPU: {gpu_id}',
                         fillcolor='lightblue')
            
            # Connect TP components
            self.dot.edge(mha_norm, q_proj)
            self.dot.edge(mha_norm, k_proj)
            self.dot.edge(mha_norm, v_proj)
            self.dot.edge(q_proj, attention)
            self.dot.edge(k_proj, attention)
            self.dot.edge(v_proj, attention)
            self.dot.edge(attention, o_proj)
        
        # All-reduce for attention output
        attn_allreduce = f'layer_{layer_num}_attn_allreduce'
        self.dot.node(attn_allreduce,
                     f'Attention All-Reduce\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024] x8\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     shape='parallelogram', fillcolor='orange')
        
        # Attention residual
        attn_residual = f'layer_{layer_num}_attn_residual'
        self.dot.node(attn_residual,
                     f'Attention Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192] x2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     fillcolor='lightyellow')
        
        # Connect attention components
        for tp_rank in range(8):
            self.dot.edge(f'layer_{layer_num}_o_proj_{tp_rank}', attn_allreduce)
        self.dot.edge(attn_allreduce, attn_residual)
        self.dot.edge(layer_input, attn_residual)  # residual connection
        
        # Expert Layer Norm
        expert_norm = f'layer_{layer_num}_expert_norm'
        self.dot.node(expert_norm,
                     f'Expert LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     fillcolor='lightcoral')
        
        # 8 Experts per GPU with tensor parallelism
        for gpu_offset in range(8):
            gpu_id = gpu_start + gpu_offset
            expert_start = 0 if stage == 0 else 8
            
            for expert_id in range(8):
                actual_expert_id = expert_start + expert_id
                
                # Gate for expert selection
                gate = f'layer_{layer_num}_gate_gpu{gpu_id}_expert{actual_expert_id}'
                self.dot.node(gate,
                             f'Gate {actual_expert_id}\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nOutput: [batch_size=1024, seq_len=10000, gate_score=1]\nGPU: {gpu_id}',
                             shape='diamond', fillcolor='yellow')
                
                # Expert computation
                expert_mlp = f'layer_{layer_num}_expert{actual_expert_id}_gpu{gpu_id}'
                self.dot.node(expert_mlp,
                             f'Expert {actual_expert_id}\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024]\nGPU: {gpu_id}',
                             fillcolor='lightpink')
                
                # Expert gate connection (dashed)
                self.dot.edge(gate, expert_mlp, style='dashed', label=f'route tokens')
                self.dot.edge(expert_norm, gate)
        
        # Expert aggregation
        expert_agg = f'layer_{layer_num}_expert_agg'
        self.dot.node(expert_agg,
                     f'Expert Output Aggregation\nInput: [batch_size=1024, seq_len=10000, hidden=8192/8=1024] x64 (8x8)\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     shape='parallelogram', fillcolor='orange')
        
        # Connect expert computations
        for gpu_offset in range(8):
            gpu_id = gpu_start + gpu_offset
            expert_start = 0 if stage == 0 else 8
            for expert_id in range(8):
                actual_expert_id = expert_start + expert_id
                self.dot.edge(f'layer_{layer_num}_expert{actual_expert_id}_gpu{gpu_id}', expert_agg)
        
        # Expert residual
        expert_residual = f'layer_{layer_num}_expert_residual'
        self.dot.node(expert_residual,
                     f'Expert Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192] x2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {gpu_start}-{gpu_start+7}',
                     fillcolor='lightyellow')
        
        # Connect final flow
        self.dot.edge(attn_residual, expert_norm)
        self.dot.edge(expert_agg, expert_residual)
        self.dot.edge(attn_residual, expert_residual)  # residual connection
        
        return expert_residual
    
    def add_pipeline_communication(self, from_layer: int, to_layer: int, from_gpu: int, to_gpu: int):
        comm_node = f'pipeline_comm_{from_layer}_to_{to_layer}'
        self.dot.node(comm_node,
                     f'Pipeline Communication\nLayer {from_layer} to {to_layer}\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {from_gpu} to {to_gpu}',
                     shape='ellipse', fillcolor='lightgreen')
        return comm_node
    
    def add_output_layer(self):
        self.dot.node('output',
                     'Output\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, vocab_size]\nGPU: 8-15',
                     shape='ellipse', fillcolor='lightgreen')
    
    def generate_dag(self):
        self.add_input_layer()
        
        # Layer 1 (Stage 0, GPUs 0-7)
        layer1_residual = self.add_layer_structure(1, 0, 0)
        
        # Layer 2 (Stage 0, GPUs 0-7)
        layer2_residual = self.add_layer_structure(2, 0, 0)
        
        # Pipeline communication from stage 0 to stage 1
        pipeline_comm_2_3 = self.add_pipeline_communication(2, 3, 7, 8)
        
        # Layer 3 (Stage 1, GPUs 8-15)
        layer3_residual = self.add_layer_structure(3, 1, 8)
        
        # Layer 4 (Stage 1, GPUs 8-15)
        layer4_residual = self.add_layer_structure(4, 1, 8)
        
        self.add_output_layer()
        
        # Connect the flow
        self.dot.edge('input', 'layer_1_input')
        self.dot.edge(layer1_residual, 'layer_2_input')
        self.dot.edge(layer2_residual, pipeline_comm_2_3)
        self.dot.edge(pipeline_comm_2_3, 'layer_3_input')
        self.dot.edge(layer3_residual, 'layer_4_input')
        self.dot.edge(layer4_residual, 'output')
        
        # Save files
        self.dot.render('../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2', format='svg', cleanup=False)
        self.dot.save('../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2.dot')
        
        return '../outputs/2025-10-19-21-59-45/moe_baseline_tp8_pp2.dot'

if __name__ == '__main__':
    generator = MoEBaselineDAGGenerator()
    dag_path = generator.generate_dag()
    print(f"Generated baseline DAG: {dag_path}")