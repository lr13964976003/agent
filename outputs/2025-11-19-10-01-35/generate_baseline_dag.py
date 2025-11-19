#!/usr/bin/env python3

import graphviz
from typing import Dict, List, Tuple

class BaselineDAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph(
            'baseline_tensor_pipeline',
            filename='baseline_tensor_pipeline',
            format='svg',
            graph_attr={
                'rankdir': 'TB',
                'bgcolor': 'white',
                'margin': '0.2',
                'pad': '0.5',
                'nodesep': '0.5',
                'ranksep': '1.0'
            },
            node_attr={
                'fontname': 'Courier',
                'fontsize': '10',
                'margin': '0.1,0.05'
            },
            edge_attr={
                'fontname': 'Courier',
                'fontsize': '8'
            }
        )
        
        # Model parameters
        self.batch_size = 128
        self.seq_len = 10000
        self.hidden_size = 4096
        self.num_heads = 32
        self.head_dim = 128
        self.tensor_parallel_degree = 8
        self.pipeline_stages = 2
        self.intermediate_size = 32768
        
    def add_node(self, node_id: str, label: str, shape: str = 'rectangle', **attrs):
        """Add a node with standardized formatting"""
        node_attrs = {'shape': shape, 'label': label}
        node_attrs.update(attrs)
        self.dot.node(node_id, **node_attrs)

    def generate_model_dag(self, num_layers: int = 2):
        """Generate baseline model with tensor + pipeline parallelism"""
        
        # Model input
        self.add_node(
            'model_input',
            'model_input\\nInput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0',
            shape='ellipse',
            fillcolor='darkblue',
            fontcolor='white',
            style='filled'
        )
        
        # Embedding (stage 0, device 0)
        self.add_node(
            'embedding',
            'embedding\\nInput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0',
            shape='rectangle',
            fillcolor='lightcyan',
            style='filled'
        )
        self.dot.edge('model_input', 'embedding')
        
        # Pipeline stage 0 (layers 0 to 1)
        self.generate_pipeline_stage(0, list(range(num_layers // 2)), 'embedding')
        
        # Pipeline stage 1 (layers 1 to 2)
        self.generate_pipeline_stage(1, list(range(num_layers // 2, num_layers)), f'stage0_output')
        
        # Final layer norm
        self.add_node(
            'final_layer_norm',
            'final_layer_norm\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge('stage1_output', 'final_layer_norm')
        
        # Model output
        self.add_node(
            'model_output',
            'model_output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nGPU: 8',
            shape='ellipse',
            fillcolor='darkgreen',
            fontcolor='white',
            style='filled'
        )
        self.dot.edge('final_layer_norm', 'model_output')

    def generate_pipeline_stage(self, stage_id: int, layer_indices: List[int], prev_node: str):
        """Generate a pipeline stage with tensor-parallel layers"""
        
        base_device = 0 if stage_id == 0 else 8
        
        # First layer norm for the stage
        self.add_node(
            f'stage{stage_id}_input',
            f'stage{stage_id}_input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='ellipse',
            fillcolor='lightblue',
            style='filled'
        )
        self.dot.edge(prev_node, f'stage{stage_id}_input')
        
        prev_layer_node = f'stage{stage_id}_input'
        
        for layer_idx in layer_indices:
            self.generate_tensor_parallel_layer(stage_id, layer_idx, prev_layer_node)
            prev_layer_node = f'layer_{layer_idx}_output'
        
        # Stage output
        self.add_node(
            f'stage{stage_id}_output',
            f'stage{stage_id}_output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device+7}',
            shape='ellipse',
            fillcolor='lightgreen',
            style='filled'
        )
        self.dot.edge(prev_layer_node, f'stage{stage_id}_output')

    def generate_tensor_parallel_layer(self, stage_id: int, layer_idx: int, prev_node: str):
        """Generate a tensor-parallel layer with MHA and FFN"""
        
        base_device = 0 if stage_id == 0 else 8
        
        # Layer input
        self.add_node(
            f'layer_{layer_idx}_input',
            f'layer_{layer_idx}_input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='ellipse',
            fillcolor='lightblue',
            style='filled'
        )
        self.dot.edge(prev_node, f'layer_{layer_idx}_input')
        
        # Layer norm
        self.add_node(
            f'layer_{layer_idx}_layer_norm_mha',
            f'layer_{layer_idx}_layer_norm_mha\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_input', f'layer_{layer_idx}_layer_norm_mha')
        
        # Generate tensor-parallel MHA
        self.generate_tensor_parallel_mha(layer_idx, base_device)
        
        # All-reduce for MHA output
        self.add_node(
            f'layer_{layer_idx}_mha_all_reduce',
            f'layer_{layer_idx}_mha_all_reduce\\nInput: [8×[batch_size=16, seq_len=10000, hidden_size=512]]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='parallelogram',
            fillcolor='lightgreen',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_mha_output_7', f'layer_{layer_idx}_mha_all_reduce')
        
        # Residual connection
        self.add_node(
            f'layer_{layer_idx}_residual_add_mha',
            f'layer_{layer_idx}_residual_add_mha\\nInput1: [batch_size=128, seq_len=10000, hidden_size=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='rectangle',
            fillcolor='lightcoral',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_layer_norm_mha', f'layer_{layer_idx}_residual_add_mha')
        self.dot.edge(f'layer_{layer_idx}_mha_all_reduce', f'layer_{layer_idx}_residual_add_mha')
        
        # Layer norm for FFN
        self.add_node(
            f'layer_{layer_idx}_layer_norm_ffn',
            f'layer_{layer_idx}_layer_norm_ffn\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_residual_add_mha', f'layer_{layer_idx}_layer_norm_ffn')
        
        # Generate tensor-parallel FFN
        self.generate_tensor_parallel_ffn(layer_idx, base_device)
        
        # All-reduce for FFN output
        self.add_node(
            f'layer_{layer_idx}_ffn_all_reduce',
            f'layer_{layer_idx}_ffn_all_reduce\\nInput: [8×[batch_size=16, seq_len=10000, hidden_size=512]]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='parallelogram',
            fillcolor='lightgreen',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_ffn_down_proj_7', f'layer_{layer_idx}_ffn_all_reduce')
        
        # Final residual connection
        self.add_node(
            f'layer_{layer_idx}_residual_add_ffn',
            f'layer_{layer_idx}_residual_add_ffn\\nInput1: [batch_size=128, seq_len=10000, hidden_size=4096]\\nInput2: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device}-{base_device+7}',
            shape='rectangle',
            fillcolor='lightcoral',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_layer_norm_ffn', f'layer_{layer_idx}_residual_add_ffn')
        self.dot.edge(f'layer_{layer_idx}_ffn_all_reduce', f'layer_{layer_idx}_residual_add_ffn')
        
        # Layer output
        self.add_node(
            f'layer_{layer_idx}_output',
            f'layer_{layer_idx}_output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {base_device+7}',
            shape='ellipse',
            fillcolor='lightgreen',
            style='filled'
        )
        self.dot.edge(f'layer_{layer_idx}_residual_add_ffn', f'layer_{layer_idx}_output')

    def generate_tensor_parallel_mha(self, layer_idx: int, base_device: int):
        """Generate tensor-parallel MHA across 8 GPUs"""
        
        for device in range(8):
            device_id = base_device + device
            
            # Each device handles 1/8 of the computation
            heads_per_device = self.num_heads // self.tensor_parallel_degree
            hidden_per_device = self.hidden_size // self.tensor_parallel_degree
            
            # Q projection
            self.add_node(
                f'layer_{layer_idx}_q_proj_{device}',
                f'layer_{layer_idx}_q_proj_{device}\\nInput: [batch_size=16, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # K projection
            self.add_node(
                f'layer_{layer_idx}_k_proj_{device}',
                f'layer_{layer_idx}_k_proj_{device}\\nInput: [batch_size=16, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # V projection
            self.add_node(
                f'layer_{layer_idx}_v_proj_{device}',
                f'layer_{layer_idx}_v_proj_{device}\\nInput: [batch_size=16, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # Attention computation
            self.add_node(
                f'layer_{layer_idx}_attention_{device}',
                f'layer_{layer_idx}_attention_{device}\\nInput1: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nInput2: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nInput3: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nOutput: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightpink',
                style='filled'
            )
            
            # Output projection
            self.add_node(
                f'layer_{layer_idx}_mha_output_{device}',
                f'layer_{layer_idx}_mha_output_{device}\\nInput: [batch_size=16, seq_len=10000, heads=4, d_k=128]\\nOutput: [batch_size=16, seq_len=10000, hidden_size=512]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # Connect MHA components
            layer_norm = f'layer_{layer_idx}_layer_norm_mha'
            self.dot.edge(layer_norm, f'layer_{layer_idx}_q_proj_{device}')
            self.dot.edge(layer_norm, f'layer_{layer_idx}_k_proj_{device}')
            self.dot.edge(layer_norm, f'layer_{layer_idx}_v_proj_{device}')
            
            self.dot.edge(f'layer_{layer_idx}_q_proj_{device}', f'layer_{layer_idx}_attention_{device}')
            self.dot.edge(f'layer_{layer_idx}_k_proj_{device}', f'layer_{layer_idx}_attention_{device}')
            self.dot.edge(f'layer_{layer_idx}_v_proj_{device}', f'layer_{layer_idx}_attention_{device}')
            self.dot.edge(f'layer_{layer_idx}_attention_{device}', f'layer_{layer_idx}_mha_output_{device}')

    def generate_tensor_parallel_ffn(self, layer_idx: int, base_device: int):
        """Generate tensor-parallel FFN across 8 GPUs"""
        
        for device in range(8):
            device_id = base_device + device
            hidden_per_device = self.hidden_size // self.tensor_parallel_degree
            intermediate_per_device = self.intermediate_size // self.tensor_parallel_degree
            
            # Gate projection (column parallel)
            self.add_node(
                f'layer_{layer_idx}_gate_proj_{device}',
                f'layer_{layer_idx}_gate_proj_{device}\\nInput: [batch_size=16, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # Up projection (column parallel)
            self.add_node(
                f'layer_{layer_idx}_up_proj_{device}',
                f'layer_{layer_idx}_up_proj_{device}\\nInput: [batch_size=16, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # Activation
            self.add_node(
                f'layer_{layer_idx}_activation_{device}',
                f'layer_{layer_idx}_activation_{device}\\nInput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nOutput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightpink',
                style='filled'
            )
            
            # Element-wise multiply
            self.add_node(
                f'layer_{layer_idx}_elementwise_mul_{device}',
                f'layer_{layer_idx}_elementwise_mul_{device}\\nInput1: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nInput2: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nOutput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightpink',
                style='filled'
            )
            
            # Down projection (row parallel)
            self.add_node(
                f'layer_{layer_idx}_ffn_down_proj_{device}',
                f'layer_{layer_idx}_ffn_down_proj_{device}\\nInput: [batch_size=16, seq_len=10000, intermediate_size=4096]\\nOutput: [batch_size=16, seq_len=10000, hidden_size=512]\\nGPU: {device_id}',
                shape='rectangle',
                fillcolor='lightcyan',
                style='filled'
            )
            
            # Connect FFN components
            layer_norm = f'layer_{layer_idx}_layer_norm_ffn'
            self.dot.edge(layer_norm, f'layer_{layer_idx}_gate_proj_{device}')
            self.dot.edge(layer_norm, f'layer_{layer_idx}_up_proj_{device}')
            
            self.dot.edge(f'layer_{layer_idx}_gate_proj_{device}', f'layer_{layer_idx}_activation_{device}')
            self.dot.edge(f'layer_{layer_idx}_activation_{device}', f'layer_{layer_idx}_elementwise_mul_{device}')
            self.dot.edge(f'layer_{layer_idx}_up_proj_{device}', f'layer_{layer_idx}_elementwise_mul_{device}')
            self.dot.edge(f'layer_{layer_idx}_elementwise_mul_{device}', f'layer_{layer_idx}_ffn_down_proj_{device}')

if __name__ == '__main__':
    generator = BaselineDAGGenerator()
    generator.generate_model_dag(num_layers=2)
    
    # Save DOT file
    with open('../outputs/2025-11-19-10-01-35/baseline_tensor_pipeline.dot', 'w') as f:
        f.write(generator.dot.source)
    
    # Save SVG
    generator.dot.render('../outputs/2025-11-19-10-01-35/baseline_tensor_pipeline', format='svg', cleanup=True)
    
    print("Baseline DAG generated successfully!")