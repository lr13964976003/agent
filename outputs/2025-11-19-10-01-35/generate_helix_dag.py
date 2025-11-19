#!/usr/bin/env python3

import graphviz
from typing import Dict, List, Tuple
import json

class HelixDAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph(
            'helix_two_level_attention_partitioning',
            filename='helix_two_level_attention_partitioning',
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
        self.head_partitions = 4
        self.dim_partitions = 4
        self.heads_per_group = 8
        self.dim_per_slice = 32
        
    def add_node(self, node_id: str, label: str, shape: str = 'rectangle', **attrs):
        """Add a node with standardized formatting"""
        node_attrs = {'shape': shape, 'label': label}
        node_attrs.update(attrs)
        self.dot.node(node_id, **node_attrs)

    def generate_layer_dag(self, layer_num: int):
        """Generate a complete layer with MHA and FFN"""
        
        # Layer prefix
        prefix = f'layer_{layer_num}'
        
        # Input to layer
        self.add_node(
            f'{prefix}_input',
            f'{prefix}_input\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='ellipse',
            fillcolor='lightblue',
            style='filled'
        )
        
        # Layer normalization
        self.add_node(
            f'{prefix}_layer_norm_mha',
            f'{prefix}_layer_norm_mha\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge(f'{prefix}_input', f'{prefix}_layer_norm_mha')
        
        # Generate MHA with Helix partitioning
        self._generate_mha_partition(prefix)
        
        # Residual connection
        self.add_node(
            f'{prefix}_residual_add_mha',
            f'{prefix}_residual_add_mha\\nInput1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nInput2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightcoral',
            style='filled'
        )
        self.dot.edge(f'{prefix}_layer_norm_mha', f'{prefix}_residual_add_mha')
        self.dot.edge(f'{prefix}_mha_aggregate', f'{prefix}_residual_add_mha')
        
        # Layer norm for FFN
        self.add_node(
            f'{prefix}_layer_norm_ffn',
            f'{prefix}_layer_norm_ffn\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge(f'{prefix}_residual_add_mha', f'{prefix}_layer_norm_ffn')
        
        # FFN (unchanged from distributed perspective)
        self._generate_ffn(prefix)
        
        # Final residual add
        self.add_node(
            f'{prefix}_residual_add_ffn',
            f'{prefix}_residual_add_ffn\\nInput1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nInput2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightcoral',
            style='filled'
        )
        self.dot.edge(f'{prefix}_layer_norm_ffn', f'{prefix}_residual_add_ffn')
        self.dot.edge(f'{prefix}_ffn_output', f'{prefix}_residual_add_ffn')
        
        # Layer output
        self.add_node(
            f'{prefix}_output',
            f'{prefix}_output\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='ellipse',
            fillcolor='lightgreen',
            style='filled'
        )
        self.dot.edge(f'{prefix}_residual_add_ffn', f'{prefix}_output')

    def _generate_mha_partition(self, prefix: str):
        """Generate the Helix-partitioned MHA"""
        
        # Input projection matrices split by Helix partitioning
        for head_group in range(self.head_partitions):
            for dim_slice in range(self.dim_partitions):
                device_id = head_group * self.dim_partitions + dim_slice
                
                # Q projection partition
                self.add_node(
                    f'{prefix}_q_proj_{head_group}_{dim_slice}',
                    f'{prefix}_q_proj_{head_group}_{dim_slice}\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nGPU: {device_id}',
                    shape='rectangle',
                    fillcolor='lightcyan',
                    style='filled'
                )
                
                # K projection partition
                self.add_node(
                    f'{prefix}_k_proj_{head_group}_{dim_slice}',
                    f'{prefix}_k_proj_{head_group}_{dim_slice}\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nGPU: {device_id}',
                    shape='rectangle',
                    fillcolor='lightcyan',
                    style='filled'
                )
                
                # V projection partition
                self.add_node(
                    f'{prefix}_v_proj_{head_group}_{dim_slice}',
                    f'{prefix}_v_proj_{head_group}_{dim_slice}\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nGPU: {device_id}',
                    shape='rectangle',
                    fillcolor='lightcyan',
                    style='filled'
                )
                
                # Attention computation per partition
                self.add_node(
                    f'{prefix}_attention_{head_group}_{dim_slice}',
                    f'{prefix}_attention_{head_group}_{dim_slice}\\nInput1: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nInput2: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nInput3: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]\\nGPU: {device_id}',
                    shape='rectangle',
                    fillcolor='lightpink',
                    style='filled'
                )
                
                # Connect inputs to projections
                self.dot.edge(f'{prefix}_layer_norm_mha', f'{prefix}_q_proj_{head_group}_{dim_slice}')
                self.dot.edge(f'{prefix}_layer_norm_mha', f'{prefix}_k_proj_{head_group}_{dim_slice}')
                self.dot.edge(f'{prefix}_layer_norm_mha', f'{prefix}_v_proj_{head_group}_{dim_slice}')
                
                # Connect projections to attention
                self.dot.edge(f'{prefix}_q_proj_{head_group}_{dim_slice}', f'{prefix}_attention_{head_group}_{dim_slice}')
                self.dot.edge(f'{prefix}_k_proj_{head_group}_{dim_slice}', f'{prefix}_attention_{head_group}_{dim_slice}')
                self.dot.edge(f'{prefix}_v_proj_{head_group}_{dim_slice}', f'{prefix}_attention_{head_group}_{dim_slice}')
        
        # Communication nodes for aggregation
        
        # First level: aggregate dimension slices within head groups
        for head_group in range(self.head_partitions):
            self.add_node(
                f'{prefix}_dim_aggregate_{head_group}',
                f'{prefix}_dim_aggregate_{head_group}\\nInput: [{self.dim_partitions}×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.dim_per_slice}]]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.head_dim}]\\nGPU: all',
                shape='parallelogram',
                fillcolor='lightgreen',
                style='filled'
            )
            
            # Connect attention outputs to dimension aggregation
            for dim_slice in range(self.dim_partitions):
                device_id = head_group * self.dim_partitions + dim_slice
                self.dot.edge(f'{prefix}_attention_{head_group}_{dim_slice}', f'{prefix}_dim_aggregate_{head_group}')
        
        # Second level: aggregate all head groups
        self.add_node(
            f'{prefix}_mha_aggregate',
            f'{prefix}_mha_aggregate\\nInput: [{self.head_partitions}×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.head_dim}]]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='parallelogram',
            fillcolor='lightgreen',
            style='filled'
        )
        
        # Connect dimension aggregates to final aggregate
        for head_group in range(self.head_partitions):
            self.dot.edge(f'{prefix}_dim_aggregate_{head_group}', f'{prefix}_mha_aggregate')

    def _generate_ffn(self, prefix: str):
        """Generate FFN layer (simplified as it's not partitioned by Helix)"""
        
        # Gate projection
        self.add_node(
            f'{prefix}_gate_proj',
            f'{prefix}_gate_proj\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightcyan',
            style='filled'
        )
        
        # Up projection
        self.add_node(
            f'{prefix}_up_proj',
            f'{prefix}_up_proj\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightcyan',
            style='filled'
        )
        
        # Activation
        self.add_node(
            f'{prefix}_activation',
            f'{prefix}_activation\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightpink',
            style='filled'
        )
        
        # Element-wise multiply
        self.add_node(
            f'{prefix}_elementwise_mul',
            f'{prefix}_elementwise_mul\\nInput1: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nInput2: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightpink',
            style='filled'
        )
        
        # Down projection
        self.add_node(
            f'{prefix}_down_proj',
            f'{prefix}_down_proj\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, intermediate_size={32768}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='rectangle',
            fillcolor='lightcyan',
            style='filled'
        )
        
        # FFN output
        self.add_node(
            f'{prefix}_ffn_output',
            f'{prefix}_ffn_output\\nInput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nOutput: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_size={self.hidden_size}]\\nGPU: all',
            shape='ellipse',
            fillcolor='lightgreen',
            style='filled'
        )
        
        # Connect FFN
        self.dot.edge(f'{prefix}_layer_norm_ffn', f'{prefix}_gate_proj')
        self.dot.edge(f'{prefix}_layer_norm_ffn', f'{prefix}_up_proj')
        self.dot.edge(f'{prefix}_gate_proj', f'{prefix}_activation')
        self.dot.edge(f'{prefix}_activation', f'{prefix}_elementwise_mul')
        self.dot.edge(f'{prefix}_up_proj', f'{prefix}_elementwise_mul')
        self.dot.edge(f'{prefix}_elementwise_mul', f'{prefix}_down_proj')
        self.dot.edge(f'{prefix}_down_proj', f'{prefix}_ffn_output')

    def generate_complete_model(self, num_layers: int = 4):
        """Generate complete model DAG"""
        
        # Model input
        self.add_node(
            'model_input',
            'model_input\\nInput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0',
            shape='ellipse',
            fillcolor='darkblue',
            fontcolor='white',
            style='filled'
        )
        
        # Embedding
        self.add_node(
            'embedding',
            'embedding\\nInput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0',
            shape='rectangle',
            fillcolor='lightcyan',
            style='filled'
        )
        self.dot.edge('model_input', 'embedding')
        
        # Generate layers
        prev_node = 'embedding'
        for layer in range(num_layers):
            self.generate_layer_dag(layer)
            self.dot.edge(prev_node, f'layer_{layer}_input')
            prev_node = f'layer_{layer}_output'
        
        # Final layer norm
        self.add_node(
            'final_layer_norm',
            'final_layer_norm\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0',
            shape='rectangle',
            fillcolor='lightyellow',
            style='filled'
        )
        self.dot.edge(prev_node, 'final_layer_norm')
        
        # Model output
        self.add_node(
            'model_output',
            'model_output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, vocab_size=50257]\\nGPU: 0',
            shape='ellipse',
            fillcolor='darkgreen',
            fontcolor='white',
            style='filled'
        )
        self.dot.edge('final_layer_norm', 'model_output')

if __name__ == '__main__':
    generator = HelixDAGGenerator()
    generator.generate_complete_model(num_layers=4)
    
    # Save DOT file
    with open('../outputs/2025-11-19-10-01-35/helix_two_level_attention_partitioning.dot', 'w') as f:
        f.write(generator.dot.source)
    
    # Save SVG
    generator.dot.render('../outputs/2025-11-19-10-01-35/helix_two_level_attention_partitioning', format='svg', cleanup=True)
    
    print("Helix DAG generated successfully!")