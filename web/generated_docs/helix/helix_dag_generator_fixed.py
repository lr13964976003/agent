#!/usr/bin/env python3
"""
Helix DAG Generator for Two-Level Attention Partitioning
Generates complete deployment DAGs showing GPU allocation and communication paths
Fixed version with proper graphviz syntax
"""

import os
from graphviz import Digraph

class HelixDAGGenerator:
    def __init__(self):
        self.gpu_grid = (4, 4)  # 4x4 grid as per paper
        self.heads_total = 16
        self.heads_per_group = 4  # 16/4 = 4
        self.d_per_head = 512
        self.d_per_slice = 128  # 512/4 = 
        self.batch_size = 1024
        self.seq_len = 10000
        
    def create_complete_dag(self):
        """Create complete DAG for Helix deployment"""
        graph = Digraph('Helix_Complete_DAG', 
                       filename='helix_complete_dag',
                       node_attr={'shape': 'rectangle', 'style': 'filled'})
        graph.attr(rankdir='TB', size='100,100')
        
        # Input node
        input_label = f"INPUT\\nbatch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192"
        graph.node('input', input_label, 
                  shape='ellipse', fillcolor='lightblue', color='blue')
        
        # Generate detailed attention and MLP nodes for each layer
        self.add_layer1(graph)
        self.add_layer2(graph)
        
        # Output node
        output_label = f"OUTPUT\\nbatch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192"
        graph.node('output', output_label, 
                  shape='ellipse', fillcolor='lightblue', color='blue')
        
        return graph
        
    def add_layer1(self, graph):
        """Add Layer 1 with both MHA and MLP"""
        
        # Layer 1 MHA
        for i in range(4):  # head groups
            for j in range(4):  # dimension slices
                gpu_id = i * 4 + j
                
                # Input split
                split_label = f"Split_L1_{i}_{j}\\nGPU:{gpu_id}\\n" \
                             f"Input: [B={self.batch_size}, L={self.seq_len}, D=8192]\\n" \
                             f"Output: [B={self.batch_size}, L={self.seq_len}, D=2048]"
                graph.node(f'split_l1_{i}_{j}', split_label, 
                          shape='parallelogram', fillcolor='lightyellow', color='orange')
                
                # Q projection
                q_label = f"Q_Proj_L1_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'q_proj_l1_{i}_{j}', q_label, fillcolor='lightgreen')
                
                # K projection  
                k_label = f"K_Proj_L1_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'k_proj_l1_{i}_{j}', k_label, fillcolor='lightgreen')
                
                # V projection
                v_label = f"V_Proj_L1_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'v_proj_l1_{i}_{j}', v_label, fillcolor='lightgreen')
                
                # Local attention
                attn_label = f"Attention_L1_{i}_{j}\\nGPU:{gpu_id}\\n" \
                           f"Input: Q/K/V [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]\\n" \
                           f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'attn_l1_{i}_{j}', attn_label, fillcolor='lightcoral')
                
        # Concatenation within head groups (dimension slices)
        for i in range(4):
            concat_dim_label = f"Concat_Dim_L1_{i}\\n" \
                             f"Input: 4×[B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]\\n" \
                             f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice*4}]"
            graph.node(f'concat_dim_l1_{i}', concat_dim_label, 
                      shape='parallelogam', fillcolor='lightyellow', color='orange')
            
        # Concatenation across head groups
        concat_heads_label = f"Concat_Heads_L1\\n" \
                           f"Input: 4×[B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice*4}]\\n" \
                           f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('concat_heads_l1', concat_heads_label, 
                  shape='parallelogram', fillcolor='lightyellow', color='orange')
        
        # Residual connection
        residual_label = f"Residual_Add_L1\\n" \
                       f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                       f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('residual_l1', residual_label, fillcolor='lightpink')
        
        # Layer 1 MLP
        for gpu in range(16):
            linear1_label = f"Linear1_L1_{gpu}\\nGPU:{gpu}\\n" \
                          f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                          f"Output: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]"
            graph.node(f'linear1_l1_{gpu}', linear1_label, fillcolor='lightblue')
            
            gelu_label = f"GELU_L1_{gpu}\\nGPU:{gpu}\\n" \
                       f"Input: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]\\n" \
                       f"Output: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]"
            graph.node(f'gelu_l1_{gpu}', gelu_label, fillcolor='lightgreen')
            
            linear2_label = f"Linear2_L1_{gpu}\\nGPU:{gpu}\\n" \
                          f"Input: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]\\n" \
                          f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=512]"
            graph.node(f'linear2_l1_{gpu}', linear2_label, fillcolor='lightblue')
            
        # All-reduce sum
        allreduce_label = f"AllReduce_L1\\n" \
                        f"Input: 16×[B={self.batch_size}, L={self.seq_len}, hidden_dim=512]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('allreduce_l1', allreduce_label, 
                  shape='parallelogram', fillcolor='lightyellow', color='orange')
        
        residual2_label = f"Residual_Add_L1_MLP\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('residual2_l1', residual2_label, fillcolor='lightpink')
        
        # Create connections for Layer 1
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                graph.edge('input', f'split_l1_{i}_{j}')
                graph.edge(f'split_l1_{i}_{j}', f'q_proj_l1_{i}_{j}')
                graph.edge(f'split_l1_{i}_{j}', f'k_proj_l1_{i}_{j}')
                graph.edge(f'split_l1_{i}_{j}', f'v_proj_l1_{i}_{j}')
                graph.edge(f'q_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                graph.edge(f'k_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                graph.edge(f'v_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                graph.edge(f'attn_l1_{i}_{j}', f'concat_dim_l1_{i}')
        
        for i in range(4):
            graph.edge(f'concat_dim_l1_{i}', 'concat_heads_l1')
        
        graph.edge('concat_heads_l1', 'residual_l1')
        graph.edge('input', 'residual_l1')  # Skip connection
        
        for gpu in range(16):
            graph.edge('residual_l1', f'linear1_l1_{gpu}')
            graph.edge(f'linear1_l1_{gpu}', f'gelu_l1_{gpu}')
            graph.edge(f'gelu_l1_{gpu}', f'linear2_l1_{gpu}')
            graph.edge(f'linear2_l1_{gpu}', 'allreduce_l1')
        
        graph.edge('allreduce_l1', 'residual2_l1')
        graph.edge('residual_l1', 'residual2_l1')  # Skip connection
        
    def add_layer2(self, graph):
        """Add Layer 2 with both MHA and MLP"""
        
        # Layer 2 MHA
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                
                split_label = f"Split_L2_{i}_{j}\\nGPU:{gpu_id}\\n" \
                             f"Input: [B={self.batch_size}, L={self.seq_len}, D=8192]\\n" \
                             f"Output: [B={self.batch_size}, L={self.seq_len}, D=2048]"
                graph.node(f'split_l2_{i}_{j}', split_label, 
                          shape='parallelogram', fillcolor='lightyellow', color='orange')
                
                q_label = f"Q_Proj_L2_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'q_proj_l2_{i}_{j}', q_label, fillcolor='lightgreen')
                
                k_label = f"K_Proj_L2_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'k_proj_l2_{i}_{j}', k_label, fillcolor='lightgreen')
                
                v_label = f"V_Proj_L2_{i}_{j}\\nGPU:{gpu_id}\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, D=2048]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'v_proj_l2_{i}_{j}', v_label, fillcolor='lightgreen')
                
                attn_label = f"Attention_L2_{i}_{j}\\nGPU:{gpu_id}\\n" \
                           f"Input: Q/K/V [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]\\n" \
                           f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]"
                graph.node(f'attn_l2_{i}_{j}', attn_label, fillcolor='lightcoral')
                
        # Concatenation within head groups
        for i in range(4):
            concat_dim_label = f"Concat_Dim_L2_{i}\\n" \
                             f"Input: 4×[B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice}]\\n" \
                             f"Output: [B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice*4}]"
            graph.node(f'concat_dim_l2_{i}', concat_dim_label, 
                      shape='parallelogram', fillcolor='lightyellow', color='orange')
            
        concat_heads_label = f"Concat_Heads_L2\\n" \
                           f"Input: 4×[B={self.batch_size}, L={self.seq_len}, h={self.heads_per_group}, d={self.d_per_slice*4}]\\n" \
                           f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('concat_heads_l2', concat_heads_label, 
                  shape='parallelogram', fillcolor='lightyellow', color='orange')
        
        residual_label = f"Residual_Add_L2\\n" \
                       f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                       f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('residual_l2', residual_label, fillcolor='lightpink')
        
        # Layer 2 MLP
        for gpu in range(16):
            linear1_label = f"Linear1_L2_{gpu}\\nGPU:{gpu}\\n" \
                          f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                          f"Output: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]"
            graph.node(f'linear1_l2_{gpu}', linear1_label, fillcolor='lightblue')
            
            gelu_label = f"GELU_L2_{gpu}\\nGPU:{gpu}\\n" \
                       f"Input: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]\\n" \
                       f"Output: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]"
            graph.node(f'gelu_l2_{gpu}', gelu_label, fillcolor='lightgreen')
            
            linear2_label = f"Linear2_L2_{gpu}\\nGPU:{gpu}\\n" \
                          f"Input: [B={self.batch_size}, L={self.seq_len}, ffn_hidden=2048]\\n" \
                          f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=512]"
            graph.node(f'linear2_l2_{gpu}', linear2_label, fillcolor='lightblue')
            
        allreduce_label = f"AllReduce_L2\\n" \
                        f"Input: 16×[B={self.batch_size}, L={self.seq_len}, hidden_dim=512]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('allreduce_l2', allreduce_label, 
                  shape='parallelogram', fillcolor='lightyellow', color='orange')
        
        residual2_label = f"Residual_Add_L2_MLP\\n" \
                        f"Input: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]\\n" \
                        f"Output: [B={self.batch_size}, L={self.seq_len}, hidden_dim=8192]"
        graph.node('residual2_l2', residual2_label, fillcolor='lightpink')
        
        # Create connections for Layer 2
        for i in range(4):
            for j in range(4):
                gpu_id = i * 4 + j
                graph.edge('residual2_l1', f'split_l2_{i}_{j}')
                graph.edge(f'split_l2_{i}_{j}', f'q_proj_l2_{i}_{j}')
                graph.edge(f'split_l2_{i}_{j}', f'k_proj_l2_{i}_{j}')
                graph.edge(f'split_l2_{i}_{j}', f'v_proj_l2_{i}_{j}')
                graph.edge(f'q_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
                graph.edge(f'k_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
                graph.edge(f'v_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
                graph.edge(f'attn_l2_{i}_{j}', f'concat_dim_l2_{i}')
        
        for i in range(4):
            graph.edge(f'concat_dim_l2_{i}', 'concat_heads_l2')
        
        graph.edge('concat_heads_l2', 'residual_l2')
        graph.edge('residual2_l1', 'residual_l2')  # Skip connection
        
        for gpu in range(16):
            graph.edge('residual_l2', f'linear1_l2_{gpu}')
            graph.edge(f'linear1_l2_{gpu}', f'gelu_l2_{gpu}')
            graph.edge(f'gelu_l2_{gpu}', f'linear2_l2_{gpu}')
            graph.edge(f'linear2_l2_{gpu}', 'allreduce_l2')
        
        graph.edge('allreduce_l2', 'residual2_l2')
        graph.edge('residual_l2', 'residual2_l2')  # Skip connection
        graph.edge('residual2_l2', 'output')
        
    def generate_files(self):
        """Generate DOT and SVG files"""
        # Create complete DAG
        graph = self.create_complete_dag()
        
        # Save DOT file
        dot_path = './generated_docs/helix/helix_complete_dag.dot'
        with open(dot_path, 'w') as f:
            f.write(str(graph.source))
            
        # Save SVG
        svg_path = './generated_docs/helix/helix_complete_dag.svg'
        try:
            graph.render(directory='./generated_docs/helix', filename='helix_complete_dag', 
                        format='svg', cleanup=True)
        except:
            # Fallback: just save DOT
            pass
            
        # Create layer-specific DAGs
        layer1_path = self.create_layer1_dag()
        layer2_path = self.create_layer2_dag()
        
        return {
            'complete_dag_dot': dot_path,
            'complete_dag_svg': svg_path,
            'layer1_dag_dot': layer1_path,
            'layer2_dag_dot': './generated_docs/helix/helix_layer2_dag.dot'
        }
        
    def create_layer1_dag(self):
        """Create detailed Layer 1 DAG"""
        layer1 = Digraph('Helix_Layer1_DAG', 
                        filename='helix_layer1_dag',
                        node_attr={'shape': 'rectangle', 'style': 'filled'})
        layer1.attr(rankdir='TB', size='50,50')
        
        # Input
        layer1.node('input', 'INPUT', shape='ellipse', fillcolor='lightblue')
        
        # Add detailed layer1 components
        self.add_layer1(layer1)
        
        dot_path = './generated_docs/helix/helix_layer1_dag.dot'
        with open(dot_path, 'w') as f:
            f.write(str(layer1.source))
            
        return dot_path
        
    def create_layer2_dag(self):
        """Create detailed Layer 2 DAG"""
        layer2 = Digraph('Helix_Layer2_DAG', 
                        filename='helix_layer2_dag',
                        node_attr={'shape': 'rectangle', 'style': 'filled'})
        layer2.attr(rankdir='TB', size='50,50')
        
        # Input to layer2
        layer2.node('input_l2', 'Layer2_Input', shape='ellipse', fillcolor='lightblue')
        
        # Add detailed layer2 components
        self.add_layer2(layer2)
        
        dot_path = './generated_docs/helix/helix_layer2_dag.dot'
        with open(dot_path, 'w') as f:
            f.write(str(layer2.source))
            
        return dot_path

if __name__ == "__main__":
    os.makedirs('./generated_docs/helix', exist_ok=True)
    
    generator = HelixDAGGenerator()
    paths = generator.generate_files()
    
    print("Generated DAGs:")
    for name, path in paths.items():
        print(f"{name}: {path}")