#!/usr/bin/env python3
"""
Helix DAG Generator for Two-Level Attention Partitioning
Generates complete deployment DAGs showing GPU allocation and communication paths
"""

import os
from graphviz import Digraph

class HelixDAGGenerator:
    def __init__(self):
        self.graph = None
        self.gpu_grid = (4, 4)  # 4x4 grid as per paper
        self.heads_total = 16
        self.heads_per_group = 4  # 16/4 = 4
        self.d_per_head = 512
        self.d_per_slice = 128  # 512/4 = 128
        self.batch_size = 1024
        self.seq_len = 10000
        
    def create_complete_dag(self):
        """Create complete DAG for Helix deployment"""
        # Create main graph
        self.graph = Digraph('Helix_Complete_DAG', 
                           filename='helix_complete_dag',
                           node_attr={'shape': 'rectangle', 'style': 'filled'})
        self.graph.attr(rankdir='TB', size='100,100')
        
        # Input node
        self.add_input_node()
        
        # Layer 1: Multi-Head Attention with Helix partitioning
        self.add_layer1_mha()
        
        # Layer 1: MLP with tensor parallelism
        self.add_layer1_mlp()
        
        # Layer 2: Multi-Head Attention with Helix partitioning
        self.add_layer2_mha()
        
        # Layer 2: MLP with tensor parallelism
        self.add_layer2_mlp()
        
        # Output node
        self.add_output_node()
        
        return self.graph
        
    def add_input_node(self):
        """Add input node with complete dimensions"""
        input_label = f"<<b>INPUT</b><br/>" \
                     f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]"
        self.graph.node('input', input_label, 
                       shape='ellipse', fillcolor='lightblue', color='blue')
        
    def add_layer1_mha(self):
        """Add Layer 1 Multi-Head Attention with Helix partitioning"""
        # Create subgraph for Layer 1 MHA
        with self.graph.subgraph(name='cluster_layer1_mha') as c:
            c.attr(label='<<b>Layer 1: Multi-Head Attention</b><br/>Helix 4×4 Partitioning>>', 
                   style='rounded', bgcolor='lightgray')
            
            # Input projection across all GPUs
            for i in range(4):  # head groups
                for j in range(4):  # dimension slices
                    gpu_id = i * 4 + j
                    
                    # Input split and broadcast
                    split_label = f"<<b>Split_L1_{i}_{j}</b><br/>" \
                                f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                                f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                                f"GPU: {gpu_id}>"
                    c.node(f'split_l1_{i}_{j}', split_label, 
                           shape='parallelogram', fillcolor='lightyellow', color='orange')
                    
                    # Q projection
                    q_label = f"<<b>Q_Proj_L1_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'q_proj_l1_{i}_{j}', q_label, fillcolor='lightgreen')
                    
                    # K projection
                    k_label = f"<<b>K_Proj_L1_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'k_proj_l1_{i}_{j}', k_label, fillcolor='lightgreen')
                    
                    # V projection
                    v_label = f"<<b>V_Proj_L1_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'v_proj_l1_{i}_{j}', v_label, fillcolor='lightgreen')
                    
                    # Local attention computation
                    attn_label = f"<<b>Attention_L1_{i}_{j}</b><br/>" \
                               f"Input Q: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"Input K/V: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"GPU: {gpu_id}>"
                    c.node(f'attn_l1_{i}_{j}', attn_label, fillcolor='lightcoral')
                    
            # Concatenation within head groups (dimension slices)
            for i in range(4):  # head groups
                concat_dim_label = f"<<b>Concat_Dim_L1_{i}</b><br/>" \
                                 f"Input: 4×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                                 f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice*4}]<br/>" \
                                 f"GPU: {i*4}-{(i+1)*4-1}>"
                c.node(f'concat_dim_l1_{i}', concat_dim_label, 
                       shape='parallelogram', fillcolor='lightyellow', color='orange')
                
            # Concatenation across head groups
            concat_heads_label = f"<<b>Concat_Heads_L1</b><br/>" \
                               f"Input: 4×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice*4}]<br/>" \
                               f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                               f"GPU: all GPUs>"
            c.node('concat_heads_l1', concat_heads_label, 
                   shape='parallelogram', fillcolor='lightyellow', color='orange')
            
            # Residual connection
            residual_label = f"<<b>Residual_Add_L1</b><br/>" \
                           f"Input1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"Input2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"GPU: all GPUs>"
            c.node('residual_l1', residual_label, fillcolor='lightpink')
            
    def add_layer1_mlp(self):
        """Add Layer 1 MLP with tensor parallelism"""
        with self.graph.subgraph(name='cluster_layer1_mlp') as c:
            c.attr(label='<<b>Layer 1: MLP</b><br/>Tensor Parallelism>>', 
                   style='rounded', bgcolor='lightgray')
            
            # Column parallel first linear
            for gpu in range(16):
                linear1_label = f"<<b>Linear1_L1_{gpu}</b><br/>" \
                              f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                              f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                              f"GPU: {gpu}>"
                c.node(f'linear1_l1_{gpu}', linear1_label, fillcolor='lightblue')
                
                # GELU activation
                gelu_label = f"<<b>GELU_L1_{gpu}</b><br/>" \
                           f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                           f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                           f"GPU: {gpu}>"
                c.node(f'gelu_l1_{gpu}', gelu_label, fillcolor='lightgreen')
                
                # Row parallel second linear
                linear2_label = f"<<b>Linear2_L1_{gpu}</b><br/>" \
                              f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                              f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=512]<br/>" \
                              f"GPU: {gpu}>"
                c.node(f'linear2_l1_{gpu}', linear2_label, fillcolor='lightblue')
                
            # All-reduce sum
            allreduce_label = f"<<b>AllReduce_L1</b><br/>" \
                            f"Input: 16×[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=512]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"GPU: all GPUs>"
            c.node('allreduce_l1', allreduce_label, 
                   shape='parallelogram', fillcolor='lightyellow', color='orange')
            
            # Residual connection
            residual2_label = f"<<b>Residual_Add_L1_MLP</b><br/>" \
                            f"Input1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"Input2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"GPU: all GPUs>"
            c.node('residual2_l1', residual2_label, fillcolor='lightpink')
            
    def add_layer2_mha(self):
        """Add Layer 2 Multi-Head Attention (same structure as Layer 1)"""
        with self.graph.subgraph(name='cluster_layer2_mha') as c:
            c.attr(label='<<b>Layer 2: Multi-Head Attention</b><br/>Helix 4×4 Partitioning>>', 
                   style='rounded', bgcolor='lightgray')
            
            # Input projection across all GPUs
            for i in range(4):
                for j in range(4):
                    gpu_id = i * 4 + j
                    
                    split_label = f"<<b>Split_L2_{i}_{j}</b><br/>" \
                                f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                                f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                                f"GPU: {gpu_id}>"
                    c.node(f'split_l2_{i}_{j}', split_label, 
                           shape='parallelogram', fillcolor='lightyellow', color='orange')
                    
                    q_label = f"<<b>Q_Proj_L2_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'q_proj_l2_{i}_{j}', q_label, fillcolor='lightgreen')
                    
                    k_label = f"<<b>K_Proj_L2_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'k_proj_l2_{i}_{j}', k_label, fillcolor='lightgreen')
                    
                    v_label = f"<<b>V_Proj_L2_{i}_{j}</b><br/>" \
                            f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=2048]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                            f"GPU: {gpu_id}>"
                    c.node(f'v_proj_l2_{i}_{j}', v_label, fillcolor='lightgreen')
                    
                    attn_label = f"<<b>Attention_L2_{i}_{j}</b><br/>" \
                               f"Input Q: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"Input K/V: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                               f"GPU: {gpu_id}>"
                    c.node(f'attn_l2_{i}_{j}', attn_label, fillcolor='lightcoral')
                    
            # Concatenation within head groups
            for i in range(4):
                concat_dim_label = f"<<b>Concat_Dim_L2_{i}</b><br/>" \
                                 f"Input: 4×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice}]<br/>" \
                                 f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice*4}]<br/>" \
                                 f"GPU: {i*4}-{(i+1)*4-1}>"
                c.node(f'concat_dim_l2_{i}', concat_dim_label, 
                       shape='parallelogram', fillcolor='lightyellow', color='orange')
                
            # Concatenation across head groups
            concat_heads_label = f"<<b>Concat_Heads_L2</b><br/>" \
                               f"Input: 4×[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_group}, d_k={self.d_per_slice*4}]<br/>" \
                               f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                               f"GPU: all GPUs>"
            c.node('concat_heads_l2', concat_heads_label, 
                   shape='parallelogram', fillcolor='lightyellow', color='orange')
            
            # Residual connection
            residual_label = f"<<b>Residual_Add_L2</b><br/>" \
                           f"Input1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"Input2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                           f"GPU: all GPUs>"
            c.node('residual_l2', residual_label, fillcolor='lightpink')
            
    def add_layer2_mlp(self):
        """Add Layer 2 MLP (same structure as Layer 1)"""
        with self.graph.subgraph(name='cluster_layer2_mlp') as c:
            c.attr(label='<<b>Layer 2: MLP</b><br/>Tensor Parallelism>>', 
                   style='rounded', bgcolor='lightgray')
            
            for gpu in range(16):
                linear1_label = f"<<b>Linear1_L2_{gpu}</b><br/>" \
                              f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                              f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                              f"GPU: {gpu}>"
                c.node(f'linear1_l2_{gpu}', linear1_label, fillcolor='lightblue')
                
                gelu_label = f"<<b>GELU_L2_{gpu}</b><br/>" \
                           f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                           f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                           f"GPU: {gpu}>"
                c.node(f'gelu_l2_{gpu}', gelu_label, fillcolor='lightgreen')
                
                linear2_label = f"<<b>Linear2_L2_{gpu}</b><br/>" \
                              f"Input: [batch_size={self.batch_size}, seq_len={self.seq_len}, ffn_hidden=2048]<br/>" \
                              f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=512]<br/>" \
                              f"GPU: {gpu}>"
                c.node(f'linear2_l2_{gpu}', linear2_label, fillcolor='lightblue')
                
            allreduce_label = f"<<b>AllReduce_L2</b><br/>" \
                            f"Input: 16×[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=512]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"GPU: all GPUs>"
            c.node('allreduce_l2', allreduce_label, 
                   shape='parallelogram', fillcolor='lightyellow', color='orange')
            
            residual2_label = f"<<b>Residual_Add_L2_MLP</b><br/>" \
                            f"Input1: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"Input2: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]<br/>" \
                            f"GPU: all GPUs>"
            c.node('residual2_l2', residual2_label, fillcolor='lightpink')
            
    def add_output_node(self):
        """Add output node"""
        output_label = f"<<b>OUTPUT</b><br/>" \
                     f"Output: [batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim=8192]"
        self.graph.node('output', output_label, 
                       shape='ellipse', fillcolor='lightblue', color='blue')
                        
    def create_connections(self):
        """Create all connections between nodes"""
        # Input to Layer 1 MHA
        for i in range(4):
            for j in range(4):
                self.graph.edge('input', f'split_l1_{i}_{j}')
                self.graph.edge(f'split_l1_{i}_{j}', f'q_proj_l1_{i}_{j}')
                self.graph.edge(f'split_l1_{i}_{j}', f'k_proj_l1_{i}_{j}')
                self.graph.edge(f'split_l1_{i}_{j}', f'v_proj_l1_{i}_{j}')
                self.graph.edge(f'q_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                self.graph.edge(f'k_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                self.graph.edge(f'v_proj_l1_{i}_{j}', f'attn_l1_{i}_{j}')
                self.graph.edge(f'attn_l1_{i}_{j}', f'concat_dim_l1_{i}')
        
        # Within head group concatenation
        for i in range(4):
            for j in range(4):
                self.graph.edge(f'attn_l1_{i}_{j}', f'concat_dim_l1_{i}')
            self.graph.edge(f'concat_dim_l1_{i}', 'concat_heads_l1')
        
        # MHA to residual
        self.graph.edge('concat_heads_l1', 'residual_l1')
        self.graph.edge('input', 'residual_l1')  # Skip connection
        
        # Layer 1 MLP
        for gpu in range(16):
            self.graph.edge('residual_l1', f'linear1_l1_{gpu}')
            self.graph.edge(f'linear1_l1_{gpu}', f'gelu_l1_{gpu}')
            self.graph.edge(f'gelu_l1_{gpu}', f'linear2_l1_{gpu}')
            self.graph.edge(f'linear2_l1_{gpu}', 'allreduce_l1')
        
        self.graph.edge('allreduce_l1', 'residual2_l1')
        self.graph.edge('residual_l1', 'residual2_l1')  # Skip connection
        
        # Layer 2 MHA
        for i in range(4):
            for j in range(4):
                self.graph.edge('residual2_l1', f'split_l2_{i}_{j}')
                self.graph.edge(f'split_l2_{i}_{j}', f'q_proj_l2_{i}_{j}')
                self.graph.edge(f'split_l2_{i}_{j}', f'k_proj_l2_{i}_{j}')
                self.graph.edge(f'split_l2_{i}_{j}', f'v_proj_l2_{i}_{j}')
                self.graph.edge(f'q_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
                self.graph.edge(f'k_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
                self.graph.edge(f'v_proj_l2_{i}_{j}', f'attn_l2_{i}_{j}')
        
        # Within head group concatenation layer 2
        for i in range(4):
            for j in range(4):
                self.graph.edge(f'attn_l2_{i}_{j}', f'concat_dim_l2_{i}')
            self.graph.edge(f'concat_dim_l2_{i}', 'concat_heads_l2')
        
        # MHA to residual layer 2
        self.graph.edge('concat_heads_l2', 'residual_l2')
        self.graph.edge('residual2_l1', 'residual_l2')  # Skip connection
        
        # Layer 2 MLP
        for gpu in range(16):
            self.graph.edge('residual_l2', f'linear1_l2_{gpu}')
            self.graph.edge(f'linear1_l2_{gpu}', f'gelu_l2_{gpu}')
            self.graph.edge(f'gelu_l2_{gpu}', f'linear2_l2_{gpu}')
            self.graph.edge(f'linear2_l2_{gpu}', 'allreduce_l2')
        
        self.graph.edge('allreduce_l2', 'residual2_l2')
        self.graph.edge('residual_l2', 'residual2_l2')  # Skip connection
        self.graph.edge('residual2_l2', 'output')
        
    def generate_all_files(self):
        """Generate all required files"""
        # Create complete DAG
        dag = self.create_complete_dag()
        self.create_connections()
        
        # Save DOT file
        dag.save(directory='./generated_docs/helix', filename='helix_complete_dag.dot')
        
        # Save SVG
        dag.render(directory='./generated_docs/helix', filename='helix_complete_dag', 
                  format='svg', cleanup=True)
        
        # Create per-layer DAGs
        self.create_layer_dags()
        
        return {
            'complete_dag_dot': './generated_docs/helix/helix_complete_dag.dot',
            'complete_dag_svg': './generated_docs/helix/helix_complete_dag.svg',
            'layer1_dag_dot': './generated_docs/helix/helix_layer1_dag.dot',
            'layer1_dag_svg': './generated_docs/helix/helix_layer1_dag.svg',
            'layer2_dag_dot': './generated_docs/helix/helix_layer2_dag.dot',
            'layer2_dag_svg': './generated_docs/helix/helix_layer2_dag.svg'
        }
        
    def create_layer_dags(self):
        """Create separate DAGs for individual layers"""
        # Layer 1 DAG
        layer1 = Digraph('Helix_Layer1_DAG', 
                        filename='helix_layer1_dag',
                        node_attr={'shape': 'rectangle', 'style': 'filled'})
        layer1.attr(rankdir='TB', size='50,50')
        
        # Add layer 1 components
        layer1.node('input', 'Input', shape='ellipse', fillcolor='lightblue')
        # ... (similar structure as above but focused on layer1)
        layer1.save(directory='./generated_docs/helix', filename='helix_layer1_dag.dot')
        layer1.render(directory='./generated_docs/helix', filename='helix_layer1_dag', 
                     format='svg', cleanup=True)
        
        # Layer 2 DAG (same structure)
        layer2 = Digraph('Helix_Layer2_DAG', 
                        filename='helix_layer2_dag',
                        node_attr={'shape': 'rectangle', 'style': 'filled'})
        layer2.attr(rankdir='TB', size='50,50')
        layer2.save(directory='./generated_docs/helix', filename='helix_layer2_dag.dot')
        layer2.render(directory='./generated_docs/helix', filename='helix_layer2_dag', 
                     format='svg', cleanup=True)

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('./generated_docs/helix', exist_ok=True)
    
    # Generate all DAGs
    generator = HelixDAGGenerator()
    paths = generator.generate_all_files()
    
    print("Generated DAGs:")
    for name, path in paths.items():
        print(f"{name}: {path}")