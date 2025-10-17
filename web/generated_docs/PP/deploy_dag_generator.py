#!/usr/bin/env python3
"""
DAG Generator for Layer-wise Cache-Aware Deployment
Following the paper: Layer-wise Distribution Strategy for Cache-Aware Deployment
"""

import graphviz
from typing import List, Dict, Tuple

class DAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph(comment='Layer-wise Cache-Aware Deployment DAG')
        self.dot.attr(rankdir='TB', splines='ortho', nodesep='1.0', ranksep='1.5')
        self.gpu_counter = 0
        self.node_counter = 0
        
    def add_node(self, name: str, label: str, gpu_id: int, shape: str = 'rectangle', 
                 input_dims: str = '', output_dims: str = ''):
        """Add a computation node to the DAG"""
        full_label = f"{name}\\nGPU: {gpu_id}\\nInput: {input_dims}\\nOutput: {output_dims}"
        self.dot.node(name, full_label, shape=shape, style='filled', fillcolor='lightblue')
        return name
    
    def add_comm_node(self, name: str, label: str, from_gpu: int, to_gpu: int,
                     input_dims: str = '', output_dims: str = ''):
        """Add a communication node to the DAG"""
        full_label = f"{label}\\nFrom GPU {from_gpu} to GPU {to_gpu}\\nInput: {input_dims}\\nOutput: {output_dims}"
        self.dot.node(name, full_label, shape='ellipse', style='filled', fillcolor='lightyellow')
        return name
    
    def add_routing_node(self, name: str, label: str, input_dims: str = '', output_dims: str = ''):
        """Add a routing/aggregation node to the DAG"""
        full_label = f"{label}\\nInput: {input_dims}\\nOutput: {output_dims}"
        self.dot.node(name, full_label, shape='parallelogram', style='filled', fillcolor='lightgray')
        return name
    
    def add_edge(self, from_node: str, to_node: str, label: str = ''):
        """Add an edge between nodes"""
        self.dot.edge(from_node, to_node, label=label)
    
    def generate_dense_model_dag(self):
        """Generate complete DAG for 16-layer dense model"""
        
        # Model dimensions
        batch_size = 1024
        seq_len = 10000
        hidden_size = 8192  # 16 heads * 512 d_k
        ffn_hidden = 32768
        heads = 16
        d_k = 512
        
        # Input node
        input_node = self.add_node(
            "input", "Total Model Input", 0,
            input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
            output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        )
        
        prev_node = input_node
        
        # Generate 16 layers distributed across 16 GPUs
        for layer_id in range(16):
            gpu_id = layer_id  # Each layer on separate GPU
            
            # Layer prefix
            prefix = f"layer{layer_id}"
            
            # LayerNorm 1
            ln1_name = f"{prefix}_ln1"
            ln1_node = self.add_node(
                ln1_name, f"LayerNorm {layer_id}.1", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # Multi-Head Attention
            # Q projection
            q_proj_name = f"{prefix}_q_proj"
            q_proj_node = self.add_node(
                q_proj_name, f"Q Projection {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}"
            )
            
            # K projection
            k_proj_name = f"{prefix}_k_proj"
            k_proj_node = self.add_node(
                k_proj_name, f"K Projection {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}"
            )
            
            # V projection
            v_proj_name = f"{prefix}_v_proj"
            v_proj_node = self.add_node(
                v_proj_name, f"V Projection {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}"
            )
            
            # Attention computation
            attn_name = f"{prefix}_attn"
            attn_node = self.add_node(
                attn_name, f"Attention {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}"
            )
            
            # Attention output projection
            attn_out_name = f"{prefix}_attn_out"
            attn_out_node = self.add_node(
                attn_out_name, f"Attention Output {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # Residual connection 1
            residual1_name = f"{prefix}_residual1"
            residual1_node = self.add_node(
                residual1_name, f"Residual Add 1 {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # LayerNorm 2
            ln2_name = f"{prefix}_ln2"
            ln2_node = self.add_node(
                ln2_name, f"LayerNorm {layer_id}.2", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # FFN
            ffn1_name = f"{prefix}_ffn1"
            ffn1_node = self.add_node(
                ffn1_name, f"FFN Linear 1 {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}"
            )
            
            # GELU activation
            gelu_name = f"{prefix}_gelu"
            gelu_node = self.add_node(
                gelu_name, f"GELU {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}"
            )
            
            # FFN output
            ffn2_name = f"{prefix}_ffn2"
            ffn2_node = self.add_node(
                ffn2_name, f"FFN Linear 2 {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # Residual connection 2
            residual2_name = f"{prefix}_residual2"
            residual2_node = self.add_node(
                residual2_name, f"Residual Add 2 {layer_id}", gpu_id,
                input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
            )
            
            # Communication between layers
            if layer_id > 0:
                # Add communication node from previous GPU to current GPU
                comm_name = f"{prefix}_comm"
                comm_node = self.add_comm_node(
                    comm_name, f"Layer Output Transfer",
                    from_gpu=layer_id-1, to_gpu=gpu_id,
                    input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
                    output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
                )
                self.add_edge(prev_node, comm_name)
                self.add_edge(comm_name, ln1_name)
            else:
                self.add_edge(prev_node, ln1_name)
            
            # Connect attention nodes
            self.add_edge(ln1_name, q_proj_name)
            self.add_edge(ln1_name, k_proj_name)
            self.add_edge(ln1_name, v_proj_name)
            self.add_edge(q_proj_name, attn_name)
            self.add_edge(k_proj_name, attn_name)
            self.add_edge(v_proj_name, attn_name)
            self.add_edge(attn_name, attn_out_name)
            self.add_edge(attn_out_name, residual1_name)
            
            # Add residual connection from ln1 to residual1
            if layer_id == 0:
                self.add_edge(input_node, residual1_name, label="Residual")
            else:
                prev_layer_name = f"layer{layer_id-1}_residual2"
                self.add_edge(prev_layer_name, residual1_name, label="Residual")
            
            # Connect FFN nodes
            self.add_edge(residual1_name, ln2_name)
            self.add_edge(ln2_name, ffn1_name)
            self.add_edge(ffn1_name, gelu_name)
            self.add_edge(gelu_name, ffn2_name)
            self.add_edge(ffn2_name, residual2_name)
            
            # Add residual connection from residual1 to residual2
            self.add_edge(residual1_name, residual2_name, label="Residual")
            
            prev_node = residual2_name
        
        # Output node
        output_node = self.add_node(
            "output", "Total Model Output", 15,
            input_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}",
            output_dims=f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        )
        self.add_edge(prev_node, output_node)
        
        return self.dot

if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("./generated_docs/PP", exist_ok=True)
    
    # Generate DAG
    generator = DAGGenerator()
    dag = generator.generate_dense_model_dag()
    
    # Save DOT file
    dot_file = "./generated_docs/PP/layer_wise_deployment_dense.dot"
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    
    # Generate SVG
    dag.render("./generated_docs/PP/layer_wise_deployment_dense", format='svg', cleanup=True)
    
    # Generate PNG for verification
    dag.render("./generated_docs/PP/layer_wise_deployment_dense", format='png', cleanup=True)
    
    print(f"Generated DAG files:")
    print(f"DOT: {dot_file}")
    print(f"SVG: ./generated_docs/PP/layer_wise_deployment_dense.svg")
    print(f"PNG: ./generated_docs/PP/layer_wise_deployment_dense.png")