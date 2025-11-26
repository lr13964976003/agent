#!/usr/bin/env python3
"""
Large-Scale Cross-Node Expert Parallelism DAG Generator
Generates complete deployment DAGs for MoE model with 61 layers (3 dense + 58 MoE)
"""

import graphviz
import json
import os
from typing import Dict, List, Tuple

class MoEDAGGenerator:
    def __init__(self):
        # Model specifications from deployment config
        self.token_dim = 7168
        self.mha_heads = 128
        self.head_dim = 128
        self.mlp_hidden = 2048
        self.precision = "BF16"
        self.dtype_size = 2  # bytes for BF16
        
        # Hardware specifications
        self.total_gpus = 3904
        self.nodes = 488
        self.gpus_per_node = 8
        
        # Model architecture
        self.dense_layers = 3
        self.moe_layers = 58
        self.total_layers = 61
        self.experts_per_layer = 64
        self.total_experts = 3712
        
        # Performance settings
        self.top_k = 2  # top-2 experts per token
        self.batch_size = 32  # variable, using typical value
        self.seq_len = 2048  # variable, using typical value
        
    def get_tensor_dimensions(self, tensor_type: str, layer_type: str = "mha") -> Dict[str, int]:
        """Get tensor dimensions for different operations"""
        if tensor_type == "input":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
        elif tensor_type == "mha_qkv":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "heads": self.mha_heads, "d_k": self.head_dim}
        elif tensor_type == "mha_output":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
        elif tensor_type == "ffn_intermediate":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "ffn_hidden": self.mlp_hidden}
        elif tensor_type == "expert_input":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
        elif tensor_type == "expert_output":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
        elif tensor_type == "gate_input":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
        elif tensor_type == "gate_output":
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "num_experts": self.experts_per_layer}
        else:
            return {"batch_size": self.batch_size, "seq_len": self.seq_len, "hidden_size": self.token_dim}
    
    def format_dimensions(self, dims: Dict[str, int]) -> str:
        """Format dimensions for node labels"""
        dim_str = ""
        for key, value in dims.items():
            if dim_str:
                dim_str += f", {key}={value}"
            else:
                dim_str += f"{key}={value}"
        return dim_str
    
    def generate_complete_dag(self) -> str:
        """Generate complete DAG for the entire model"""
        dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism MoE')
        dot.attr(rankdir='TB', size='100,200')
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Track nodes for connection
        prev_node = None
        
        # Input node
        input_dims = self.get_tensor_dimensions("input")
        dot.node('input', 
                f'INPUT\\nInput: {self.format_dimensions(input_dims)}\\nGPU: All', 
                shape='ellipse', fillcolor='lightgreen')
        prev_node = 'input'
        
        # Process each layer
        for layer_idx in range(self.total_layers):
            if layer_idx < self.dense_layers:
                # Dense layer (MHA + FFN)
                layer_nodes = self._add_dense_layer(dot, layer_idx, prev_node)
                prev_node = layer_nodes[-1] if isinstance(layer_nodes, list) else layer_nodes
            else:
                # MoE layer (MHA + Expert FFN)
                layer_nodes = self._add_moe_layer(dot, layer_idx, prev_node)
                prev_node = layer_nodes[-1] if isinstance(layer_nodes, list) else layer_nodes
        
        # Output node
        output_dims = self.get_tensor_dimensions("input")
        dot.node('output', 
                f'OUTPUT\\nInput: {self.format_dimensions(output_dims)}\\nGPU: All', 
                shape='ellipse', fillcolor='lightcoral')
        dot.edge(prev_node, 'output')
        
        return dot.source
    
    def _add_dense_layer(self, dot: graphviz.Digraph, layer_idx: int, prev_node: str) -> str:
        """Add a dense layer (MHA + FFN)"""
        gpu_id = 3712 + layer_idx  # Dense layers on GPUs 3712-3714
        node_id = 464
        gpu_within_node = layer_idx
        
        # MHA Layer Normalization
        ln1_dims = self.get_tensor_dimensions("input")
        dot.node(f'layer{layer_idx}_ln1', 
                f'LayerNorm\\nInput: {self.format_dimensions(ln1_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightyellow')
        dot.edge(prev_node, f'layer{layer_idx}_ln1')
        
        # MHA QKV Projection
        qkv_dims = self.get_tensor_dimensions("mha_qkv")
        dot.node(f'layer{layer_idx}_qkv', 
                f'QKV Projection\\nInput: {self.format_dimensions(ln1_dims)}\\nOutput: {self.format_dimensions(qkv_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_ln1', f'layer{layer_idx}_qkv')
        
        # MHA Attention
        attn_dims = self.get_tensor_dimensions("mha_output")
        dot.node(f'layer{layer_idx}_attn', 
                f'MHA Attention\\nInput: {self.format_dimensions(qkv_dims)}\\nOutput: {self.format_dimensions(attn_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_qkv', f'layer{layer_idx}_attn')
        
        # MHA Output Projection
        out_dims = self.get_tensor_dimensions("input")
        dot.node(f'layer{layer_idx}_mha_out', 
                f'MHA Output Proj\\nInput: {self.format_dimensions(attn_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_attn', f'layer{layer_idx}_mha_out')
        
        # Residual Add 1
        dot.node(f'layer{layer_idx}_res1', 
                f'Residual Add\\nInput1: {self.format_dimensions(out_dims)}\\nInput2: {self.format_dimensions(ln1_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightpink')
        dot.edge(f'layer{layer_idx}_mha_out', f'layer{layer_idx}_res1')
        dot.edge(prev_node, f'layer{layer_idx}_res1')
        
        # FFN Layer Normalization
        dot.node(f'layer{layer_idx}_ln2', 
                f'LayerNorm\\nInput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightyellow')
        dot.edge(f'layer{layer_idx}_res1', f'layer{layer_idx}_ln2')
        
        # FFN First Linear
        ffn1_dims = self.get_tensor_dimensions("ffn_intermediate")
        dot.node(f'layer{layer_idx}_ffn1', 
                f'FFN Linear1\\nInput: {self.format_dimensions(out_dims)}\\nOutput: {self.format_dimensions(ffn1_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_ln2', f'layer{layer_idx}_ffn1')
        
        # FFN GELU Activation
        dot.node(f'layer{layer_idx}_gelu', 
                f'GELU Activation\\nInput: {self.format_dimensions(ffn1_dims)}\\nOutput: {self.format_dimensions(ffn1_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightgreen')
        dot.edge(f'layer{layer_idx}_ffn1', f'layer{layer_idx}_gelu')
        
        # FFN Second Linear
        dot.node(f'layer{layer_idx}_ffn2', 
                f'FFN Linear2\\nInput: {self.format_dimensions(ffn1_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_gelu', f'layer{layer_idx}_ffn2')
        
        # Residual Add 2
        final_node = f'layer{layer_idx}_output'
        dot.node(final_node, 
                f'Residual Add\\nInput1: {self.format_dimensions(out_dims)}\\nInput2: {self.format_dimensions(out_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightpink')
        dot.edge(f'layer{layer_idx}_ffn2', final_node)
        dot.edge(f'layer{layer_idx}_res1', final_node)
        
        return final_node
    
    def _add_moe_layer(self, dot: graphviz.Digraph, layer_idx: int, prev_node: str) -> str:
        """Add an MoE layer (MHA + Expert FFN)"""
        # MHA part (same as dense layer)
        mha_end = self._add_mha_part(dot, layer_idx, prev_node)
        
        # Expert FFN part
        expert_start_gpu = (layer_idx - self.dense_layers) * self.experts_per_layer
        expert_end = self._add_expert_ffn(dot, layer_idx, mha_end, expert_start_gpu)
        
        return expert_end
    
    def _add_mha_part(self, dot: graphviz.Digraph, layer_idx: int, prev_node: str) -> str:
        """Add MHA part of the layer"""
        # Use GPU 3712 for MHA computation (shared across layers for load balancing)
        gpu_id = 3712
        
        # MHA Layer Normalization
        ln1_dims = self.get_tensor_dimensions("input")
        dot.node(f'layer{layer_idx}_ln1', 
                f'LayerNorm\\nInput: {self.format_dimensions(ln1_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightyellow')
        dot.edge(prev_node, f'layer{layer_idx}_ln1')
        
        # MHA QKV Projection
        qkv_dims = self.get_tensor_dimensions("mha_qkv")
        dot.node(f'layer{layer_idx}_qkv', 
                f'QKV Projection\\nInput: {self.format_dimensions(ln1_dims)}\\nOutput: {self.format_dimensions(qkv_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_ln1', f'layer{layer_idx}_qkv')
        
        # MHA Attention
        attn_dims = self.get_tensor_dimensions("mha_output")
        dot.node(f'layer{layer_idx}_attn', 
                f'MHA Attention\\nInput: {self.format_dimensions(qkv_dims)}\\nOutput: {self.format_dimensions(attn_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_qkv', f'layer{layer_idx}_attn')
        
        # MHA Output Projection
        out_dims = self.get_tensor_dimensions("input")
        dot.node(f'layer{layer_idx}_mha_out', 
                f'MHA Output Proj\\nInput: {self.format_dimensions(attn_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightblue')
        dot.edge(f'layer{layer_idx}_attn', f'layer{layer_idx}_mha_out')
        
        # Residual Add 1
        final_node = f'layer{layer_idx}_mha_output'
        dot.node(final_node, 
                f'Residual Add\\nInput1: {self.format_dimensions(out_dims)}\\nInput2: {self.format_dimensions(ln1_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {gpu_id}', 
                fillcolor='lightpink')
        dot.edge(f'layer{layer_idx}_mha_out', final_node)
        dot.edge(prev_node, final_node)
        
        return final_node
    
    def _add_expert_ffn(self, dot: graphviz.Digraph, layer_idx: int, prev_node: str, start_gpu: int) -> str:
        """Add Expert FFN part with routing and expert computation"""
        out_dims = self.get_tensor_dimensions("input")
        
        # Layer Normalization for FFN
        dot.node(f'layer{layer_idx}_ln2', 
                f'LayerNorm\\nInput: {self.format_dimensions(out_dims)}\\nGPU: {3713}', 
                fillcolor='lightyellow')
        dot.edge(prev_node, f'layer{layer_idx}_ln2')
        
        # Gate computation - selects top-k experts
        gate_dims = self.get_tensor_dimensions("gate_output")
        dot.node(f'layer{layer_idx}_gate', 
                f'Expert Gate\\nInput: {self.format_dimensions(out_dims)}\\nOutput: {self.format_dimensions(gate_dims)}\\nGPU: {3713}', 
                shape='parallelogram', fillcolor='orange')
        dot.edge(f'layer{layer_idx}_ln2', f'layer{layer_idx}_gate')
        
        # Expert routing (communication node)
        expert_input_dims = self.get_tensor_dimensions("expert_input")
        dot.node(f'layer{layer_idx}_route', 
                f'Expert Router\\nInput: {self.format_dimensions(expert_input_dims)}\\nTop-k: {self.top_k}\\nGPU: All', 
                shape='ellipse', fillcolor='lightgray')
        dot.edge(f'layer{layer_idx}_gate', f'layer{layer_idx}_route', style='dashed')
        dot.edge(f'layer{layer_idx}_ln2', f'layer{layer_idx}_route')
        
        # Expert computation nodes (one per GPU)
        expert_outputs = []
        for expert_id in range(self.experts_per_layer):
            gpu_id = start_gpu + expert_id
            node_id = gpu_id // 8
            gpu_within_node = gpu_id % 8
            
            # Expert FFN First Linear
            ffn1_dims = self.get_tensor_dimensions("ffn_intermediate")
            dot.node(f'layer{layer_idx}_expert{expert_id}_ffn1', 
                    f'Expert {expert_id} FFN1\\nInput: {self.format_dimensions(expert_input_dims)}\\nOutput: {self.format_dimensions(ffn1_dims)}\\nGPU: {gpu_id}', 
                    fillcolor='lightblue')
            dot.edge(f'layer{layer_idx}_route', f'layer{layer_idx}_expert{expert_id}_ffn1')
            
            # Expert GELU
            dot.node(f'layer{layer_idx}_expert{expert_id}_gelu', 
                    f'Expert {expert_id} GELU\\nInput: {self.format_dimensions(ffn1_dims)}\\nOutput: {self.format_dimensions(ffn1_dims)}\\nGPU: {gpu_id}', 
                    fillcolor='lightgreen')
            dot.edge(f'layer{layer_idx}_expert{expert_id}_ffn1', f'layer{layer_idx}_expert{expert_id}_gelu')
            
            # Expert FFN Second Linear
            expert_output_dims = self.get_tensor_dimensions("expert_output")
            expert_node = f'layer{layer_idx}_expert{expert_id}_output'
            dot.node(expert_node, 
                    f'Expert {expert_id} FFN2\\nInput: {self.format_dimensions(ffn1_dims)}\\nOutput: {self.format_dimensions(expert_output_dims)}\\nGPU: {gpu_id}', 
                    fillcolor='lightblue')
            dot.edge(f'layer{layer_idx}_expert{expert_id}_gelu', expert_node)
            expert_outputs.append(expert_node)
        
        # Expert aggregation
        dot.node(f'layer{layer_idx}_expert_agg', 
                f'Expert Aggregation\\nInput: {len(expert_outputs)} experts\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {3713}', 
                shape='parallelogram', fillcolor='purple')
        
        for expert_output in expert_outputs:
            dot.edge(expert_output, f'layer{layer_idx}_expert_agg')
        
        # Final residual add
        final_node = f'layer{layer_idx}_output'
        dot.node(final_node, 
                f'Final Residual Add\\nInput1: {self.format_dimensions(out_dims)}\\nInput2: {self.format_dimensions(out_dims)}\\nOutput: {self.format_dimensions(out_dims)}\\nGPU: {3713}', 
                fillcolor='lightpink')
        dot.edge(f'layer{layer_idx}_expert_agg', final_node)
        dot.edge(prev_node, final_node)
        
        return final_node
    
    def generate_layer_dag(self, layer_idx: int) -> str:
        """Generate DAG for a single layer"""
        dot = graphviz.Digraph(comment=f'Layer {layer_idx} DAG')
        dot.attr(rankdir='TB', size='50,100')
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Input node
        input_dims = self.get_tensor_dimensions("input")
        dot.node('input', 
                f'INPUT\\nInput: {self.format_dimensions(input_dims)}\\nGPU: All', 
                shape='ellipse', fillcolor='lightgreen')
        
        if layer_idx < self.dense_layers:
            # Dense layer
            self._add_dense_layer(dot, layer_idx, 'input')
        else:
            # MoE layer
            self._add_moe_layer(dot, layer_idx, 'input')
        
        return dot.source
    
    def generate_expert_parallelism_dag(self) -> str:
        """Generate DAG showing expert parallelism across GPUs"""
        dot = graphviz.Digraph(comment='Expert Parallelism Overview')
        dot.attr(rankdir='LR', size='200,50')
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Show first few MoE layers with expert distribution
        for layer_idx in range(3, min(6, self.total_layers)):  # Show layers 3-5
            layer_start_gpu = (layer_idx - self.dense_layers) * self.experts_per_layer
            
            # Layer input
            input_dims = self.get_tensor_dimensions("input")
            dot.node(f'layer{layer_idx}_input', 
                    f'Layer {layer_idx} Input\\nInput: {self.format_dimensions(input_dims)}\\nGPU: {3712}', 
                    shape='ellipse', fillcolor='lightgreen')
            
            # Show experts for this layer
            expert_nodes = []
            for expert_id in range(min(8, self.experts_per_layer)):  # Show first 8 experts
                gpu_id = layer_start_gpu + expert_id
                node_id = gpu_id // 8
                
                expert_dims = self.get_tensor_dimensions("expert_output")
                expert_node = f'layer{layer_idx}_expert{expert_id}'
                dot.node(expert_node, 
                        f'Expert {expert_id}\\nNode {node_id} GPU {gpu_id % 8}\\nOutput: {self.format_dimensions(expert_dims)}', 
                        fillcolor='lightblue')
                dot.edge(f'layer{layer_idx}_input', expert_node)
                expert_nodes.append(expert_node)
            
            # Layer output
            output_dims = self.get_tensor_dimensions("input")
            dot.node(f'layer{layer_idx}_output', 
                    f'Layer {layer_idx} Output\\nInput: {self.format_dimensions(output_dims)}\\nGPU: {3713}', 
                    shape='ellipse', fillcolor='lightcoral')
            
            for expert_node in expert_nodes:
                dot.edge(expert_node, f'layer{layer_idx}_output')
        
        return dot.source

def main():
    generator = MoEDAGGenerator()
    
    # Create output directory
    output_dir = "../outputs/2025-11-26-11-58-22"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate complete DAG
    print("Generating complete DAG...")
    complete_dag = generator.generate_complete_dag()
    
    with open(f"{output_dir}/complete_model_dag.dot", "w") as f:
        f.write(complete_dag)
    
    # Generate layer-specific DAGs
    for layer_idx in [0, 3, 10]:  # Dense layer, first MoE layer, middle MoE layer
        print(f"Generating DAG for layer {layer_idx}...")
        layer_dag = generator.generate_layer_dag(layer_idx)
        
        with open(f"{output_dir}/layer_{layer_idx}_dag.dot", "w") as f:
            f.write(layer_dag)
    
    # Generate expert parallelism overview
    print("Generating expert parallelism DAG...")
    expert_dag = generator.generate_expert_parallelism_dag()
    
    with open(f"{output_dir}/expert_parallelism_dag.dot", "w") as f:
        f.write(expert_dag)
    
    # Generate SVG images
    print("Generating SVG images...")
    try:
        import subprocess
        
        # Convert DOT files to SVG
        for dot_file in ["complete_model_dag", "layer_0_dag", "layer_3_dag", "layer_10_dag", "expert_parallelism_dag"]:
            dot_path = f"{output_dir}/{dot_file}.dot"
            svg_path = f"{output_dir}/{dot_file}.svg"
            subprocess.run(["dot", "-Tsvg", dot_path, "-o", svg_path], check=True)
            print(f"Generated {svg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Graphviz not available, skipping SVG generation")
    
    # Save paths for submission
    paths = {
        "complete_model_dag": f"{output_dir}/complete_model_dag.dot",
        "layer_0_dag": f"{output_dir}/layer_0_dag.dot",
        "layer_3_dag": f"{output_dir}/layer_3_dag.dot",
        "layer_10_dag": f"{output_dir}/layer_10_dag.dot",
        "expert_parallelism_dag": f"{output_dir}/expert_parallelism_dag.dot",
        "complete_model_svg": f"{output_dir}/complete_model_dag.svg",
        "layer_0_svg": f"{output_dir}/layer_0_dag.svg",
        "layer_3_svg": f"{output_dir}/layer_3_dag.svg",
        "layer_10_svg": f"{output_dir}/layer_10_dag.svg",
        "expert_parallelism_svg": f"{output_dir}/expert_parallelism_dag.svg"
    }
    
    with open(f"{output_dir}/generated_dags.json", "w") as f:
        json.dump(paths, f, indent=2)
    
    print("DAG generation complete!")
    return paths

if __name__ == "__main__":
    main()