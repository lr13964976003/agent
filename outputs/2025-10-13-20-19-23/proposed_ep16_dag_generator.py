#!/usr/bin/env python3
"""
Proposed MoE DAG Generator (EP=16)
Large-scale cross-node expert parallelism with one expert per GPU
"""

import graphviz
from typing import Dict, List, Tuple

class ProposedEP16DAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph('MoE_Proposed_EP16', 
                                  filename='moe_proposed_ep16.dot',
                                  format='svg',
                                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})
        self.dot.attr('node', fontname='Arial', fontsize='10')
        
        # Model parameters
        self.batch_size = 1024
        self.seq_len = 10000
        self.hidden_dim = 8192
        self.heads = 16
        self.head_dim = 512
        self.ffn_hidden = 32768
        self.num_layers = 4
        self.num_experts = 16
        
        # Parallelism configuration
        self.ep_degree = 16
        self.experts_per_gpu = 1
        
    def create_node(self, name: str, label: str, shape: str, gpu: str, 
                   input_dims: str, output_dims: str, color: str = 'black') -> None:
        """Create a node with proper attributes"""
        full_label = f"{name}\\n{label}\\nGPU: {gpu}\\nIn: {input_dims}\\nOut: {output_dims}"
        self.dot.node(name, full_label, shape=shape, color=color)
    
    def create_edge(self, from_node: str, to_node: str, label: str = "", style: str = "solid") -> None:
        """Create an edge with optional label"""
        self.dot.edge(from_node, to_node, label=label, style=style)
    
    def generate_attention_module(self, layer_idx: int, gpu_id: int) -> str:
        """Generate attention module for a specific GPU"""
        prefix = f"layer{layer_idx}_gpu{gpu_id}"
        
        # Input reshape
        self.create_node(
            f"{prefix}_attn_input_reshape",
            "Input Reshape",
            "ellipse",
            str(gpu_id),
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # Q projection
        self.create_node(
            f"{prefix}_q_proj",
            "Q Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, heads={self.heads}, head_dim={self.head_dim}"
        )
        
        # K projection
        self.create_node(
            f"{prefix}_k_proj",
            "K Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, heads={self.heads}, head_dim={self.head_dim}"
        )
        
        # V projection
        self.create_node(
            f"{prefix}_v_proj",
            "V Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, heads={self.heads}, head_dim={self.head_dim}"
        )
        
        # Multi-head attention
        self.create_node(
            f"{prefix}_mha",
            "Multi-Head Attention",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads}, head_dim={self.head_dim}",
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads}, head_dim={self.head_dim}"
        )
        
        # Output projection
        self.create_node(
            f"{prefix}_o_proj",
            "Output Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads}, head_dim={self.head_dim}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # Residual connection
        self.create_node(
            f"{prefix}_attn_residual",
            "Residual Add",
            "ellipse",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x2)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # Connect attention components
        self.create_edge(f"{prefix}_attn_input_reshape", f"{prefix}_q_proj")
        self.create_edge(f"{prefix}_attn_input_reshape", f"{prefix}_k_proj")
        self.create_edge(f"{prefix}_attn_input_reshape", f"{prefix}_v_proj")
        self.create_edge(f"{prefix}_q_proj", f"{prefix}_mha")
        self.create_edge(f"{prefix}_k_proj", f"{prefix}_mha")
        self.create_edge(f"{prefix}_v_proj", f"{prefix}_mha")
        self.create_edge(f"{prefix}_mha", f"{prefix}_o_proj")
        self.create_edge(f"{prefix}_o_proj", f"{prefix}_attn_residual")
        self.create_edge(f"{prefix}_attn_input_reshape", f"{prefix}_attn_residual", style="dashed")
        
        return f"{prefix}_attn_residual"
    
    def generate_expert_module(self, layer_idx: int, expert_id: int, gpu_id: int) -> str:
        """Generate expert module for a specific expert on a specific GPU"""
        prefix = f"layer{layer_idx}_expert{expert_id}_gpu{gpu_id}"
        
        # Expert MLP layers
        self.create_node(
            f"{prefix}_gate_proj",
            "Gate Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, ffn_hidden={self.ffn_hidden}"
        )
        
        self.create_node(
            f"{prefix}_up_proj",
            "Up Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, ffn_hidden={self.ffn_hidden}"
        )
        
        self.create_node(
            f"{prefix}_activation",
            "GELU Activation",
            "ellipse",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, ffn_hidden={self.ffn_hidden}",
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, ffn_hidden={self.ffn_hidden}"
        )
        
        self.create_node(
            f"{prefix}_down_proj",
            "Down Projection",
            "rectangle",
            str(gpu_id),
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, ffn_hidden={self.ffn_hidden}",
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}"
        )
        
        # Connect expert components
        self.create_edge(f"{prefix}_gate_proj", f"{prefix}_activation")
        self.create_edge(f"{prefix}_up_proj", f"{prefix}_down_proj")
        self.create_edge(f"{prefix}_activation", f"{prefix}_down_proj")
        
        return f"{prefix}_down_proj"
    
    def generate_gating_network(self, layer_idx: int) -> str:
        """Generate gating network that runs on all GPUs"""
        prefix = f"layer{layer_idx}_gating"
        
        # Gating network (replicated across all GPUs)
        self.create_node(
            f"{prefix}_network",
            "Gating Network",
            "parallelogram",
            "0-15",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, num_experts={self.num_experts}"
        )
        
        # Expert selection
        self.create_node(
            f"{prefix}_selection",
            "Expert Selection",
            "parallelogram",
            "0-15",
            f"batch_size={self.batch_size * self.seq_len}, num_experts={self.num_experts}",
            f"batch_size={self.batch_size * self.seq_len}, selected_experts=2"
        )
        
        self.create_edge(f"{prefix}_network", f"{prefix}_selection")
        
        return f"{prefix}_selection"
    
    def generate_expert_routing(self, layer_idx: int, gating_output: str) -> str:
        """Generate expert routing and communication"""
        prefix = f"layer{layer_idx}_routing"
        
        # Token routing to experts
        for expert_id in range(self.num_experts):
            gpu_id = expert_id  # One expert per GPU
            self.create_node(
                f"{prefix}_route_to_expert{expert_id}",
                f"Route to Expert {expert_id}",
                "parallelogram",
                str(gpu_id),
                f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
                f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}"
            )
            # Connect gating to routing with dashed line
            self.create_edge(gating_output, f"{prefix}_route_to_expert{expert_id}", style="dashed")
        
        # Expert computation
        expert_outputs = []
        for expert_id in range(self.num_experts):
            gpu_id = expert_id
            expert_output = self.generate_expert_module(layer_idx, expert_id, gpu_id)
            self.create_edge(f"{prefix}_route_to_expert{expert_id}", expert_output.split('_down_proj')[0] + "_gate_proj")
            self.create_edge(f"{prefix}_route_to_expert{expert_id}", expert_output.split('_down_proj')[0] + "_up_proj")
            expert_outputs.append(expert_output)
        
        # Expert output aggregation
        self.create_node(
            f"{prefix}_aggregate",
            "Aggregate Expert Outputs",
            "parallelogram",
            "0-15",
            f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim} (x16)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        for expert_id, expert_output in enumerate(expert_outputs):
            self.create_edge(expert_output, f"{prefix}_aggregate")
        
        # Weighted sum based on gating scores
        self.create_node(
            f"{prefix}_weighted_sum",
            "Weighted Sum",
            "ellipse",
            "0-15",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x2)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        self.create_edge(f"{prefix}_aggregate", f"{prefix}_weighted_sum")
        self.create_edge(gating_output, f"{prefix}_weighted_sum", style="dashed")
        
        return f"{prefix}_weighted_sum"
    
    def generate_layer(self, layer_idx: int) -> str:
        """Generate a complete layer with attention and expert components"""
        # Create layer input broadcast
        self.create_node(
            f"layer{layer_idx}_input_broadcast",
            "Layer Input Broadcast",
            "ellipse",
            "0-15",
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # Generate attention on all GPUs (replicated)
        attn_outputs = []
        for gpu_id in range(16):
            attn_output = self.generate_attention_module(layer_idx, gpu_id)
            self.create_edge(f"layer{layer_idx}_input_broadcast", f"layer{layer_idx}_gpu{gpu_id}_attn_input_reshape")
            attn_outputs.append(attn_output)
        
        # Broadcast attention outputs to all GPUs for expert routing
        for gpu_id in range(16):
            self.create_node(
                f"layer{layer_idx}_attn_output_broadcast_gpu{gpu_id}",
                "Attention Output Broadcast",
                "ellipse",
                str(gpu_id),
                f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
                f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
            )
            self.create_edge(attn_outputs[gpu_id], f"layer{layer_idx}_attn_output_broadcast_gpu{gpu_id}")
        
        # Generate gating and expert routing
        gating_output = self.generate_gating_network(layer_idx)
        expert_output = self.generate_expert_routing(layer_idx, gating_output)
        
        # Connect attention outputs to gating
        for gpu_id in range(16):
            self.create_edge(f"layer{layer_idx}_attn_output_broadcast_gpu{gpu_id}", f"layer{layer_idx}_gating_network")
        
        # Final residual connection for the layer
        self.create_node(
            f"layer{layer_idx}_output",
            "Layer Output",
            "ellipse",
            "0-15",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x2)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        for gpu_id in range(16):
            self.create_edge(f"layer{layer_idx}_attn_output_broadcast_gpu{gpu_id}", f"layer{layer_idx}_output", style="dashed")
        self.create_edge(expert_output, f"layer{layer_idx}_output")
        
        return f"layer{layer_idx}_output"
    
    def generate(self) -> None:
        """Generate the complete proposed EP16 DAG"""
        # Model input
        self.create_node(
            "model_input",
            "Model Input",
            "ellipse",
            "0-15",
            "batch_size=1024, seq_len=10000, hidden_dim=8192",
            "batch_size=1024, seq_len=10000, hidden_dim=8192"
        )
        
        # Generate all 4 layers
        layer_outputs = []
        for layer_idx in range(self.num_layers):
            layer_output = self.generate_layer(layer_idx)
            layer_outputs.append(layer_output)
            
            # Connect layers
            if layer_idx == 0:
                self.create_edge("model_input", "layer0_input_broadcast")
            else:
                prev_layer = layer_idx - 1
                self.create_edge(layer_outputs[prev_layer], f"layer{layer_idx}_input_broadcast")
        
        # Model output
        self.create_node(
            "model_output",
            "Model Output",
            "ellipse",
            "0-15",
            "batch_size=1024, seq_len=10000, hidden_dim=8192",
            "batch_size=1024, seq_len=10000, hidden_dim=8192"
        )
        self.create_edge(layer_outputs[-1], "model_output")
        
        # Save files
        self.dot.render('./outputs/2025-10-13-20-19-23/moe_proposed_ep16', format='svg', cleanup=False)
        self.dot.save('./outputs/2025-10-13-20-19-23/moe_proposed_ep16.dot')

if __name__ == "__main__":
    generator = ProposedEP16DAGGenerator()
    generator.generate()
    print("Proposed EP16 DAG generated successfully!")