#!/usr/bin/env python3
"""
Baseline MoE DAG Generator (TP=8, PP=2)
Traditional approach with 8 experts per GPU, tensor parallelism within experts
"""

import graphviz
from typing import Dict, List, Tuple

class BaselineMoEDAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', 
                                  filename='moe_baseline_tp8_pp2.dot',
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
        self.tp_degree = 8
        self.pp_degree = 2
        self.experts_per_gpu = 8
        
    def create_node(self, name: str, label: str, shape: str, gpu: str, 
                   input_dims: str, output_dims: str, color: str = 'black') -> None:
        """Create a node with proper attributes"""
        full_label = f"{name}\\n{label}\\nGPU: {gpu}\\nIn: {input_dims}\\nOut: {output_dims}"
        self.dot.node(name, full_label, shape=shape, color=color)
    
    def create_edge(self, from_node: str, to_node: str, label: str = "", style: str = "solid") -> None:
        """Create an edge with optional label"""
        self.dot.edge(from_node, to_node, label=label, style=style)
    
    def generate_attention_layer(self, layer_idx: int, stage: int, start_gpu: int) -> None:
        """Generate attention layer with tensor parallelism"""
        layer_prefix = f"layer{layer_idx}_stage{stage}"
        
        # Input reshape
        self.create_node(
            f"{layer_prefix}_input_reshape",
            "Input Reshape",
            "ellipse",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # QKV projection (column parallel)
        for i in range(self.tp_degree):
            gpu_id = start_gpu + i
            self.create_node(
                f"{layer_prefix}_qkv_proj_{i}",
                f"QKV Projection (TP-{i})",
                "rectangle",
                str(gpu_id),
                f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim // self.tp_degree}",
                f"batch_size={self.batch_size * self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim * 3}"
            )
            self.create_edge(f"{layer_prefix}_input_reshape", f"{layer_prefix}_qkv_proj_{i}")
        
        # QKV reshape and split heads
        for i in range(self.tp_degree):
            gpu_id = start_gpu + i
            self.create_node(
                f"{layer_prefix}_qkv_reshape_{i}",
                f"QKV Reshape (TP-{i})",
                "ellipse",
                str(gpu_id),
                f"batch_size={self.batch_size * self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim * 3}",
                f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim}"
            )
            self.create_edge(f"{layer_prefix}_qkv_proj_{i}", f"{layer_prefix}_qkv_reshape_{i}")
        
        # Attention computation
        for i in range(self.tp_degree):
            gpu_id = start_gpu + i
            self.create_node(
                f"{layer_prefix}_attention_{i}",
                f"Multi-Head Attention (TP-{i})",
                "rectangle",
                str(gpu_id),
                f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim}",
                f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim}"
            )
            self.create_edge(f"{layer_prefix}_qkv_reshape_{i}", f"{layer_prefix}_attention_{i}")
        
        # Output projection (row parallel)
        for i in range(self.tp_degree):
            gpu_id = start_gpu + i
            self.create_node(
                f"{layer_prefix}_output_proj_{i}",
                f"Output Projection (TP-{i})",
                "rectangle",
                str(gpu_id),
                f"batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads // self.tp_degree}, head_dim={self.head_dim}",
                f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim // self.tp_degree}"
            )
            self.create_edge(f"{layer_prefix}_attention_{i}", f"{layer_prefix}_output_proj_{i}")
        
        # All-reduce for output
        self.create_node(
            f"{layer_prefix}_attention_allreduce",
            "All-Reduce Sum",
            "parallelogram",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim // self.tp_degree} (x8)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        for i in range(self.tp_degree):
            self.create_edge(f"{layer_prefix}_output_proj_{i}", f"{layer_prefix}_attention_allreduce")
        
        # Residual connection
        self.create_node(
            f"{layer_prefix}_attention_residual",
            "Residual Add",
            "ellipse",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x2)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        self.create_edge(f"{layer_prefix}_input_reshape", f"{layer_prefix}_attention_residual", style="dashed")
        self.create_edge(f"{layer_prefix}_attention_allreduce", f"{layer_prefix}_attention_residual")
    
    def generate_expert_layer(self, layer_idx: int, stage: int, start_gpu: int) -> None:
        """Generate expert layer with 8 experts per GPU"""
        layer_prefix = f"layer{layer_idx}_stage{stage}"
        
        # Gating network
        self.create_node(
            f"{layer_prefix}_gating",
            "Gating Network",
            "parallelogram",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, num_experts={self.num_experts}"
        )
        
        # Expert computation (8 experts per GPU)
        for gpu_offset in range(self.tp_degree):
            gpu_id = start_gpu + gpu_offset
            for expert_idx in range(self.experts_per_gpu):
                expert_id = gpu_offset * self.experts_per_gpu + expert_idx
                self.create_node(
                    f"{layer_prefix}_expert_{expert_id}_gpu{gpu_id}",
                    f"Expert {expert_id}",
                    "rectangle",
                    str(gpu_id),
                    f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}",
                    f"batch_size={self.batch_size * self.seq_len // self.num_experts}, hidden_dim={self.hidden_dim}"
                )
                # Connect gating to expert with dashed line for routing
                self.create_edge(f"{layer_prefix}_gating", f"{layer_prefix}_expert_{expert_id}_gpu{gpu_id}", style="dashed")
        
        # Expert output aggregation
        self.create_node(
            f"{layer_prefix}_expert_aggregate",
            "Expert Output Aggregation",
            "parallelogram",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x16)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        
        # Connect all experts to aggregation
        for gpu_offset in range(self.tp_degree):
            gpu_id = start_gpu + gpu_offset
            for expert_idx in range(self.experts_per_gpu):
                expert_id = gpu_offset * self.experts_per_gpu + expert_idx
                self.create_edge(f"{layer_prefix}_expert_{expert_id}_gpu{gpu_id}", f"{layer_prefix}_expert_aggregate")
        
        # Final residual connection
        self.create_node(
            f"{layer_prefix}_expert_residual",
            "Residual Add",
            "ellipse",
            f"{start_gpu}-{start_gpu+7}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim} (x2)",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
        # Need to connect from previous layer output
    
    def generate_pipeline_communication(self, from_stage: int, to_stage: int) -> None:
        """Generate pipeline communication between stages"""
        stage0_end = 7  # Last GPU of stage 0
        stage1_start = 8  # First GPU of stage 1
        
        self.create_node(
            f"pipeline_comm_{from_stage}_to_{to_stage}",
            "Pipeline Communication",
            "parallelogram",
            f"{stage0_end}-{stage1_start}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}",
            f"batch_size={self.batch_size * self.seq_len}, hidden_dim={self.hidden_dim}"
        )
    
    def generate(self) -> None:
        """Generate the complete baseline DAG"""
        # Model input
        self.create_node(
            "model_input",
            "Model Input",
            "ellipse",
            "0-15",
            "batch_size=1024, seq_len=10000, hidden_dim=8192",
            "batch_size=1024, seq_len=10000, hidden_dim=8192"
        )
        
        # Stage 0: Layers 0-1 on GPUs 0-7
        for layer_idx in range(2):
            self.generate_attention_layer(layer_idx, 0, 0)
            self.generate_expert_layer(layer_idx, 0, 0)
            
            # Connect layers
            if layer_idx == 0:
                self.create_edge("model_input", f"layer{layer_idx}_stage0_input_reshape")
            else:
                prev_layer = layer_idx - 1
                self.create_edge(f"layer{prev_layer}_stage0_expert_residual", f"layer{layer_idx}_stage0_input_reshape")
        
        # Pipeline communication between stage 0 and 1
        self.generate_pipeline_communication(0, 1)
        self.create_edge("layer1_stage0_expert_residual", "pipeline_comm_0_to_1")
        
        # Stage 1: Layers 2-3 on GPUs 8-15
        for layer_idx in range(2, 4):
            self.generate_attention_layer(layer_idx, 1, 8)
            self.generate_expert_layer(layer_idx, 1, 8)
            
            # Connect layers
            if layer_idx == 2:
                self.create_edge("pipeline_comm_0_to_1", f"layer{layer_idx}_stage1_input_reshape")
            else:
                self.create_edge(f"layer{layer_idx-1}_stage1_expert_residual", f"layer{layer_idx}_stage1_input_reshape")
        
        # Model output
        self.create_node(
            "model_output",
            "Model Output",
            "ellipse",
            "8-15",
            "batch_size=1024, seq_len=10000, hidden_dim=8192",
            "batch_size=1024, seq_len=10000, hidden_dim=8192"
        )
        self.create_edge("layer3_stage1_expert_residual", "model_output")
        
        # Save files
        self.dot.render('./outputs/2025-10-13-20-19-23/moe_baseline_tp8_pp2', format='svg', cleanup=False)
        self.dot.save('./outputs/2025-10-13-20-19-23/moe_baseline_tp8_pp2.dot')

if __name__ == "__main__":
    generator = BaselineMoEDAGGenerator()
    generator.generate()
    print("Baseline DAG generated successfully!")