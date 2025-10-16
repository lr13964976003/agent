#!/usr/bin/env python3
"""
MA Separation DAG Generator
Generates complete deployment DAG for the MA Separation paper
"""

import graphviz
from typing import Dict, List, Tuple
import os
import json

class MASeparationDAGGenerator:
    def __init__(self):
        self.dot = graphviz.Digraph('MA_Separation_DAG', format='svg')
        self.dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        
        # Model parameters
        self.batch_size = 'batch_size'
        self.seq_len = 2048
        self.hidden_dim = 4096
        self.attention_heads = 32
        self.heads_per_gpu = 4
        self.dk = self.hidden_dim // self.attention_heads  # 128
        self.experts_total = 16
        self.experts_per_gpu = 2
        self.expert_hidden = 16384
        self.layers = 4
        
        # GPU assignments
        self.attention_gpus = list(range(8))  # GPUs 0-7
        self.moe_gpus = list(range(8, 16))    # GPUs 8-15
    
    def add_node(self, node_id: str, label: str, shape: str, gpu: str, input_dim: str, output_dim: str):
        """Add a node with proper formatting"""
        label_with_dims = f"{label}\\nInput: {input_dim}\\nOutput: {output_dim}\\nGPU: {gpu}"
        if shape == "ellipse":
            self.dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightblue')
        elif shape == "rectangle":
            if "Expert" in label:
                self.dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightcoral')
            elif "Projection" in label or "Linear" in label:
                self.dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightgreen')
            else:
                self.dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightyellow')
        elif shape == "parallelogram":
            self.dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightpink')
        else:
            self.dot.node(node_id, label_with_dims, shape=shape)
    
    def add_edge(self, from_node: str, to_node: str, label: str = "", style: str = "solid"):
        """Add an edge with optional label"""
        self.dot.edge(from_node, to_node, label=label, style=style)
    
    def create_layer_dag(self, layer_idx: int, prev_node: str) -> str:
        """Create complete DAG for one transformer layer"""
        layer_prefix = f"layer{layer_idx}_"
        
        # Attention computation across GPUs 0-7
        current_node = prev_node
        
        # Attention layer computations for each GPU
        for gpu_idx, gpu_id in enumerate(self.attention_gpus):
            gpu_str = str(gpu_id)
            head_start = gpu_idx * self.heads_per_gpu
            head_end = (gpu_idx + 1) * self.heads_per_gpu
            
            # Layer norm (replicated across all attention GPUs)
            attn_norm_id = f"{layer_prefix}attn_norm_gpu{gpu_id}"
            input_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
            self.add_node(attn_norm_id, "Layer Norm", "rectangle", gpu_str, input_shape, input_shape)
            self.add_edge(current_node, attn_norm_id)
            
            # QKV Projections
            q_proj_id = f"{layer_prefix}q_proj_gpu{gpu_id}"
            q_output = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_gpu}, d_k={self.dk}]"
            self.add_node(q_proj_id, "Q Projection", "rectangle", gpu_str, input_shape, q_output)
            self.add_edge(attn_norm_id, q_proj_id)
            
            k_proj_id = f"{layer_prefix}k_proj_gpu{gpu_id}"
            self.add_node(k_proj_id, "K Projection", "rectangle", gpu_str, input_shape, q_output)
            self.add_edge(attn_norm_id, k_proj_id)
            
            v_proj_id = f"{layer_prefix}v_proj_gpu{gpu_id}"
            self.add_node(v_proj_id, "V Projection", "rectangle", gpu_str, input_shape, q_output)
            self.add_edge(attn_norm_id, v_proj_id)
            
            # All-gather Q,K,V across GPUs (communication)
            gather_q_id = f"{layer_prefix}gather_q_gpu{gpu_id}"
            gather_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.attention_heads}, d_k={self.dk}]"
            self.add_node(gather_q_id, "All-Gather Q", "ellipse", gpu_str, q_output, gather_shape)
            self.add_edge(q_proj_id, gather_q_id)
            
            gather_k_id = f"{layer_prefix}gather_k_gpu{gpu_id}"
            self.add_node(gather_k_id, "All-Gather K", "ellipse", gpu_str, q_output, gather_shape)
            self.add_edge(k_proj_id, gather_k_id)
            
            gather_v_id = f"{layer_prefix}gather_v_gpu{gpu_id}"
            self.add_node(gather_v_id, "All-Gather V", "ellipse", gpu_str, q_output, gather_shape)
            self.add_edge(v_proj_id, gather_v_id)
            
            # Attention computation
            attn_scores_id = f"{layer_prefix}attn_scores_gpu{gpu_id}"
            score_shape = f"[batch_size={self.batch_size}, heads={self.heads_per_gpu}, seq_len={self.seq_len}, seq_len={self.seq_len}]"
            self.add_node(attn_scores_id, "QK^T / sqrt(d_k)", "rectangle", gpu_str, gather_shape, score_shape)
            self.add_edge(gather_q_id, attn_scores_id)
            self.add_edge(gather_k_id, attn_scores_id)
            
            # Softmax
            softmax_id = f"{layer_prefix}softmax_gpu{gpu_id}"
            self.add_node(softmax_id, "Softmax", "rectangle", gpu_str, score_shape, score_shape)
            self.add_edge(attn_scores_id, softmax_id)
            
            # Attention output
            attn_out_id = f"{layer_prefix}attn_out_gpu{gpu_id}"
            attn_out_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.heads_per_gpu}, d_k={self.dk}]"
            self.add_node(attn_out_id, "Attention×V", "rectangle", gpu_str, score_shape, attn_out_shape)
            self.add_edge(softmax_id, attn_out_id)
            self.add_edge(gather_v_id, attn_out_id)
            
            # Output projection
            out_proj_id = f"{layer_prefix}out_proj_gpu{gpu_id}"
            out_proj_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim//8}]"
            self.add_node(out_proj_id, "Output Projection", "rectangle", gpu_str, attn_out_shape, out_proj_shape)
            self.add_edge(attn_out_id, out_proj_id)
        
        # All-reduce across attention GPUs
        attn_all_reduce_id = f"{layer_prefix}attn_all_reduce"
        attn_reduce_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
        self.add_node(attn_all_reduce_id, "All-Reduce Attention", "ellipse", "GPUs 0-7", 
                     f"8×[{self.hidden_dim//8}]", attn_reduce_shape)
        
        for gpu_id in self.attention_gpus:
            out_proj_id = f"{layer_prefix}out_proj_gpu{gpu_id}"
            self.add_edge(out_proj_id, attn_all_reduce_id)
        
        # Residual connection
        attn_residual_id = f"{layer_prefix}attn_residual"
        self.add_node(attn_residual_id, "Residual Add", "ellipse", "all GPUs", 
                     f"{attn_reduce_shape}, {input_shape}", attn_reduce_shape)
        self.add_edge(prev_node, attn_residual_id)
        self.add_edge(attn_all_reduce_id, attn_residual_id)
        
        # MoE computation across GPUs 8-15
        # Broadcast attention output to MoE GPUs
        broadcast_id = f"{layer_prefix}broadcast_to_moe"
        self.add_node(broadcast_id, "Broadcast to MoE GPUs", "ellipse", "GPUs 8-15", attn_reduce_shape, attn_reduce_shape)
        self.add_edge(attn_residual_id, broadcast_id)
        
        # Layer norm for MoE
        moe_norm_id = f"{layer_prefix}moe_norm"
        self.add_node(moe_norm_id, "Layer Norm", "rectangle", "all GPUs", attn_reduce_shape, attn_reduce_shape)
        self.add_edge(broadcast_id, moe_norm_id)
        
        # Gate computation
        gate_id = f"{layer_prefix}gate"
        gate_output = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, experts={self.experts_total}]"
        self.add_node(gate_id, "Gate Network", "parallelogram", "all GPUs", attn_reduce_shape, gate_output)
        self.add_edge(moe_norm_id, gate_id)
        
        # Expert computation
        expert_outputs = []
        for gpu_idx, gpu_id in enumerate(self.moe_gpus):
            gpu_str = str(gpu_id)
            expert_start = gpu_idx * self.experts_per_gpu
            
            for expert_idx in range(self.experts_per_gpu):
                expert_id = expert_start + expert_idx
                
                # Route tokens based on gate
                route_id = f"{layer_prefix}route_expert{expert_id}"
                route_input_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
                route_output_shape = f"[tokens, hidden_dim={self.hidden_dim}]"
                self.add_node(route_id, f"Route to Expert {expert_id}", "parallelogram", gpu_str, route_input_shape, route_output_shape)
                self.add_edge(gate_id, route_id, style="dashed")
                self.add_edge(moe_norm_id, route_id)
                
                # First linear layer (up-projection)
                expert_up_id = f"{layer_prefix}expert{expert_id}_up"
                up_output_shape = f"[tokens, hidden_dim={self.expert_hidden}]"
                self.add_node(expert_up_id, f"Expert {expert_id} Up", "rectangle", gpu_str, route_output_shape, up_output_shape)
                self.add_edge(route_id, expert_up_id)
                
                # Activation
                activation_id = f"{layer_prefix}expert{expert_id}_activation"
                self.add_node(activation_id, "GELU", "rectangle", gpu_str, up_output_shape, up_output_shape)
                self.add_edge(expert_up_id, activation_id)
                
                # Second linear layer (down-projection)
                expert_down_id = f"{layer_prefix}expert{expert_id}_down"
                down_output_shape = f"[tokens, hidden_dim={self.hidden_dim}]"
                self.add_node(expert_down_id, f"Expert {expert_id} Down", "rectangle", gpu_str, up_output_shape, down_output_shape)
                self.add_edge(activation_id, expert_down_id)
                
                # Route back to original sequence
                route_back_id = f"{layer_prefix}route_back_expert{expert_id}"
                route_back_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
                self.add_node(route_back_id, f"Route Back Expert {expert_id}", "parallelogram", gpu_str, down_output_shape, route_back_shape)
                self.add_edge(expert_down_id, route_back_id)
                
                expert_outputs.append(route_back_id)
        
        # Aggregate expert outputs
        expert_agg_id = f"{layer_prefix}expert_agg"
        expert_agg_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
        self.add_node(expert_agg_id, "Aggregate Experts", "ellipse", "all GPUs", expert_agg_shape, expert_agg_shape)
        
        for expert_out in expert_outputs:
            self.add_edge(expert_out, expert_agg_id)
        
        # Final residual connection
        final_residual_id = f"{layer_prefix}final_residual"
        self.add_node(final_residual_id, "Final Residual Add", "ellipse", "all GPUs", 
                     f"{expert_agg_shape}, {attn_reduce_shape}", attn_reduce_shape)
        self.add_edge(attn_residual_id, final_residual_id)
        self.add_edge(expert_agg_id, final_residual_id)
        
        return final_residual_id
    
    def create_complete_dag(self):
        """Create the complete MA Separation DAG"""
        
        # Model input
        input_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}]"
        self.add_node("model_input", "Model Input", "ellipse", "CPU", "Raw tokens", input_shape)
        
        # Token embedding
        embed_id = "token_embedding"
        embed_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
        self.add_node(embed_id, "Token Embedding", "rectangle", "all GPUs", input_shape, embed_shape)
        self.add_edge("model_input", embed_id)
        
        # Position embedding
        pos_embed_id = "position_embedding"
        self.add_node(pos_embed_id, "Position Embedding", "rectangle", "all GPUs", 
                     f"[seq_len={self.seq_len}]", embed_shape)
        
        # Add embeddings
        add_embed_id = "add_embeddings"
        self.add_node(add_embed_id, "Add Embeddings", "ellipse", "all GPUs", 
                     f"{embed_shape}, {embed_shape}", embed_shape)
        self.add_edge(embed_id, add_embed_id)
        self.add_edge(pos_embed_id, add_embed_id)
        
        # Layer connections
        prev_node = add_embed_id
        
        for layer_idx in range(self.layers):
            layer_output = self.create_layer_dag(layer_idx, prev_node)
            prev_node = layer_output
        
        # Final layer norm
        final_norm_id = "final_norm"
        self.add_node(final_norm_id, "Final Layer Norm", "rectangle", "all GPUs", embed_shape, embed_shape)
        self.add_edge(prev_node, final_norm_id)
        
        # Output projection
        output_proj_id = "output_projection"
        vocab_size = "vocab_size"
        self.add_node(output_proj_id, "Output Projection", "rectangle", "all GPUs", 
                     embed_shape, f"[batch_size={self.batch_size}, seq_len={self.seq_len}, vocab_size={vocab_size}]")
        self.add_edge(final_norm_id, output_proj_id)
        
        # Final output
        self.add_node("model_output", "Model Output", "ellipse", "all GPUs", 
                     f"[batch_size={self.batch_size}, seq_len={self.seq_len}, vocab_size={vocab_size}]", 
                     f"[batch_size={self.batch_size}, seq_len={self.seq_len}, vocab_size={vocab_size}]")
        self.add_edge(output_proj_id, "model_output")
    
    def generate_files(self):
        """Generate both DOT and SVG files"""
        self.create_complete_dag()
        
        # Save DOT file
        dot_content = self.dot.source
        
        # Write DOT file
        dot_path = './outputs/2025-10-16-14-01-39/ma_separation_dag.dot'
        svg_path = './outputs/2025-10-16-14-01-39/ma_separation_dag.svg'
        
        with open(dot_path, 'w') as f:
            f.write(dot_content)
        
        # Render SVG
        self.dot.render('./outputs/2025-10-16-14-01-39/ma_separation_dag', format='svg', cleanup=True)
        
        return {
            'dot_path': dot_path,
            'svg_path': svg_path
        }

if __name__ == "__main__":
    generator = MASeparationDAGGenerator()
    paths = generator.generate_files()
    print(json.dumps(paths, indent=2))