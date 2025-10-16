#!/usr/bin/env python3
"""
Baseline DAGs Generator
Creates DAGs for tensor parallelism (TP=8), pipeline parallelism (PP=2), and hybrid (TP=8, PP=2)
"""

import graphviz
import json

class BaselineDAGGenerator:
    def __init__(self, tp_size: int = 8, pp_size: int = 2):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.layers = 4
        self.hidden_dim = 4096
        self.attention_heads = 32
        self.dk = self.hidden_dim // self.attention_heads
        self.batch_size = 'batch_size'
        self.seq_len = 2048
        self.expert_hidden = 16384
        self.experts_total = 16
    
    def add_node(self, dot, node_id: str, label: str, shape: str, gpu: str, input_dim: str, output_dim: str):
        """Add a node with proper formatting"""
        label_with_dims = f"{label}\\nInput: {input_dim}\\nOutput: {output_dim}\\nGPU: {gpu}"
        if shape == "ellipse":
            dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightblue')
        elif shape == "rectangle":
            if "Expert" in label:
                dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightcoral')
            elif "Projection" in label or "Linear" in label:
                dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightgreen')
            else:
                dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightyellow')
        elif shape == "parallelogram":
            dot.node(node_id, label_with_dims, shape=shape, style='filled', fillcolor='lightpink')
        else:
            dot.node(node_id, label_with_dims, shape=shape)
    
    def create_tensor_parallel_dag(self):
        """Create DAG for TP=8"""
        dot = graphviz.Digraph('Tensor_Parallel_DAG', format='svg')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        
        # Model input
        input_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}]"
        self.add_node(dot, "tp_input", "Model Input", "ellipse", "CPU", "Raw tokens", input_shape)
        
        # Embedding
        embed_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
        self.add_node(dot, "tp_embed", "Token + Position Embedding", "rectangle", "all GPUs", input_shape, embed_shape)
        dot.edge("tp_input", "tp_embed")
        
        prev_node = "tp_embed"
        
        for layer_idx in range(self.layers):
            layer_prefix = f"tp_layer{layer_idx}_"
            
            # Attention across all 8 GPUs with tensor parallelism
            # Layer norm (replicated)
            attn_norm_id = f"{layer_prefix}norm"
            self.add_node(dot, attn_norm_id, "Layer Norm", "rectangle", "all GPUs", embed_shape, embed_shape)
            dot.edge(prev_node, attn_norm_id)
            
            # QKV projections (tensor parallel)
            for gpu_id in range(self.tp_size):
                gpu_str = str(gpu_id)
                heads_per_gpu = self.attention_heads // self.tp_size  # 4 heads per GPU
                
                q_proj_id = f"{layer_prefix}q_proj_gpu{gpu_id}"
                q_output = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={heads_per_gpu}, d_k={self.dk}]"
                self.add_node(dot, q_proj_id, "Q Projection", "rectangle", gpu_str, embed_shape, q_output)
                dot.edge(attn_norm_id, q_proj_id)
                
                k_proj_id = f"{layer_prefix}k_proj_gpu{gpu_id}"
                self.add_node(dot, k_proj_id, "K Projection", "rectangle", gpu_str, embed_shape, q_output)
                dot.edge(attn_norm_id, k_proj_id)
                
                v_proj_id = f"{layer_prefix}v_proj_gpu{gpu_id}"
                self.add_node(dot, v_proj_id, "V Projection", "rectangle", gpu_str, embed_shape, q_output)
                dot.edge(attn_norm_id, v_proj_id)
                
                # Attention computation
                attn_scores_id = f"{layer_prefix}attn_scores_gpu{gpu_id}"
                score_shape = f"[batch_size={self.batch_size}, heads={heads_per_gpu}, seq_len={self.seq_len}, seq_len={self.seq_len}]"
                self.add_node(dot, attn_scores_id, "QK^T / sqrt(d_k)", "rectangle", gpu_str, q_output, score_shape)
                self.add_node(dot, f"{layer_prefix}gather_k_gpu{gpu_id}", "All-Gather K", "ellipse", gpu_str, q_output, q_output)
                self.add_node(dot, f"{layer_prefix}gather_q_gpu{gpu_id}", "All-Gather Q", "ellipse", gpu_str, q_output, q_output)
                dot.edge(q_proj_id, f"{layer_prefix}gather_q_gpu{gpu_id}")
                dot.edge(k_proj_id, f"{layer_prefix}gather_k_gpu{gpu_id}")
                dot.edge(f"{layer_prefix}gather_q_gpu{gpu_id}", attn_scores_id)
                dot.edge(f"{layer_prefix}gather_k_gpu{gpu_id}", attn_scores_id)
                
                # Softmax and attention output
                softmax_id = f"{layer_prefix}softmax_gpu{gpu_id}"
                self.add_node(dot, softmax_id, "Softmax", "rectangle", gpu_str, score_shape, score_shape)
                dot.edge(attn_scores_id, softmax_id)
                
                attn_out_id = f"{layer_prefix}attn_out_gpu{gpu_id}"
                attn_out_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={heads_per_gpu}, d_k={self.dk}]"
                self.add_node(dot, attn_out_id, "Attention×V", "rectangle", gpu_str, score_shape, attn_out_shape)
                self.add_node(dot, f"{layer_prefix}gather_v_gpu{gpu_id}", "All-Gather V", "ellipse", gpu_str, q_output, q_output)
                dot.edge(v_proj_id, f"{layer_prefix}gather_v_gpu{gpu_id}")
                dot.edge(softmax_id, attn_out_id)
                dot.edge(f"{layer_prefix}gather_v_gpu{gpu_id}", attn_out_id)
                
                # Output projection
                out_proj_id = f"{layer_prefix}out_proj_gpu{gpu_id}"
                out_proj_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim//self.tp_size}]"
                self.add_node(dot, out_proj_id, "Output Projection", "rectangle", gpu_str, attn_out_shape, out_proj_shape)
                dot.edge(attn_out_id, out_proj_id)
            
            # All-reduce for attention output
            attn_all_reduce_id = f"{layer_prefix}attn_all_reduce"
            self.add_node(dot, attn_all_reduce_id, "All-Reduce Attention", "ellipse", "all GPUs", 
                         f"{self.tp_size}×[{self.hidden_dim//self.tp_size}]", embed_shape)
            for gpu_id in range(self.tp_size):
                dot.edge(f"{layer_prefix}out_proj_gpu{gpu_id}", attn_all_reduce_id)
            
            # Residual connection
            attn_residual_id = f"{layer_prefix}attn_residual"
            self.add_node(dot, attn_residual_id, "Residual Add", "ellipse", "all GPUs", 
                         f"{embed_shape}, {embed_shape}", embed_shape)
            dot.edge(prev_node, attn_residual_id)
            dot.edge(attn_all_reduce_id, attn_residual_id)
            
            # MoE layer (tensor parallel)
            moe_norm_id = f"{layer_prefix}moe_norm"
            self.add_node(dot, moe_norm_id, "Layer Norm", "rectangle", "all GPUs", embed_shape, embed_shape)
            dot.edge(attn_residual_id, moe_norm_id)
            
            # MoE experts distributed
            expert_outputs = []
            for gpu_id in range(self.tp_size):
                gpu_str = str(gpu_id)
                experts_per_gpu = self.experts_total // self.tp_size  # 2 experts per GPU
                expert_start = gpu_id * experts_per_gpu
                
                for expert_idx in range(experts_per_gpu):
                    expert_id = expert_start + expert_idx
                    
                    # Expert computation
                    expert_up_id = f"{layer_prefix}expert{expert_id}_up"
                    up_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.expert_hidden}]"
                    self.add_node(dot, expert_up_id, f"Expert {expert_id} Up", "rectangle", gpu_str, embed_shape, up_shape)
                    dot.edge(moe_norm_id, expert_up_id)
                    
                    activation_id = f"{layer_prefix}expert{expert_id}_activation"
                    self.add_node(dot, activation_id, "GELU", "rectangle", gpu_str, up_shape, up_shape)
                    dot.edge(expert_up_id, activation_id)
                    
                    expert_down_id = f"{layer_prefix}expert{expert_id}_down"
                    down_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim//self.tp_size}]"
                    self.add_node(dot, expert_down_id, f"Expert {expert_id} Down", "rectangle", gpu_str, up_shape, down_shape)
                    dot.edge(activation_id, expert_down_id)
                    
                    expert_outputs.append(expert_down_id)
            
            # All-reduce for MoE
            moe_all_reduce_id = f"{layer_prefix}moe_all_reduce"
            self.add_node(dot, moe_all_reduce_id, "All-Reduce MoE", "ellipse", "all GPUs", 
                         f"{self.tp_size}×[{self.hidden_dim//self.tp_size}]", embed_shape)
            for expert_out in expert_outputs:
                dot.edge(expert_out, moe_all_reduce_id)
            
            # Final residual
            final_residual_id = f"{layer_prefix}final_residual"
            self.add_node(dot, final_residual_id, "Residual Add", "ellipse", "all GPUs", 
                         f"{embed_shape}, {embed_shape}", embed_shape)
            dot.edge(attn_residual_id, final_residual_id)
            dot.edge(moe_all_reduce_id, final_residual_id)
            
            prev_node = final_residual_id
        
        # Final output
        final_output_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, vocab_size=vocab_size]"
        self.add_node(dot, "tp_output", "Output Projection", "rectangle", "all GPUs", embed_shape, final_output_shape)
        dot.edge(prev_node, "tp_output")
        
        return dot
    
    def create_pipeline_parallel_dag(self):
        """Create DAG for PP=2"""
        dot = graphviz.Digraph('Pipeline_Parallel_DAG', format='svg')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
        
        # Model input
        input_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}]"
        self.add_node(dot, "pp_input", "Model Input", "ellipse", "CPU", "Raw tokens", input_shape)
        
        # Embedding
        embed_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
        self.add_node(dot, "pp_embed", "Token + Position Embedding", "rectangle", "GPU 0", input_shape, embed_shape)
        dot.edge("pp_input", "pp_embed")
        
        # Pipeline stages
        layers_per_stage = self.layers // self.pp_size  # 2 layers per stage
        
        prev_node = "pp_embed"
        
        for stage_idx in range(self.pp_size):
            stage_gpus = [stage_idx]  # Each stage on separate GPU
            
            for layer_idx in range(layers_per_stage):
                actual_layer_idx = stage_idx * layers_per_stage + layer_idx
                layer_prefix = f"pp_stage{stage_idx}_layer{actual_layer_idx}_"
                
                # Attention layer
                attn_norm_id = f"{layer_prefix}norm"
                self.add_node(dot, attn_norm_id, "Layer Norm", "rectangle", str(stage_idx), embed_shape, embed_shape)
                dot.edge(prev_node, attn_norm_id)
                
                # QKV projections
                q_proj_id = f"{layer_prefix}q_proj"
                q_output = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, heads={self.attention_heads}, d_k={self.dk}]"
                self.add_node(dot, q_proj_id, "Q Projection", "rectangle", str(stage_idx), embed_shape, q_output)
                dot.edge(attn_norm_id, q_proj_id)
                
                k_proj_id = f"{layer_prefix}k_proj"
                self.add_node(dot, k_proj_id, "K Projection", "rectangle", str(stage_idx), embed_shape, q_output)
                dot.edge(attn_norm_id, k_proj_id)
                
                v_proj_id = f"{layer_prefix}v_proj"
                self.add_node(dot, v_proj_id, "V Projection", "rectangle", str(stage_idx), embed_shape, q_output)
                dot.edge(attn_norm_id, v_proj_id)
                
                # Attention computation
                attn_scores_id = f"{layer_prefix}attn_scores"
                score_shape = f"[batch_size={self.batch_size}, heads={self.attention_heads}, seq_len={self.seq_len}, seq_len={self.seq_len}]"
                self.add_node(dot, attn_scores_id, "QK^T / sqrt(d_k)", "rectangle", str(stage_idx), q_output, score_shape)
                dot.edge(q_proj_id, attn_scores_id)
                dot.edge(k_proj_id, attn_scores_id)
                
                # Softmax and attention output
                softmax_id = f"{layer_prefix}softmax"
                self.add_node(dot, softmax_id, "Softmax", "rectangle", str(stage_idx), score_shape, score_shape)
                dot.edge(attn_scores_id, softmax_id)
                
                attn_out_id = f"{layer_prefix}attn_out"
                self.add_node(dot, attn_out_id, "Attention×V", "rectangle", str(stage_idx), score_shape, q_output)
                dot.edge(softmax_id, attn_out_id)
                dot.edge(v_proj_id, attn_out_id)
                
                # Output projection
                out_proj_id = f"{layer_prefix}out_proj"
                self.add_node(dot, out_proj_id, "Output Projection", "rectangle", str(stage_idx), q_output, embed_shape)
                dot.edge(attn_out_id, out_proj_id)
                
                # Residual
                attn_residual_id = f"{layer_prefix}attn_residual"
                self.add_node(dot, attn_residual_id, "Residual Add", "ellipse", str(stage_idx), 
                             f"{embed_shape}, {embed_shape}", embed_shape)
                dot.edge(prev_node, attn_residual_id)
                dot.edge(out_proj_id, attn_residual_id)
                
                # MoE layer
                moe_norm_id = f"{layer_prefix}moe_norm"
                self.add_node(dot, moe_norm_id, "Layer Norm", "rectangle", str(stage_idx), embed_shape, embed_shape)
                dot.edge(attn_residual_id, moe_norm_id)
                
                # Expert selection and routing
                gate_id = f"{layer_prefix}gate"
                gate_output = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, experts={self.experts_total}]"
                self.add_node(dot, gate_id, "Gate Network", "parallelogram", str(stage_idx), embed_shape, gate_output)
                dot.edge(moe_norm_id, gate_id)
                
                # Expert computation (all on same GPU for this stage)
                expert_outputs = []
                experts_per_gpu = self.experts_total
                for expert_idx in range(experts_per_gpu):
                    expert_up_id = f"{layer_prefix}expert{expert_idx}_up"
                    up_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.expert_hidden}]"
                    self.add_node(dot, expert_up_id, f"Expert {expert_idx} Up", "rectangle", str(stage_idx), embed_shape, up_shape)
                    dot.edge(moe_norm_id, expert_up_id)
                    
                    activation_id = f"{layer_prefix}expert{expert_idx}_activation"
                    self.add_node(dot, activation_id, "GELU", "rectangle", str(stage_idx), up_shape, up_shape)
                    dot.edge(expert_up_id, activation_id)
                    
                    expert_down_id = f"{layer_prefix}expert{expert_idx}_down"
                    down_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}]"
                    self.add_node(dot, expert_down_id, f"Expert {expert_idx} Down", "rectangle", str(stage_idx), up_shape, down_shape)
                    dot.edge(activation_id, expert_down_id)
                    
                    # Route back
                    route_back_id = f"{layer_prefix}route_back_expert{expert_idx}"
                    self.add_node(dot, route_back_id, f"Route Back Expert {expert_idx}", "parallelogram", str(stage_idx), down_shape, down_shape)
                    dot.edge(expert_down_id, route_back_id)
                    dot.edge(gate_id, route_back_id, style="dashed")
                    
                    expert_outputs.append(route_back_id)
                
                # Aggregate experts
                expert_agg_id = f"{layer_prefix}expert_agg"
                self.add_node(dot, expert_agg_id, "Aggregate Experts", "ellipse", str(stage_idx), embed_shape, embed_shape)
                for exp_out in expert_outputs:
                    dot.edge(exp_out, expert_agg_id)
                
                # Final residual
                final_residual_id = f"{layer_prefix}final_residual"
                self.add_node(dot, final_residual_id, "Final Residual Add", "ellipse", str(stage_idx), 
                             f"{embed_shape}, {embed_shape}", embed_shape)
                dot.edge(attn_residual_id, final_residual_id)
                dot.edge(expert_agg_id, final_residual_id)
                
                prev_node = final_residual_id
                
                # Pipeline communication between stages
                if stage_idx < self.pp_size - 1:
                    comm_id = f"{layer_prefix}pipeline_comm"
                    self.add_node(dot, comm_id, "Pipeline Communication", "ellipse", f"GPU {stage_idx+1}", embed_shape, embed_shape)
                    dot.edge(final_residual_id, comm_id)
                    prev_node = comm_id
        
        # Final output
        final_output_shape = f"[batch_size={self.batch_size}, seq_len={self.seq_len}, vocab_size=vocab_size]"
        self.add_node(dot, "pp_output", "Output Projection", "rectangle", str(self.pp_size-1), embed_shape, final_output_shape)
        dot.edge(prev_node, "pp_output")
        
        return dot
    
    def generate_all_baseline_dags(self):
        """Generate all baseline DAGs"""
        results = {}
        
        # Tensor Parallel (TP=8)
        tp_dot = self.create_tensor_parallel_dag()
        tp_dot_path = './outputs/2025-10-16-14-01-39/tensor_parallel_dag.dot'
        tp_svg_path = './outputs/2025-10-16-14-01-39/tensor_parallel_dag.svg'
        
        with open(tp_dot_path, 'w') as f:
            f.write(tp_dot.source)
        tp_dot.render('./outputs/2025-10-16-14-01-39/tensor_parallel_dag', format='svg', cleanup=True)
        
        results['tensor_parallel'] = {
            'dot_path': tp_dot_path,
            'svg_path': tp_svg_path
        }
        
        # Pipeline Parallel (PP=2)
        pp_dot = self.create_pipeline_parallel_dag()
        pp_dot_path = './outputs/2025-10-16-14-01-39/pipeline_parallel_dag.dot'
        pp_svg_path = './outputs/2025-10-16-14-01-39/pipeline_parallel_dag.svg'
        
        with open(pp_dot_path, 'w') as f:
            f.write(pp_dot.source)
        pp_dot.render('./outputs/2025-10-16-14-01-39/pipeline_parallel_dag', format='svg', cleanup=True)
        
        results['pipeline_parallel'] = {
            'dot_path': pp_dot_path,
            'svg_path': pp_svg_path
        }
        
        return results

if __name__ == "__main__":
    generator = BaselineDAGGenerator()
    results = generator.generate_all_baseline_dags()
    print(json.dumps(results, indent=2))