#!/usr/bin/env python3
"""
LLM Deployment DAG Generator for 30B MoE Model
Generates a comprehensive DAG showing all parallel strategies with GPU boundaries
"""

import graphviz
import json
from typing import Dict, List, Tuple

class LLMDeploymentDAGGenerator:
    def __init__(self):
        # Model configuration from deployment
        self.model_config = {
            'total_parameters': '30B',
            'layers': 16,
            'experts_per_layer': 64,
            'hidden_size': 1024,
            'ffn_hidden_size': 2048,
            'num_heads': 16,
            'head_dim': 64,
            'batch_size': 128,
            'sequence_length': 1024,
            'precision': 'FP16'
        }
        
        # Parallel configuration
        self.parallel_config = {
            'tensor_parallel_size': 8,
            'pipeline_parallel_size': 4,
            'expert_parallel_size': 8,
            'data_parallel_size': 2,
            'total_gpus': 512
        }
        
        # Hardware configuration
        self.gpu_memory = 64  # GB
        self.gpu_compute = 400  # TFlops
        
        # Node tracking
        self.nodes = {}
        self.edges = []
        
    def get_gpu_id(self, dp: int, pp: int, tp: int, ep: int) -> int:
        """Calculate GPU ID from parallel dimensions"""
        return (dp * self.parallel_config['pipeline_parallel_size'] * 
                self.parallel_config['tensor_parallel_size'] * self.parallel_config['expert_parallel_size'] +
                pp * self.parallel_config['tensor_parallel_size'] * self.parallel_config['expert_parallel_size'] +
                tp * self.parallel_config['expert_parallel_size'] +
                ep)
    
    def create_node_attributes(self, name: str, node_type: str, gpu_id: int, 
                              input_dim: Dict, output_dim: Dict, label: str = None) -> Dict:
        """Create node attributes for graphviz"""
        # Format dimensions
        input_str = f"[batch_size={input_dim.get('batch_size', '?')}, seq_len={input_dim.get('seq_len', '?')}, heads={input_dim.get('heads', '?')}, d_k={input_dim.get('d_k', '?')}]"
        output_str = f"[batch_size={output_dim.get('batch_size', '?')}, seq_len={output_dim.get('seq_len', '?')}, heads={output_dim.get('heads', '?')}, d_k={output_dim.get('d_k', '?')}]"
        
        # Node attributes based on type
        if node_type == 'computation':
            shape = 'rectangle'
            style = 'filled'
            fillcolor = 'lightblue'
        elif node_type == 'communication':
            shape = 'ellipse'
            style = 'dashed'
            fillcolor = 'lightyellow'
        elif node_type == 'routing':
            shape = 'parallelogram'
            style = 'filled'
            fillcolor = 'lightgreen'
        elif node_type == 'input':
            shape = 'ellipse'
            style = 'filled'
            fillcolor = 'lightpink'
        elif node_type == 'output':
            shape = 'ellipse'
            style = 'filled'
            fillcolor = 'orange'
        else:
            shape = 'rectangle'
            style = 'filled'
            fillcolor = 'white'
        
        # Create full label with dimensions
        full_label = f"{label or name}\\nInput: {input_str}\\nOutput: {output_str}"
        
        return {
            'label': full_label,
            'shape': shape,
            'style': style,
            'fillcolor': fillcolor
        }
    
    def add_edge(self, from_node: str, to_node: str, edge_type: str = 'solid', label: str = None):
        """Add an edge between nodes"""
        self.edges.append({
            'from': from_node,
            'to': to_node,
            'type': edge_type,
            'label': label
        })
    
    def generate_dag(self):
        """Generate the complete DAG"""
        dot = graphviz.Digraph(comment='LLM 30B MoE Deployment DAG')
        dot.attr(rankdir='TB', size='20,30')
        dot.attr('node', fontsize='9')
        dot.attr('edge', fontsize='8')
        
        # Create a simplified but comprehensive DAG showing all key components
        # Input node
        input_attrs = self.create_node_attributes(
            'input', 'input', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Input'
        )
        dot.node('input', **input_attrs)
        
        # Token embedding (computation)
        embed_attrs = self.create_node_attributes(
            'token_embedding', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 1, 'd_k': 1},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Token Embedding\nVocab:32K→Hidden:1024'
        )
        dot.node('token_embedding', **embed_attrs)
        dot.edge('input', 'token_embedding')
        
        # Position embedding (computation)
        pos_embed_attrs = self.create_node_attributes(
            'position_embedding', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Position Embedding\nSeq:1024→Hidden:1024'
        )
        dot.node('position_embedding', **pos_embed_attrs)
        dot.edge('token_embedding', 'position_embedding')
        
        # Layer normalization (computation)
        layernorm1_attrs = self.create_node_attributes(
            'layer_norm_1', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Layer Norm\nHidden:1024'
        )
        dot.node('layer_norm_1', **layernorm1_attrs)
        dot.edge('position_embedding', 'layer_norm_1')
        
        # Multi-head attention components (tensor parallel)
        q_proj_attrs = self.create_node_attributes(
            'q_projection', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            'Q Projection\nTP: Hidden:1024→Heads:2×64'
        )
        dot.node('q_projection', **q_proj_attrs)
        dot.edge('layer_norm_1', 'q_projection')
        
        k_proj_attrs = self.create_node_attributes(
            'k_projection', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            'K Projection\nTP: Hidden:1024→Heads:2×64'
        )
        dot.node('k_projection', **k_proj_attrs)
        dot.edge('layer_norm_1', 'k_projection')
        
        v_proj_attrs = self.create_node_attributes(
            'v_projection', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            'V Projection\nTP: Hidden:1024→Heads:2×64'
        )
        dot.node('v_projection', **v_proj_attrs)
        dot.edge('layer_norm_1', 'v_projection')
        
        # Attention computation (computation)
        attention_attrs = self.create_node_attributes(
            'attention', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            'Attention\nQK^T V: Softmax'
        )
        dot.node('attention', **attention_attrs)
        dot.edge('q_projection', 'attention')
        dot.edge('k_projection', 'attention')
        dot.edge('v_projection', 'attention')
        
        # Attention output projection (tensor parallel)
        attn_out_attrs = self.create_node_attributes(
            'attention_output', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 2, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Attention Output\nTP: Heads:2×64→Hidden:1024'
        )
        dot.node('attention_output', **attn_out_attrs)
        dot.edge('attention', 'attention_output')
        
        # Add attention all-reduce across tensor groups (communication)
        attn_allreduce_attrs = self.create_node_attributes(
            'attention_allreduce', 'communication', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'All-Reduce\nAttention Output'
        )
        dot.node('attention_allreduce', **attn_allreduce_attrs)
        dot.edge('attention_output', 'attention_allreduce')
        
        # Residual connection (computation)
        residual1_attrs = self.create_node_attributes(
            'residual_1', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Residual Add\nHidden:1024'
        )
        dot.node('residual_1', **residual1_attrs)
        dot.edge('attention_allreduce', 'residual_1')
        dot.edge('position_embedding', 'residual_1')  # Skip connection
        
        # Layer normalization 2 (computation)
        layernorm2_attrs = self.create_node_attributes(
            'layer_norm_2', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Layer Norm\nHidden:1024'
        )
        dot.node('layer_norm_2', **layernorm2_attrs)
        dot.edge('residual_1', 'layer_norm_2')
        
        # Expert routing (routing)
        router_attrs = self.create_node_attributes(
            'expert_router', 'routing', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Expert Router\nTop-K:2 Selection'
        )
        dot.node('expert_router', **router_attrs)
        dot.edge('layer_norm_2', 'expert_router')
        
        # Expert computation (computation)
        for expert_idx in range(4):  # Show 4 experts
            expert_attrs = self.create_node_attributes(
                f'expert_{expert_idx}', 'computation', 0,
                {'batch_size': 32, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
                {'batch_size': 32, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
                f'Expert {expert_idx}\nMLP: 1024→2048→1024'
            )
            dot.node(f'expert_{expert_idx}', **expert_attrs)
            dot.edge('expert_router', f'expert_{expert_idx}')
            
            # Expert all-to-all communication (communication)
            expert_all2all_attrs = self.create_node_attributes(
                f'expert_all2all_{expert_idx}', 'communication', 0,
                {'batch_size': 32, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
                {'batch_size': 32, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
                f'All-to-All\nExpert {expert_idx}'
            )
            dot.node(f'expert_all2all_{expert_idx}', **expert_all2all_attrs)
            dot.edge(f'expert_{expert_idx}', f'expert_all2all_{expert_idx}', style='dashed')
        
        # Expert aggregation (routing)
        expert_agg_attrs = self.create_node_attributes(
            'expert_aggregation', 'routing', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Expert Aggregation\nWeighted Sum'
        )
        dot.node('expert_aggregation', **expert_agg_attrs)
        
        # Connect all expert all-to-all to aggregation
        for expert_idx in range(4):
            dot.edge(f'expert_all2all_{expert_idx}', 'expert_aggregation')
        
        # MLP output projection (tensor parallel)
        mlp_out_attrs = self.create_node_attributes(
            'mlp_output', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'MLP Output\nTP: Hidden:1024→Hidden:1024'
        )
        dot.node('mlp_output', **mlp_out_attrs)
        dot.edge('expert_aggregation', 'mlp_output')
        
        # Add MLP all-reduce across tensor groups (communication)
        mlp_allreduce_attrs = self.create_node_attributes(
            'mlp_allreduce', 'communication', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'All-Reduce\nMLP Output'
        )
        dot.node('mlp_allreduce', **mlp_allreduce_attrs)
        dot.edge('mlp_output', 'mlp_allreduce')
        
        # Residual connection 2 (computation)
        residual2_attrs = self.create_node_attributes(
            'residual_2', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Residual Add\nHidden:1024'
        )
        dot.node('residual_2', **residual2_attrs)
        dot.edge('mlp_allreduce', 'residual_2')
        dot.edge('residual_1', 'residual_2')  # Skip connection
        
        # Final layer norm (computation)
        final_layernorm_attrs = self.create_node_attributes(
            'final_layer_norm', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            'Final Layer Norm\nHidden:1024'
        )
        dot.node('final_layer_norm', **final_layernorm_attrs)
        dot.edge('residual_2', 'final_layer_norm')
        
        # Output projection (computation)
        output_proj_attrs = self.create_node_attributes(
            'output_projection', 'computation', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 16, 'd_k': 64},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 1, 'd_k': 32000},
            'Output Projection\nHidden:1024→Vocab:32K'
        )
        dot.node('output_projection', **output_proj_attrs)
        dot.edge('final_layer_norm', 'output_projection')
        
        # Final output (output)
        output_attrs = self.create_node_attributes(
            'output', 'output', 0,
            {'batch_size': 128, 'seq_len': 1024, 'heads': 1, 'd_k': 32000},
            {'batch_size': 128, 'seq_len': 1024, 'heads': 1, 'd_k': 32000},
            'Output\nLogits:32K'
        )
        dot.node('output', **output_attrs)
        dot.edge('output_projection', 'output')
        
        return dot
    
    def save_dag(self, dot, output_dir: str):
        """Save DAG in both DOT and SVG formats"""
        # Save as DOT file
        dot_file = f"{output_dir}/llm_deployment_dag.dot"
        with open(dot_file, 'w') as f:
            f.write(dot.source)
        
        # Save as SVG file
        svg_file = f"{output_dir}/llm_deployment_dag.svg"
        dot.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
        
        return dot_file, svg_file

def main():
    """Main function to generate the DAG"""
    print("Generating LLM Deployment DAG...")
    
    generator = LLMDeploymentDAGGenerator()
    dot = generator.generate_dag()
    
    output_dir = "../outputs/2025-12-04-14-18-19"
    dot_file, svg_file = generator.save_dag(dot, output_dir)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    # Save submission paths
    submission_paths = {
        "dag_dot_file": dot_file,
        "dag_svg_file": svg_file,
        "generated_at": "2025-12-04 14:18:19",
        "model_configuration": generator.model_config,
        "parallel_configuration": generator.parallel_config
    }
    
    with open(f"{output_dir}/final_dag_submission_paths.json", 'w') as f:
        json.dump(submission_paths, f, indent=2)
    
    return dot_file, svg_file

if __name__ == "__main__":
    main()