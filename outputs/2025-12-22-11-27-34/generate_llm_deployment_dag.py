#!/usr/bin/env python3
"""
LLM Deployment DAG Generator for EP1-TP1-PP1-DP8 Configuration
Generates Graphviz code for the final optimized deployment strategy
"""

import graphviz
from typing import Dict, List, Tuple

class LLMDeploymentDAGGenerator:
    defga = ""
    def __init__(self):
        # Model configuration from final optimized strategy
        self.num_layers = 16
        self.num_heads = 16
        self.head_dim = 64
        self.hidden_size = 2048
        self.token_dim = 1024
        self.experts_per_layer = 64
        self.batch_size = 2048
        self.seq_length = 512  # Average sequence length
        
        # Parallel configuration: EP1-TP1-PP1-DP8
        self.ep = 1  # Expert Parallelism
        self.tp = 1  # Tensor Parallelism
        self.pp = 1  # Pipeline Parallelism
        self.dp = 8  # Data Parallelism
        self.num_gpus = self.ep * self.tp * self.pp * self.dp
        
        # GPU assignments
        self.gpu_ids = list(range(self.num_gpus))
        
    def create_dag(self) -> graphviz.Digraph:
        """Create the complete DAG for LLM deployment"""
        dot = graphviz.Digraph(comment='LLM Deployment DAG - EP1-TP1-PP1-DP8')
        dot.attr(rankdir='TB')
        dot.attr('graph', bgcolor='white', fontname='Arial')
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Create subgraphs for each GPU
        for gpu_id in self.gpu_ids:
            with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
                c.attr(label=f'GPU {gpu_id}', style='rounded', fillcolor='lightblue', color='blue')
                self._add_gpu_nodes(c, gpu_id)
        
        # Add inter-GPU communication edges
        self._add_inter_gpu_edges(dot)
        
        return dot
    
    def _add_gpu_nodes(self, subgraph, gpu_id: int):
        """Add all nodes for a specific GPU"""        
        # Input processing
        input_node = f'input_gpu{gpu_id}'
        subgraph.node(input_node, 
                     label=f'Input\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.token_dim}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.token_dim}]',
                     shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Token embedding
        embed_node = f'embed_gpu{gpu_id}'
        subgraph.node(embed_node,
                     label=f'Token Embedding\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Add position encoding
        pos_enc_node = f'pos_enc_gpu{gpu_id}'
        subgraph.node(pos_enc_node,
                     label=f'Position Encoding\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Create transformer layers
        prev_node = pos_enc_node
        for layer_idx in range(self.num_layers):
            layer_nodes = self._add_transformer_layer(subgraph, gpu_id, layer_idx, prev_node)
            prev_node = layer_nodes[-1]
        
        # Final layer norm
        final_ln_node = f'final_ln_gpu{gpu_id}'
        subgraph.node(final_ln_node,
                     label=f'Final Layer Norm\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightyellow')
        
        # Add edge from last layer to final layer norm
        last_layer_node = f'layer_{self.num_layers-1}_output_gpu{gpu_id}'
        subgraph.edge(last_layer_node, final_ln_node)
        
        # Language modeling head
        lm_head_node = f'lm_head_gpu{gpu_id}'
        subgraph.node(lm_head_node,
                     label=f'LM Head\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, vocab_size]',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Output processing
        output_node = f'output_gpu{gpu_id}'
        subgraph.node(output_node,
                     label=f'Output\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, vocab_size]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, vocab_size]',
                     shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Connect nodes within GPU
        subgraph.edge(input_node, embed_node)
        subgraph.edge(embed_node, pos_enc_node)
        subgraph.edge(final_ln_node, lm_head_node)
        subgraph.edge(lm_head_node, output_node)
    
    def _add_transformer_layer(self, subgraph, gpu_id: int, layer_idx: int, input_node: str) -> List[str]:
        """Add a complete transformer layer with all operators"""
        nodes = []        
        # Layer norm 1
        ln1_node = f'layer_{layer_idx}_ln1_gpu{gpu_id}'
        subgraph.node(ln1_node,
                     label=f'Layer {layer_idx} LayerNorm1\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Self-attention components
        # Q projection
        q_proj_node = f'layer_{layer_idx}_q_proj_gpu{gpu_id}'
        subgraph.node(q_proj_node,
                     label=f'Layer {layer_idx} Q Projection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]',
                     shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # K projection
        k_proj_node = f'layer_{layer_idx}_k_proj_gpu{gpu_id}'
        subgraph.node(k_proj_node,
                     label=f'Layer {layer_idx} K Projection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]',
                     shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # V projection
        v_proj_node = f'layer_{layer_idx}_v_proj_gpu{gpu_id}'
        subgraph.node(v_proj_node,
                     label=f'Layer {layer_idx} V Projection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]',
                     shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # Scaled dot-product attention
        attn_node = f'layer_{layer_idx}_attn_gpu{gpu_id}'
        subgraph.node(attn_node,
                     label=f'Layer {layer_idx} Scaled Dot-Product Attention\\nGPU {gpu_id}\\nInput Q: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]\\nInput K,V: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]',
                     shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # Attention output projection
        attn_out_node = f'layer_{layer_idx}_attn_out_proj_gpu{gpu_id}'
        subgraph.node(attn_out_node,
                     label=f'Layer {layer_idx} Attention Output Projection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, heads={self.num_heads}, d_k={self.head_dim}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # Residual connection after attention
        residual1_node = f'layer_{layer_idx}_residual1_gpu{gpu_id}'
        subgraph.node(residual1_node,
                     label=f'Layer {layer_idx} Residual1\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='parallelogram', style='filled', fillcolor='lightpink')
        
        # Layer norm 2
        ln2_node = f'layer_{layer_idx}_ln2_gpu{gpu_id}'
        subgraph.node(ln2_node,
                     label=f'Layer {layer_idx} LayerNorm2\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightblue')
        
        # MoE components (since EP=1, all experts are on this GPU)
        # Routing
        router_node = f'layer_{layer_idx}_router_gpu{gpu_id}'
        subgraph.node(router_node,
                     label=f'Layer {layer_idx} Router\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, num_experts={self.experts_per_layer}]',
                     shape='parallelogram', style='filled', fillcolor='lightsteelblue')
        
        # Expert selection (gate) - shown with dashed line
        gate_node = f'layer_{layer_idx}_gate_gpu{gpu_id}'
        subgraph.node(gate_node,
                     label=f'Layer {layer_idx} Gate Selection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, num_experts={self.experts_per_layer}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, top_k=2]',
                     shape='parallelogram', style='dashed', fillcolor='lightsteelblue')
        
        # Expert computations (all 64 experts on this GPU since EP=1)
        expert_nodes = []
        for expert_id in range(min(4, self.experts_per_layer)):  # Show first 4 experts for clarity
            expert_node = f'layer_{layer_idx}_expert_{expert_id}_gpu{gpu_id}'
            subgraph.node(expert_node,
                         label=f'Layer {layer_idx} Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                         shape='rectangle', style='filled', fillcolor='lightsalmon')
            expert_nodes.append(expert_node)
        
        # Expert aggregation
        expert_agg_node = f'layer_{layer_idx}_expert_agg_gpu{gpu_id}'
        subgraph.node(expert_agg_node,
                     label=f'Layer {layer_idx} Expert Aggregation\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='parallelogram', style='filled', fillcolor='lightsteelblue')
        
        # MoE output projection
        moe_out_node = f'layer_{layer_idx}_moe_out_proj_gpu{gpu_id}'
        subgraph.node(moe_out_node,
                     label=f'Layer {layer_idx} MoE Output Projection\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='rectangle', style='filled', fillcolor='lightsalmon')
        
        # Residual connection after MoE
        residual2_node = f'layer_{layer_idx}_residual2_gpu{gpu_id}'
        subgraph.node(residual2_node,
                     label=f'Layer {layer_idx} Residual2\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='parallelogram', style='filled', fillcolor='lightpink')
        
        # Connect nodes within the layer
        if layer_idx == 0:
            subgraph.edge(input_node, ln1_node)
        else:
            prev_layer_output = f'layer_{layer_idx-1}_output_gpu{gpu_id}'
            subgraph.edge(prev_layer_output, ln1_node)
        
        subgraph.edge(ln1_node, q_proj_node)
        subgraph.edge(ln1_node, k_proj_node)
        subgraph.edge(ln1_node, v_proj_node)
        
        subgraph.edge(q_proj_node, attn_node)
        subgraph.edge(k_proj_node, attn_node)
        subgraph.edge(v_proj_node, attn_node)
        
        subgraph.edge(attn_node, attn_out_node)
        subgraph.edge(attn_out_node, residual1_node)
        
        # Connect residual to ln2
        subgraph.edge(residual1_node, ln2_node)
        
        # MoE connections
        subgraph.edge(ln2_node, router_node)
        subgraph.edge(router_node, gate_node)
        
        # Connect gate to experts with dashed lines
        for expert_node in expert_nodes:
            subgraph.edge(gate_node, expert_node, style='dashed')
        
        # Connect experts to aggregation
        for expert_node in expert_nodes:
            subgraph.edge(expert_node, expert_agg_node)
        
        subgraph.edge(expert_agg_node, moe_out_node)
        subgraph.edge(moe_out_node, residual2_node)
        
        # Layer output node
        layer_output_node = f'layer_{layer_idx}_output_gpu{gpu_id}'
        subgraph.node(layer_output_node,
                     label=f'Layer {layer_idx} Output\\nGPU {gpu_id}\\nInput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]\\nOutput: [batch={self.batch_size//self.dp}, seq={self.seq_length}, dim={self.hidden_size}]',
                     shape='ellipse', style='filled', fillcolor='lightgray')
        
        subgraph.edge(residual2_node, layer_output_node)
        
        nodes = [ln1_node, q_proj_node, k_proj_node, v_proj_node, attn_node, attn_out_node, 
                residual1_node, ln2_node, router_node, gate_node, expert_agg_node, 
                moe_out_node, residual2_node, layer_output_node]
        nodes.extend(expert_nodes)
        
        return nodes
    
    def _add_inter_gpu_edges(self, dot):
        """Add communication edges between GPUs"""
        # Since we have DP=8, we need to show data parallelism communication
        # In this configuration, each GPU processes a different subset of the batch
        # and results are aggregated at the end
        
        # Add aggregation node for final output
        agg_node = 'final_aggregation'
        dot.node(agg_node,
                label='Final Output Aggregation\\nAll GPUs\\nInput: 8×[batch=256, seq=512, vocab_size]\\nOutput: [batch=2048, seq=512, vocab_size]',
                shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Connect each GPU's output to the aggregation node
        for gpu_id in self.gpu_ids:
            output_node = f'output_gpu{gpu_id}'
            dot.edge(output_node, agg_node)
    
    def generate_graphviz_code(self) -> str:
        """Generate the Graphviz DOT code"""
        dot = self.create_dag()
        return dot.source
    
    def save_files(self, output_dir: str):
        """Save both DOT and SVG files"""
        dot = self.create_dag()
        
        # Save DOT file
        dot_file = f'{output_dir}/llm_deployment_dag.dot'
        with open(dot_file, 'w') as f:
            f.write(dot.source)
        
        # Save SVG file
        svg_file = f'{output_dir}/llm_deployment_dag.svg'
        dot.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
        
        return dot_file, svg_file

def main():
    """Main function to generate the DAG"""
    generator = LLMDeploymentDAGGenerator()
    
    # Generate Graphviz code
    graphviz_code = generator.generate_graphviz_code()
    
    print("=== LLM DEPLOYMENT DAG GENERATED ===")
    print(f"Configuration: EP{generator.ep}-TP{generator.tp}-PP{generator.pp}-DP{generator.dp}")
    print(f"Total GPUs: {generator.num_gpus}")
    print(f"Batch size per GPU: {generator.batch_size // generator.dp}")
    print(f"Sequence length: {generator.seq_length}")
    print(f"Hidden size: {generator.hidden_size}")
    print(f"Number of layers: {generator.num_layers}")
    print(f"Experts per layer: {generator.experts_per_layer}")
    print()
    
    # Save files
    output_dir = "../outputs/2025-12-22-11-27-34"
    dot_file, svg_file = generator.save_files(output_dir)
    
    print(f"DOT file saved: {dot_file}")
    print(f"SVG file saved: {svg_file}")
    
    # Also save the raw Graphviz code
    dot_code_file = f"{output_dir}/llm_deployment_dag_code.dot"
    with open(dot_code_file, 'w') as f:
        f.write(graphviz_code)
    
    print(f"Graphviz code saved: {dot_code_file}")
    
    return {
        'dot_file': dot_file,
        'svg_file': svg_file,
        'dot_code_file': dot_code_file,
        'graphviz_code': graphviz_code
    }

if __name__ == "__main__":
    result = main()
    print("\n=== DAG GENERATION COMPLETE ===")