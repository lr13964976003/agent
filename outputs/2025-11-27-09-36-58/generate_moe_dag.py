#!/usr/bin/env python3
"""
Generate comprehensive DAG for large-scale cross-node expert parallelism deployment
based on the provided paper and deployment configuration.
"""

import graphviz
import os

class MoEDeploymentDAG:
    def __init__(self):
        self.dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
        self.dot.attr(rankdir='TB', splines='ortho', compound='true')
        
        # Define node shapes and styles
        self.dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        self.dot.attr('edge', arrowhead='normal', arrowsize='0.5')
        
        # Track GPU nodes for each layer
        self.gpu_nodes = {}
        self.layer_nodes = {}
        
    def add_input_node(self):
        """Add input node with initial tensor dimensions"""
        self.dot.node('input', 'Input\n[batch_size=8, seq_len=1024, token_dim=7168]', 
                     shape='ellipse', fillcolor='lightgreen')
    
    def add_dense_layers(self, layer_indices=[0, 1, 2]):
        """Add initial dense layers"""
        for layer_idx in layer_indices:
            layer_name = f'dense_{layer_idx}'
            
            # Layer normalization
            ln_node = f'{layer_name}_ln'
            self.dot.node(ln_node, f'LayerNorm\nGPU: 0-3\nInput: [8,1024,7168]\nOutput: [8,1024,7168]', 
                         shape='rectangle', fillcolor='lightyellow')
            
            # Multi-head attention
            for gpu_id in range(4):
                mha_node = f'{layer_name}_mha_gpu{gpu_id}'
                self.dot.node(mha_node, 
                             f'MHA\nGPU: {gpu_id}\nInput: [8,1024,7168]\nOutput: [8,1024,7168]', 
                             shape='rectangle', fillcolor='coral')
                
                # FFN nodes in tensor parallelism
                ffn_nodes = self.add_tensor_parallel_mlp(layer_name, gpu_id)
                
                # Connect MHA to FFN
                for ffn_node in ffn_nodes:
                    self.dot.edge(mha_node, ffn_node, style='solid')
    
    def add_tensor_parallel_mlp(self, layer_name, gpu_id, tp_degree=2):
        """Add tensor parallel MLP for dense layers"""
        base_gpu = gpu_id * tp_degree
        ffn_nodes = []
        
        for tp_id in range(tp_degree):
            actual_gpu = base_gpu + tp_id
            
            # Column parallel linear 1
            col_linear = f'{layer_name}_col_linear_gpu{actual_gpu}'
            self.dot.node(col_linear, 
                         f'ColLinear\nGPU: {actual_gpu}\nInput: [8,1024,7168]\nOutput: [8,1024,1024]', 
                         shape='rectangle', fillcolor='lightblue')
            
            # Activation
            activation = f'{layer_name}_activation_gpu{actual_gpu}'
            self.dot.node(activation, 
                         f'GELU\nGPU: {actual_gpu}\nInput: [8,1024,1024]\nOutput: [8,1024,1024]', 
                         shape='rectangle', fillcolor='lightblue')
            
            # Row parallel linear 2
            row_linear = f'{layer_name}_row_linear_gpu{actual_gpu}'
            self.dot.node(row_linear, 
                         f'RowLinear\nGPU: {actual_gpu}\nInput: [8,1024,1024]\nOutput: [8,1024,7168]', 
                         shape='rectangle', fillcolor='lightblue')
            
            # All-reduce
            all_reduce = f'{layer_name}_all_reduce_gpu{actual_gpu}'
            self.dot.node(all_reduce, 
                         f'AllReduce\nGPU: {actual_gpu}\nInput: [8,1024,7168]\nOutput: [8,1024,7168]', 
                         shape='parallelogram', fillcolor='orange')
            
            # Connect the chain
            self.dot.edge(col_linear, activation)
            self.dot.edge(activation, row_linear)
            self.dot.edge(row_linear, all_reduce)
            
            ffn_nodes.append(all_reduce)
        
        return ffn_nodes
    
    def add_moe_layers(self, layer_indices=[3, 4, 5]):
        """Add MoE layers with expert parallelism"""
        for layer_idx in layer_indices:
            layer_name = f'moe_{layer_idx}'
            
            # Add input routing
            routing_node = f'{layer_name}_routing'
            self.dot.node(routing_node, 
                         f'TopK Routing\nGPU: 0\nInput: [8,1024,7168]\nOutput: [8,1024,7168]', 
                         shape='parallelogram', fillcolor='lightgreen')
            
            # Add experts distributed across 64 GPUs
            expert_nodes = []
            for expert_id in range(64):
                gpu_id = expert_id
                node_id = expert_id // 8  # 8 GPUs per node
                
                expert_name = f'{layer_name}_expert_{expert_id}_gpu{gpu_id}'
                self.dot.node(expert_name, 
                             f'Expert {expert_id}\nGPU: {gpu_id}\nNode: {node_id}\nInput: [1,variable,7168]\nOutput: [1,variable,7168]', 
                             shape='rectangle', fillcolor='lightblue')
                expert_nodes.append(expert_name)
            
            # Add gate operations for each expert
            gate_nodes = []
            for expert_id in range(64):
                gpu_id = expert_id
                gate_name = f'{layer_name}_gate_{expert_id}_gpu{gpu_id}'
                self.dot.node(gate_name, 
                             f'Gate Selection\nGPU: {gpu_id}\nInput: [1,variable,7168]\nOutput: [1,variable,7168]', 
                             shape='parallelogram', fillcolor='yellow')
                gate_nodes.append(gate_name)
                
                # Connect gate to expert with dashed line
                self.dot.edge(gate_name, f'{layer_name}_expert_{expert_id}_gpu{gpu_id}', 
                             style='dashed', color='red')
            
            # Add aggregation nodes
            agg_nodes = []
            for expert_id in range(64):
                gpu_id = expert_id
                agg_name = f'{layer_name}_agg_{expert_id}_gpu{gpu_id}'
                self.dot.node(agg_name, 
                             f'Aggregate\nGPU: {gpu_id}\nInput: [1,variable,7168]\nOutput: [1,variable,7168]', 
                             shape='parallelogram', fillcolor='orange')
                agg_nodes.append(agg_name)
                
                # Connect expert to aggregation
                self.dot.edge(f'{layer_name}_expert_{expert_id}_gpu{gpu_id}', agg_name)
            
            # Add communication patterns
            self.add_expert_communication(layer_name, expert_id, gpu_id)
    
    def add_expert_communication(self, layer_name, expert_id, gpu_id):
        """Add communication patterns between experts"""
        # Token sharding communication
        for src_gpu in range(64):
            for dst_gpu in range(64):
                if src_gpu != dst_gpu:
                    comm_node = f'{layer_name}_comm_{src_gpu}_to_{dst_gpu}'
                    self.dot.node(comm_node, 
                                 f'Token Transfer\nFrom GPU: {src_gpu}\nTo GPU: {dst_gpu}\nInput: [1,variable,7168]\nOutput: [1,variable,7168]', 
                                 shape='ellipse', fillcolor='purple', style='dashed')
    
    def add_pipeline_parallelism(self):
        """Add pipeline parallelism for 61 layers"""
        # We'll represent 3 representative layers as requested
        layers_to_show = [0, 30, 60]  # First, middle, and last layers
        
        for layer_idx in layers_to_show:
            if layer_idx < 3:
                # Dense layers
                layer_type = 'dense'
                node_name = f'{layer_type}_{layer_idx}'
                self.dot.node(f'pipeline_{layer_idx}', 
                             f'Pipeline Stage {layer_idx}\nType: Dense\nGPUs: 0-3', 
                             shape='rectangle', fillcolor='cyan')
            else:
                # MoE layers
                layer_type = 'moe'
                node_name = f'{layer_type}_{layer_idx}'
                self.dot.node(f'pipeline_{layer_idx}', 
                             f'Pipeline Stage {layer_idx}\nType: MoE\nGPUs: 0-63', 
                             shape='rectangle', fillcolor='cyan')
    
    def add_output_node(self):
        """Add final output node"""
        self.dot.node('output', 'Output\n[batch_size=8, seq_len=1024, token_dim=7168]', 
                     shape='ellipse', fillcolor='lightgreen')
    
    def generate_dag(self):
        """Generate the complete DAG"""
        print("Generating comprehensive MoE deployment DAG...")
        
        # Add all components
        self.add_input_node()
        self.add_dense_layers()
        self.add_moe_layers()
        self.add_pipeline_parallelism()
        self.add_output_node()
        
        # Connect components
        self.dot.edge('input', 'dense_0_ln')
        self.dot.edge('dense_2_row_linear_gpu3', 'moe_3_routing')
        self.dot.edge('moe_5_agg_63_gpu63', 'output')
        
        # Save DAG
        output_dir = "../outputs/2025-11-27-09-36-58"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as DOT
        dot_path = os.path.join(output_dir, "moe_deployment_dag.dot")
        with open(dot_path, 'w') as f:
            f.write(self.dot.source)
        
        # Save as SVG
        svg_path = os.path.join(output_dir, "moe_deployment_dag.svg")
        self.dot.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
        
        print(f"DAG saved to: {dot_path}")
        print(f"SVG saved to: {svg_path}")
        
        return dot_path, svg_path

# Generate the DAG
dag = MoEDeploymentDAG()
dot_path, svg_path = dag.generate_dag()

# Create a more detailed version focusing on 3 representative layers
def create_detailed_layer_dag():
    """Create detailed DAG for 3 representative layers"""
    detailed_dot = graphviz.Digraph(comment='Detailed MoE Deployment - 3 Representative Layers')
    detailed_dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Define precise tensor dimensions
    batch_size = 8
    seq_len = 1024
    token_dim = 7168
    heads = 128
    d_k = 128
    ffn_hidden = 2048
    
    # Layer 0 (Dense Layer)
    with detailed_dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0 (Dense)', style='rounded')
        
        # Input
        layer0.node('layer0_input', 
                   f'Input\nGPU: 0-3\n[{batch_size},{seq_len},{token_dim}]',
                   shape='ellipse', fillcolor='lightgreen')
        
        # Layer Norm
        layer0.node('layer0_ln', 
                   f'LayerNorm\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                   shape='rectangle', fillcolor='lightyellow')
        
        # MHA across 4 GPUs with tensor parallelism
        for gpu_id in range(4):
            # Q projection
            q_proj = f'layer0_q_proj_gpu{gpu_id}'
            layer0.node(q_proj, 
                       f'Q Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # K projection
            k_proj = f'layer0_k_proj_gpu{gpu_id}'
            layer0.node(k_proj, 
                       f'K Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # V projection
            v_proj = f'layer0_v_proj_gpu{gpu_id}'
            layer0.node(v_proj, 
                       f'V Proj\nGPU: {gpu_id}\n[{batch_size},{seq_len},{heads},{d_k}]',
                       shape='rectangle', fillcolor='coral')
            
            # Attention computation
            attn = f'layer0_attn_gpu{gpu_id}'
            layer0.node(attn, 
                       f'Attention\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='rectangle', fillcolor='coral')
            
            # FFN with tensor parallelism
            ffn1 = f'layer0_ffn1_gpu{gpu_id}'
            layer0.node(ffn1, 
                       f'FFN1\nGPU: {gpu_id}\n[{batch_size},{seq_len},{ffn_hidden}]',
                       shape='rectangle', fillcolor='lightblue')
            
            ffn2 = f'layer0_ffn2_gpu{gpu_id}'
            layer0.node(ffn2, 
                       f'FFN2\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='rectangle', fillcolor='lightblue')
            
            # All-reduce for tensor parallel
            all_reduce = f'layer0_all_reduce_gpu{gpu_id}'
            layer0.node(all_reduce, 
                       f'AllReduce\nGPU: {gpu_id}\n[{batch_size},{seq_len},{token_dim}]',
                       shape='parallelogram', fillcolor='orange')
    
    # Layer 30 (MoE Layer)
    with detailed_dot.subgraph(name='cluster_layer30') as layer30:
        layer30.attr(label='Layer 30 (MoE)', style='rounded')
        
        # Input
        layer30.node('layer30_input', 
                    f'Input\nGPU: 0-63\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
        
        # Gate computation
        gate = 'layer30_gate'
        layer30.node(gate, 
                    f'Gate\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='yellow')
        
        # Expert distribution across 64 GPUs
        for expert_id in range(64):
            gpu_id = expert_id
            node_id = expert_id // 8
            
            # Expert computation
            expert = f'layer30_expert_{expert_id}'
            layer30.node(expert, 
                        f'Expert {expert_id}\nGPU: {gpu_id}\nNode: {node_id}\n[{batch_size//64},{seq_len},{token_dim}]',
                        shape='rectangle', fillcolor='lightblue')
            
            # Communication for token routing
            comm = f'layer30_comm_{expert_id}'
            layer30.node(comm, 
                        f'Token Transfer\nFrom GPU: 0\nTo GPU: {gpu_id}\n[{batch_size//64},{seq_len},{token_dim}]',
                        shape='ellipse', fillcolor='purple', style='dashed')
            
            # Expert computation
            expert_ffn1 = f'layer30_expert_{expert_id}_ffn1'
            layer30.node(expert_ffn1, 
                        f'Expert FFN1\nGPU: {gpu_id}\n[{batch_size//64},{seq_len},{ffn_hidden}]',
                        shape='rectangle', fillcolor='lightblue')
            
            expert_ffn2 = f'layer30_expert_{expert_id}_ffn2'
            layer30.node(expert_ffn2, 
                        f'Expert FFN2\nGPU: {gpu_id}\n[{batch_size//64},{seq_len},{token_dim}]',
                        shape='rectangle', fillcolor='lightblue')
            
            # Aggregation
            agg = f'layer30_agg_{expert_id}'
            layer30.node(agg, 
                        f'Aggregate\nGPU: {gpu_id}\n[{batch_size//64},{seq_len},{token_dim}]',
                        shape='parallelogram', fillcolor='orange')
            
            # Connections
            layer30.edge(comm, expert_ffn1, style='dashed')
            layer30.edge(expert_ffn1, expert_ffn2)
            layer30.edge(expert_ffn2, agg)
    
    # Layer 60 (Final MoE Layer)
    with detailed_dot.subgraph(name='cluster_layer60') as layer60:
        layer60.attr(label='Layer 60 (Final MoE)', style='rounded')
        
        # Similar structure to layer 30 but final output
        layer60.node('layer60_input', 
                    f'Input\nGPU: 0-63\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
        
        layer60.node('layer60_gate', 
                    f'Gate\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='yellow')
        
        # Final output aggregation
        layer60.node('layer60_final_agg', 
                    f'Final Aggregation\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='parallelogram', fillcolor='orange')
        
        layer60.node('output', 
                    f'Output\nGPU: 0\n[{batch_size},{seq_len},{token_dim}]',
                    shape='ellipse', fillcolor='lightgreen')
    
    # Connect layers
    detailed_dot.edge('layer0_all_reduce_gpu3', 'layer30_input')
    detailed_dot.edge('layer30_agg_63', 'layer60_input')
    detailed_dot.edge('layer60_final_agg', 'output')
    
    # Save detailed DAG
    detailed_dot_path = os.path.join(output_dir, "detailed_moe_layers.dot")
    detailed_svg_path = os.path.join(output_dir, "detailed_moe_layers.svg")
    
    with open(detailed_dot_path, 'w') as f:
        f.write(detailed_dot.source)
    
    detailed_dot.render(detailed_svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    return detailed_dot_path, detailed_svg_path

# Create detailed DAG
detailed_dot_path, detailed_svg_path = create_detailed_layer_dag()

print("Detailed DAG generation complete!")

# Create submission JSON
import json
submission = {
    "dag_files": [
        dot_path,
        svg_path,
        detailed_dot_path,
        detailed_svg_path
    ],
    "deployment_summary": {
        "strategy": "large_scale_cross_node_expert_parallelism",
        "expert_parallelism": 64,
        "gpus": 64,
        "layers_represented": 3,
        "node_distribution": "topology_aware",
        "communication_overlap": "asynchronous_cuda_streams",
        "load_balancing": "dynamic_gate_probabilities"
    }
}

submission_path = os.path.join(output_dir, "submission.json")
with open(submission_path, 'w') as f:
    json.dump(submission, f, indent=2)

print(f"Submission saved to: {submission_path}")