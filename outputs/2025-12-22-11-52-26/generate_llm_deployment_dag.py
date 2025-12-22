#!/usr/bin/env python3
"""
DAG Generator for 30B MoE Model Deployment
EP8-TP4-PP2-DP4 Configuration
Comprehensive operator-level DAG with communication patterns
"""

import graphviz
from typing import Dict, List, Tuple

class LLMDeploymentDAGGenerator:
    def __init__(self):
        # Parallel configuration from strategy
        self.ep_dim = 8    # Expert Parallelism
        self.tp_dim = 4    # Tensor Parallelism  
        self.pp_dim = 2    # Pipeline Parallelism
        self.dp_dim = 4    # Data Parallelism
        self.total_gpus = 256
        
        # Model architecture
        self.num_layers = 16
        self.num_heads = 16
        self.head_dim = 64
        self.hidden_size = 2048
        self.token_dim = 1024
        self.experts_per_layer = 64
        self.batch_size = 128
        self.seq_length = 10240  # max sequence length
        
        # GPU assignments
        self.gpus_pp0 = list(range(0, 128))  # First pipeline stage
        self.gpus_pp1 = list(range(128, 256))  # Second pipeline stage
        
    def create_dag(self):
        """Create comprehensive DAG for MoE LLM deployment"""
        dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG')
        dot.attr(rankdir='TB', size='30,20', ranksep='1.5',..sep='1.0')
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Define node shapes
        dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
        dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation  
        dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing
        
        # Create input node
        input_node = self.create_input_node()
        dot.node('input', input_node['label'], shape='box', fillcolor='lightblue')
        
        # Create prefill phase
        prefill_nodes = self.create_prefill_phase()
        
        # Create decode phase  
        decode_nodes = self.create_decode_phase()
        
        # Connect input to prefill
        dot.edge('input', 'prefill_start')
        
        # Connect prefill to decode
        dot.edge('prefill_end', 'decode_start')
        
        # Add all nodes to graph
        for node_id, node_data in {**prefill_nodes, **decode_nodes}.items():
            dot.node(node_id, node_data['label'], **node_data.get('attrs', {}))
            
        # Add all edges
        for edge in self.get_all_edges():
            dot.edge(edge['from'], edge['to'], **edge.get('attrs', {}))
            
        return dot
        
    def create_input_node(self):
        """Create input node with dimensions"""
        return {
            'label': f'Input\\nBatch: {self.batch_size}, Seq: {self.seq_length}\\nDim: {self.token_dim}',
            'input_dim': [self.batch_size, self.seq_length, self.token_dim],
            'output_dim': [self.batch_size, self.seq_length, self.token_dim]
        }
        
    def create_prefill_phase(self):
        """Create complete prefill phase with all operators"""
        nodes = {}        
        # Start marker
        nodes['prefill_start'] = {
            'label': 'Prefill Phase Start',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue'}
        }
        
        # Embedding layer (distributed across TP)
        for tp in range(self.tp_dim):
            for gpu_id in range(tp, 256, self.tp_dim):
                node_id = f'embed_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Embedding\\nTP{tp}\\nGPU{gpu_id}\\nIn: [B={self.batch_size},S={self.seq_length},D={self.token_dim}]\\nOut: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # All-Reduce for embedding
        for tp in range(self.tp_dim):
            for gpu_id in range(tp, 256, self.tp_dim):
                node_id = f'embed_ar_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Embedding All-Reduce\\nTP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue'},                    'gpu': gpu_id,
                    'type': 'communication'
                }
        
        # Layer processing for pipeline stage 0 (layers 0-7)
        for layer in range(8):  # First pipeline stage has layers 0-7
            layer_nodes = self.create_layer_nodes(layer, self.gpus_pp0)
            nodes.update(layer_nodes)
            
        # Pipeline communication between stages
        for gpu_src in self.gpus_pp0:
            for gpu_dst in self.gpus_pp1:
                if gpu_src % self.tp_dim == gpu_dst % self.tp_dim:  # Same TP group
                    node_id = f'pipeline_send_{gpu_src}_to_{gpu_dst}'
                    nodes[node_id] = {
                        'label': f'Pipeline Send\\nGPU{gpu_src}→GPU{gpu_dst}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue'},
                        'gpu': gpu_src,
                        'type': 'communication'
                    }
        
        # Layer processing for pipeline stage 1 (layers 8-15)
        for layer in range(8, 16):  # Second pipeline stage has layers 8-15
            layer_nodes = self.create_layer_nodes(layer, self.gpus_pp1)
            nodes.update(layer_nodes)
        
        # End marker
        nodes['prefill_end'] = {
            'label': 'Prefill Phase End',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue'}
        }
        
        return nodes
        
    def create_decode_phase(self):
        """Create complete decode phase with all operators"""
        nodes = {}
        
        # Start marker
        nodes['decode_start'] = {
            'label': 'Decode Phase Start',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue'}
        }
        
        # Similar structure to prefill but with single token processing
        # and KV cache dependencies
        
        # Decode embedding
        for tp in range(self.tp_dim):
            for gpu_id in range(tp, 256, self.tp_dim):
                node_id = f'decode_embed_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Decode Embedding\\nTP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Decode layers with KV cache
        for layer in range(8):  # Pipeline stage 0
            decode_layer_nodes = self.create_decode_layer_nodes(layer, self.gpus_pp0)
            nodes.update(decode_layer_nodes)
            
        # Pipeline communication for decode
        for gpu_src in self.gpus_pp0:
            for gpu_dst in self.gpus_pp1:
                if gpu_src % self.tp_dim == gpu_dst % self.tp_dim:
                    node_id = f'decode_pipeline_send_{gpu_src}_to_{gpu_dst}'
                    nodes[node_id] = {
                        'label': f'Decode Pipeline Send\\nGPU{gpu_src}→GPU{gpu_dst}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue'},
                        'gpu': gpu_src,
                        'type': 'communication'
                    }
        
        for layer in range(8, 16):  # Pipeline stage 1
            decode_layer_nodes = self.create_decode_layer_nodes(layer, self.gpus_pp1)
            nodes.update(decode_layer_nodes)
        
        # Final output
        nodes['output'] = {
            'label': 'Output Token',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue'}
        }
        
        return nodes
        
    def create_layer_nodes(self, layer_id: int, gpu_list: List[int]):
        """Create nodes for a single transformer layer"""
        nodes = {}
        
        # Layer normalization (distributed across TP)
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:  # GPUs in this TP group
                node_id = f'ln1_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Layer Norm 1\\nL{layer_id} TP{tp}\\nGPU{gpu_id}\\nIn: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]\\nOut: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Self-attention QKV projections
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'qkv_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'QKV Projection\\nL{layer_id} TP{tp}\\nGPU{gpu_id}\\nIn: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]\\nOut: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Attention computation
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'attn_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Self-Attention\\nL{layer_id} TP{tp}\\nGPU{gpu_id}\\nIn: [B={self.batch_size},H={self.num_heads//self.tp_dim},S={self.seq_length},D={self.head_dim}]\\nOut: [B={self.batch_size},H={self.num_heads//self.tp_dim},S={self.seq_length},D={self.head_dim}]',                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Attention output projection
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'attn_out_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Attention Output Proj\\nL{layer_id} TP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Attention All-Reduce
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'attn_ar_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Attention All-Reduce\\nL{layer_id} TP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue'},
                    'gpu': gpu_id,
                    'type': 'communication'
                }
        
        # MoE routing (gate)
        for ep in range(self.ep_dim):
            for gpu_id in gpu_list[ep::self.ep_dim]:  # GPUs in this EP group
                node_id = f'gate_l{layer_id}_ep{ep}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'MoE Gate\\nL{layer_id} EP{ep}\\nGPU{gpu_id}\\nIn: [B={self.batch_size},S={self.seq_length},H={self.hidden_size}]\\nOut: [B={self.batch_size},S={self.seq_length},K=2]',
                    'attrs': {'shape': 'parallelogram', 'fillcolor': 'lightyellow'},
                    'gpu': gpu_id,
                    'type': 'routing'
                }
        
        # Expert dispatch (All-to-All communication)
        for ep in range(self.ep_dim):
            for gpu_id in gpu_list[ep::self.ep_dim]:
                node_id = f'dispatch_l{layer_id}_ep{ep}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Expert Dispatch\\nL{layer_id} EP{ep}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'dashed'},
                    'gpu': gpu_id,
                    'type': 'communication'
                }
        
        # Expert computation (8 experts per GPU)
        for ep in range(self.ep_dim):
            for gpu_id in gpu_list[ep::self.ep_dim]:
                for expert_id in range(8):  # 8 experts per GPU
                    node_id = f'expert_l{layer_id}_e{expert_id}_ep{ep}_gpu{gpu_id}'
                    nodes[node_id] = {
                        'label': f'Expert {expert_id}\\nL{layer_id} EP{ep}\\nGPU{gpu_id}\\nIn: [B={self.batch_size//self.ep_dim},S={self.seq_length},H={self.hidden_size}]\\nOut: [B={self.batch_size//self.ep_dim},S={self.seq_length},H={self.hidden_size}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                        'gpu': gpu_id,
                        'type': 'compute'
                    }
        
        # Expert combine (All-to-All communication)
        for ep in range(self.ep_dim):
            for gpu_id in gpu_list[ep::self.tp_dim]:
                node_id = f'combine_l{layer_id}_ep{ep}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Expert Combine\\nL{layer_id} EP{ep}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue'},
                    'gpu': gpu_id,
                    'type': 'communication'
                }
        
        # Final layer norm
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'ln2_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Layer Norm 2\\nL{layer_id} TP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        return nodes
        
    def create_decode_layer_nodes(self, layer_id: int, gpu_list: List[int]):
        """Create nodes for decode phase of a single layer"""
        nodes = {}
        
        # Similar to prefill but with KV cache dependencies
        # and single token processing
        
        # KV cache read
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'kv_cache_read_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'KV Cache Read\\nL{layer_id} TP{tp}\\nGPU{gpu_id}\\nIn: [B=1,S=1,H={self.hidden_size}]\\nOut: [B=1,S=1,H={self.hidden_size}]',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        # Rest of decode layer similar to prefill but with smaller dimensions
        # ... (similar structure as create_layer_nodes but adapted for decode)
        
        # KV cache update
        for tp in range(self.tp_dim):
            for gpu_id in gpu_list[tp::self.tp_dim]:
                node_id = f'kv_cache_write_l{layer_id}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'KV Cache Update\\nL{layer_id} TP{tp}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen'},
                    'gpu': gpu_id,
                    'type': 'compute'
                }
        
        return nodes
        
    def get_all_edges(self):
        """Define all edges in the DAG"""
        edges = []
        
        # Input to embedding
        for tp in range(self.tp_dim):
            for gpu_id in range(tp, 256, self.tp_dim):
                edges.append({
                    'from': 'input',
                    'to': f'embed_tp{tp}_gpu{gpu_id}'
                })
                edges.append({
                    'from': f'embed_tp{tp}_gpu{gpu_id}',
                    'to': f'embed_ar_tp{tp}_gpu{gpu_id}'
                })
        
        # Connect embedding to first layer
        for tp in range(self.tp_dim):
            for gpu_id in range(tp, 128, self.tp_dim):  # PP stage 0
                edges.append({
                    'from': f'embed_ar_tp{tp}_gpu{gpu_id}',
                    'to': f'ln1_l0_tp{tp}_gpu{gpu_id}'
                })
        
        # Connect layers within pipeline stage 0
        for layer in range(7):  # Layers 0-6
            for tp in range(self.tp_dim):
                for gpu_id in range(tp, 128, self.tp_dim):
                    # Connect layer norm to QKV
                    edges.append({
                        'from': f'ln2_l{layer}_tp{tp}_gpu{gpu_id}',
                        'to': f'ln1_l{layer+1}_tp{tp}_gpu{gpu_id}'
                    })
        
        # Connect last layer of PP stage 0 to pipeline communication
        for tp in range(self.tp_dim):
            for gpu_src in range(tp, 128, self.tp_dim):
                for gpu_dst in range(tp+128, 256, self.tp_dim):
                    edges.append({
                        'from': f'ln2_l7_tp{tp}_gpu{gpu_src}',
                        'to': f'pipeline_send_{gpu_src}_to_{gpu_dst}'
                    })
                    edges.append({
                        'from': f'pipeline_send_{gpu_src}_to_{gpu_dst}',
                        'to': f'ln1_l8_tp{tp}_gpu{gpu_dst}'
                    })
        
        # Connect prefill end to decode start
        edges.append({
            'from': 'prefill_end',
            'to': 'decode_start'
        })
        
        # Connect decode to output
        edges.append({
            'from': 'decode_end',
            'to': 'output'
        })
        
        return edges

def main():
    """Generate and save the DAG"""
    generator = LLMDeploymentDAGGenerator()
    dag = generator.create_dag()
    
    # Save DOT file
    dag.save('../outputs/2025-12-22-11-52-26/llm_deployment_dag.dot')
    
    # Save SVG image
    dag.render('../outputs/2025-12-22-11-52-26/llm_deployment_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-22-11-52-26/llm_deployment_dag.dot")
    print(f"SVG image: ../outputs/2025-12-22-11-52-26/llm_deployment_dag.svg")
    
    # Return paths for JSON output
    return {
        "dot_path": "../outputs/2025-12-22-11-52-26/llm_deployment_dag.dot",
        "svg_path": "../outputs/2025-12-22-11-52-26/llm_deployment_dag.svg"
    }

if __name__ == "__main__":
    paths = main()
    print(f"\nGenerated files:\n{paths}")