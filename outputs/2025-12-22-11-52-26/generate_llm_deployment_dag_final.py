#!/usr/bin/env python3
"""
Comprehensive DAG Generator for 30B MoE Model Deployment
EP8-TP4-PP2-DP4 Configuration with Complete GPU Mapping
All parallel dimensions represented with operator-level granularity
"""

import graphviz
from typing import Dict, List, Tuple, Set
import itertools

class ComprehensiveLLMDeploymentDAGGenerator:
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
        self.vocab_size = 32000  # Typical vocabulary size
        
        # Calculate per-dimension sizes
        self.batch_per_dp = self.batch_size // self.dp_dim
        self.experts_per_ep = self.experts_per_layer // self.ep_dim
        self.layers_per_pp = self.num_layers // self.pp_dim
        self.hidden_per_tp = self.hidden_size // self.tp_dim
        self.heads_per_tp = self.num_heads // self.tp_dim
        
        # GPU indexing function
        self.get_gpu_id = lambda ep, tp, pp, dp: ep * (self.tp_dim * self.pp_dim * self.dp_dim) + tp * (self.pp_dim * self.dp_dim) + pp * self.dp_dim + dp
        
    def create_dag(self):
        """Create comprehensive DAG for MoE LLM deployment"""
        dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG - Complete Configuration')
        dot.attr(rankdir='TB', size='50,40', ranksep='1.0', nodesep='0.5')
        dot.attr('node', fontname='Arial', fontsize='9')
        
        # Create comprehensive input node
        input_node = self.create_comprehensive_input_node()
        dot.node('input', input_node['label'], shape='box', fillcolor='lightblue', style='filled', width='2')
        
        # Create comprehensive prefill phase with all parallel dimensions
        prefill_nodes = self.create_comprehensive_prefill_phase()
        
        # Create comprehensive decode phase
        decode_nodes = self.create_comprehensive_decode_phase()
        
        # Create final output phase
        output_nodes = self.create_output_phase()
        
        # Add all nodes to graph
        all_nodes = {**prefill_nodes, **decode_nodes, **output_nodes}
        for node_id, node_data in all_nodes.items():
            attrs = node_data.get('attrs', {})
            if 'width' not in attrs:
                attrs['width'] = '1.5'
            dot.node(node_id, node_data['label'], **attrs)
            
        # Add all edges
        edges = self.get_comprehensive_edges()
        for edge in edges:
            attrs = edge.get('attrs', {})
            dot.edge(edge['from'], edge['to'], **attrs)
            
        # Connect phases
        dot.edge('input', 'prefill_start')
        dot.edge('prefill_end', 'decode_start')
        dot.edge('decode_end', 'output_start')
        
        return dot
        
    def create_comprehensive_input_node(self):
        """Create comprehensive input node with all dimensions"""
        total_tokens = self.batch_size * self.seq_length
        return {
            'label': f'Input Data\\nBatch: {self.batch_size} (DP×{self.dp_dim}={self.batch_per_dp})\\nSeq: {self.seq_length}\\nTotal: {self.batch_size}×{self.seq_length}={total_tokens} tokens',
            'input_dim': [self.batch_size, self.seq_length, self.token_dim],
            'output_dim': [self.batch_size, self.seq_length, self.token_dim]
        }
        
    def create_comprehensive_prefill_phase(self):
        """Create complete prefill phase with all parallel dimensions"""
        nodes = {}
        
        # Start marker
        nodes['prefill_start'] = {
            'label': 'Prefill Phase Start\\nAll 256 GPUs Active',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue', 'style': 'filled', 'width': '3'}
        }
        
        # Create embedding layer for all DP and TP dimensions
        for dp in range(self.dp_dim):
            for tp in range(self.tp_dim):
                # Find representative GPUs for this DP-TP combination
                gpu_id = self.get_gpu_id(0, tp, 0, dp)  # EP=0, PP=0
                node_id = f'embed_dp{dp}_tp{tp}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Embedding Layer\\nDP{dp}-TP{tp}\\nGPU{gpu_id}\\nIn: [B={self.batch_per_dp},S={self.seq_length},D={self.token_dim}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                    'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                }
                
        # All-Reduce for embedding across TP dimensions
        for dp in range(self.dp_dim):
            for tp_group in range(self.tp_dim):
                gpu_id = self.get_gpu_id(0, tp_group, 0, dp)
                node_id = f'embed_ar_dp{dp}_tp{tp_group}_gpu{gpu_id}'
                nodes[node_id] = {
                    'label': f'Embedding All-Reduce\\nDP{dp} TP Group {tp_group}\\nGPU{gpu_id}',
                    'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled'}
                }
                
        # Create all transformer layers for all parallel dimensions
        for layer in range(self.num_layers):
            pp_stage = layer // self.layers_per_pp
            layer_in_stage = layer % self.layers_per_pp
            
            layer_nodes = self.create_comprehensive_layer_nodes(layer, pp_stage, layer_in_stage)
            nodes.update(layer_nodes)
            
        # End marker
        nodes['prefill_end'] = {
            'label': 'Prefill Phase End\\nKV Cache Constructed',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue', 'style': 'filled', 'width': '3'}
        }
        
        return nodes
        
    def create_comprehensive_layer_nodes(self, layer_id: int, pp_stage: int, layer_in_stage: int):
        """Create complete layer nodes for all parallel dimensions"""
        nodes = {}
        
        # Create nodes for each parallel combination
        for ep in range(self.ep_dim):
            for tp in range(self.tp_dim):
                for dp in range(self.dp_dim):
                    base_gpu = self.get_gpu_id(ep, tp, pp_stage, dp)
                    
                    # Layer normalization 1 (pre-attention)
                    node_id = f'ln1_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Layer Norm 1\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # QKV projections with TP decomposition
                    node_id = f'qkv_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'QKV Projection\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Multi-head attention computation
                    node_id = f'attn_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Multi-Head Attention\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},H={self.heads_per_tp},S={self.seq_length},D={self.head_dim}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Attention output projection
                    node_id = f'attn_out_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Attention Output Proj\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Attention All-Reduce across TP
                    node_id = f'attn_ar_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Attention All-Reduce\\nL{layer_id} EP{ep} TP Group {tp} PP{pp_stage} DP{dp}\\nGPU{base_gpu}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled'}
                    }
                    
                    # Residual connection addition
                    node_id = f'residual1_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Residual Add 1\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]×2\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Layer normalization 2 (pre-FFN)
                    node_id = f'ln2_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Layer Norm 2\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # MoE Gate (routing)
                    node_id = f'gate_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'MoE Gate (Routing)\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},K=2]',
                        'attrs': {'shape': 'parallelogram', 'fillcolor': 'lightyellow', 'style': 'filled'}
                    }
                    
                    # Expert dispatch (All-to-All communication)
                    node_id = f'dispatch_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Expert Dispatch\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled,dashed'}
                    }
                    
                    # Expert computation (multiple experts per GPU)
                    for expert in range(min(4, self.experts_per_ep)):  # Show first 4 experts per GPU
                        node_id = f'expert_l{layer_id}_e{expert}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                        nodes[node_id] = {
                            'label': f'Expert {expert}\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp//self.ep_dim},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp//self.ep_dim},S={self.seq_length},H={self.hidden_per_tp}]',
                            'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                        }
                    
                    # Expert combine (All-to-All communication)
                    node_id = f'combine_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Expert Combine\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled'}
                    }
                    
                    # FFN (Feed-Forward Network)
                    node_id = f'ffn_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'FFN (Up-proj + GeLU + Down-proj)\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # FFN All-Reduce
                    node_id = f'ffn_ar_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'FFN All-Reduce\\nL{layer_id} EP{ep} TP Group {tp} PP{pp_stage} DP{dp}\\nGPU{base_gpu}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled'}
                    }
                    
                    # Residual connection 2
                    node_id = f'residual2_l{layer_id}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{base_gpu}'
                    nodes[node_id] = {
                        'label': f'Residual Add 2\\nL{layer_id} EP{ep}TP{tp}PP{pp_stage}DP{dp}\\nGPU{base_gpu}\\nIn: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]×2\\nOut: [B={self.batch_per_dp},S={self.seq_length},H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
        return nodes
        
    def create_comprehensive_decode_phase(self):
        """Create complete decode phase with all parallel dimensions"""
        nodes = {}
        
        # Start marker
        nodes['decode_start'] = {
            'label': 'Decode Phase Start\\nSingle Token Processing',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue', 'style': 'filled', 'width': '3'}
        }
        
        # Create decode nodes for all parallel dimensions
        for ep in range(self.ep_dim):
            for tp in range(self.tp_dim):
                for pp in range(self.pp_dim):
                    for dp in range(self.dp_dim):
                        gpu_id = self.get_gpu_id(ep, tp, pp, dp)
                        
                        # KV cache read
                        node_id = f'kv_read_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        nodes[node_id] = {
                            'label': f'KV Cache Read\\nEP{ep}TP{tp}PP{pp}DP{dp}\\nGPU{gpu_id}\\nIn: [B=1,S=1,H={self.hidden_per_tp}]\\nOut: [B=1,S=1,H={self.hidden_per_tp}]',
                            'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                        }
                        
                        # Decode attention (single token)
                        node_id = f'decode_attn_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        nodes[node_id] = {
                            'label': f'Decode Attention\\nEP{ep}TP{tp}PP{pp}DP{dp}\\nGPU{gpu_id}\\nIn: [B=1,H={self.heads_per_tp},S=1,D={self.head_dim}]\\nOut: [B=1,H={self.heads_per_tp},S=1,D={self.head_dim}]',
                            'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                        }
                        
                        # KV cache update
                        node_id = f'kv_write_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        nodes[node_id] = {
                            'label': f'KV Cache Update\\nEP{ep}TP{tp}PP{pp}DP{dp}\\nGPU{gpu_id}',
                            'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                        }
                        
        # End marker
        nodes['decode_end'] = {
            'label': 'Decode Phase End\\nAll Tokens Processed',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue', 'style': 'filled', 'width': '3'}
        }
        
        return nodes
        
    def create_output_phase(self):
        """Create final output phase"""
        nodes = {}
        
        # Start marker
        nodes['output_start'] = {
            'label': 'Output Phase Start',
            'attrs': {'shape': 'box', 'fillcolor': 'lightblue', 'style': 'filled'}
        }
        
        # Final layer normalization
        for ep in range(self.ep_dim):
            for tp in range(self.tp_dim):
                for dp in range(self.dp_dim):
                    gpu_id = self.get_gpu_id(ep, tp, 1, dp)  # Use PP stage 1
                    node_id = f'final_ln_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    nodes[node_id] = {
                        'label': f'Final Layer Norm\\nEP{ep}TP{tp}DP{dp}\\nGPU{gpu_id}\\nIn: [B={self.batch_per_dp},S=1,H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S=1,H={self.hidden_per_tp}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Output projection head
                    node_id = f'output_proj_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    nodes[node_id] = {
                        'label': f'Output Projection\\nEP{ep}TP{tp}DP{dp}\\nGPU{gpu_id}\\nIn: [B={self.batch_per_dp},S=1,H={self.hidden_per_tp}]\\nOut: [B={self.batch_per_dp},S=1,V={self.vocab_size//self.tp_dim}]',
                        'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled'}
                    }
                    
                    # Final All-Reduce for output
                    node_id = f'final_ar_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    nodes[node_id] = {
                        'label': f'Final All-Reduce\\nEP{ep} TP Group {tp} DP{dp}\\nGPU{gpu_id}',
                        'attrs': {'shape': 'ellipse', 'fillcolor': 'lightblue', 'style': 'filled'}
                    }
                    
        # Final output
        nodes['final_output'] = {
            'label': f'Final Output\\nBatch: {self.batch_size}, Vocab: {self.vocab_size}',
            'attrs': {'shape': 'box', 'fillcolor': 'lightgreen', 'style': 'filled', 'width': '3'}
        }
        
        return nodes
        
    def get_comprehensive_edges(self):
        """Define comprehensive edges for all nodes"""
        edges = []
        
        # Input to embedding
        edges.append({'from': 'input', 'to': 'prefill_start'})
        
        # Embedding flow for all dimensions
        for dp in range(self.dp_dim):
            for tp in range(self.tp_dim):
                gpu_id = self.get_gpu_id(0, tp, 0, dp)
                embed_id = f'embed_dp{dp}_tp{tp}_gpu{gpu_id}'
                ar_id = f'embed_ar_dp{dp}_tp{tp}_gpu{gpu_id}'
                
                edges.append({'from': 'prefill_start', 'to': embed_id})
                edges.append({'from': embed_id, 'to': ar_id})
                
        # Layer processing for all layers and dimensions
        for layer in range(self.num_layers):
            pp_stage = layer // self.layers_per_pp
            
            # Connect first layer from embedding
            if layer == 0:
                for dp in range(self.dp_dim):
                    for tp in range(self.tp_dim):
                        gpu_id = self.get_gpu_id(0, tp, pp_stage, dp)
                        prev_id = f'embed_ar_dp{dp}_tp{tp}_gpu{gpu_id}'
                        next_id = f'ln1_l{layer}_ep0_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        edges.append({'from': prev_id, 'to': next_id})
            
            # Complete layer flow for all parallel dimensions
            for ep in range(self.ep_dim):
                for tp in range(self.tp_dim):
                    for dp in range(self.dp_dim):
                        gpu_id = self.get_gpu_id(ep, tp, pp_stage, dp)
                        
                        # Layer norm 1 → QKV → Attention → Attention output
                        ln1_id = f'ln1_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        qkv_id = f'qkv_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        attn_id = f'attn_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        attn_out_id = f'attn_out_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        attn_ar_id = f'attn_ar_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        residual1_id = f'residual1_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        
                        edges.append({'from': ln1_id, 'to': qkv_id})
                        edges.append({'from': qkv_id, 'to': attn_id})
                        edges.append({'from': attn_id, 'to': attn_out_id})
                        edges.append({'from': attn_out_id, 'to': attn_ar_id})
                        edges.append({'from': attn_ar_id, 'to': residual1_id})
                        
                        # MoE components
                        ln2_id = f'ln2_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        gate_id = f'gate_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        dispatch_id = f'dispatch_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        combine_id = f'combine_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        ffn_id = f'ffn_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        ffn_ar_id = f'ffn_ar_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        residual2_id = f'residual2_l{layer}_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'
                        
                        edges.append({'from': residual1_id, 'to': ln2_id})
                        edges.append({'from': ln2_id, 'to': gate_id})
                        edges.append({'from': gate_id, 'to': dispatch_id, 'attrs': {'style': 'dashed'}})
                        edges.append({'from': dispatch_id, 'to': f'expert_l{layer}_e0_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}'})
                        edges.append({'from': f'expert_l{layer}_e0_ep{ep}_tp{tp}_pp{pp_stage}_dp{dp}_gpu{gpu_id}', 'to': combine_id})
                        edges.append({'from': combine_id, 'to': ffn_id})
                        edges.append({'from': ffn_id, 'to': ffn_ar_id})
                        edges.append({'from': ffn_ar_id, 'to': residual2_id})
                        
        # Connect prefill to decode
        edges.append({'from': 'residual2_l15_ep7_tp3_pp1_dp3_gpu255', 'to': 'prefill_end'})
        edges.append({'from': 'prefill_end', 'to': 'decode_start'})
        
        # Decode phase edges
        for ep in range(self.ep_dim):
            for tp in range(self.tp_dim):
                for pp in range(self.pp_dim):
                    for dp in range(self.dp_dim):
                        gpu_id = self.get_gpu_id(ep, tp, pp, dp)
                        
                        kv_read_id = f'kv_read_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        decode_attn_id = f'decode_attn_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        kv_write_id = f'kv_write_ep{ep}_tp{tp}_pp{pp}_dp{dp}_gpu{gpu_id}'
                        
                        edges.append({'from': 'decode_start', 'to': kv_read_id})
                        edges.append({'from': kv_read_id, 'to': decode_attn_id})
                        edges.append({'from': decode_attn_id, 'to': kv_write_id})
                        
        # Connect decode to output
        edges.append({'from': 'kv_write_ep7_tp3_pp1_dp3_gpu255', 'to': 'decode_end'})
        edges.append({'from': 'decode_end', 'to': 'output_start'})
        
        # Output phase edges
        for ep in range(self.ep_dim):
            for tp in range(self.tp_dim):
                for dp in range(self.dp_dim):
                    gpu_id = self.get_gpu_id(ep, tp, 1, dp)
                    
                    final_ln_id = f'final_ln_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    output_proj_id = f'output_proj_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    final_ar_id = f'final_ar_ep{ep}_tp{tp}_dp{dp}_gpu{gpu_id}'
                    
                    edges.append({'from': 'output_start', 'to': final_ln_id})
                    edges.append({'from': final_ln_id, 'to': output_proj_id})
                    edges.append({'from': output_proj_id, 'to': final_ar_id})
                    edges.append({'from': final_ar_id, 'to': 'final_output'})
                    
        return edges

def main():
    """Generate and save the comprehensive DAG"""
    generator = ComprehensiveLLMDeploymentDAGGenerator()
    dag = generator.create_dag()
    
    # Save DOT file
    dot_path = '../outputs/2025-12-22-11-52-26/llm_deployment_dag_final.dot'
    dag.save(dot_path)
    
    # Save SVG image
    svg_path = '../outputs/2025-12-22-11-52-26/llm_deployment_dag_final.svg'
    dag.render('../outputs/2025-12-22-11-52-26/llm_deployment_dag_final', format='svg', cleanup=True)
    
    print("Comprehensive DAG generated successfully!")
    print(f"DOT file: {dot_path}")
    print(f"SVG image: {svg_path}")
    
    # Return paths for JSON output
    return {
        "dot_path": dot_path,
        "svg_path": svg_path
    }

if __name__ == "__main__":
    paths = main()
    print(f"\nGenerated files:\n{paths}")