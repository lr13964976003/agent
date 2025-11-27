#!/usr/bin/env python3
"""
Generate complete deployment DAGs for Large-Scale Cross-Node Expert Parallelism
"""

import graphviz
import os
import json

class MoEDeploymentDAGGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.model_config = {
            'total_layers': 61,
            'dense_layers': [0, 1, 2],
            'moe_layers': list(range(3, 61)),
            'token_dim': 7168,
            'mha_heads': 128,
            'head_dim': 128,
            'mlp_hidden': 2048,
            'experts_per_layer': 64,
            'batch_size': 4,
            'seq_len': 2048
        }
        
        self.hardware_config = {
            'total_gpus': 128,
            'gpus_per_node': 8,
            'total_nodes': 16,
            'ep_degree': 64,
            'dp_degree': 2,
            'tp_degree': 1,
            'pp_degree': 1
        }
        
    def _get_gpu_id(self, expert_id: int, dp_rank: int) -> int:
        return expert_id + dp_rank * self.hardware_config['ep_degree']
    
    def _get_node_id(self, gpu_id: int) -> int:
        return gpu_id // self.hardware_config['gpus_per_node']
    
    def generate_all_dags(self):
        """Generate all required DAGs"""
        os.makedirs(self.output_dir, exist_ok=True)
        generated_files = []
        
        # 1. System overview DAG
        dot = graphviz.Digraph('system_overview')
        dot.attr(rankdir='TB')
        
        for dp_rank in range(self.hardware_config['dp_degree']):
            with dot.subgraph(name=f'cluster_dp_{dp_rank}') as c:
                c.attr(label=f'Data Parallel Replica {dp_rank}', style='rounded', color='blue')
                
                # Dense layers
                for layer_idx in self.model_config['dense_layers']:
                    c.node(f'dense_{layer_idx}_{dp_rank}', 
                          f'Dense Layer {layer_idx}\\nGPU:{dp_rank*8}-{(dp_rank+1)*8-1}\\nFull MHA+MLP',
                          shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # MoE layers
                for layer_idx in self.model_config['moe_layers']:
                    for expert_id in range(min(8, self.model_config['experts_per_layer'])):
                        gpu_id = self._get_gpu_id(expert_id, dp_rank)
                        node_id = self._get_node_id(gpu_id)
                        c.node(f'expert_{layer_idx}_{expert_id}_{dp_rank}',
                              f'Expert {expert_id}\\nGPU:{gpu_id}\\nNode:{node_id}',
                              shape='rectangle', style='filled', fillcolor='lightcoral')
                    
                    c.node(f'gate_{layer_idx}_{dp_rank}',
                          f'Gate\\nLayer {layer_idx}\\nDP:{dp_rank}',
                          shape='parallelogram', style='filled', fillcolor='orange')
                    c.node(f'agg_{layer_idx}_{dp_rank}',
                          f'Aggregate\\nLayer {layer_idx}\\nDP:{dp_rank}',
                          shape='parallelogram', style='filled', fillcolor='lightgrey')
        
        # Connections
        for dp_rank in range(self.hardware_config['dp_degree']):
            prev = f'dense_0_{dp_rank}'
            for layer_idx in self.model_config['dense_layers']:
                if layer_idx > 0:
                    dot.edge(f'dense_{layer_idx-1}_{dp_rank}', f'dense_{layer_idx}_{dp_rank}')
                    prev = f'dense_{layer_idx}_{dp_rank}'
            
            for layer_idx in self.model_config['moe_layers']:
                dot.edge(prev, f'gate_{layer_idx}_{dp_rank}')
                for expert_id in range(min(8, self.model_config['experts_per_layer'])):
                    dot.edge(f'gate_{layer_idx}_{dp_rank}', 
                           f'expert_{layer_idx}_{expert_id}_{dp_rank}', 
                           style='dashed')
                    dot.edge(f'expert_{layer_idx}_{expert_id}_{dp_rank}', 
                           f'agg_{layer_idx}_{dp_rank}', 
                           style='dashed')
                prev = f'agg_{layer_idx}_{dp_rank}'
        
        overview_path = os.path.join(self.output_dir, "system_overview")
        dot.render(overview_path, format='dot')
        dot.render(overview_path, format='svg')
        generated_files.extend([f"{overview_path}.dot", f"{overview_path}.svg"])
        
        # 2. Detailed attention DAG
        attn_dot = graphviz.Digraph('detailed_attention')
        attn_dot.attr(rankdir='TB')
        
        input_shape = f"batch_size={self.model_config['batch_size']}, seq_len={self.model_config['seq_len']}, heads={self.model_config['mha_heads']}, d_k={self.model_config['head_dim']}"
        
        # LayerNorm
        attn_dot.node('ln', f'LayerNorm\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{input_shape}]', 
                     shape='rectangle', style='filled', fillcolor='lightblue')
        
        # QKV projections
        attn_dot.node('q_proj', f'Q Projection\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{input_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        attn_dot.node('k_proj', f'K Projection\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{input_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        attn_dot.node('v_proj', f'V Projection\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{input_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention computation
        attn_dot.node('scale', f'Scale\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{input_shape}]',
                     shape='rectangle', style='filled', fillcolor='yellow')
        attn_dot.node('matmul1', f'QK^T MatMul\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [batch_size=4,seq_len=2048,seq_len=2048]',
                     shape='rectangle', style='filled', fillcolor='yellow')
        attn_dot.node('softmax', f'Softmax\\nGPU:0\\nInput: [batch_size=4,seq_len=2048,seq_len=2048]\\nOutput: [batch_size=4,seq_len=2048,seq_len=2048]',
                     shape='rectangle', style='filled', fillcolor='yellow')
        attn_dot.node('matmul2', f'Attention V MatMul\\nGPU:0\\nInput: [batch_size=4,seq_len=2048,seq_len=2048]\\nOutput: [{input_shape}]',
                     shape='rectangle', style='filled', fillcolor='yellow')
        
        # Output projection
        output_shape = f"batch_size={self.model_config['batch_size']}, seq_len={self.model_config['seq_len']}, hidden={self.model_config['token_dim']}"
        attn_dot.node('out_proj', f'Output Projection\\nGPU:0\\nInput: [{input_shape}]\\nOutput: [{output_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        attn_dot.node('residual', f'Residual Add\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgrey')
        
        # Connections
        attn_dot.edge('ln', 'q_proj')
        attn_dot.edge('ln', 'k_proj')
        attn_dot.edge('ln', 'v_proj')
        attn_dot.edge('q_proj', 'scale')
        attn_dot.edge('k_proj', 'matmul1')
        attn_dot.edge('scale', 'matmul1')
        attn_dot.edge('matmul1', 'softmax')
        attn_dot.edge('softmax', 'matmul2')
        attn_dot.edge('v_proj', 'matmul2')
        attn_dot.edge('matmul2', 'out_proj')
        attn_dot.edge('out_proj', 'residual')
        
        attn_path = os.path.join(self.output_dir, "detailed_attention")
        attn_dot.render(attn_path, format='dot')
        attn_dot.render(attn_path, format='svg')
        generated_files.extend([f"{attn_path}.dot", f"{attn_path}.svg"])
        
        # 3. Detailed expert MLP DAG
        expert_dot = graphviz.Digraph('detailed_expert_mlp')
        expert_dot.attr(rankdir='TB')
        
        # Input
        expert_dot.node('input', f'Expert Input\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                       shape='ellipse', style='filled', fillcolor='lightblue')
        
        # LayerNorm
        expert_dot.node('ln2', f'LayerNorm\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                       shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Gate
        gate_shape = f"batch_size={self.model_config['batch_size']}, seq_len={self.model_config['seq_len']}, experts=64"
        expert_dot.node('gate', f'Gating Network\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{gate_shape}]',
                       shape='parallelogram', style='filled', fillcolor='orange')
        
        # MLP layers
        hidden_shape = f"batch_size={self.model_config['batch_size']}, seq_len={self.model_config['seq_len']}, hidden={self.model_config['mlp_hidden']}"
        expert_dot.node('fc1', f'FC1 (Up-project)\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{hidden_shape}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
        expert_dot.node('gelu', f'GELU Activation\\nGPU:0\\nInput: [{hidden_shape}]\\nOutput: [{hidden_shape}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
        expert_dot.node('fc2', f'FC2 (Down-project)\\nGPU:0\\nInput: [{hidden_shape}]\\nOutput: [{output_shape}]',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Aggregation
        expert_dot.node('agg', f'Expert Aggregation\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                       shape='parallelogram', style='filled', fillcolor='lightgrey')
        expert_dot.node('residual2', f'Residual Add\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                       shape='rectangle', style='filled', fillcolor='lightgrey')
        
        # Connections
        expert_dot.edge('input', 'ln2')
        expert_dot.edge('ln2', 'gate')
        expert_dot.edge('ln2', 'fc1')
        expert_dot.edge('fc1', 'gelu')
        expert_dot.edge('gelu', 'fc2')
        expert_dot.edge('fc2', 'agg')
        expert_dot.edge('agg', 'residual2')
        
        expert_path = os.path.join(self.output_dir, "detailed_expert_mlp")
        expert_dot.render(expert_path, format='dot')
        expert_dot.render(expert_path, format='svg')
        generated_files.extend([f"{expert_path}.dot", f"{expert_path}.svg"])
        
        # 4. Token routing DAG
        routing_dot = graphviz.Digraph('token_routing')
        routing_dot.attr(rankdir='LR')
        
        # Token distribution
        routing_dot.node('distributor', f'Token Distributor\\nLayer:3, DP:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                        shape='parallelogram', style='filled', fillcolor='gold')
        
        # Expert routing
        for expert_id in range(8):  # Show first 8 experts
            gpu_id = self._get_gpu_id(expert_id, 0)
            node_id = self._get_node_id(gpu_id)
            routing_dot.node(f'expert_{expert_id}',
                           f'Expert {expert_id}\\nGPU:{gpu_id}\\nNode:{node_id}\\nInput: [batch_size=var,seq_len=var,hidden=7168]\\nOutput: [batch_size=var,seq_len=var,hidden=7168]',
                           shape='rectangle', style='filled', fillcolor='lightcoral')
            routing_dot.edge('distributor', f'expert_{expert_id}', style='dashed')
        
        # Aggregation
        routing_dot.node('aggregator', f'Expert Aggregation\\nLayer:3, DP:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                        shape='parallelogram', style='filled', fillcolor='lightgrey')
        
        for expert_id in range(8):
            routing_dot.edge(f'expert_{expert_id}', 'aggregator', style='dashed')
        
        routing_path = os.path.join(self.output_dir, "token_routing")
        routing_dot.render(routing_path, format='dot')
        routing_dot.render(routing_path, format='svg')
        generated_files.extend([f"{routing_path}.dot", f"{routing_path}.svg"])
        
        # 5. Complete layer DAG
        complete_dot = graphviz.Digraph('complete_layer3_dp0')
        complete_dot.attr(rankdir='TB', compound='true')
        
        # Attention phase
        with complete_dot.subgraph(name='cluster_attention') as attn:
            attn.attr(label='Multi-Head Attention Phase', style='rounded')
            attn.node('attn_input', f'Layer Input\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                     shape='ellipse', style='filled', fillcolor='lightblue')
            attn.node('mha', f'MHA Complete\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            attn.edge('attn_input', 'mha')
        
        # Expert phase
        with complete_dot.subgraph(name='cluster_experts') as experts:
            experts.attr(label='Expert Parallel MoE Phase', style='rounded')
            experts.node('distribute', f'Token Distribution\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                        shape='parallelogram', style='filled', fillcolor='gold')
            
            for expert_id in range(4):
                gpu_id = self._get_gpu_id(expert_id, 0)
                node_id = self._get_node_id(gpu_id)
                experts.node(f'expert_{expert_id}_comp',
                           f'Expert {expert_id} MLP\\nGPU:{gpu_id}\\nNode:{node_id}',
                           shape='rectangle', style='filled', fillcolor='lightcoral')
            
            experts.node('aggregate', f'Expert Aggregation\\nGPU:0\\nInput: [{output_shape}]\\nOutput: [{output_shape}]',
                        shape='parallelogram', style='filled', fillcolor='lightgrey')
        
        # Connections
        complete_dot.edge('mha', 'distribute', ltail='cluster_attention', lhead='cluster_experts')
        
        for expert_id in range(4):
            complete_dot.edge('distribute', f'expert_{expert_id}_comp', style='dashed')
            complete_dot.edge(f'expert_{expert_id}_comp', 'aggregate', style='dashed')
        
        complete_path = os.path.join(self.output_dir, "complete_layer3_dp0")
        complete_dot.render(complete_path, format='dot')
        complete_dot.render(complete_path, format='svg')
        generated_files.extend([f"{complete_path}.dot", f"{complete_path}.svg"])
        
        # Create manifest
        manifest = {
            "generated_dags": [f.replace('../outputs/2025-11-26-17-03-32/', '') for f in generated_files],
            "deployment_strategy": "large_scale_cross_node_expert_parallelism",
            "ep_degree": self.hardware_config['ep_degree'],
            "dp_degree": self.hardware_config['dp_degree'],
            "total_gpus": self.hardware_config['total_gpus'],
            "experts_per_gpu": 1,
            "timestamp": "2025-11-26-17-03-32"
        }
        
        manifest_path = os.path.join(self.output_dir, "dag_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return generated_files + [manifest_path]

def main():
    output_dir = "../outputs/2025-11-26-17-03-32"
    generator = MoEDeploymentDAGGenerator(output_dir)
    files = generator.generate_all_dags()
    
    print(f"Generated {len(files)} DAG files in {output_dir}")
    return files

if __name__ == "__main__":
    main()