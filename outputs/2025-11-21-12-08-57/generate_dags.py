#!/usr/bin/env python3
"""
Generate deployment DAGs for Large-Scale Cross-Node Expert Parallelism paper
This script creates detailed DAGs showing tensor dimensions and GPU assignments
"""

import graphviz
from typing import Dict, List, Tuple

class DAGGenerator:
    def __init__(self):
        self.dag_counter = 0
        
    def create_baseline_dag(self) -> graphviz.Digraph:
        """Create baseline DAG with TP=8, PP=2"""
        dot = graphviz.Digraph('baseline_moe_deployment', 
                              comment='Baseline MoE Deployment: TP8_PP2')
        dot.attr(rankdir='TB', size='20,30')
        dot.attr('node', shape='rectangle')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]', 
                shape='ellipse')
        
        # Stage 0 (GPUs 0-7)
        self._add_stage(dot, 0, 8, [0,1,2,3,4,5,6,7], layer_range='0-7')
        
        # Cross-stage communication
        dot.node('stage0_to_stage1', 'Cross-Stage Transfer\\nAll-to-All\\n128×10000×4096 BF16\\nGPUs 0-7 → 8-15', 
                shape='parallelogram')
        
        # Stage 1 (GPUs 8-15)
        self._add_stage(dot, 1, 8, [8,9,10,11,12,13,14,15], layer_range='8-15')
        
        # Output node
        dot.node('output', 'Output\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
                shape='ellipse')
        
        # Connections
        dot.edge('input', 'layer0_mha_qkv_0')
        dot.edge('layer7_ffn2_7', 'stage0_to_stage1')
        dot.edge('stage0_to_stage1', 'layer8_mha_qkv_8')
        dot.edge('layer15_ffn2_15', 'output')
        
        return dot
    
    def create_proposed_dag(self) -> graphviz.Digraph:
        """Create proposed DAG with EP=16"""
        dot = graphviz.Digraph('proposed_moe_deployment', 
                              comment='Proposed MoE Deployment: EP16_OneExpertPerGPU')
        dot.attr(rankdir='TB', size='20,30')
        dot.attr('node', shape='rectangle')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden=4096]', 
                shape='ellipse')
        
        # Routing layer
        dot.node('routing', 'Gating\\nTop-2 Expert Selection\\nInput: [batch_size=128, seq_len=10000, hidden=4096]\\nOutput: [batch_size=128, seq_len=10000, expert_routing=2]', 
                shape='diamond')
        
        # Add all 16 layers, each with 16 experts distributed across 16 GPUs
        for layer_id in range(16):
            self._add_layer_with_experts(dot, layer_id)
        
        # Output node
        dot.node('output', 'Output\\nOutput: [batch_size=128, seq_len=10000, hidden=4096]', 
                shape='ellipse')
        
        # Connections
        dot.edge('input', 'routing')
        dot.edge('routing', 'layer0_mha_qkv')
        dot.edge('layer15_mha_out', 'output')
        
        return dot
    
    def _add_stage(self, dot: graphviz.Digraph, stage_id: int, tp_size: int, 
                   gpus: List[int], layer_range: str):
        """Add a complete stage with tensor parallelism"""
        
        for layer in range(int(layer_range.split('-')[0]), int(layer_range.split('-')[1]) + 1):
            # MHA QKV projection (column parallel)
            for gpu_idx, gpu_id in enumerate(gpus):
                dot.node(f'layer{layer}_mha_qkv_{gpu_id}', 
                        f'MHA QKV Proj\\nTP Shard {gpu_idx}/{tp_size}\\n' +
                        f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                        f'Output: [batch_size=128, seq_len=10000, heads=32, d_k=128, width=1/{tp_size}]\\n' +
                        f'GPU {gpu_id}',
                        style='filled', fillcolor='lightblue')
            
            # MHA attention
            dot.node(f'layer{layer}_mha_attn_{gpu_id}', 
                    f'MHA Attention\\nAll heads combined\\n' +
                    f'Input: [batch_size=128, seq_len=10000, heads=32, d_k=128]\\n' +
                    f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'All GPUs',
                    shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # MHA output projection (row parallel)
            for gpu_idx, gpu_id in enumerate(gpus):
                dot.node(f'layer{layer}_mha_out_{gpu_id}', 
                        f'MHA Output Proj\\nTP Shard {gpu_idx}/{tp_size}\\n' +
                        f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                        f'Output: [batch_size=128, seq_len=10000, hidden=4096/{tp_size}]\\n' +
                        f'GPU {gpu_id}',
                        style='filled', fillcolor='lightblue')
            
            # Residual connection
            dot.node(f'layer{layer}_res1_{gpu_id}', 
                    f'Residual Add\\n' +
                    f'Input1: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'Input2: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'All GPUs',
                    shape='parallelogram')
            
            # Layer norm 1
            dot.node(f'layer{layer}_ln1_{gpu_id}', 
                    f'Layer Norm 1\\n' +
                    f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'All GPUs',
                    style='filled', fillcolor='yellow')
            
            # FFN Gate (column parallel)
            for gpu_idx, gpu_id in enumerate(gpus):
                dot.node(f'layer{layer}_ffn_gate_{gpu_id}', 
                        f'FFN Gate\\nTP Shard {gpu_idx}/{tp_size}\\n' +
                        f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                        f'Output: [batch_size=128, seq_len=10000, ffn_hidden=16384/{tp_size}]\\n' +
                        f'GPU {gpu_id}',
                        style='filled', fillcolor='lightblue')
            
            # FFN Up (column parallel)
            for gpu_idx, gpu_id in enumerate(gpus):
                dot.node(f'layer{layer}_ffn_up_{gpu_id}', 
                        f'FFN Up\\nTP Shard {gpu_idx}/{tp_size}\\n' +
                        f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                        f'Output: [batch_size=128, seq_len=10000, ffn_hidden=16384/{tp_size}]\\n' +
                        f'GPU {gpu_id}',
                        style='filled', fillcolor='lightblue')
            
            # FFN activation (SiLU on gate * up)
            dot.node(f'layer{layer}_ffn_act_{gpu_id}', 
                    f'FFN Activation\\nSiLU(gate) * up\\n' +
                    f'Input: [batch_size=128, seq_len=10000, ffn_hidden=16384]\\n' +
                    f'Output: [batch_size=128, seq_len=10000, ffn_hidden=16384]\\n' +
                    f'All GPUs',
                    shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # FFN Down (row parallel)
            for gpu_idx, gpu_id in enumerate(gpus):
                dot.node(f'layer{layer}_ffn_down_{gpu_id}', 
                        f'FFN Down\\nTP Shard {gpu_idx}/{tp_size}\\n' +
                        f'Input: [batch_size=128, seq_len=10000, ffn_hidden=16384]\\n' +
                        f'Output: [batch_size=128, seq_len=10000, hidden=4096/{tp_size}]\\n' +
                        f'GPU {gpu_id}',
                        style='filled', fillcolor='lightblue')
            
            # Residual connection 2
            dot.node(f'layer{layer}_res2_{gpu_id}', 
                    f'Residual Add\\n' +
                    f'Input1: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'Input2: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                    f'All GPUs',
                    shape='parallelogram')
            
            # Connections within layer
            for gpu_idx, gpu_id in enumerate(gpus):
                if layer == 0 and stage_id == 0 and gpu_idx == 0:
                    prev_node = 'input'
                elif layer > int(layer_range.split('-')[0]) or stage_id > 0:
                    prev_node = f'layer{layer-1}_res2_{gpu_id}' if layer > 0 else f'layer{layer}_mha_qkv_{gpu_id}'
                else:
                    prev_node = 'input'
                
                dot.edge(prev_node, f'layer{layer}_mha_qkv_{gpu_id}')
                dot.edge(f'layer{layer}_mha_qkv_{gpu_id}', f'layer{layer}_mha_attn_{gpu_id}')
                dot.edge(f'layer{layer}_mha_attn_{gpu_id}', f'layer{layer}_mha_out_{gpu_id}')
                dot.edge(f'layer{layer}_mha_out_{gpu_id}', f'layer{layer}_res1_{gpu_id}')
                dot.edge(f'layer{layer}_res1_{gpu_id}', f'layer{layer}_ln1_{gpu_id}')
                dot.edge(f'layer{layer}_ln1_{gpu_id}', f'layer{layer}_ffn_gate_{gpu_id}')
                dot.edge(f'layer{layer}_ln1_{gpu_id}', f'layer{layer}_ffn_up_{gpu_id}')
                dot.edge(f'layer{layer}_ffn_gate_{gpu_id}', f'layer{layer}_ffn_act_{gpu_id}')
                dot.edge(f'layer{layer}_ffn_up_{gpu_id}', f'layer{layer}_ffn_act_{gpu_id}')
                dot.edge(f'layer{layer}_ffn_act_{gpu_id}', f'layer{layer}_ffn_down_{gpu_id}')
                dot.edge(f'layer{layer}_ffn_down_{gpu_id}', f'layer{layer}_res2_{gpu_id}')
    
    def _add_layer_with_experts(self, dot: graphviz.Digraph, layer_id: int):
        """Add a complete layer with expert selection and computation"""
        
        # MHA components (replicated across all GPUs)
        dot.node(f'layer{layer_id}_mha_qkv', 
                f'MHA QKV Projection\\n' +
                f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, heads=32, d_k=128]\\n' +
                f'All GPUs',
                style='filled', fillcolor='lightblue')
        
        dot.node(f'layer{layer_id}_mha_attn', 
                f'MHA Attention\\n' +
                f'Input: [batch_size=128, seq_len=10000, heads=32, d_k=128]\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                style='filled', fillcolor='lightgreen')
        
        dot.node(f'layer{layer_id}_mha_out', 
                f'MHA Output Projection\\n' +
                f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                style='filled', fillcolor='lightblue')
        
        dot.node(f'layer{layer_id}_res1', 
                f'Residual Add\\n' +
                f'Input1: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Input2: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                shape='parallelogram')
        
        dot.node(f'layer{layer_id}_ln1', 
                f'Layer Norm\\n' +
                f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                style='filled', fillcolor='yellow')
        
        # Expert routing and selection
        dot.node(f'layer{layer_id}_expert_routing', 
                f'Expert Routing\\nTop-2 Selection\\n' +
                f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, expert_id=2]\\n' +
                f'All GPUs',
                shape='diamond', style='filled', fillcolor='orange')
        
        # Token distribution to experts
        dot.node(f'layer{layer_id}_token_distribution', 
                f'Token Distribution\\n' +
                f'Input: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: Tokens routed to 2 experts\\n' +
                f'All GPUs → Expert GPUs',
                shape='parallelogram', style='filled', fillcolor='lightcoral')
        
        # Expert computations (one per GPU)
        for expert_id in range(16):
            gpu_id = expert_id
            dot.node(f'layer{layer_id}_expert{expert_id}_gate', 
                    f'Expert {expert_id} Gate\\n' +
                    f'Input: [dynamic_batch, seq_len, hidden=4096]\\n' +
                    f'Output: [dynamic_batch, seq_len, ffn_hidden=16384]\\n' +
                    f'GPU {gpu_id}',
                    style='filled', fillcolor='lightblue')
            
            dot.node(f'layer{layer_id}_expert{expert_id}_up', 
                    f'Expert {expert_id} Up\\n' +
                    f'Input: [dynamic_batch, seq_len, hidden=4096]\\n' +
                    f'Output: [dynamic_batch, seq_len, ffn_hidden=16384]\\n' +
                    f'GPU {gpu_id}',
                    style='filled', fillcolor='lightblue')
            
            dot.node(f'layer{layer_id}_expert{expert_id}_act', 
                    f'Expert {expert_id} Activation\\n' +
                    f'Input: [dynamic_batch, seq_len, ffn_hidden=16384]\\n' +
                    f'Output: [dynamic_batch, seq_len, ffn_hidden=16384]\\n' +
                    f'GPU {gpu_id}',
                    style='filled', fillcolor='lightgreen')
            
            dot.node(f'layer{layer_id}_expert{expert_id}_down', 
                    f'Expert {expert_id} Down\\n' +
                    f'Input: [dynamic_batch, seq_len, ffn_hidden=16384]\\n' +
                    f'Output: [dynamic_batch, seq_len, hidden=4096]\\n' +
                    f'GPU {gpu_id}',
                    style='filled', fillcolor='lightblue')
        
        # Token aggregation from experts
        dot.node(f'layer{layer_id}_token_aggregation', 
                f'Token Aggregation\\n' +
                f'Input: [dynamic_batch, seq_len, hidden=4096] from 2 experts\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                shape='parallelogram', style='filled', fillcolor='lightcoral')
        
        # Final residual
        dot.node(f'layer{layer_id}_res2', 
                f'Residual Add\\n' +
                f'Input1: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Input2: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'Output: [batch_size=128, seq_len=10000, hidden=4096]\\n' +
                f'All GPUs',
                shape='parallelogram')
        
        # Connections within layer
        if layer_id == 0:
            prev_node = 'routing'
        else:
            prev_node = f'layer{layer_id-1}_res2'
        
        dot.edge(prev_node, f'layer{layer_id}_mha_qkv')
        dot.edge(f'layer{layer_id}_mha_qkv', f'layer{layer_id}_mha_attn')
        dot.edge(f'layer{layer_id}_mha_attn', f'layer{layer_id}_mha_out')
        dot.edge(f'layer{layer_id}_mha_out', f'layer{layer_id}_res1')
        dot.edge(f'layer{layer_id}_res1', f'layer{layer_id}_ln1')
        dot.edge(f'layer{layer_id}_ln1', f'layer{layer_id}_expert_routing')
        dot.edge(f'layer{layer_id}_expert_routing', f'layer{layer_id}_token_distribution')
        
        # Expert connections
        for expert_id in range(16):
            dot.edge(f'layer{layer_id}_token_distribution', 
                    f'layer{layer_id}_expert{expert_id}_gate', 
                    style='dashed')
            dot.edge(f'layer{layer_id}_token_distribution', 
                    f'layer{layer_id}_expert{expert_id}_up',
                    style='dashed')
            dot.edge(f'layer{layer_id}_expert{expert_id}_gate', 
                    f'layer{layer_id}_expert{expert_id}_act')
            dot.edge(f'layer{layer_id}_expert{expert_id}_up', 
                    f'layer{layer_id}_expert{expert_id}_act')
            dot.edge(f'layer{layer_id}_expert{expert_id}_act', 
                    f'layer{layer_id}_expert{expert_id}_down')
            dot.edge(f'layer{layer_id}_expert{expert_id}_down', 
                    f'layer{layer_id}_token_aggregation')
        
        dot.edge(f'layer{layer_id}_token_aggregation', f'layer{layer_id}_res2')

def main():
    """Generate all DAGs"""
    generator = DAGGenerator()
    
    # Create baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = generator.create_baseline_dag()
    
    # Save baseline DAG
    baseline_dag.render('../outputs/2025-11-21-12-08-57/baseline_moe_deployment', 
                       format='svg', cleanup=True)
    
    with open('../outputs/2025-11-21-12-08-57/baseline_moe_deployment.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    # Create proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = generator.create_proposed_dag()
    
    # Save proposed DAG
    proposed_dag.render('../outputs/2025-11-21-12-08-57/proposed_moe_deployment', 
                       format='svg', cleanup=True)
    
    with open('../outputs/2025-11-21-12-08-57/proposed_moe_deployment.dot', 'w') as f:
        f.write(proposed_dag.source)
    
    print("DAG generation complete!")
    
    # Generate summary
    summary = {
        "baselines": [
            {
                "name": "baseline_moe_deployment",
                "description": "TP8_PP2 baseline with 2 experts per GPU",
                "total_gpus": 16,
                "experts_per_gpu": 2,
                "parallelism": "TP=8, PP=2"
            }
        ],
        "proposed": [
            {
                "name": "proposed_moe_deployment", 
                "description": "EP16 with 1 expert per GPU",
                "total_gpus": 16,
                "experts_per_gpu": 1,
                "parallelism": "EP=16"
            }
        ],
        "files_generated": [
            "../outputs/2025-11-21-12-08-57/baseline_moe_deployment.dot",
            "../outputs/2025-11-21-12-08-57/baseline_moe_deployment.svg",
            "../outputs/2025-11-21-12-08-57/proposed_moe_deployment.dot", 
            "../outputs/2025-11-21-12-08-57/proposed_moe_deployment.svg"
        ]
    }
    
    import json
    with open('../outputs/2025-11-21-12-08-57/dags_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()