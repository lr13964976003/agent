#!/usr/bin/env python3

import graphviz
from typing import Dict, List, Tuple
import os
import json

class MoEDAGGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model dimensions from paper
        self.batch_size = 128
        self.seq_len = 10000
        self.token_dim = 4096
        self.mha_heads = 32
        self.mha_head_dim = 128
        self.mha_total_dim = self.mha_heads * self.mha_head_dim
        self.ffn_hidden = 16384
        self.num_layers = 16
        self.num_experts = 16
        
    def _create_baseline_dag(self):
        """Create DAG for baseline: TP=8, PP=2, 8 experts per GPU"""
        dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', 
                              comment='MoE Baseline with Tensor Parallel=8, Pipeline Parallel=2')
        
        dot.attr(rankdir='TB', size='20,30')
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Input node
        dot.node('input', 'Model Input\\nbatch_size=128, seq_len=10000, token_dim=4096', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Pipeline stages
        dot.attr('node', shape='rectangle', fillcolor='lightblue')
        
        # Stage 0 (GPUs 0-7, layers 0-7)
        with dot.subgraph(name='cluster_stage0') as stage0:
            stage0.attr(label='Pipeline Stage 0\\nGPUs 0-7', color='red', style='dashed')
            
            # Each layer in stage 0
            for layer_idx in range(8):
                layer_name = f'layer_{layer_idx}'
                
                # MHA for this layer (tensor parallel across 8 GPUs)
                mha_name = f'{layer_name}_mha'
                stage0.node(mha_name, 
                           f'Layer {layer_idx}\\nMulti-Head Attention\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nTP=8 across GPUs 0-7')
                
                # Expert selection gate
                gate_name = f'{layer_name}_gate'
                stage0.node(gate_name,
                           f'Layer {layer_idx}\\nExpert Gate\\nInput: [128,10000,4096]\\nOutput: routing decisions\\nGPU: all 8 GPUs')
                
                # 8 experts (2 per GPU in reality, simplified as 8 total)
                for expert_id in range(8):
                    expert_name = f'{layer_name}_expert_{expert_id}'
                    gpu_id = expert_id % 8
                    stage0.node(expert_name,
                               f'Expert {expert_id}\\nMLP\\nInput: [128,subset_tokens,4096]\\nOutput: [128,subset_tokens,4096]\\nGPU: {gpu_id}')
                
                # Expert aggregation
                agg_name = f'{layer_name}_expert_agg'
                stage0.node(agg_name,
                           f'Layer {layer_idx}\\nExpert Aggregation\\nInput: [128,subset_tokens,4096]\\nOutput: [128,10000,4096]\\nGPU: all 8 GPUs')
                
                # Residual connection
                residual_name = f'{layer_name}_residual'
                stage0.node(residual_name,
                           f'Layer {layer_idx}\\nResidual Add\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: all 8 GPUs')
        
        # Stage 1 (GPUs 8-15, layers 8-15)
        with dot.subgraph(name='cluster_stage1') as stage1:
            stage1.attr(label='Pipeline Stage 1\\nGPUs 8-15', color='blue', style='dashed')
            
            for layer_idx in range(8, 16):
                layer_name = f'layer_{layer_idx}'
                
                # MHA for this layer
                mha_name = f'{layer_name}_mha'
                stage1.node(mha_name, 
                           f'Layer {layer_idx}\\nMulti-Head Attention\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nTP=8 across GPUs 8-15')
                
                # Expert selection gate
                gate_name = f'{layer_name}_gate'
                stage1.node(gate_name,
                           f'Layer {layer_idx}\\nExpert Gate\\nInput: [128,10000,4096]\\nOutput: routing decisions\\nGPU: all 8 GPUs')
                
                # 8 experts
                for expert_id in range(8):
                    expert_name = f'{layer_name}_expert_{expert_id}'
                    gpu_id = 8 + (expert_id % 8)
                    stage1.node(expert_name,
                               f'Expert {expert_id}\\nMLP\\nInput: [128,subset_tokens,4096]\\nOutput: [128,subset_tokens,4096]\\nGPU: {gpu_id}')
                
                # Expert aggregation
                agg_name = f'{layer_name}_expert_agg'
                stage1.node(agg_name,
                           f'Layer {layer_idx}\\nExpert Aggregation\\nInput: [128,subset_tokens,4096]\\nOutput: [128,10000,4096]\\nGPU: all 8 GPUs')
                
                # Residual connection
                residual_name = f'{layer_name}_residual'
                stage1.node(residual_name,
                           f'Layer {layer_idx}\\nResidual Add\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: all 8 GPUs')
        
        # Output node
        dot.node('output', 'Model Output\\nbatch_size=128, seq_len=10000, token_dim=4096', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Connections
        prev_node = 'input'
        
        for layer_idx in range(16):
            layer_name = f'layer_{layer_idx}'
            
            # Connect to MHA
            mha_name = f'{layer_name}_mha'
            dot.edge(prev_node, mha_name)
            
            # MHA to gate
            gate_name = f'{layer_name}_gate'
            dot.edge(mha_name, gate_name)
            
            # Gate to experts (showing routing)
            for expert_id in range(8):
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.edge(gate_name, expert_name, style='dashed', label='routed tokens')
            
            # Experts to aggregation
            agg_name = f'{layer_name}_expert_agg'
            for expert_id in range(8):
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.edge(expert_name, agg_name)
            
            # Aggregation to residual
            residual_name = f'{layer_name}_residual'
            dot.edge(mha_name, residual_name, label='identity')
            dot.edge(agg_name, residual_name)
            
            prev_node = residual_name
        
        dot.edge(prev_node, 'output')
        
        # Pipeline communication
        dot.edge('layer_7_residual', 'layer_8_mha', 
                label='Pipeline stage boundary\\nGPU 7 → GPU 8')
        
        return dot
    
    def _create_proposed_dag(self):
        """Create DAG for proposed: EP=16, 1 expert per GPU"""
        dot = graphviz.Digraph('MoE_Large_EP16_CrossNode',
                              comment='MoE with Expert Parallel=16 (1 expert per GPU)')
        
        dot.attr(rankdir='TB', size='25,30')
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Input node
        dot.node('input', 'Model Input\\nbatch_size=128, seq_len=10000, token_dim=4096', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Create nodes for each layer and GPU
        for layer_idx in range(16):
            layer_name = f'layer_{layer_idx}'
            
            # MHA for this layer (no TP, runs on all GPUs)
            mha_name = f'{layer_name}_mha'
            dot.node(mha_name,
                    f'Layer {layer_idx}\\nMulti-Head Attention\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nAll 16 GPUs')
            
            # Expert selection gate (runs on all GPUs)
            gate_name = f'{layer_name}_gate'
            dot.node(gate_name,
                    f'Layer {layer_idx}\\nExpert Gate\\nInput: [128,10000,4096]\\nOutput: routing decisions\\nAll 16 GPUs')
            
            # All-to-all communication for routing
            all2all_name = f'{layer_name}_all2all'
            dot.node(all2all_name,
                    f'Layer {layer_idx}\\nAll-to-All Communication\\nGPU: all 16 GPUs\\nShape: [128,10000,4096]\\nvia NVLink/InfiniBand',
                    shape='ellipse', fillcolor='yellow')
            
            # 16 experts, one per GPU
            for expert_id in range(16):
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.node(expert_name,
                        f'Expert {expert_id}\\nMLP\\nInput: [128,subset_tokens,4096]\\nOutput: [128,subset_tokens,4096]\\nGPU: {expert_id}')
            
            # Another all-to-all for results
            all2all_result_name = f'{layer_name}_all2all_result'
            dot.node(all2all_result_name,
                    f'Layer {layer_idx}\\nAll-to-All Result Merge\\nGPU: all 16 GPUs\\nShape: [128,10000,4096]\\nvia NVLink/InfiniBand',
                    shape='ellipse', fillcolor='yellow')
            
            # Expert aggregation (runs on all GPUs)
            agg_name = f'{layer_name}_expert_agg'
            dot.node(agg_name,
                    f'Layer {layer_idx}\\nExpert Aggregation\\nInput: [128,subset_tokens,4096]\\nOutput: [128,10000,4096]\\nAll 16 GPUs')
            
            # Residual connection
            residual_name = f'{layer_name}_residual'
            dot.node(residual_name,
                    f'Layer {layer_idx}\\nResidual Add\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nAll 16 GPUs')
        
        # Output node
        dot.node('output', 'Model Output\\nbatch_size=128, seq_len=10000, token_dim=4096', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Connections
        prev_node = 'input'
        
        for layer_idx in range(16):
            layer_name = f'layer_{layer_idx}'
            
            # Input to MHA
            mha_name = f'{layer_name}_mha'
            dot.edge(prev_node, mha_name)
            
            # MHA to gate
            gate_name = f'{layer_name}_gate'
            dot.edge(mha_name, gate_name)
            
            # Gate to all-to-all
            all2all_name = f'{layer_name}_all2all'
            dot.edge(gate_name, all2all_name)
            
            # All-to-all to experts (routing)
            for expert_id in range(16):
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.edge(all2all_name, expert_name, style='dashed', 
                        label=f'tokens routed to expert {expert_id}')
            
            # Experts to results all-to-all
            all2all_result_name = f'{layer_name}_all2all_result'
            for expert_id in range(16):
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.edge(expert_name, all2all_result_name)
            
            # Results to aggregation
            agg_name = f'{layer_name}_expert_agg'
            dot.edge(all2all_result_name, agg_name)
            
            # Aggregation to residual
            residual_name = f'{layer_name}_residual'
            dot.edge(mha_name, residual_name, label='identity')
            dot.edge(agg_name, residual_name)
            
            prev_node = residual_name
        
        dot.edge(prev_node, 'output')
        
        return dot
    
    def generate_dags(self):
        """Generate both DAGs and save them"""
        
        # Generate baseline DAG
        baseline_dag = self._create_baseline_dag()
        baseline_dag.render(os.path.join(self.output_dir, 'moe_baseline_tp8_pp2'), 
                           format='svg', cleanup=True)
        
        # Save DOT file for baseline
        with open(os.path.join(self.output_dir, 'moe_baseline_tp8_pp2.dot'), 'w') as f:
            f.write(baseline_dag.source)
        
        # Generate proposed DAG
        proposed_dag = self._create_proposed_dag()
        proposed_dag.render(os.path.join(self.output_dir, 'moe_proposed_ep16'), 
                           format='svg', cleanup=True)
        
        # Save DOT file for proposed
        with open(os.path.join(self.output_dir, 'moe_proposed_ep16.dot'), 'w') as f:
            f.write(proposed_dag.source)
        
        return [
            os.path.join(self.output_dir, 'moe_baseline_tp8_pp2.dot'),
            os.path.join(self.output_dir, 'moe_proposed_ep16.dot')
        ]

if __name__ == '__main__':
    generator = MoEDAGGenerator('../outputs/2025-11-24-12-00-17')
    paths = generator.generate_dags()
    print(json.dumps(paths))