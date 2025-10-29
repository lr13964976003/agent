#!/usr/bin/env python3
"""
Helix Parallelism DAG Generator
Generates comprehensive DAGs for Helix models and baselines
"""

import graphviz
from typing import Dict, List, Tuple
import os

class HelixDAGGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_deepseek_helix_dag(self) -> str:
        """
        Generate DAG for DeepSeek-R1 with Helix parallelism (16 GPUs)
        """
        dot = graphviz.Digraph('DeepSeek_R1_Helix', 
                              comment='DeepSeek-R1 with Helix Parallelism (16 GPUs)')
        
        # Graph attributes
        dot.attr(rankdir='TB', size='20,30', compound='true')
        dot.attr('node', fontsize='10', height='0.8', width='2.5')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=B, seq_len=1, hidden_dim=16384]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # ===== ATTENTION PHASE =====
        with dot.subgraph(name='cluster_attention') as c:
            c.attr(label='Attention Phase (KV Parallelism)', style='dashed')
            
            # QKV projections - each GPU computes full projections
            for gpu_id in range(16):
                gpu_name = f'kvp_{gpu_id}'
                c.node(f'qkv_{gpu_id}', 
                      f'QKV Projection\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 128, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # Attention computation - each GPU processes S/16 tokens
            for gpu_id in range(16):
                c.node(f'attn_{gpu_id}', 
                      f'Attention\\nGPU {gpu_id}\\nKV slice: [tokens {gpu_id*16}:{(gpu_id+1)*16}S/16]\\nInput: [B, 1, 128, 128]\\nOutput: [B, 1, 128, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # HOP-B All-to-All communication
            c.node('hopb', 'HOP-B All-to-All\\nCommunication\\nVolume: [B, 16384]\\nOverlap with next token', 
                  shape='parallelogram', style='dashed', fillcolor='lightcoral')
            
            # Connections for attention phase
            for gpu_id in range(16):
                dot.edge('input', f'qkv_{gpu_id}')
                dot.edge(f'qkv_{gpu_id}', f'attn_{gpu_id}')
                dot.edge(f'attn_{gpu_id}', 'hopb')
        
        # ===== FFN PHASE =====
        with dot.subgraph(name='cluster_ffn') as c:
            c.attr(label='FFN Phase (TP×EP)', style='dashed')
            
            # Gate computation
            for gpu_id in range(16):
                c.node(f'gate_{gpu_id}', 
                      f'Gate\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 256]', 
                      shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Expert routing
            for expert_id in range(8):
                for tp_id in range(2):
                    gpu_id = expert_id * 2 + tp_id
                    c.node(f'expert_{expert_id}_tp_{tp_id}', 
                          f'Expert {expert_id}\\nTP {tp_id}\\nGPU {gpu_id}\\nExperts: [{expert_id*2}:{expert_id*2+2}]\\nInput: [B/{8}, 1, 2048]\\nOutput: [B/{8}, 1, 2048]', 
                          shape='rectangle', style='filled', fillcolor='lightcyan')
            
            # FFN computation for each expert
            for expert_id in range(8):
                for tp_id in range(2):
                    gpu_id = expert_id * 2 + tp_id
                    c.node(f'ffn_{expert_id}_tp_{tp_id}', 
                          f'FFN\\nGPU {gpu_id}\\nUP: [B/{8}, 2048, 16384/2]\\nGate: [B/{8}, 2048, 16384/2]\\nDown: [B/{8}, 16384/2, 2048]\\nOutput: [B/{8}, 1, 2048]', 
                          shape='rectangle', style='filled', fillcolor='lightsteelblue')
            
            # All-reduce for TP within each expert
            for expert_id in range(8):
                c.node(f'allreduce_{expert_id}', 
                      f'TP All-Reduce\\nExpert {expert_id}\\nTPF=2\\nInput: [B/{8}, 1, 2048]\\nOutput: [B/{8}, 1, 2048]', 
                      shape='parallelogram', style='filled', fillcolor='gold')
            
            # All-gather for EP
            c.node('ep_allgather', 'EP All-Gather\\n8 experts\\nInput: [B/{8}, 1, 2048]\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
            
            # Connections for FFN phase
            dot.edge('hopb', 'gate_0', lhead='cluster_ffn')
            for gpu_id in range(16):
                expert_id = gpu_id // 2
                tp_id = gpu_id % 2
                dot.edge(f'gate_{gpu_id}', f'expert_{expert_id}_tp_{tp_id}')
                dot.edge(f'expert_{expert_id}_tp_{tp_id}', f'ffn_{expert_id}_tp_{tp_id}')
                dot.edge(f'ffn_{expert_id}_tp_{tp_id}', f'allreduce_{expert_id}')
                dot.edge(f'allreduce_{expert_id}', 'ep_allgather')
        
        # Output node
        dot.node('output', 'Output\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16384]', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
        
        dot.edge('ep_allgather', 'output')
        
        return dot.source
    
    def generate_llama_helix_dag(self) -> str:
        """
        Generate DAG for Llama-405B with Helix parallelism (64 GPUs)
        """
        dot = graphviz.Digraph('Llama_405B_Helix', 
                              comment='Llama-405B with Helix Parallelism (64 GPUs)')
        
        # Graph attributes
        dot.attr(rankdir='TB', size='25,35', compound='true')
        dot.attr('node', fontsize='9', height='0.7', width='2.0')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=B, seq_len=1, hidden_dim=16384]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # ===== ATTENTION PHASE =====
        with dot.subgraph(name='cluster_attention') as c:
            c.attr(label='Attention Phase (KV×TP)', style='dashed')
            
            # Group by KVP (8 groups)
            for kvp_id in range(8):
                with c.subgraph(name=f'cluster_kvp_{kvp_id}') as kvp:
                    kvp.attr(label=f'KVP {kvp_id} (tokens [{kvp_id}S/8:{(kvp_id+1)}S/8])')
                    
                    # Within each KVP, TP across 8 GPUs
                    for tp_id in range(8):
                        gpu_id = kvp_id * 8 + tp_id
                        
                        kvp.node(f'qkv_{gpu_id}', 
                               f'QKV Projection\\nGPU {gpu_id}\\nQ heads: [{tp_id*16}:{(tp_id+1)*16}]\\nKV head: [{tp_id}:{tp_id+1}]\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16, 128]', 
                               shape='rectangle', style='filled', fillcolor='lightyellow')
                        
                        kvp.node(f'attn_{gpu_id}', 
                               f'Attention\\nGPU {gpu_id}\\nInput: [B, 1, 16, 128]\\nOutput: [B, 1, 16, 128]', 
                               shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # HOP-B All-to-All
            c.node('hopb', 'HOP-B All-to-All\\nTPA=8, KVP=8\\nVolume: [B, 2048]\\nOverlap with next token', 
                  shape='parallelogram', style='dashed', fillcolor='lightcoral')
        
        # ===== FFN PHASE =====
        with dot.subgraph(name='cluster_ffn') as c:
            c.attr(label='FFN Phase (TP=64)', style='dashed')
            
            # FFN across 64 GPUs in pure TP
            for gpu_id in range(64):
                c.node(f'ffn_up_{gpu_id}', 
                      f'FFN UP\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nWeights: [16384, 1024]\\nOutput: [B, 1, 1024]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                c.node(f'ffn_gate_{gpu_id}', 
                      f'FFN Gate\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nWeights: [16384, 1024]\\nOutput: [B, 1, 1024]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                c.node(f'ffn_down_{gpu_id}', 
                      f'FFN Down\\nGPU {gpu_id}\\nInput: [B, 1, 1024]\\nWeights: [1024, 16384]\\nOutput: [B, 1, 256]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
            
            # TP All-Reduce
            c.node('tp_allreduce', 'TP All-Reduce\\nTP=64\\nInput: [B, 1, 256]×64\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
        
        # Output node
        dot.node('output', 'Output\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16384]', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Connections
        for kvp_id in range(8):
            for tp_id in range(8):
                gpu_id = kvp_id * 8 + tp_id
                dot.edge('input', f'qkv_{gpu_id}')
                dot.edge(f'qkv_{gpu_id}', f'attn_{gpu_id}')
                dot.edge(f'attn_{gpu_id}', 'hopb')
        
        for gpu_id in range(64):
            dot.edge('hopb', f'ffn_up_{gpu_id}', lhead='cluster_ffn')
            dot.edge('hopb', f'ffn_gate_{gpu_id}')
            dot.edge(f'ffn_up_{gpu_id}', f'ffn_down_{gpu_id}')
            dot.edge(f'ffn_gate_{gpu_id}', f'ffn_down_{gpu_id}')
            dot.edge(f'ffn_down_{gpu_id}', 'tp_allreduce')
        
        dot.edge('tp_allreduce', 'output')
        
        return dot.source
    
    def generate_deepseek_baseline_dag(self) -> str:
        """
        Generate baseline DAG for DeepSeek-R1 with pure Tensor Parallelism (8 GPUs)
        """
        dot = graphviz.Digraph('DeepSeek_R1_TP_Baseline', 
                              comment='DeepSeek-R1 Baseline Tensor Parallelism (8 GPUs)')
        
        # Graph attributes
        dot.attr(rankdir='TB', size='15,20', compound='true')
        dot.attr('node', fontsize='10', height='0.8', width='2.5')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=B, seq_len=1, hidden_dim=16384]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # ===== ATTENTION PHASE (TP) =====
        with dot.subgraph(name='cluster_attention') as c:
            c.attr(label='Attention Phase (TP=8)', style='dashed')
            
            # QKV projections - TP across 8 GPUs
            for gpu_id in range(8):
                c.node(f'qkv_{gpu_id}', 
                      f'QKV Projection\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # KV cache (duplicated across GPUs)
            for gpu_id in range(8):
                c.node(f'kv_cache_{gpu_id}', 
                      f'KV Cache\\nGPU {gpu_id}\\nFull cache: [B, S, 8, 128]\\n(duplicated)', 
                      shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Attention computation
            for gpu_id in range(8):
                c.node(f'attn_{gpu_id}', 
                      f'Attention\\nGPU {gpu_id}\\nInput: [B, 1, 16, 128]\\nOutput: [B, 1, 16, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # TP All-reduce
            c.node('attn_allreduce', 'TP All-Reduce\\nTP=8\\nInput: [B, 1, 16, 128]×8\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
        
        # ===== FFN PHASE (TP×EP) =====
        with dot.subgraph(name='cluster_ffn') as c:
            c.attr(label='FFN Phase (TP=8×EP=1)', style='dashed')
            
            # Gate
            for gpu_id in range(8):
                c.node(f'gate_{gpu_id}', 
                      f'Gate\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 256]', 
                      shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Expert selection (all experts on all GPUs due to TP)
            for gpu_id in range(8):
                c.node(f'expert_{gpu_id}', 
                      f'All Experts\\nGPU {gpu_id}\\nExperts: [0:255]\\nInput: [B, 8, 2048]\\nOutput: [B, 8, 2048]', 
                      shape='rectangle', style='filled', fillcolor='lightcyan')
            
            # FFN computation
            for gpu_id in range(8):
                c.node(f'ffn_{gpu_id}', 
                      f'FFN\\nGPU {gpu_id}\\nUP: [B, 8, 2048, 16384/8]\\nGate: [B, 8, 2048, 16384/8]\\nDown: [B, 8, 16384/8, 2048]\\nOutput: [B, 8, 2048]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
            
            # TP All-reduce
            c.node('ffn_allreduce', 'TP All-Reduce\\nTP=8\\nInput: [B, 8, 2048]×8\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
        
        # Output node
        dot.node('output', 'Output\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16384]', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Connections
        for gpu_id in range(8):
            dot.edge('input', f'qkv_{gpu_id}')
            dot.edge(f'qkv_{gpu_id}', f'kv_cache_{gpu_id}')
            dot.edge(f'kv_cache_{gpu_id}', f'attn_{gpu_id}')
            dot.edge(f'attn_{gpu_id}', 'attn_allreduce')
            dot.edge('attn_allreduce', f'gate_{gpu_id}', lhead='cluster_ffn')
            dot.edge(f'gate_{gpu_id}', f'expert_{gpu_id}')
            dot.edge(f'expert_{gpu_id}', f'ffn_{gpu_id}')
            dot.edge(f'ffn_{gpu_id}', 'ffn_allreduce')
        
        dot.edge('ffn_allreduce', 'output')
        
        return dot.source
    
    def generate_llama_baseline_dag(self) -> str:
        """
        Generate baseline DAG for Llama-405B with pure Tensor Parallelism (8 GPUs)
        """
        dot = graphviz.Digraph('Llama_405B_TP_Baseline', 
                              comment='Llama-405B Baseline Tensor Parallelism (8 GPUs)')
        
        # Graph attributes
        dot.attr(rankdir='TB', size='15,20', compound='true')
        dot.attr('node', fontsize='10', height='0.8', width='2.5')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=B, seq_len=1, hidden_dim=16384]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # ===== ATTENTION PHASE (TP) =====
        with dot.subgraph(name='cluster_attention') as c:
            c.attr(label='Attention Phase (TP=8)', style='dashed')
            
            # QKV projections - TP across 8 GPUs
            for gpu_id in range(8):
                c.node(f'qkv_{gpu_id}', 
                      f'QKV Projection\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # KV cache (duplicated across GPUs)
            for gpu_id in range(8):
                c.node(f'kv_cache_{gpu_id}', 
                      f'KV Cache\\nGPU {gpu_id}\\nFull cache: [B, S, 8, 128]\\n(duplicated)', 
                      shape='rectangle', style='filled', fillcolor='lightgray')
            
            # Attention computation
            for gpu_id in range(8):
                c.node(f'attn_{gpu_id}', 
                      f'Attention\\nGPU {gpu_id}\\nInput: [B, 1, 16, 128]\\nOutput: [B, 1, 16, 128]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # TP All-reduce
            c.node('attn_allreduce', 'TP All-Reduce\\nTP=8\\nInput: [B, 1, 16, 128]×8\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
        
        # ===== FFN PHASE (TP) =====
        with dot.subgraph(name='cluster_ffn') as c:
            c.attr(label='FFN Phase (TP=8)', style='dashed')
            
            # FFN layers
            for gpu_id in range(8):
                c.node(f'ffn_up_{gpu_id}', 
                      f'FFN UP\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nWeights: [16384, 8192]\\nOutput: [B, 1, 8192]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                c.node(f'ffn_gate_{gpu_id}', 
                      f'FFN Gate\\nGPU {gpu_id}\\nInput: [B, 1, 16384]\\nWeights: [16384, 8192]\\nOutput: [B, 1, 8192]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                c.node(f'ffn_down_{gpu_id}', 
                      f'FFN Down\\nGPU {gpu_id}\\nInput: [B, 1, 8192]\\nWeights: [8192, 2048]\\nOutput: [B, 1, 2048]', 
                      shape='rectangle', style='filled', fillcolor='lightsteelblue')
            
            # TP All-reduce
            c.node('tp_allreduce', 'TP All-Reduce\\nTP=8\\nInput: [B, 1, 2048]×8\\nOutput: [B, 1, 16384]', 
                  shape='parallelogram', style='filled', fillcolor='gold')
        
        # Output node
        dot.node('output', 'Output\\nInput: [B, 1, 16384]\\nOutput: [B, 1, 16384]', 
                shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Connections
        for gpu_id in range(8):
            dot.edge('input', f'qkv_{gpu_id}')
            dot.edge(f'qkv_{gpu_id}', f'kv_cache_{gpu_id}')
            dot.edge(f'kv_cache_{gpu_id}', f'attn_{gpu_id}')
            dot.edge(f'attn_{gpu_id}', 'attn_allreduce')
            dot.edge('attn_allreduce', f'ffn_up_{gpu_id}', lhead='cluster_ffn')
            dot.edge('attn_allreduce', f'ffn_gate_{gpu_id}')
            dot.edge(f'ffn_up_{gpu_id}', f'ffn_down_{gpu_id}')
            dot.edge(f'ffn_gate_{gpu_id}', f'ffn_down_{gpu_id}')
            dot.edge(f'ffn_down_{gpu_id}', 'tp_allreduce')
        
        dot.edge('tp_allreduce', 'output')
        
        return dot.source
    
    def generate_all_dags(self):
        """Generate all DAGs and save to files"""
        
        # Generate Helix DAGs
        deepseek_helix = self.generate_deepseek_helix_dag()
        llama_helix = self.generate_llama_helix_dag()
        
        # Generate baseline DAGs
        deepseek_baseline = self.generate_deepseek_baseline_dag()
        llama_baseline = self.generate_llama_baseline_dag()
        
        # Save DOT files
        with open(os.path.join(self.output_dir, 'deepseek_r1_helix.dot'), 'w') as f:
            f.write(deepseek_helix)
        
        with open(os.path.join(self.output_dir, 'llama_405b_helix.dot'), 'w') as f:
            f.write(llama_helix)
        
        with open(os.path.join(self.output_dir, 'deepseek_r1_baseline.dot'), 'w') as f:
            f.write(deepseek_baseline)
        
        with open(os.path.join(self.output_dir, 'llama_405b_baseline.dot'), 'w') as f:
            f.write(llama_baseline)
        
        # Generate SVGs
        for dot_file in ['deepseek_r1_helix.dot', 'llama_405b_helix.dot', 
                        'deepseek_r1_baseline.dot', 'llama_405b_baseline.dot']:
            svg_file = dot_file.replace('.dot', '.svg')
            os.system(f'dot -Tsvg {os.path.join(self.output_dir, dot_file)} -o {os.path.join(self.output_dir, svg_file)}')
        
        return {
            'deepseek_r1_helix_dot': os.path.join(self.output_dir, 'deepseek_r1_helix.dot'),
            'llama_405b_helix_dot': os.path.join(self.output_dir, 'llama_405b_helix.dot'),
            'deepseek_r1_baseline_dot': os.path.join(self.output_dir, 'deepseek_r1_baseline.dot'),
            'llama_405b_baseline_dot': os.path.join(self.output_dir, 'llama_405b_baseline.dot'),
            'deepseek_r1_helix_svg': os.path.join(self.output_dir, 'deepseek_r1_helix.svg'),
            'llama_405b_helix_svg': os.path.join(self.output_dir, 'llama_405b_helix.svg'),
            'deepseek_r1_baseline_svg': os.path.join(self.output_dir, 'deepseek_r1_baseline.svg'),
            'llama_405b_baseline_svg': os.path.join(self.output_dir, 'llama_405b_baseline.svg')
        }

if __name__ == '__main__':
    generator = HelixDAGGenerator('../outputs/2025-10-29-11-58-32')
    paths = generator.generate_all_dags()
    
    print("Generated DAGs:")
    for name, path in paths.items():
        print(f"{name}: {path}")