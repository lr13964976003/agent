#!/usr/bin/env python3
"""
Detailed FA Pool DAG Generator
Creates complete engineering-level DAGs with proper dimensions and GPU mappings
"""

import os
from graphviz import Digraph

class DetailedFAPoolDAGGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model specifications from paper
        self.hidden_dim = 4096
        self.ffn_hidden_dim = 16384
        self.num_heads = 32
        self.d_k = self.hidden_dim // self.num_heads  # 128
        self.num_layers = 4
        self.batch_size = 1024
        
    def create_complete_8gpu_dag(self):
        """Complete DAG for 8 GPU base configuration (seq_len < 4096)"""
        dot = Digraph('FA_Pool_8GPU_Config')
        dot.attr(rankdir='TB', splines='spline', overlap='false')
        
        # Node styles
        dot.attr('node', shape='rectangle', style='filled')
        
        # Input
        dot.node('input', 'Model Input\n[batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7',
                shape='ellipse', fillcolor='lightgreen', color='green', penwidth='2')
        
        # Token embeddings (tensor parallel across 8 GPUs)
        for i in range(8):
            dot.node(f'embed_{i}', f'Token Embedding\nShard {i}\n[batch_size=1024, seq_len=?, d_model=512]\nGPU: {i}',
                    fillcolor='lightblue', color='blue')
        
        # Positional encoding
        for i in range(8):
            dot.node(f'pos_{i}', f'Positional Encoding\nShard {i}\n[batch_size=1024, seq_len=?, d_model=512]\nGPU: {i}',
                    fillcolor='lightblue', color='blue')
        
        # RMSNorm for each layer
        for layer in range(4):
            for i in range(8):
                dot.node(f'prenorm_{layer}_{i}', f'Pre-Norm Layer {layer}\nShard {i}\n[batch_size=1024, seq_len=?, d_model=512]\nGPU: {i}',
                        fillcolor='lightblue', color='blue')
        
        # Attention blocks (8-way tensor parallel)
        for layer in range(4):
            for head_group in range(4):  # 32 heads / 8 GPUs = 4 heads per GPU
                for gpu in range(8):
                    head_start = gpu * 4 + head_group
                    dot.node(f'attn_q_{layer}_{gpu}_{head_group}', 
                            f'Q Proj Layer {layer}\nHead {head_start}-{head_start+3}\n[batch_size=1024, seq_len=?, d_k=128]\nGPU: {gpu}',
                            fillcolor='lightcoral', color='red')
                    dot.node(f'attn_k_{layer}_{gpu}_{head_group}', 
                            f'K Proj Layer {layer}\nHead {head_start}-{head_start+3}\n[batch_size=1024, seq_len=?, d_k=128]\nGPU: {gpu}',
                            fillcolor='lightcoral', color='red')
                    dot.node(f'attn_v_{layer}_{gpu}_{head_group}', 
                            f'V Proj Layer {layer}\nHead {head_start}-{head_start+3}\n[batch_size=1024, seq_len=?, d_k=128]\nGPU: {gpu}',
                            fillcolor='lightcoral', color='red')
            
            # Flash attention computation
            for gpu in range(8):
                dot.node(f'flash_attn_{layer}_{gpu}', 
                        f'Flash Attention Layer {layer}\nGPU {gpu}\n[batch_size=1024, seq_len=?, heads=4, d_k=128]\nGPU: {gpu}',
                        fillcolor='lightcoral', color='red')
            
            # Attention output projection
            for gpu in range(8):
                dot.node(f'attn_out_{layer}_{gpu}', 
                        f'Attention Output Layer {layer}\nShard {gpu}\n[batch_size=1024, seq_len=?, d_model=512]\nGPU: {gpu}',
                        fillcolor='lightcoral', color='red')
            
            # All-reduce for attention
            dot.node(f'attn_allreduce_{layer}', 
                    f'Attention All-Reduce\nLayer {layer}\n[batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7',
                    shape='parallelogram', fillcolor='yellow', color='orange', penwidth='2')
            
            # Attention residual
            dot.node(f'attn_residual_{layer}', 
                    f'Attention Residual\nLayer {layer}\n[batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7',
                    shape='ellipse', fillcolor='orange', color='orange')
        
        # FFN layers
        for layer in range(4):
            # Gate layer (column parallel)
            for gpu in range(8):
                dot.node(f'ffn_gate_{layer}_{gpu}', 
                        f'FFN Gate Layer {layer}\nShard {gpu}\n[batch_size=1024, seq_len=?, ffn_dim=2048]\nGPU: {gpu}',
                        fillcolor='lightblue', color='blue')
            
            # Up layer (column parallel)
            for gpu in range(8):
                dot.node(f'ffn_up_{layer}_{gpu}', 
                        f'FFN Up Layer {layer}\nShard {gpu}\n[batch_size=1024, seq_len=?, ffn_dim=2048]\nGPU: {gpu}',
                        fillcolor='lightblue', color='blue')
            
            # GELU activation
            for gpu in range(8):
                dot.node(f'gelu_{layer}_{gpu}', 
                        f'GELU Layer {layer}\nShard {gpu}\n[batch_size=1024, seq_len=?, ffn_dim=2048]\nGPU: {gpu}',
                        fillcolor='lightblue', color='blue')
            
            # Down layer (row parallel)
            for gpu in range(8):
                dot.node(f'ffn_down_{layer}_{gpu}', 
                        f'FFN Down Layer {layer}\nShard {gpu}\n[batch_size=1024, seq_len=?, d_model=512]\nGPU: {gpu}',
                        fillcolor='lightblue', color='blue')
            
            # FFN all-reduce
            dot.node(f'ffn_allreduce_{layer}', 
                    f'FFN All-Reduce\nLayer {layer}\n[batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7',
                    shape='parallelogram', fillcolor='yellow', color='orange', penwidth='2')
            
            # FFN residual
            dot.node(f'ffn_residual_{layer}', 
                    f'FFN Residual\nLayer {layer}\n[batch_size=1024, seq_len=?, d_model=4096]\nGPU: 0-7',
                    shape='ellipse', fillcolor='orange', color='orange')
        
        # Output
        dot.node('output', 'Model Output\n[batch_size=1024, seq_len=?, vocab_size=full_vocab]\nGPU: 0-7',
                shape='ellipse', fillcolor='lightgreen', color='green', penwidth='2')
        
        # Connect the graph
        dot.edge('input', 'embed_0')
        for i in range(1, 8):
            dot.edge('input', f'embed_{i}')
        
        for i in range(8):
            dot.edge(f'embed_{i}', f'pos_{i}')
        
        # Layer connections
        for layer in range(4):
            if layer == 0:
                for i in range(8):
                    dot.edge(f'pos_{i}', f'prenorm_{layer}_{i}')
            else:
                for i in range(8):
                    dot.edge(f'ffn_residual_{layer-1}', f'prenorm_{layer}_{i}')
            
            # Attention flow
            for gpu in range(8):
                dot.edge(f'prenorm_{layer}_{gpu}', f'attn_q_{layer}_{gpu}_0')
                dot.edge(f'prenorm_{layer}_{gpu}', f'attn_k_{layer}_{gpu}_0')
                dot.edge(f'prenorm_{layer}_{gpu}', f'attn_v_{layer}_{gpu}_0')
                
                # Connect attention components
                for head_group in range(4):
                    dot.edge(f'attn_q_{layer}_{gpu}_{head_group}', f'flash_attn_{layer}_{gpu}')
                    dot.edge(f'attn_k_{layer}_{gpu}_{head_group}', f'flash_attn_{layer}_{gpu}')
                    dot.edge(f'attn_v_{layer}_{gpu}_{head_group}', f'flash_attn_{layer}_{gpu}')
                
                dot.edge(f'flash_attn_{layer}_{gpu}', f'attn_out_{layer}_{gpu}')
                dot.edge(f'attn_out_{layer}_{gpu}', f'attn_allreduce_{layer}')
            
            # Attention residual
            if layer == 0:
                dot.edge('input', f'attn_residual_{layer}')
            else:
                dot.edge(f'ffn_residual_{layer-1}', f'attn_residual_{layer}')
            dot.edge(f'attn_allreduce_{layer}', f'attn_residual_{layer}')
            
            # FFN flow
            for gpu in range(8):
                dot.edge(f'attn_residual_{layer}', f'ffn_gate_{layer}_{gpu}')
                dot.edge(f'attn_residual_{layer}', f'ffn_up_{layer}_{gpu}')
                dot.edge(f'ffn_gate_{layer}_{gpu}', f'gelu_{layer}_{gpu}')
                dot.edge(f'ffn_up_{layer}_{gpu}', f'gelu_{layer}_{gpu}')
                dot.edge(f'gelu_{layer}_{gpu}', f'ffn_down_{layer}_{gpu}')
                dot.edge(f'ffn_down_{layer}_{gpu}', f'ffn_allreduce_{layer}')
            
            dot.edge(f'attn_residual_{layer}', f'ffn_residual_{layer}')
            dot.edge(f'ffn_allreduce_{layer}', f'ffn_residual_{layer}')
        
        dot.edge(f'ffn_residual_{3}', 'output')
        
        return dot
    
    def create_16gpu_dag(self):
        """Complete DAG for 16 GPU configuration (8 base + 8 pool)"""
        dot = Digraph('FA_Pool_16GPU_Config')
        dot.attr(rankdir='TB', splines='spline')
        
        # Node styles
        dot.attr('node', shape='rectangle', style='filled')
        
        seq_len = 4096
        block_size = seq_len // 8
        
        # Input
        dot.node('input', f'Model Input\nSeqLen: {seq_len}\n[batch_size=1024, seq_len={seq_len}, d_model=4096]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Resource manager
        dot.node('manager', f'Resource Manager\nSeqLen: {seq_len}\nPool GPUs: 8',
                shape='diamond', fillcolor='gold')
        
        # Base layer embeddings
        dot.node('embeddings', 'Token Embeddings\n[batch_size=1024, seq_len=4096, d_model=4096]\nGPU: 0-7',
                fillcolor='lightblue')
        
        # Layer processing for 4 layers
        for layer in range(4):
            # Pre-norm
            dot.node(f'prenorm_{layer}', f'Pre-Norm Layer {layer}\nGPU: 0-7',
                    fillcolor='lightblue')
            
            # Query split for attention pool
            dot.node(f'query_split_{layer}', f'Query Split Layer {layer}\nBlock size: {block_size}\nGPU: 8-15',
                    shape='parallelogram', fillcolor='yellow')
            
            # KV broadcast
            dot.node(f'kv_broadcast_{layer}', f'KV Broadcast Layer {layer}\nGPU: 8-15',
                    shape='parallelogram', fillcolor='yellow')
            
            # Attention blocks in pool
            for gpu in range(8):
                gpu_id = 8 + gpu
                start_pos = gpu * block_size
                end_pos = start_pos + block_size
                
                dot.node(f'attn_q_{layer}_{gpu}', 
                        f'Q Proj Layer {layer}\nGPU {gpu_id}\nRange: {start_pos}-{end_pos}',
                        fillcolor='lightcoral')
                dot.node(f'attn_k_{layer}_{gpu}', 
                        f'K Proj Layer {layer}\nGPU {gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'attn_v_{layer}_{gpu}', 
                        f'V Proj Layer {layer}\nGPU {gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'flash_attn_{layer}_{gpu}', 
                        f'Flash Attention\nLayer {layer} GPU {gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'attn_out_{layer}_{gpu}', 
                        f'Output Proj\nLayer {layer} GPU {gpu_id}',
                        fillcolor='lightcoral')
            
            # Gather results
            dot.node(f'gather_{layer}', f'Gather Attention\nLayer {layer}\nGPU: 8-15',
                    shape='parallelogram', fillcolor='yellow')
            
            # Residual connections
            dot.node(f'attn_residual_{layer}', f'Attention Residual\nLayer {layer}',
                    shape='ellipse', fillcolor='orange')
            
            # FFN (back on base GPUs)
            dot.node(f'ffn_{layer}', f'FFN Layer {layer}\nGPU: 0-7',
                    fillcolor='lightblue')
            dot.node(f'ffn_residual_{layer}', f'FFN Residual\nLayer {layer}',
                    shape='ellipse', fillcolor='orange')
        
        # Output
        dot.node('output', 'Model Output\n[batch_size=1024, seq_len=4096, vocab_size=full_vocab]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Connect flow
        dot.edge('input', 'manager')
        dot.edge('manager', 'embeddings')
        
        for layer in range(4):
            if layer == 0:
                dot.edge('embeddings', f'prenorm_{layer}')
            else:
                dot.edge(f'ffn_residual_{layer-1}', f'prenorm_{layer}')
            
            dot.edge(f'prenorm_{layer}', f'query_split_{layer}')
            dot.edge(f'prenorm_{layer}', f'kv_broadcast_{layer}')
            
            for gpu in range(8):
                dot.edge(f'query_split_{layer}', f'attn_q_{layer}_{gpu}')
                dot.edge(f'kv_broadcast_{layer}', f'attn_k_{layer}_{gpu}')
                dot.edge(f'kv_broadcast_{layer}', f'attn_v_{layer}_{gpu}')
                dot.edge(f'attn_q_{layer}_{gpu}', f'flash_attn_{layer}_{gpu}')
                dot.edge(f'attn_k_{layer}_{gpu}', f'flash_attn_{layer}_{gpu}')
                dot.edge(f'attn_v_{layer}_{gpu}', f'flash_attn_{layer}_{gpu}')
                dot.edge(f'flash_attn_{layer}_{gpu}', f'attn_out_{layer}_{gpu}')
                dot.edge(f'attn_out_{layer}_{gpu}', f'gather_{layer}')
            
            if layer == 0:
                dot.edge('embeddings', f'attn_residual_{layer}')
            else:
                dot.edge(f'ffn_residual_{layer-1}', f'attn_residual_{layer}')
            dot.edge(f'gather_{layer}', f'attn_residual_{layer}')
            
            dot.edge(f'attn_residual_{layer}', f'ffn_{layer}')
            dot.edge(f'attn_residual_{layer}', f'ffn_residual_{layer}')
            dot.edge(f'ffn_{layer}', f'ffn_residual_{layer}')
        
        dot.edge(f'ffn_residual_{3}', 'output')
        
        return dot
    
    def create_24gpu_dag(self):
        """Complete DAG for 24 GPU configuration (8 base + 16 pool)"""
        dot = Digraph('FA_Pool_24GPU_Config')
        dot.attr(rankdir='TB', splines='spline')
        
        seq_len = 8192
        pool_gpus = 16
        block_size = seq_len // pool_gpus
        
        # Input
        dot.node('input', f'Model Input\nSeqLen: {seq_len}\n[batch_size=1024, seq_len={seq_len}, d_model=4096]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Resource manager
        dot.node('manager', f'Resource Manager\nSeqLen: {seq_len}\nPool GPUs: {pool_gpus}',
                shape='diamond', fillcolor='gold')
        
        # Processing layers
        for layer in range(4):
            dot.node(f'layer{layer}_prenorm', f'Pre-Norm L{layer}\nGPU: 0-7', fillcolor='lightblue')
            dot.node(f'layer{layer}_qsplit', f'Query Split L{layer}\nBlock: {block_size}\nGPU: 8-23',
                    shape='parallelogram', fillcolor='yellow')
            dot.node(f'layer{layer}_kvbroadcast', f'KV Broadcast L{layer}\nGPU: 8-23',
                    shape='parallelogram', fillcolor='yellow')
            
            for gpu in range(pool_gpus):
                gpu_id = 8 + gpu
                dot.node(f'layer{layer}_attn_q_{gpu}', f'Q L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'layer{layer}_attn_k_{gpu}', f'K L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'layer{layer}_attn_v_{gpu}', f'V L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'layer{layer}_flash_{gpu}', f'Flash L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'layer{layer}_out_{gpu}', f'Output L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
            
            dot.node(f'layer{layer}_gather', f'Gather L{layer}\nGPU: 8-23',
                    shape='parallelogram', fillcolor='yellow')
            dot.node(f'layer{layer}_attn_res', f'Attn Res L{layer}',
                    shape='ellipse', fillcolor='orange')
            dot.node(f'layer{layer}_ffn', f'FFN L{layer}\nGPU: 0-7', fillcolor='lightblue')
            dot.node(f'layer{layer}_ffn_res', f'FFN Res L{layer}',
                    shape='ellipse', fillcolor='orange')
        
        dot.node('output', f'Model Output\n[batch_size=1024, seq_len={seq_len}, vocab_size=full_vocab]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Connect flow
        dot.edge('input', 'manager')
        dot.edge('manager', 'layer0_prenorm')
        
        for layer in range(4):
            if layer > 0:
                dot.edge(f'layer{layer-1}_ffn_res', f'layer{layer}_prenorm')
            
            dot.edge(f'layer{layer}_prenorm', f'layer{layer}_qsplit')
            dot.edge(f'layer{layer}_prenorm', f'layer{layer}_kvbroadcast')
            
            for gpu in range(pool_gpus):
                dot.edge(f'layer{layer}_qsplit', f'layer{layer}_attn_q_{gpu}')
                dot.edge(f'layer{layer}_kvbroadcast', f'layer{layer}_attn_k_{gpu}')
                dot.edge(f'layer{layer}_kvbroadcast', f'layer{layer}_attn_v_{gpu}')
                dot.edge(f'layer{layer}_attn_q_{gpu}', f'layer{layer}_flash_{gpu}')
                dot.edge(f'layer{layer}_attn_k_{gpu}', f'layer{layer}_flash_{gpu}')
                dot.edge(f'layer{layer}_attn_v_{gpu}', f'layer{layer}_flash_{gpu}')
                dot.edge(f'layer{layer}_flash_{gpu}', f'layer{layer}_out_{gpu}')
                dot.edge(f'layer{layer}_out_{gpu}', f'layer{layer}_gather')
            
            if layer == 0:
                dot.edge('input', f'layer{layer}_attn_res')
            else:
                dot.edge(f'layer{layer-1}_ffn_res', f'layer{layer}_attn_res')
            dot.edge(f'layer{layer}_gather', f'layer{layer}_attn_res')
            
            dot.edge(f'layer{layer}_attn_res', f'layer{layer}_ffn')
            dot.edge(f'layer{layer}_attn_res', f'layer{layer}_ffn_res')
            dot.edge(f'layer{layer}_ffn', f'layer{layer}_ffn_res')
        
        dot.edge('layer3_ffn_res', 'output')
        
        return dot
    
    def create_40gpu_dag(self):
        """Complete DAG for 40 GPU configuration (8 base + 32 pool)"""
        dot = Digraph('FA_Pool_40GPU_Config')
        dot.attr(rankdir='TB', splines='spline')
        
        seq_len = 32768
        pool_gpus = 32
        block_size = seq_len // pool_gpus
        
        # Input
        dot.node('input', f'Model Input\nSeqLen: {seq_len}\n[batch_size=1024, seq_len={seq_len}, d_model=4096]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Resource manager
        dot.node('manager', f'Resource Manager\nSeqLen: {seq_len}\nPool GPUs: {pool_gpus}',
                shape='diamond', fillcolor='gold')
        
        # Processing
        for layer in range(4):
            dot.node(f'L{layer}_prenorm', f'Pre-Norm L{layer}\nGPU: 0-7', fillcolor='lightblue')
            dot.node(f'L{layer}_qsplit', f'Query Split L{layer}\nBlock: {block_size}\nGPU: 8-39',
                    shape='parallelogram', fillcolor='yellow')
            dot.node(f'L{layer}_kvbroadcast', f'KV Broadcast L{layer}\nGPU: 8-39',
                    shape='parallelogram', fillcolor='yellow')
            
            # Attention processing for 32 GPUs
            for gpu in range(32):
                gpu_id = 8 + gpu
                dot.node(f'L{layer}_Q_{gpu}', f'Q L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'L{layer}_K_{gpu}', f'K L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'L{layer}_V_{gpu}', f'V L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'L{layer}_FLASH_{gpu}', f'Flash L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
                dot.node(f'L{layer}_OUT_{gpu}', f'Output L{layer} GPU{gpu_id}',
                        fillcolor='lightcoral')
            
            dot.node(f'L{layer}_gather', f'Gather L{layer}\nGPU: 8-39',
                    shape='parallelogram', fillcolor='yellow')
            dot.node(f'L{layer}_attn_res', f'Attn Res L{layer}',
                    shape='ellipse', fillcolor='orange')
            dot.node(f'L{layer}_ffn', f'FFN L{layer}\nGPU: 0-7', fillcolor='lightblue')
            dot.node(f'L{layer}_ffn_res', f'FFN Res L{layer}',
                    shape='ellipse', fillcolor='orange')
        
        dot.node('output', f'Model Output\n[batch_size=1024, seq_len={seq_len}, vocab_size=full_vocab]',
                shape='ellipse', fillcolor='lightgreen')
        
        # Connect
        dot.edge('input', 'manager')
        dot.edge('manager', 'L0_prenorm')
        
        for layer in range(4):
            if layer > 0:
                dot.edge(f'L{layer-1}_ffn_res', f'L{layer}_prenorm')
            
            dot.edge(f'L{layer}_prenorm', f'L{layer}_qsplit')
            dot.edge(f'L{layer}_prenorm', f'L{layer}_kvbroadcast')
            
            for gpu in range(32):
                dot.edge(f'L{layer}_qsplit', f'L{layer}_Q_{gpu}')
                dot.edge(f'L{layer}_kvbroadcast', f'L{layer}_K_{gpu}')
                dot.edge(f'L{layer}_kvbroadcast', f'L{layer}_V_{gpu}')
                dot.edge(f'L{layer}_Q_{gpu}', f'L{layer}_FLASH_{gpu}')
                dot.edge(f'L{layer}_K_{gpu}', f'L{layer}_FLASH_{gpu}')
                dot.edge(f'L{layer}_V_{gpu}', f'L{layer}_FLASH_{gpu}')
                dot.edge(f'L{layer}_FLASH_{gpu}', f'L{layer}_OUT_{gpu}')
                dot.edge(f'L{layer}_OUT_{gpu}', f'L{layer}_gather')
            
            if layer == 0:
                dot.edge('input', f'L{layer}_attn_res')
            else:
                dot.edge(f'L{layer-1}_ffn_res', f'L{layer}_attn_res')
            dot.edge(f'L{layer}_gather', f'L{layer}_attn_res')
            
            dot.edge(f'L{layer}_attn_res', f'L{layer}_ffn')
            dot.edge(f'L{layer}_attn_res', f'L{layer}_ffn_res')
            dot.edge(f'L{layer}_ffn', f'L{layer}_ffn_res')
        
        dot.edge('L3_ffn_res', 'output')
        
        return dot
    
    def generate_all_configs(self):
        """Generate all DAG configurations"""
        configs = [
            ('fa_pool_8gpu_base.dot', self.create_complete_8gpu_dag()),
            ('fa_pool_16gpu_4096seq.dot', self.create_16gpu_dag()),
            ('fa_pool_24gpu_8192seq.dot', self.create_24gpu_dag()),
            ('fa_pool_40gpu_32768seq.dot', self.create_40gpu_dag())
        ]
        
        generated_files = []
        for filename, dag in configs:
            filepath = os.path.join(self.output_dir, filename)
            dag.save(filepath)
            generated_files.append(filepath)
            
            # Try to generate simplified SVG
            try:
                simple_dag = Digraph()
                simple_dag.attr(rankdir='TB', splines='spline')
                
                # Copy key nodes for SVG
                for node_name, node_attrs in dag.body:
                    if 'shape' in str(node_attrs) and ('ellipse' in str(node_attrs) or 'diamond' in str(node_attrs)):
                        simple_dag.node(node_name, label=str(node_attrs))
                
                simple_dag.render(filepath.replace('.dot', ''), format='svg', cleanup=True)
                generated_files.append(filepath.replace('.dot', '.svg'))
            except:
                pass
        
        return generated_files

if __name__ == "__main__":
    generator = DetailedFAPoolDAGGenerator("./outputs/2025-10-16-11-38-04")
    files = generator.generate_all_configs()
    print(f"Generated {len(files)} DAG files:")
    for f in files:
        print(f"  {f}")