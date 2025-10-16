#!/usr/bin/env python3
"""
FA Pool DAG Generator
Generates complete DAGs for FA Pool dynamic parallel strategy
"""

import os
from graphviz import Digraph

class FAPoolDAGGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model specifications
        self.hidden_dim = 4096
        self.ffn_hidden_dim = 16384
        self.num_heads = 32
        self.d_k = self.hidden_dim // self.num_heads  # 128
        self.num_layers = 4
        self.batch_size = 1024
        
        # GPU configurations
        self.base_gpus = list(range(8))
        self.max_pool_gpus = 32
        
    def create_base_layer_dag(self):
        """Create DAG for the base layer (8 GPUs with tensor parallelism)"""
        dot = Digraph(comment='FA Pool Base Layer - Tensor Parallelism (8 GPUs)')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.5')
        
        # Define node styles
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Input node
        dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=?, d_model=4096]\\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\\nGPU: all GPUs', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Embedding layer (tensor parallel across 8 GPUs)
        for gpu_id in range(8):
            dot.node(f'embedding_{gpu_id}', 
                    f'Embedding Partition {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=4096]\\nOutput: [batch_size=1024, seq_len=?, d_model=512]\\nGPU: {gpu_id}')
        
        # Positional encoding (tensor parallel)
        for gpu_id in range(8):
            dot.node(f'pos_enc_{gpu_id}', 
                    f'Positional Encoding {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=512]\\nOutput: [batch_size=1024, seq_len=?, d_model=512]\\nGPU: {gpu_id}')
        
        # Layer norm (tensor parallel)
        for layer_idx in range(self.num_layers):
            for gpu_id in range(8):
                dot.node(f'layer_norm_{layer_idx}_{gpu_id}', 
                        f'Layer Norm {layer_idx} Part {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=512]\\nOutput: [batch_size=1024, seq_len=?, d_model=512]\\nGPU: {gpu_id}')
        
        # FFN layers (tensor parallel)
        for layer_idx in range(self.num_layers):
            # FFN Gate (column parallel)
            for gpu_id in range(8):
                dot.node(f'ffn_gate_{layer_idx}_{gpu_id}', 
                        f'FFN Gate {layer_idx} Part {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\\nGPU: {gpu_id}')
            
            # FFN Up (column parallel)
            for gpu_id in range(8):
                dot.node(f'ffn_up_{layer_idx}_{gpu_id}', 
                        f'FFN Up {layer_idx} Part {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=512]\\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\\nGPU: {gpu_id}')
            
            # GELU activation
            for gpu_id in range(8):
                dot.node(f'gelu_{layer_idx}_{gpu_id}', 
                        f'GELU {layer_idx} Part {gpu_id}\\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\\nOutput: [batch_size=1024, seq_len=?, ffn_dim=1024]\\nGPU: {gpu_id}')
            
            # FFN Down (row parallel)
            for gpu_id in range(8):
                dot.node(f'ffn_down_{layer_idx}_{gpu_id}', 
                        f'FFN Down {layer_idx} Part {gpu_id}\\nInput: [batch_size=1024, seq_len=?, ffn_dim=1024]\\nOutput: [batch_size=1024, seq_len=?, d_model=512]\\nGPU: {gpu_id}')
                
            # All-reduce for FFN
            dot.node(f'ffn_all_reduce_{layer_idx}', 
                    f'FFN All-Reduce {layer_idx}\\nInput: [batch_size=1024, seq_len=?, d_model=512] x 8\\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\\nGPU: all GPUs', 
                    shape='parallelogram', fillcolor='lightyellow')
            
            # Residual add
            dot.node(f'residual_ffn_{layer_idx}', 
                    f'Residual Add FFN {layer_idx}\\nInput: [batch_size=1024, seq_len=?, d_model=4096] x 2\\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\\nGPU: all GPUs', 
                    shape='ellipse', fillcolor='orange')
        
        # Output projection
        for gpu_id in range(8):
            dot.node(f'output_proj_{gpu_id}', 
                    f'Output Projection {gpu_id}\\nInput: [batch_size=1024, seq_len=?, d_model=512]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=dim/8]\\nGPU: {gpu_id}')
        
        # Final all-reduce and output
        dot.node('output_all_reduce', 'Output All-Reduce\\nInput: [batch_size=1024, seq_len=?, vocab_size=dim/8] x 8\\nOutput: [batch_size=1024, seq_len=?, vocab_size=full_vocab]\\nGPU: all GPUs', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=?, vocab_size=full_vocab]\\nOutput: [batch_size=1024, seq_len=?, vocab_size=full_vocab]\\nGPU: all GPUs', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Connect base layer components
        dot.edge('input', 'embedding_0')
        for gpu_id in range(1, 8):
            dot.edge('input', f'embedding_{gpu_id}')
        
        for gpu_id in range(8):
            dot.edge(f'embedding_{gpu_id}', f'pos_enc_{gpu_id}')
        
        # Connect through layers
        for layer_idx in range(self.num_layers):
            for gpu_id in range(8):
                if layer_idx == 0:
                    dot.edge(f'pos_enc_{gpu_id}', f'layer_norm_{layer_idx}_{gpu_id}')
                dot.edge(f'layer_norm_{layer_idx}_{gpu_id}', f'ffn_gate_{layer_idx}_{gpu_id}')
                dot.edge(f'layer_norm_{layer_idx}_{gpu_id}', f'ffn_up_{layer_idx}_{gpu_id}')
                dot.edge(f'ffn_gate_{layer_idx}_{gpu_id}', f'gelu_{layer_idx}_{gpu_id}')
                dot.edge(f'ffn_up_{layer_idx}_{gpu_id}', f'gelu_{layer_idx}_{gpu_id}')
                dot.edge(f'gelu_{layer_idx}_{gpu_id}', f'ffn_down_{layer_idx}_{gpu_id}')
                dot.edge(f'ffn_down_{layer_idx}_{gpu_id}', f'ffn_all_reduce_{layer_idx}')
            
            # Residual connections
            if layer_idx == 0:
                dot.edge('input', f'residual_ffn_{layer_idx}')
            else:
                dot.edge(f'residual_ffn_{layer_idx-1}', f'residual_ffn_{layer_idx}')
            dot.edge(f'ffn_all_reduce_{layer_idx}', f'residual_ffn_{layer_idx}')
        
        # Connect to output
        for gpu_id in range(8):
            dot.edge(f'residual_ffn_{self.num_layers-1}', f'output_proj_{gpu_id}')
            dot.edge(f'output_proj_{gpu_id}', 'output_all_reduce')
        dot.edge('output_all_reduce', 'output')
        
        return dot
    
    def create_attention_pool_dag(self, seq_len, pool_gpus):
        """Create DAG for attention pool with specified number of GPUs"""
        dot = Digraph(comment=f'FA Pool Attention Pool - {pool_gpus} GPUs - SeqLen {seq_len}')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.5')
        
        # Define node styles
        dot.attr('node', shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Calculate block size
        block_size = (seq_len + pool_gpus - 1) // pool_gpus
        
        # Input from base layer
        dot.node('base_input', 'From Base Layer\\nInput: [batch_size=1024, seq_len=?, d_model=4096]\\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\\nGPU: 0-7', 
                shape='ellipse', fillcolor='lightblue')
        
        # KV cache replication
        dot.node('kv_cache', 'KV Cache Replication\\nInput: [batch_size=1024, seq_len=?, d_model=4096]\\nOutput: [batch_size=1024, seq_len=?, d_model=4096]\\nGPU: all pool GPUs', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Split query across pool GPUs
        dot.node('query_split', 'Query Split\\nInput: [batch_size=1024, seq_len=?, d_model=4096]\\nOutput: [batch_size=1024, block_size=?, d_model=4096]\\nGPU: split across pool', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Attention blocks for each GPU
        for gpu_id in range(pool_gpus):
            start_pos = gpu_id * block_size
            end_pos = min((gpu_id + 1) * block_size, seq_len)
            actual_block_size = end_pos - start_pos
            
            # Q/K/V projections
            for head_id in range(self.num_heads):
                dot.node(f'q_proj_{gpu_id}_{head_id}', 
                        f'Q Projection GPU{gpu_id+8} Head{head_id}\\nInput: [batch_size=1024, block_size={actual_block_size}, d_model=4096]\\nOutput: [batch_size=1024, block_size={actual_block_size}, d_k=128]\\nGPU: {gpu_id+8}')
                dot.node(f'k_proj_{gpu_id}_{head_id}', 
                        f'K Projection GPU{gpu_id+8} Head{head_id}\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_k=128]\\nGPU: {gpu_id+8}')
                dot.node(f'v_proj_{gpu_id}_{head_id}', 
                        f'V Projection GPU{gpu_id+8} Head{head_id}\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_k=128]\\nGPU: {gpu_id+8}')
            
            # Multi-head attention
            dot.node(f'mha_{gpu_id}', 
                    f'Multi-Head Attention GPU{gpu_id+8}\\nInput: [batch_size=1024, block_size={actual_block_size}, seq_len={seq_len}, heads=32, d_k=128]\\nOutput: [batch_size=1024, block_size={actual_block_size}, d_model=4096]\\nGPU: {gpu_id+8}')
            
            # Output projection
            dot.node(f'attn_out_{gpu_id}', 
                    f'Attention Output GPU{gpu_id+8}\\nInput: [batch_size=1024, block_size={actual_block_size}, d_model=4096]\\nOutput: [batch_size=1024, block_size={actual_block_size}, d_model=4096]\\nGPU: {gpu_id+8}')
        
        # Gather attention results
        dot.node('gather', 'Gather Attention Results\\nInput: [batch_size=1024, block_size=?, d_model=4096] x {pool_gpus}\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: all pool GPUs', 
                shape='parallelogram', fillcolor='lightyellow')
        
        # Send back to base layer
        dot.node('to_base', 'Send to Base Layer\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7', 
                shape='ellipse', fillcolor='lightblue')
        
        # Connect nodes
        dot.edge('base_input', 'kv_cache')
        dot.edge('base_input', 'query_split')
        
        for gpu_id in range(pool_gpus):
            dot.edge('kv_cache', f'k_proj_{gpu_id}_0')
            dot.edge('kv_cache', f'v_proj_{gpu_id}_0')
            dot.edge('query_split', f'q_proj_{gpu_id}_0')
            
            for head_id in range(self.num_heads):
                if head_id > 0:
                    dot.edge('kv_cache', f'k_proj_{gpu_id}_{head_id}')
                    dot.edge('kv_cache', f'v_proj_{gpu_id}_{head_id}')
                    dot.edge('query_split', f'q_proj_{gpu_id}_{head_id}')
                
                # Connect to MHA
                dot.edge(f'q_proj_{gpu_id}_{head_id}', f'mha_{gpu_id}')
                dot.edge(f'k_proj_{gpu_id}_{head_id}', f'mha_{gpu_id}')
                dot.edge(f'v_proj_{gpu_id}_{head_id}', f'mha_{gpu_id}')
            
            dot.edge(f'mha_{gpu_id}', f'attn_out_{gpu_id}')
            dot.edge(f'attn_out_{gpu_id}', 'gather')
        
        dot.edge('gather', 'to_base')
        
        return dot
    
    def create_complete_fa_pool_dag(self, seq_len, pool_gpus):
        """Create complete FA Pool DAG showing base layer + attention pool"""
        dot = Digraph(comment=f'Complete FA Pool Deployment - {8 + pool_gpus} GPUs - SeqLen {seq_len}')
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', compound='true')
        
        # Create subgraphs for GPU groups
        with dot.subgraph(name='cluster_base') as base:
            base.attr(label='Base Layer (8 GPUs)', style='dashed', color='blue')
            # Include base layer components here
            base.node('base_start', 'Base Layer Processing\\nGPU 0-7', shape='folder')
        
        if pool_gpus > 0:
            with dot.subgraph(name='cluster_pool') as pool:
                pool.attr(label=f'Attention Pool ({pool_gpus} GPUs)', style='dashed', color='red')
                pool.node('pool_start', f'Attention Pool Processing\\nGPU 8-{8+pool_gpus-1}', shape='folder')
        
        # Input node
        dot.node('input_total', 'Model Input\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Resource manager
        dot.node('resource_manager', f'Resource Manager\\nSequence Length: {seq_len}\\nPool GPUs: {pool_gpus}\\nTotal GPUs: {8 + pool_gpus}', 
                shape='diamond', fillcolor='gold')
        
        # Processing flow
        dot.node('embeddings', 'Token Embeddings\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7 (tensor parallel)', 
                shape='rectangle', fillcolor='lightblue')
        
        # Create layer components
        for layer_idx in range(self.num_layers):
            # Attention layer
            dot.node(f'attention_{layer_idx}', 
                    f'Layer {layer_idx} Attention\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: {"0-7" if pool_gpus == 0 else f"8-{8+pool_gpus-1}"}', 
                    shape='rectangle', fillcolor='lightcoral' if pool_gpus > 0 else 'lightblue')
            
            # FFN layer
            dot.node(f'ffn_{layer_idx}', 
                    f'Layer {layer_idx} FFN\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7 (tensor parallel)', 
                    shape='rectangle', fillcolor='lightblue')
            
            # Residual connections
            if layer_idx == 0:
                dot.node(f'residual_attn_{layer_idx}', 
                        f'Residual Add Attention {layer_idx}\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096] x 2\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7', 
                        shape='ellipse', fillcolor='orange')
            
            dot.node(f'residual_ffN_{layer_idx}', 
                    f'Residual Add FFN {layer_idx}\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096] x 2\\nOutput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nGPU: 0-7', 
                    shape='ellipse', fillcolor='orange')
        
        # Output
        dot.node('output_total', 'Model Output\\nInput: [batch_size=1024, seq_len={seq_len}, d_model=4096]\\nOutput: [batch_size=1024, seq_len={seq_len}, vocab_size=full_vocab]\\nGPU: 0-7', 
                shape='ellipse', fillcolor='lightgreen')
        
        # Connect the flow
        dot.edge('input_total', 'resource_manager')
        dot.edge('resource_manager', 'embeddings')
        dot.edge('embeddings', 'attention_0')
        dot.edge('attention_0', 'residual_attn_0')
        dot.edge('embeddings', 'residual_attn_0')
        dot.edge('residual_attn_0', 'ffn_0')
        dot.edge('residual_attn_0', 'residual_ffN_0')
        dot.edge('ffn_0', 'residual_ffN_0')
        
        for layer_idx in range(1, self.num_layers):
            prev_idx = layer_idx - 1
            dot.edge(f'residual_ffN_{prev_idx}', f'attention_{layer_idx}')
            dot.edge(f'attention_{layer_idx}', f'residual_attn_{layer_idx}' if layer_idx > 0 else f'residual_ffN_{layer_idx}')
            dot.edge(f'residual_ffN_{prev_idx}', f'residual_attn_{layer_idx}' if layer_idx > 0 else f'residual_ffN_{layer_idx}')
            dot.edge(f'residual_attn_{layer_idx}' if layer_idx > 0 else f'residual_ffN_{layer_idx}', f'ffn_{layer_idx}')
            dot.edge(f'ffn_{layer_idx}', f'residual_ffN_{layer_idx}')
            dot.edge(f'residual_attn_{layer_idx}' if layer_idx > 0 else f'residual_ffN_{layer_idx}', f'residual_ffN_{layer_idx}')
        
        dot.edge(f'residual_ffN_{self.num_layers-1}', 'output_total')
        
        return dot
    
    def generate_all_dags(self):
        """Generate all required DAG configurations"""
        configurations = [
            (512, 0, "short_sequence_no_pool"),
            (4096, 8, "medium_sequence_8_pool"),
            (8192, 16, "long_sequence_16_pool"),
            (16384, 24, "very_long_sequence_24_pool"),
            (32768, 32, "extreme_sequence_32_pool")
        ]
        
        generated_files = []
        
        for seq_len, pool_gpus, name in configurations:
            print(f"Generating DAG for seq_len={seq_len}, pool_gpus={pool_gpus}")
            
            # Generate base layer DAG
            base_dag = self.create_base_layer_dag()
            base_filename = f"fa_pool_base_layer_8gpus.dot"
            base_path = os.path.join(self.output_dir, base_filename)
            base_dag.save(base_path)
            generated_files.append(base_path)
            
            # Generate attention pool DAG if needed
            if pool_gpus > 0:
                pool_dag = self.create_attention_pool_dag(seq_len, pool_gpus)
                pool_filename = f"fa_pool_attention_{pool_gpus}gpus_seq{seq_len}.dot"
                pool_path = os.path.join(self.output_dir, pool_filename)
                pool_dag.save(pool_path)
                generated_files.append(pool_path)
            
            # Generate complete DAG
            complete_dag = self.create_complete_fa_pool_dag(seq_len, pool_gpus)
            complete_filename = f"fa_pool_complete_{name}.dot"
            complete_path = os.path.join(self.output_dir, complete_filename)
            complete_dag.save(complete_path)
            generated_files.append(complete_path)
            
            # Also generate SVG files
            try:
                base_dag.render(base_path.replace('.dot', ''), format='svg', cleanup=True)
                if pool_gpus > 0:
                    pool_dag.render(pool_path.replace('.dot', ''), format='svg', cleanup=True)
                complete_dag.render(complete_path.replace('.dot', ''), format='svg', cleanup=True)
            except Exception as e:
                print(f"Warning: Could not generate SVG files: {e}")
        
        return generated_files

if __name__ == "__main__":
    generator = FAPoolDAGGenerator("./outputs/2025-10-16-11-38-04")
    files = generator.generate_all_dags()
    print(f"Generated {len(files)} DAG files:")
    for f in files:
        print(f"  {f}")