#!/usr/bin/env python3
"""
DAG Generator for 10B MoE Model with TP×EP×PP = 2×2×2 Parallelism
Generates complete execution graph with operator-level granularity
"""

import graphviz
from graphviz import Digraph

def create_moe_dag():
    # Create directed graph
    dot = Digraph(comment='10B MoE Model TP×EP×PP=2×2×2 Parallelism DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    dot.attr('node', fontsize='10', height='0.6', width='1.2')
    dot.attr('edge', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rect', style='filled', fillcolor='lightgreen')    # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # GPU assignments
    gpu_colors = {
        0: 'lightcoral',
        1: 'lightsteelblue', 
        2: 'lightseagreen',
        3: 'lightsalmon'
    }
    
    # ====================================================================================
    # PREFILL PHASE
    # ====================================================================================
    
    with dot.subgraph(name='cluster_prefill') as prefill:
        prefill.attr(label='PREFILL PHASE', style='rounded,filled', fillcolor='lightgray', labeljust='l')
        
        # Input distribution across TP dimension
        prefill.node('input_split', 'Input Split\n[batch=128, seq=10240, dim=512]→[batch=128, seq=10240, dim=256]\nGPU: ALL', 
                    shape='parallelogram', fillcolor='lightyellow')
        
        # PP Stage 0: Layers 0-7 on GPUs 0,1
        with prefill.subgraph(name='cluster_pp0') as pp0:
            pp0.attr(label='Pipeline Stage 0 (Layers 0-7)', style='rounded,filled', fillcolor='lightcyan')
            
            # GPU 0: TP group 0, PP stage 0, Experts 0-7
            with pp0.subgraph(name='cluster_gpu0') as gpu0:
                gpu0.attr(label='GPU 0 (TP-0, PP-0, EP-0)', style='rounded,filled', fillcolor=gpu_colors[0])
                
                # Layer 0 operations
                gpu0.node('gpu0_l0_attn_q', 'Attention Q Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_attn_k', 'Attention K Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_attn_v', 'Attention V Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                gpu0.node('gpu0_l0_attn_score', 'Attention Score\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                gpu0.node('gpu0_l0_attn_softmax', 'Attention Softmax\nInput: [128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                gpu0.node('gpu0_l0_attn_out', 'Attention Output\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,32]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                gpu0.node('gpu0_l0_attn_proj', 'Attention Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,512]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                # TP All-Reduce for attention
                gpu0.node('gpu0_l0_attn_ar', 'TP All-Reduce\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0-1',
                         shape='ellipse', fillcolor='lightblue')
                
                # MoE routing
                gpu0.node('gpu0_l0_gate', 'MoE Gate\nInput: [128,10240,512]\nOutput: [128,10240,16]\nGPU: 0',
                         shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                # Expert computations (simplified - showing 2 experts per GPU)
                gpu0.node('gpu0_l0_exp0_fc1', 'Expert 0 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_exp0_act', 'Expert 0 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_exp0_fc2', 'Expert 0 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                gpu0.node('gpu0_l0_exp1_fc1', 'Expert 1 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_exp1_act', 'Expert 1 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_exp1_fc2', 'Expert 1 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                
                # EP All-to-All communication
                gpu0.node('gpu0_l0_ep_a2a', 'EP All-to-All\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0-1',
                         shape='ellipse', fillcolor='lightblue')
                
                # Expert aggregation
                gpu0.node('gpu0_l0_exp_agg', 'Expert Aggregation\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0',
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu0.node('gpu0_l0_ln1', 'Layer Norm 1\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
                gpu0.node('gpu0_l0_ln2', 'Layer Norm 2\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0',
                         shape='rect', fillcolor='lightgreen')
            
            # GPU 1: TP group 1, PP stage 0, Experts 8-15
            with pp0.subgraph(name='cluster_gpu1') as gpu1:
                gpu1.attr(label='GPU 1 (TP-1, PP-0, EP-1)', style='rounded,filled', fillcolor=gpu_colors[1])
                
                # Similar operations for GPU 1
                gpu1.node('gpu1_l0_attn_q', 'Attention Q Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_attn_k', 'Attention K Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_attn_v', 'Attention V Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                gpu1.node('gpu1_l0_attn_score', 'Attention Score\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                gpu1.node('gpu1_l0_attn_softmax', 'Attention Softmax\nInput: [128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                gpu1.node('gpu1_l0_attn_out', 'Attention Output\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,32]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                gpu1.node('gpu1_l0_attn_proj', 'Attention Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,512]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                # TP All-Reduce for attention
                gpu1.node('gpu1_l0_attn_ar', 'TP All-Reduce\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0-1',
                         shape='ellipse', fillcolor='lightblue')
                
                # MoE routing
                gpu1.node('gpu1_l0_gate', 'MoE Gate\nInput: [128,10240,512]\nOutput: [128,10240,16]\nGPU: 1',
                         shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                # Expert computations
                gpu1.node('gpu1_l0_exp8_fc1', 'Expert 8 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_exp8_act', 'Expert 8 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_exp8_fc2', 'Expert 8 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                gpu1.node('gpu1_l0_exp9_fc1', 'Expert 9 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_exp9_act', 'Expert 9 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_exp9_fc2', 'Expert 9 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                
                # EP All-to-All communication
                gpu1.node('gpu1_l0_ep_a2a', 'EP All-to-All\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 0-1',
                         shape='ellipse', fillcolor='lightblue')
                
                # Expert aggregation
                gpu1.node('gpu1_l0_exp_agg', 'Expert Aggregation\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 1',
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu1.node('gpu1_l0_ln1', 'Layer Norm 1\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
                gpu1.node('gpu1_l0_ln2', 'Layer Norm 2\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 1',
                         shape='rect', fillcolor='lightgreen')
        
        # PP Stage 1: Layers 8-15 on GPUs 2,3
        with prefill.subgraph(name='cluster_pp1') as pp1:
            pp1.attr(label='Pipeline Stage 1 (Layers 8-15)', style='rounded,filled', fillcolor='lightcyan')
            
            # GPU 2: TP group 0, PP stage 1, Experts 0-7
            with pp1.subgraph(name='cluster_gpu2') as gpu2:
                gpu2.attr(label='GPU 2 (TP-0, PP-1, EP-0)', style='rounded,filled', fillcolor=gpu_colors[2])
                
                # Layer 8 operations (simplified representation)
                gpu2.node('gpu2_l8_attn_q', 'Attention Q Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                gpu2.node('gpu2_l8_attn_k', 'Attention K Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                gpu2.node('gpu2_l8_attn_v', 'Attention V Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                gpu2.node('gpu2_l8_attn_score', 'Attention Score\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                gpu2.node('gpu2_l8_attn_softmax', 'Attention Softmax\nInput: [128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                gpu2.node('gpu2_l8_attn_out', 'Attention Output\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,32]\nGPU: 2',
                         shape='rect', phases='lightgreen')
                
                gpu2.node('gpu2_l8_attn_proj', 'Attention Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,512]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                # TP All-Reduce for attention
                gpu2.node('gpu2_l8_attn_ar', 'TP All-Reduce\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2-3',
                         shape='ellipse', fillcolor='lightblue')
                
                # MoE routing
                gpu2.node('gpu2_l8_gate', 'MoE Gate\nInput: [128,10240,512]\nOutput: [128,10240,16]\nGPU: 2',
                         shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                # Expert computations
                gpu2.node('gpu2_l8_exp0_fc1', 'Expert 0 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                gpu2.node('gpu2_l8_exp0_act', 'Expert 0 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                gpu2.node('gpu2_l8_exp0_fc2', 'Expert 0 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                # EP All-to-All communication
                gpu2.node('gpu2_l8_ep_a2a', 'EP All-to-All\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2-3',
                         shape='ellipse', fillcolor='lightblue')
                
                # Expert aggregation
                gpu2.node('gpu2_l8_exp_agg', 'Expert Aggregation\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2',
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu2.node('gpu2_l8_ln1', 'Layer Norm 1\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                gpu2.node('gpu2_l8_ln2', 'Layer Norm 2\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
                
                # Output projection
                gpu2.node('gpu2_output_proj', 'Output Projection\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2',
                         shape='rect', fillcolor='lightgreen')
            
            # GPU 3: TP group 1, PP stage 1, Experts 8-15
            with pp1.subgraph(name='cluster_gpu3') as gpu3:
                gpu3.attr(label='GPU 3 (TP-1, PP-1, EP-1)', style='rounded,filled', fillcolor=gpu_colors[3])
                
                # Layer 8 operations
                gpu3.node('gpu3_l8_attn_q', 'Attention Q Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                gpu3.node('gpu3_l8_attn_k', 'Attention K Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                gpu3.node('gpu3_l8_attn_v', 'Attention V Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,16,16]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                gpu3.node('gpu3_l8_attn_score', 'Attention Score\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                gpu3.node('gpu3_l8_attn_softmax', 'Attention Softmax\nInput: [128,10240,16,16]\nOutput: [128,10240,16,16]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                gpu3.node('gpu3_l8_attn_out', 'Attention Output\nInput: [128,10240,16,16]×[128,10240,16,16]\nOutput: [128,10240,16,32]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                gpu3.node('gpu3_l8_attn_proj', 'Attention Projection\nInput: [128,10240,16,32]\nOutput: [128,10240,512]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                # TP All-Reduce for attention
                gpu3.node('gpu3_l8_attn_ar', 'TP All-Reduce\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2-3',
                         shape='ellipse', fillcolor='lightblue')
                
                # MoE routing
                gpu3.node('gpu3_l8_gate', 'MoE Gate\nInput: [128,10240,512]\nOutput: [128,10240,16]\nGPU: 3',
                         shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                # Expert computations
                gpu3.node('gpu3_l8_exp8_fc1', 'Expert 8 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                gpu3.node('gpu3_l8_exp8_act', 'Expert 8 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                gpu3.node('gpu3_l8_exp8_fc2', 'Expert 8 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                # EP All-to-All communication
                gpu3.node('gpu3_l8_ep_a2a', 'EP All-to-All\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 2-3',
                         shape='ellipse', fillcolor='lightblue')
                
                # Expert aggregation
                gpu3.node('gpu3_l8_exp_agg', 'Expert Aggregation\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 3',
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu3.node('gpu3_l8_ln1', 'Layer Norm 1\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                gpu3.node('gpu3_l8_ln2', 'Layer Norm 2\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                # Output projection
                gpu3.node('gpu3_output_proj', 'Output Projection\nInput: [128,10240,512]\nOutput: [128,10240,512]\nGPU: 3',
                         shape='rect', fillcolor='lightgreen')
                
                # Final output aggregation
                gpu3.node('output_agg', 'Output Aggregation\nInput: [128,10240,512]×2\nOutput: [128,10240,512]\nGPU: ALL',
                         shape='parallelogram', fillcolor='lightyellow')
    
    # ====================================================================================
    # DECODE PHASE
    # ====================================================================================
    
    with dot.subgraph(name='cluster_decode') as decode:
        decode.attr(label='DECODE PHASE', style='rounded,filled', fillcolor='lightgray', labeljust='l')
        
        # Decode starts with single token
        decode.node('decode_input', 'Decode Input\n[batch=128, seq=1, dim=512]→[batch=128, seq=1, dim=256]\nGPU: ALL',
                   shape='parallelogram', fillcolor='lightyellow')
        
        # PP Stage 0: Decode layers 0-7
        with decode.subgraph(name='cluster_decode_pp0') as dec_pp0:
            dec_pp0.attr(label='Decode Pipeline Stage 0 (Layers 0-7)', style='rounded,filled', fillcolor='lightcyan')
            
            # GPU 0 decode operations
            with dec_pp0.subgraph(name='cluster_decode_gpu0') as dec_gpu0:
                dec_gpu0.attr(label='GPU 0 Decode (TP-0, PP-0, EP-0)', style='rounded,filled', fillcolor=gpu_colors[0])
                
                dec_gpu0.node('dec_gpu0_l0_attn_q', 'Attention Q\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu0.node('dec_gpu0_l0_attn_k', 'Attention K\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu0.node('dec_gpu0_l0_attn_v', 'Attention V\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_kv_cache', 'KV Cache Update\nInput: [128,1,16,16]\nOutput: [128,seq+1,16,16]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_attn_score', 'Attention Score\nInput: [128,1,16,16]×[128,seq+1,16,16]\nOutput: [128,1,16,seq+1]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_attn_softmax', 'Attention Softmax\nInput: [128,1,16,seq+1]\nOutput: [128,1,16,seq+1]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_attn_out', 'Attention Output\nInput: [128,1,16,seq+1]×[128,seq+1,16,16]\nOutput: [128,1,16,32]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_attn_proj', 'Attention Projection\nInput: [128,1,16,32]\nOutput: [128,1,512]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_attn_ar', 'TP All-Reduce\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0-1',
                             shape='ellipse', fillcolor='lightblue')
                
                # MoE for decode
                dec_gpu0.node('dec_gpu0_l0_gate', 'MoE Gate\nInput: [128,1,512]\nOutput: [128,1,16]\nGPU: 0',
                             shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                dec_gpu0.node('dec_gpu0_l0_exp0_fc1', 'Expert 0 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu0.node('dec_gpu0_l0_exp0_act', 'Expert 0 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu0.node('dec_gpu0_l0_exp0_fc2', 'Expert 0 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu0.node('dec_gpu0_l0_ep_a2a', 'EP All-to-All\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0-1',
                             shape='ellipse', fillcolor='lightblue')
                
                dec_gpu0.node('dec_gpu0_l0_exp_agg', 'Expert Aggregation\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0',
                             shape='parallelogram', fillcolor='lightyellow')
                
                dec_gpu0.node('dec_gpu0_l0_ln1', 'Layer Norm 1\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu0.node('dec_gpu0_l0_ln2', 'Layer Norm 2\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0',
                             shape='rect', fillcolor='lightgreen')
            
            # GPU 1 decode operations
            with dec_pp0.subgraph(name='cluster_decode_gpu1') as dec_gpu1:
                dec_gpu1.attr(label='GPU 1 Decode (TP-1, PP-0, EP-1)', style='rounded,filled', fillcolor=gpu_colors[1])
                
                dec_gpu1.node('dec_gpu1_l0_attn_q', 'Attention Q\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu1.node('dec_gpu1_l0_attn_k', 'Attention K\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu1.node('dec_gpu1_l0_attn_v', 'Attention V\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_kv_cache', 'KV Cache Update\nInput: [128,1,16,16]\nOutput: [128,seq+1,16,16]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_attn_score', 'Attention Score\nInput: [128,1,16,16]×[128,seq+1,16,16]\nOutput: [128,1,16,seq+1]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_attn_softmax', 'Attention Softmax\nInput: [128,1,16,seq+1]\nOutput: [128,1,16,seq+1]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_attn_out', 'Attention Output\nInput: [128,1,16,seq+1]×[128,seq+1,16,16]\nOutput: [128,1,16,32]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_attn_proj', 'Attention Projection\nInput: [128,1,16,32]\nOutput: [128,1,512]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_attn_ar', 'TP All-Reduce\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0-1',
                             shape='ellipse', fillcolor='lightblue')
                
                # MoE for decode
                dec_gpu1.node('dec_gpu1_l0_gate', 'MoE Gate\nInput: [128,1,512]\nOutput: [128,1,16]\nGPU: 1',
                             shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                
                dec_gpu1.node('dec_gpu1_l0_exp8_fc1', 'Expert 8 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu1.node('dec_gpu1_l0_exp8_act', 'Expert 8 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu1.node('dec_gpu1_l0_exp8_fc2', 'Expert 8 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                
                dec_gpu1.node('dec_gpu1_l0_ep_a2a', 'EP All-to-All\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 0-1',
                             shape='ellipse', fillcolor='lightblue')
                
                dec_gpu1.node('dec_gpu1_l0_exp_agg', 'Expert Aggregation\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 1',
                             shape='parallelogram', fillcolor='lightyellow')
                
                dec_gpu1.node('dec_gpu1_l0_ln1', 'Layer Norm 1\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu1.node('dec_gpu1_l0_ln2', 'Layer Norm 2\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 1',
                             shape='rect', fillcolor='lightgreen')
        
        # PP Stage 1: Decode layers 8-15
        with decode.subgraph(name='cluster_decode_pp1') as dec_pp1:
            dec_pp1.attr(label='Decode Pipeline Stage 1 (Layers 8-15)', style='rounded,filled', fillcolor='lightcyan')
            
            # GPU 2 decode operations
            with dec_pp1.subgraph(name='cluster_decode_gpu2') as dec_gpu2:
                dec_gpu2.attr(label='GPU 2 Decode (TP-0, PP-1, EP-0)', style='rounded,filled', fillcolor=gpu_colors[2])
                
                dec_gpu2.node('dec_gpu2_l8_attn_q', 'Attention Q\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_kv_cache', 'KV Cache Update\nInput: [128,1,16,16]\nOutput: [128,seq+1,16,16]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_attn_score', 'Attention Score\nInput: [128,1,16,16]×[128,seq+1,16,16]\nOutput: [128,1,16,seq+1]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_attn_softmax', 'Attention Softmax\nInput: [128,1,16,seq+1]\nOutput: [128,1,16,seq+1]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_attn_out', 'Attention Output\nInput: [128,1,16,seq+1]×[128,seq+1,16,16]\nOutput: [128,1,16,32]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_attn_proj', 'Attention Projection\nInput: [128,1,16,32]\nOutput: [128,1,512]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_attn_ar', 'TP All-Reduce\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2-3',
                             shape='ellipse', fillcolor='lightblue')
                
                # MoE for decode layer 8
                dec_gpu2.node('dec_gpu2_l8_gate', 'MoE Gate\nInput: [128,1,512]\nOutput: [128,1,16]\nGPU: 2',
                             shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                dec_gpu2.node('dec_gpu2_l8_exp0_fc1', 'Expert 0 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_exp0_act', 'Expert 0 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_exp0_fc2', 'Expert 0 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_ep_a2a', 'EP All-to-All\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2-3',
                             shape='ellipse', fillcolor='lightblue')
                dec_gpu2.node('dec_gpu2_l8_exp_agg', 'Expert Aggregation\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2',
                             shape='parallelogram', fillcolor='lightyellow')
                dec_gpu2.node('dec_gpu2_l8_ln1', 'Layer Norm 1\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu2.node('dec_gpu2_l8_ln2', 'Layer Norm 2\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
                
                # Final decode output
                dec_gpu2.node('dec_gpu2_output', 'Decode Output\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2',
                             shape='rect', fillcolor='lightgreen')
            
            # GPU 3 decode operations
            with dec_pp1.subgraph(name='cluster_decode_gpu3') as dec_gpu3:
                dec_gpu3.attr(label='GPU 3 Decode (TP-1, PP-1, EP-1)', style='rounded,filled', fillcolor=gpu_colors[3])
                
                dec_gpu3.node('dec_gpu3_l8_attn_q', 'Attention Q\nInput: [128,1,16,32]\nOutput: [128,1,16,16]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_kv_cache', 'KV Cache Update\nInput: [128,1,16,16]\nOutput: [128,seq+1,16,16]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_attn_score', 'Attention Score\nInput: [128,1,16,16]×[128,seq+1,16,16]\nOutput: [128,1,16,seq+1]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_attn_softmax', 'Attention Softmax\nInput: [128,1,16,seq+1]\nOutput: [128,1,16,seq+1]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_attn_out', 'Attention Output\nInput: [128,1,16,seq+1]×[128,seq+1,16,16]\nOutput: [128,1,16,32]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_attn_proj', 'Attention Projection\nInput: [128,1,16,32]\nOutput: [128,1,512]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_attn_ar', 'TP All-Reduce\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2-3',
                             shape='ellipse', fillcolor='lightblue')
                
                # MoE for decode layer 8
                dec_gpu3.node('dec_gpu3_l8_gate', 'MoE Gate\nInput: [128,1,512]\nOutput: [128,1,16]\nGPU: 3',
                             shape='parallelogram', fillcolor='lightyellow', style='dashed,filled')
                dec_gpu3.node('dec_gpu3_l8_exp8_fc1', 'Expert 8 FC1\nInput: [tokens,512]\nOutput: [tokens,1024]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_exp8_act', 'Expert 8 Activation\nInput: [tokens,1024]\nOutput: [tokens,1024]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_exp8_fc2', 'Expert 8 FC2\nInput: [tokens,1024]\nOutput: [tokens,512]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_ep_a2a', 'EP All-to-All\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 2-3',
                             shape='ellipse', fillcolor='lightblue')
                dec_gpu3.node('dec_gpu3_l8_exp_agg', 'Expert Aggregation\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 3',
                             shape='parallelogram', fillcolor='lightyellow')
                dec_gpu3.node('dec_gpu3_l8_ln1', 'Layer Norm 1\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                dec_gpu3.node('dec_gpu3_l8_ln2', 'Layer Norm 2\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                
                # Final decode output
                dec_gpu3.node('dec_gpu3_output', 'Decode Output\nInput: [128,1,512]\nOutput: [128,1,512]\nGPU: 3',
                             shape='rect', fillcolor='lightgreen')
                
                # Final output aggregation
                dec_gpu3.node('decode_output_agg', 'Decode Output Agg\nInput: [128,1,512]×2\nOutput: [128,1,512]\nGPU: ALL',
                             shape='parallelogram', fillcolor='lightyellow')
    
    # ====================================================================================
    # EDGES - PREFILL PHASE
    # ====================================================================================
    
    # Input split to first layer
    dot.edge('input_split', 'gpu0_l0_attn_q')
    dot.edge('input_split', 'gpu0_l0_attn_k')
    dot.edge('input_split', 'gpu0_l0_attn_v')
    dot.edge('input_split', 'gpu1_l0_attn_q')
    dot.edge('input_split', 'gpu1_l0_attn_k')
    dot.edge('input_split', 'gpu1_l0_attn_v')
    
    # Attention computation flow
    dot.edge('gpu0_l0_attn_q', 'gpu0_l0_attn_score')
    dot.edge('gpu0_l0_attn_k', 'gpu0_l0_attn_score')
    dot.edge('gpu0_l0_attn_v', 'gpu0_l0_attn_out')
    dot.edge('gpu0_l0_attn_score', 'gpu0_l0_attn_softmax')
    dot.edge('gpu0_l0_attn_softmax', 'gpu0_l0_attn_out')
    dot.edge('gpu0_l0_attn_out', 'gpu0_l0_attn_proj')
    dot.edge('gpu0_l0_attn_proj', 'gpu0_l0_attn_ar')
    
    # Same for GPU 1
    dot.edge('gpu1_l0_attn_q', 'gpu1_l0_attn_score')
    dot.edge('gpu1_l0_attn_k', 'gpu1_l0_attn_score')
    dot.edge('gpu1_l0_attn_v', 'gpu1_l0_attn_out')
    dot.edge('gpu1_l0_attn_score', 'gpu1_l0_attn_softmax')
    dot.edge('gpu1_l0_attn_softmax', 'gpu1_l0_attn_out')
    dot.edge('gpu1_l0_attn_out', 'gpu1_l0_attn_proj')
    dot.edge('gpu1_l0_attn_proj', 'gpu1_l0_attn_ar')
    
    # Attention to MoE
    dot.edge('gpu0_l0_attn_ar', 'gpu0_l0_gate')
    dot.edge('gpu0_l0_attn_ar', 'gpu0_l0_ln1')
    dot.edge('gpu1_l0_attn_ar', 'gpu1_l0_gate')
    dot.edge('gpu1_l0_attn_ar', 'gpu1_l0_ln1')
    
    # MoE expert flow
    dot.edge('gpu0_l0_gate', 'gpu0_l0_exp0_fc1', style='dashed')
    dot.edge('gpu0_l0_gate', 'gpu0_l0_exp1_fc1', style='dashed')
    dot.edge('gpu0_l0_exp0_fc1', 'gpu0_l0_exp0_act')
    dot.edge('gpu0_l0_exp0_act', 'gpu0_l0_exp0_fc2')
    dot.edge('gpu0_l0_exp1_fc1', 'gpu0_l0_exp1_act')
    dot.edge('gpu0_l0_exp1_act', 'gpu0_l0_exp1_fc2')
    dot.edge('gpu0_l0_exp0_fc2', 'gpu0_l0_ep_a2a')
    dot.edge('gpu0_l0_exp1_fc2', 'gpu0_l0_ep_a2a')
    
    # Expert aggregation
    dot.edge('gpu0_l0_ep_a2a', 'gpu0_l0_exp_agg')
    dot.edge('gpu0_l0_exp_agg', 'gpu0_l0_ln2')
    
    # Pipeline to next stage
    dot.edge('gpu0_l0_ln2', 'gpu2_l8_attn_q')
    dot.edge('gpu0_l0_ln2', 'gpu2_l8_attn_k')
    dot.edge('gpu0_l0_ln2', 'gpu2_l8_attn_v')
    dot.edge('gpu1_l0_ln2', 'gpu3_l8_attn_q')
    dot.edge('gpu1_l0_ln2', 'gpu3_l8_attn_k')
    dot.edge('gpu1_l0_ln2', 'gpu3_l8_attn_v')
    
    # Final output
    dot.edge('gpu2_output_proj', 'output_agg')
    dot.edge('gpu3_output_proj', 'output_agg')
    
    # ====================================================================================
    # EDGES - DECODE PHASE
    # ====================================================================================
    
    # Decode input to first layer
    dot.edge('decode_input', 'dec_gpu0_l0_attn_q')
    dot.edge('decode_input', 'dec_gpu0_l0_attn_k')
    dot.edge('decode_input', 'dec_gpu0_l0_attn_v')
    dot.edge('decode_input', 'dec_gpu1_l0_attn_q')
    dot.edge('decode_input', 'dec_gpu1_l0_attn_k')
    dot.edge('decode_input', 'dec_gpu1_l0_attn_v')
    
    # Decode attention flow
    dot.edge('dec_gpu0_l0_attn_q', 'dec_gpu0_l0_attn_score')
    dot.edge('dec_gpu0_l0_attn_k', 'dec_gpu0_l0_kv_cache')
    dot.edge('dec_gpu0_l0_attn_v', 'dec_gpu0_l0_kv_cache')
    dot.edge('dec_gpu0_l0_kv_cache', 'dec_gpu0_l0_attn_score')
    dot.edge('dec_gpu0_l0_attn_score', 'dec_gpu0_l0_attn_softmax')
    dot.edge('dec_gpu0_l0_attn_softmax', 'dec_gpu0_l0_attn_out')
    dot.edge('dec_gpu0_l0_attn_out', 'dec_gpu0_l0_attn_proj')
    dot.edge('dec_gpu0_l0_attn_proj', 'dec_gpu0_l0_attn_ar')
    
    # Decode MoE
    dot.edge('dec_gpu0_l0_attn_ar', 'dec_gpu0_l0_gate')
    dot.edge('dec_gpu0_l0_attn_ar', 'dec_gpu0_l0_ln1')
    dot.edge('dec_gpu0_l0_gate', 'dec_gpu0_l0_exp0_fc1', style='dashed')
    dot.edge('dec_gpu0_l0_exp0_fc1', 'dec_gpu0_l0_exp0_act')
    dot.edge('dec_gpu0_l0_exp0_act', 'dec_gpu0_l0_exp0_fc2')
    dot.edge('dec_gpu0_l0_exp0_fc2', 'dec_gpu0_l0_ep_a2a')
    dot.edge('dec_gpu0_l0_ep_a2a', 'dec_gpu0_l0_exp_agg')
    dot.edge('dec_gpu0_l0_exp_agg', 'dec_gpu0_l0_ln2')
    
    # Decode pipeline to next stage
    dot.edge('dec_gpu0_l0_ln2', 'dec_gpu2_l8_attn_q')
    dot.edge('dec_gpu0_l0_ln2', 'dec_gpu2_l8_attn_k')
    dot.edge('dec_gpu0_l0_ln2', 'dec_gpu2_l8_attn_v')
    dot.edge('dec_gpu1_l0_ln2', 'dec_gpu3_l8_attn_q')
    dot.edge('dec_gpu1_l0_ln2', 'dec_gpu3_l8_attn_k')
    dot.edge('dec_gpu1_l0_ln2', 'dec_gpu3_l8_attn_v')
    
    # Final decode output
    dot.edge('dec_gpu2_output', 'decode_output_agg')
    dot.edge('dec_gpu3_output', 'decode_output_agg')
    
    return dot

def main():
    print("Generating 10B MoE Model DAG with TP×EP×PP = 2×2×2 parallelism...")
    
    # Create the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-24-15-21-58/moe_model_dag.dot'
    dag.save(dot_file)
    print(f"DOT file saved: {dot_file}")
    
    # Render as SVG
    svg_file = '../outputs/2025-12-24-15-21-58/moe_model_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved: {svg_file}")
    
    # Also render as PNG for easier viewing
    png_file = '../outputs/2025-12-24-15-21-58/moe_model_dag.png'
    dag.render(png_file.replace('.png', ''), format='png', cleanup=True)
    print(f"PNG file saved: {png_file}")
    
    print("DAG generation completed successfully!")
    
    # Verify DAG has no cycles
    try:
        import networkx as nx
        from networkx.drawing.nx_agraph import read_dot
        
        # Read the DOT file and check for cycles
        G = read_dot(dot_file)
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"WARNING: Found {len(cycles)} cycles in the DAG!")
            for cycle in cycles[:3]:  # Show first 3 cycles
                print(f"Cycle: {cycle}")
        else:
            print("✓ DAG verification: No cycles found")
            
        # Check connectivity
        if nx.is_weakly_connected(G):
            print("✓ DAG verification: Graph is connected")
        else:
            print("WARNING: Graph is not fully connected")
            
    except ImportError:
        print("NetworkX not available for cycle detection")
    except Exception as e:
        print(f"Error during DAG verification: {e}")

if __name__ == "__main__":
    main()