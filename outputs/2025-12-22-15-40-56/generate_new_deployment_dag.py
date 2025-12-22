#!/usr/bin/env python3
"""
Corrected DAG Generator for EP32-TP16-PP4-DP4 New Deployment Strategy
Addresses all critical issues found in previous submissions
"""

import os
from graphviz import Digraph

def generate_new_deployment_dag():
    """Generate corrected DAG for EP32-TP16-PP4-DP4 deployment strategy"""
    
    dot = Digraph(comment='EP32-TP16-PP4-DP4 MoE LLM Inference DAG (Corrected)')
    dot.attr(bgcolor='white', fontname='Arial', rankdir='TB')
    dot.attr('node', fontname='Arial', shape='rectangle', style='filled')
    
    # Input node - DP splits the batch
    dot.node('input', 
             'Input\nInput: [batch_size=128, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]\nDP=4: Split batch into 4 groups of 32 sequences each',
             fillcolor='lightblue', shape='ellipse')
    
    # DP Split node
    dot.node('dp_split',
             'Data Parallelism Split\nGPU: DP0-DP3\nSplit batch_size=128 → 4×32\nNo communication required for inference',
             fillcolor='lightgreen', shape='parallelogram')
    
    # CRITICAL FIX: Complete attention mechanism for all DP branches with correct dimensions
    
    # DP0 Branch - Complete attention mechanism
    dot.node('embed_dp0',
             'Embedding Layer DP0\nGPU: DP0_TP0-DP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=256]\nTP=16: hidden_size=4096 → 256 per TP rank',
             fillcolor='lightgreen')
    
    dot.node('hidden_up_dp0',
             'Hidden Upscaling DP0\nGPU: DP0_TP0-DP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=256]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]\nTP=16: All-Gather operation',
             fillcolor='lightgreen')
    
    dot.node('pp0_dp0_start',
             'PP Stage 0 DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nLayers 0-3 (4 layers)\n4 layers × 32 EP groups × 2 experts/GPU = 256 expert instances',
             fillcolor='yellow', shape='parallelogram')
    
    # CRITICAL FIX: Complete Layer 0 attention for DP0 with correct dimensions (d_k=128)
    dot.node('layer0_norm_dp0',
             'Layer 0: RMSNorm DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, hidden_size=4096]',
             fillcolor='lightcoral')
    
    dot.node('layer0_q_proj_dp0',
             'Layer 0: Q Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_k_proj_dp0',
             'Layer 0: K Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=32, seq_len=1024, hidden_size=4096]\nOutput: [batch_size=32, seq_len=1024, heads=32, d_k=128]',
             fillcolor='lightcoral')
    
    dot.node('layer0_v_proj_dp0',
             'Layer 0: V Projection DP0\nGPU: DP0_PP0_TP0-DP0_PP0_TP15\nInput: [batch_size=权限不足，请稍后重试。