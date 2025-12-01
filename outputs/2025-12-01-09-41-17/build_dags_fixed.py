#!/usr/bin/env python3
"""
Generate complete operator-level DAGs for:
1. Baseline: 16-layer MoE, TP=8, PP=2, 4 experts/GPU
2. Proposed: 16-layer MoE, EP=16, 1 expert/GPU

Output:
- baseline.dot  &  baseline.svg
- proposed.dot  &  proposed.svg
"""

import os
import subprocess

def shape(batch, seq, heads=None, dk=None, hidden=None, ffn=None):
    """Helper to format dimension string"""
    parts = [f"batch_size={batch}", f"seq_len={seq}"]
    if heads is not None:
        parts.append(f"heads={heads}")
    if dk is not None:
        parts.append(f"d_k={dk}")
    if hidden is not None:
        parts.append(f"hidden={hidden}")
    if ffn is not None:
        parts.append(f"ffn={ffn}")
    return "[" + ",".join(parts) + "]"

# ---------- Baseline DAG (TP=8, PP=2) ----------
def build_baseline():
    dot = []
    dot.append("digraph Baseline {")
    dot.append("  rank_policy=TB;")
    dot.append("  node [shape=rectangle];")
    dot.append("")
    # Global constants
    B, S, H, Dk, FFN = 128, 128, 16, 64, 2048
    TP, PP = 8, 2
    L = 16
    # GPU layout: 2 stages × 8 TP ranks = 16 GPUs
    # Stage-0: GPU0..7, Stage-1: GPU8..15
    def gpu(stage, tp): return stage*TP + tp
    # Experts: 64 per layer → 4 per GPU
    experts_per_gpu = 64 // 16
    def expert_id(gpu_num, local_idx): return gpu_num*experts_per_gpu + local_idx
    # Input node
    dot.append(f'  Input [shape=ellipse, label="Input\\n{shape(B,S,hidden=H*Dk)}"];')
    prev = "Input"
    for layer in range(1, L+1):
        # ---- Attention across 8 GPUs (stage0 for layer<=8, stage1 for layer>8) ----
        stage = 0 if layer <= 8 else 1
        # QKV concat split
        for tp in range(TP):
            dot.append(f'  QKV_s{stage}_l{layer}_tp{tp} [label="QKV_linear\\nGPU{gpu(stage,tp)}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  {prev} -> QKV_s{stage}_l{layer}_tp{tp};')
        # Split heads
        for tp in range(TP):
            dot.append(f'  SplitHeads_s{stage}_l{layer}_tp{tp} [label="SplitHeads\\nGPU{gpu(stage,tp)}\\n{shape(B,S,H,Dk)}"];')
            dot.append(f'  QKV_s{stage}_l{layer}_tp{tp} -> SplitHeads_s{stage}_l{layer}_tp{tp};')
        # All-to-all for attention (simplified as one comm node)
        dot.append(f'  A2A_s{stage}_l{layer} [shape=ellipse, label="AllToAll\\nstage{stage}\\n{shape(B,S,H,Dk)}"];')
        for tp in range(TP):
            dot.append(f'  SplitHeads_s{stage}_l{layer}_tp{tp} -> A2A_s{stage}_l{layer};')
        # Attention compute
        for tp in range(TP):
            dot.append(f'  Attn_s{stage}_l{layer}_tp{tp} [label="Attention\\nGPU{gpu(stage,tp)}\\n{shape(B,S,H,Dk)}"];')
            dot.append(f'  A2A_s{stage}_l{layer} -> Attn_s{stage}_l{layer}_tp{tp};')
        # All-reduce after attention
        dot.append(f'  AR_attn_s{stage}_l{layer} [shape=ellipse, label="AllReduce\\nstage{stage}\\n{shape(B,S,H,Dk)}"];')
        for tp in range(TP):
            dot.append(f'  Attn_s{stage}_l{layer}_tp{tp} -> AR_attn_s{stage}_l{layer};')
        # O projection
        for tp in range(TP):
            dot.append(f'  O_s{stage}_l{layer}_tp{tp} [label="O_linear\\nGPU{gpu(stage,tp)}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  AR_attn_s{stage}_l{layer} -> O_s{stage}_l{layer}_tp{tp};')
        # Residual add (needs both O and residual input)
        for tp in range(TP):
            dot.append(f'  ResAttn_s{stage}_l{layer}_tp{tp} [shape=parallelogram, label="ResAdd\\nGPU{gpu(stage,tp)}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  O_s{stage}_l{layer}_tp{tp} -> ResAttn_s{stage}_l{layer}_tp{tp};')
            dot.append(f'  {prev} -> ResAttn_s{stage}_l{layer}_tp{tp};')
        # ---- MoE: 4 experts per GPU ----
        # Gate (on each GPU)
        for tp in range(TP):
            dot.append(f'  Gate_s{stage}_l{layer}_tp{tp} [label="Gate\\nGPU{gpu(stage,tp)}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  ResAttn_s{stage}_l{layer}_tp{tp} -> Gate_s{stage}_l{layer}_tp{tp};')
        # Expert compute (4 per GPU)
        for tp in range(TP):
            for e in range(experts_per_gpu):
                g = gpu(stage,tp)
                eid = expert_id(g, e)
                dot.append(f'  Expert{eid}_s{stage}_l{layer}_tp{tp} [label="Expert{eid}\\nGPU{g}\\n{shape(B,S,ffn=FFN)}"];')
                # dashed line from gate to expert
                dot.append(f'  Gate_s{stage}_l{layer}_tp{tp} -> Expert{eid}_s{stage}_l{layer}_tp{tp} [style=dashed];')
        # All-reduce across experts (within stage)
        dot.append(f'  AR_exp_s{stage}_l{layer} [shape=ellipse, label="AllReduceExperts\\nstage{stage}\\n{shape(B,S,hidden=H*Dk)}"];')
        for tp in range(TP):
            for e in range(experts_per_gpu):
                eid = expert_id(gpu(stage,tp), e)
                dot.append(f'  Expert{eid}_s{stage}_l{layer}_tp{tp} -> AR_exp_s{stage}_l{layer};')
        # Residual add after MoE
        for tp in range(TP):
            dot.append(f'  ResMOE_s{stage}_l{layer}_tp{tp} [shape=parallelogram, label="ResAdd\\nGPU{gpu(stage,tp)}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  AR_exp_s{stage}_l{layer} -> ResMOE_s{stage}_l{layer}_tp{tp};')
            dot.append(f'  ResAttn_s{stage}_l{layer}_tp{tp} -> ResMOE_s{stage}_l{layer}_tp{tp};')
        prev = f'ResMOE_s{stage}_l{layer}_tp0'  # arbitrary anchor
    # Final output
    dot.append(f'  Output [shape=ellipse, label="Output\\n{shape(B,S,hidden=H*Dk)}"];')
    dot.append(f'  {prev} -> Output;')
    dot.append("}")
    return "\n".join(dot)

# ---------- Proposed DAG (EP=16) ----------
def build_proposed():
    dot = []
    dot.append("digraph Proposed {")
    dot.append("  rank_policy=TB;")
    dot.append("  node [shape=rectangle];")
    dot.append("")
    B, S, H, Dk, FFN = 128, 128, 16, 64, 2048
    EP = 16
    L = 16
    # Each GPU holds exactly one expert per layer
    def gpu(e): return e % EP
    # Input node
    dot.append(f'  Input [shape=ellipse, label="Input\\n{shape(B,S,hidden=H*Dk)}"];')
    prev = "Input"
    for layer in range(1, L+1):
        # ---- Attention (replicated on all 16 GPUs) ----
        # QKV
        for g in range(EP):
            dot.append(f'  QKV_l{layer}_gpu{g} [label="QKV_linear\\nGPU{g}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  {prev} -> QKV_l{layer}_gpu{g};')
        # Split heads
        for g in range(EP):
            dot.append(f'  SplitHeads_l{layer}_gpu{g} [label="SplitHeads\\nGPU{g}\\n{shape(B,S,H,Dk)}"];')
            dot.append(f'  QKV_l{layer}_gpu{g} -> SplitHeads_l{layer}_gpu{g};')
        # All-to-all
        dot.append(f'  A2A_l{layer} [shape=ellipse, label="AllToAll\\nall16GPUs\\n{shape(B,S,H,Dk)}"];')
        for g in range(EP):
            dot.append(f'  SplitHeads_l{layer}_gpu{g} -> A2A_l{layer};')
        # Attention
        for g in range(EP):
            dot.append(f'  Attn_l{layer}_gpu{g} [label="Attention\\nGPU{g}\\n{shape(B,S,H,Dk)}"];')
            dot.append(f'  A2A_l{layer} -> Attn_l{layer}_gpu{g};')
        # All-reduce
        dot.append(f'  AR_attn_l{layer} [shape=ellipse, label="AllReduce\\nall16GPUs\\n{shape(B,S,H,Dk)}"];')
        for g in range(EP):
            dot.append(f'  Attn_l{layer}_gpu{g} -> AR_attn_l{layer};')
        # O proj
        for g in range(EP):
            dot.append(f'  O_l{layer}_gpu{g} [label="O_linear\\nGPU{g}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  AR_attn_l{layer} -> O_l{layer}_gpu{g};')
        # ResAdd
        for g in range(EP):
            dot.append(f'  ResAttn_l{layer}_gpu{g} [shape=parallelogram, label="ResAdd\\nGPU{g}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  O_l{layer}_gpu{g} -> ResAttn_l{layer}_gpu{g};')
            dot.append(f'  {prev} -> ResAttn_l{layer}_gpu{g};')
        # ---- MoE: one expert per GPU ----
        # Gate (each GPU computes its own gate)
        for g in range(EP):
            dot.append(f'  Gate_l{layer}_gpu{g} [label="Gate\\nGPU{g}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  ResAttn_l{layer}_gpu{g} -> Gate_l{layer}_gpu{g};')
        # Expert (only one per GPU)
        for g in range(EP):
            dot.append(f'  Expert_l{layer}_gpu{g} [label="Expert_l{layer}\\nGPU{g}\\n{shape(B,S,ffn=FFN)}"];')
            dot.append(f'  Gate_l{layer}_gpu{g} -> Expert_l{layer}_gpu{g} [style=dashed];')
        # All-reduce across experts
        dot.append(f'  AR_exp_l{layer} [shape=ellipse, label="AllReduceExperts\\nall16GPUs\\n{shape(B,S,hidden=H*Dk)}"];')
        for g in range(EP):
            dot.append(f'  Expert_l{layer}_gpu{g} -> AR_exp_l{layer};')
        # ResAdd after MoE
        for g in range(EP):
            dot.append(f'  ResMOE_l{layer}_gpu{g} [shape=parallelogram, label="ResAdd\\nGPU{g}\\n{shape(B,S,hidden=H*Dk)}"];')
            dot.append(f'  AR_exp_l{layer} -> ResMOE_l{layer}_gpu{g};')
            dot.append(f'  ResAttn_l{layer}_gpu{g} -> ResMOE_l{layer}_gpu{g};')
        prev = f'ResMOE_l{layer}_gpu0'
    # Output
    dot.append(f'  Output [shape=ellipse, label="Output\\n{shape(B,S,hidden=H*Dk)}"];')
    dot.append(f'  {prev} -> Output;')
    dot.append("}")
    return "\n".join(dot)

if __name__ == "__main__":
    out_dir = "../outputs/2025-12-01-09-41-17"
    os.makedirs(out_dir, exist_ok=True)
    
    # Baseline
    bl_dot = build_baseline()
    bl_path = os.path.join(out_dir, "baseline_complete.dot")
    with open(bl_path, "w") as f:
        f.write(bl_dot)
    
    # Proposed
    pr_dot = build_proposed()
    pr_path = os.path.join(out_dir, "proposed.dot")
    with open(pr_path, "w") as f:
        f.write(pr_dot)
    
    # Generate SVG files
    try:
        subprocess.run(["dot", "-Tsvg", bl_path, "-o", os.path.join(out_dir, "baseline_complete.svg")], check=True)
        print(f"Generated: {os.path.join(out_dir, 'baseline_complete.svg')}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating baseline SVG: {e}")
    
    try:
        subprocess.run(["dot", "-Tsvg", pr_path, "-o", os.path.join(out_dir, "proposed.svg")], check=True)
        print(f"Generated: {os.path.join(out_dir, 'proposed.svg')}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating proposed SVG: {e}")
    
    print("Wrote:")
    print(f"  {bl_path}")
    print(f"  {pr_path}")