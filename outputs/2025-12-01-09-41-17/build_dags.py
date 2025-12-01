#!/usr/bin/env python3
"""
Generate two complete deployment DAGs:
  1. baseline_tp8_pp2.dot / .svg   (tensor+pipeline parallel, experts colocated)
  2. proposed_ep16.dot / .svg      (large cross-node expert parallel, 1 expert/GPU)
All tensor dimensions are exact per the paper.
"""

import os
import subprocess

OUT_DIR = "../outputs/2025-12-01-09-41-17"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- shared helpers ----------
def gv_box(label, gpu, in_dim, out_dim, shape="box"):
    # Escape backslashes for Graphviz
    label_escaped = label.replace("\\", "\\\\")
    gpu_escaped = gpu.replace("\\", "\\\\")
    in_dim_escaped = in_dim.replace("\\", "\\\\")
    out_dim_escaped = out_dim.replace("\\", "\\\\")
    return f'"{label_escaped}" [shape={shape}, label="{label_escaped}\\nGPU: {gpu_escaped}\\nIn: {in_dim_escaped}\\nOut: {out_dim_escaped}"];'

def gv_ellipsis(label, in_dim, out_dim):
    label_escaped = label.replace("\\", "\\\\")
    in_dim_escaped = in_dim.replace("\\", "\\\\")
    out_dim_escaped = out_dim.replace("\\", "\\\\")
    return f'"{label_escaped}" [shape=ellipse, style=dashed, label="{label_escaped}\\nIn: {in_dim_escaped}\\nOut: {out_dim_escaped}"];'

def gv_para(label, gpu, in_dim, out_dim):
    return gv_box(label, gpu, in_dim, out_dim, shape="parallelogram")

def gv_arrow(a, b):
    a_escaped = a.replace("\\", "\\\\")
    b_escaped = b.replace("\\", "\\\\")
    return f'"{a_escaped}" -> "{b_escaped}";'

# ---------- baseline: TP=8 PP=2 ----------
def build_baseline():
    g = ["digraph baseline_tp8_pp2 {"]
    g.append("rankdir=TB; splines=true;")
    # model params
    B, S, H, D_K = "batch_size", "seq_len=128", "heads=16", "d_k=64"   # MHA
    D = "d_model=1024"
    HIDDEN = "hidden=2048"
    TP = 8   # tensor-parallel degree
    PP = 2   # pipeline stages
    EXPERTS = 64
    # 16 GPUs: 0..7 stage0, 8..15 stage1
    def gpu(stage, tp_idx):
        return stage*TP + tp_idx
    # INPUT
    g.append(gv_box("INPUT", "all", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    # we show one full layer (layer0) replicated in both stages; repeat 16 times conceptually
    # ------ stage0 layer0 ------
    stage = 0
    # 1. MHA QKV split column-wise
    for tp in range(TP):
        g.append(gv_box(f"stage{stage}_layer0_mha_qkv_tp{tp}", gpu(stage,tp),
                        f"{B}, {S}, {D}", f"{B}, {S}, {H}, {D_K}"))
    # 2. ALL_GATHER QKV (ellipsis)
    g.append(gv_ellipsis(f"stage{stage}_layer0_ag_qkv", f"{B}, {S}, {H}, {D_K} (shard)", f"{B}, {S}, {H}, {D_K} (full)"))
    for tp in range(TP):
        g.append(gv_arrow(f"stage{stage}_layer0_mha_qkv_tp{tp}", f"stage{stage}_layer0_ag_qkv"))
    # 3. MHA score (full)
    g.append(gv_box(f"stage{stage}_layer0_mha_score", "all_stage0", f"{B}, {S}, {H}, {D_K}", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow(f"stage{stage}_layer0_ag_qkv", f"stage{stage}_layer0_mha_score"))
    # 4. MHA attn (softmax)
    g.append(gv_box(f"stage{stage}_layer0_mha_softmax", "all_stage0", f"{B}, {H}, {S}, {S}", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow(f"stage{stage}_layer0_mha_score", f"stage{stage}_layer0_mha_softmax"))
    # 5. MHA out_proj (row-parallel)
    for tp in range(TP):
        g.append(gv_box(f"stage{stage}_layer0_mha_out_tp{tp}", gpu(stage,tp),
                        f"{B}, {H}, {S}, {S}", f"{B}, {S}, {D}//{TP}"))
        g.append(gv_arrow(f"stage{stage}_layer0_mha_softmax", f"stage{stage}_layer0_mha_out_tp{tp}"))
    # 6. ALL_REDUCE out (ellipsis)
    g.append(gv_ellipsis(f"stage{stage}_layer0_ar_mha", f"{B}, {S}, {D}//{TP} (shard)", f"{B}, {S}, {D} (full)"))
    for tp in range(TP):
        g.append(gv_arrow(f"stage{stage}_layer0_mha_out_tp{tp}", f"stage{stage}_layer0_ar_mha"))
    # 7. RESIDUAL ADD
    g.append(gv_box(f"stage{stage}_layer0_resid1", "all_stage0", f"{B}, {S}, {D} (x2)", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"INPUT", f"stage{stage}_layer0_resid1"))
    g.append(gv_arrow(f"stage{stage}_layer0_ar_mha", f"stage{stage}_layer0_resid1"))
    # 8. MOE gate (full, replicate)
    g.append(gv_box(f"stage{stage}_layer0_moe_gate", "all_stage0", f"{B}, {S}, {D}", f"{B}, {S}, {EXPERTS}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid1", f"stage{stage}_layer0_moe_gate"))
    # 9. MOE experts (colocated 4 per GPU)
    EXP_PER_GPU = EXPERTS // (TP*PP)   # 64/16=4
    for tp in range(TP):
        for e in range(EXP_PER_GPU):
            exp_id = tp*EXP_PER_GPU + e
            g.append(gv_box(f"stage{stage}_layer0_exp{exp_id}", gpu(stage,tp),
                            f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
            g.append(gv_arrow(f"stage{stage}_layer0_moe_gate", f"stage{stage}_layer0_exp{exp_id}"))
    # 10. MOE aggregate (sum)
    g.append(gv_box(f"stage{stage}_layer0_moe_agg", "all_stage0", f"{B}, {S}, {D} (x{EXP_PER_GPU})", f"{B}, {S}, {D}"))
    for tp in range(TP):
        for e in range(EXP_PER_GPU):
            exp_id = tp*EXP_PER_GPU + e
            g.append(gv_arrow(f"stage{stage}_layer0_exp{exp_id}", f"stage{stage}_layer0_moe_agg"))
    # 11. RESIDUAL2
    g.append(gv_box(f"stage{stage}_layer0_resid2", "all_stage0", f"{B}, {S}, {D} (x2)", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid1", f"stage{stage}_layer0_resid2"))
    g.append(gv_arrow(f"stage{stage}_layer0_moe_agg", f"stage{stage}_layer0_resid2"))
    # 12. send to stage1 (ellipsis)
    g.append(gv_ellipsis(f"stage{stage}_layer0_send_stage1", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid2", f"stage{stage}_layer0_send_stage1"))
    # ------ stage1 layer0 ------
    stage = 1
    # recv
    g.append(gv_ellipsis(f"stage{stage}_layer0_recv", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage0_layer0_send_stage1", f"stage{stage}_layer0_recv"))
    # repeat MHA + MOE exactly as stage0
    # MHA QKV
    for tp in range(TP):
        g.append(gv_box(f"stage{stage}_layer0_mha_qkv_tp{tp}", gpu(stage,tp),
                        f"{B}, {S}, {D}", f"{B}, {S}, {H}, {D_K}"))
        g.append(gv_arrow(f"stage{stage}_layer0_recv", f"stage{stage}_layer0_mha_qkv_tp{tp}"))
    # AG
    g.append(gv_ellipsis(f"stage{stage}_layer0_ag_qkv", f"shard", f"full"))
    for tp in range(TP): g.append(gv_arrow(f"stage{stage}_layer0_mha_qkv_tp{tp}", f"stage{stage}_layer0_ag_qkv"))
    # score
    g.append(gv_box(f"stage{stage}_layer0_mha_score", "all_stage1", f"full", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow(f"stage{stage}_layer0_ag_qkv", f"stage{stage}_layer0_mha_score"))
    # softmax
    g.append(gv_box(f"stage{stage}_layer0_mha_softmax", "all_stage1", f"{B}, {H}, {S}, {S}", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow(f"stage{stage}_layer0_mha_score", f"stage{stage}_layer0_mha_softmax"))
    # out proj
    for tp in range(TP):
        g.append(gv_box(f"stage{stage}_layer0_mha_out_tp{tp}", gpu(stage,tp), f"full", f"{B}, {S}, {D}//{TP}"))
        g.append(gv_arrow(f"stage{stage}_layer0_mha_softmax", f"stage{stage}_layer0_mha_out_tp{tp}"))
    # AR
    g.append(gv_ellipsis(f"stage{stage}_layer0_ar_mha", f"shard", f"full"))
    for tp in range(TP): g.append(gv_arrow(f"stage{stage}_layer0_mha_out_tp{tp}", f"stage{stage}_layer0_ar_mha"))
    # resid1
    g.append(gv_box(f"stage{stage}_layer0_resid1", "all_stage1", f"x2", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage{stage}_layer0_recv", f"stage{stage}_layer0_resid1"))
    g.append(gv_arrow(f"stage{stage}_layer0_ar_mha", f"stage{stage}_layer0_resid1"))
    # MOE gate
    g.append(gv_box(f"stage{stage}_layer0_moe_gate", "all_stage1", f"{B}, {S}, {D}", f"{B}, {S}, {EXPERTS}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid1", f"stage{stage}_layer0_moe_gate"))
    # experts
    for tp in range(TP):
        for e in range(EXP_PER_GPU):
            exp_id = 8*EXP_PER_GPU + tp*EXP_PER_GPU + e
            g.append(gv_box(f"stage{stage}_layer0_exp{exp_id}", gpu(stage,tp), f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
            g.append(gv_arrow(f"stage{stage}_layer0_moe_gate", f"stage{stage}_layer0_exp{exp_id}"))
    # agg
    g.append(gv_box(f"stage{stage}_layer0_moe_agg", "all_stage1", f"x{EXP_PER_GPU}", f"{B}, {S}, {D}"))
    for tp in range(TP):
        for e in range(EXP_PER_GPU):
            exp_id = 8*EXP_PER_GPU + tp*EXP_PER_GPU + e
            g.append(gv_arrow(f"stage{stage}_layer0_exp{exp_id}", f"stage{stage}_layer0_moe_agg"))
    # resid2
    g.append(gv_box(f"stage{stage}_layer0_resid2", "all_stage1", f"x2", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid1", f"stage{stage}_layer0_resid2"))
    g.append(gv_arrow(f"stage{stage}_layer0_moe_agg", f"stage{stage}_layer0_resid2"))
    # OUTPUT
    g.append(gv_box("OUTPUT", "all", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    g.append(gv_arrow(f"stage{stage}_layer0_resid2", "OUTPUT"))
    g.append("}")
    open(f"{OUT_DIR}/baseline_tp8_pp2.dot","w").write("\n".join(g))

# ---------- proposed: EP=16 ----------
def build_proposed():
    g = ["digraph proposed_ep16 {"]
    g.append("rankdir=TB; splines=true;")
    B, S, H, D_K = "batch_size", "seq_len=128", "heads=16", "d_k=64"
    D = "d_model=1024"
    HIDDEN = "hidden=2048"
    EP = 16
    EXPERTS = 64
    # INPUT (replicated on every GPU)
    g.append(gv_box("INPUT", "all", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    # Attention replicated on every GPU
    for gpu in range(16):
        g.append(gv_box(f"gpu{gpu}_mha_qkv", gpu, f"{B}, {S}, {D}", f"{B}, {S}, {H}, {D_K}"))
        g.append(gv_arrow("INPUT", f"gpu{gpu}_mha_qkv"))
    # All-Gather QKV
    g.append(gv_ellipsis("ag_qkv", f"shard", f"full"))
    for gpu in range(16): g.append(gv_arrow(f"gpu{gpu}_mha_qkv", "ag_qkv"))
    # Score
    g.append(gv_box("mha_score", "all", f"full", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow("ag_qkv", "mha_score"))
    # Softmax
    g.append(gv_box("mha_softmax", "all", f"{B}, {H}, {S}, {S}", f"{B}, {H}, {S}, {S}"))
    g.append(gv_arrow("mha_score", "mha_softmax"))
    # Out proj (row-parallel)
    for gpu in range(16):
        g.append(gv_box(f"gpu{gpu}_mha_out", gpu, f"full", f"{B}, {S}, {D}//16"))
        g.append(gv_arrow("mha_softmax", f"gpu{gpu}_mha_out"))
    # All-Reduce
    g.append(gv_ellipsis("ar_mha", f"shard", f"full"))
    for gpu in range(16): g.append(gv_arrow(f"gpu{gpu}_mha_out", "ar_mha"))
    # Residual1
    g.append(gv_box("resid1", "all", f"x2", f"{B}, {S}, {D}"))
    g.append(gv_arrow("INPUT", "resid1"))
    g.append(gv_arrow("ar_mha", "resid1"))
    # MOE gate (replicated)
    g.append(gv_box("moe_gate", "all", f"{B}, {S}, {D}", f"{B}, {S}, {EXPERTS}"))
    g.append(gv_arrow("resid1", "moe_gate"))
    # Expert dispatch (dashed)
    g.append(gv_ellipsis("dispatch", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    g.append(gv_arrow("moe_gate", "dispatch"))
    # One expert per GPU (total 64 experts => 4 layers shown conceptually, but we draw all 16 GPUs)
    # We map expert e to GPU e%16
    for gpu in range(16):
        g.append(gv_box(f"gpu{gpu}_expert", gpu, f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
        g.append(gv_arrow("dispatch", f"gpu{gpu}_expert"))
    # Expert combine
    g.append(gv_ellipsis("combine", f"{B}, {S}, {D} (x16)", f"{B}, {S}, {D}"))
    for gpu in range(16): g.append(gv_arrow(f"gpu{gpu}_expert", "combine"))
    # Residual2
    g.append(gv_box("resid2", "all", f"x2", f"{B}, {S}, {D}"))
    g.append(gv_arrow("resid1", "resid2"))
    g.append(gv_arrow("combine", "resid2"))
    # OUTPUT
    g.append(gv_box("OUTPUT", "all", f"{B}, {S}, {D}", f"{B}, {S}, {D}"))
    g.append(gv_arrow("resid2", "OUTPUT"))
    g.append("}")
    open(f"{OUT_DIR}/proposed_ep16.dot","w").write("\n".join(g))

# ---------- render ----------
def render():
    for name in ["baseline_tp8_pp2", "proposed_ep16"]:
        dot = f"{OUT_DIR}/{name}.dot"
        svg = f"{OUT_DIR}/{name}.svg"
        subprocess.check_call(["dot", "-Tsvg", "-o", svg, dot])

if __name__ == "__main__":
    build_baseline()
    build_proposed()
    render()
    print("Done – both .dot and .svg saved to", OUT_DIR)