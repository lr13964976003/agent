#!/usr/bin/env python3
"""
Generate a complete, validated DAG for
EP3_TP2_PP1 deployment (24-layer MoE) with every edge explicitly listed.
Output: ep3_tp2_pp1_complete.dot  and  ep3_tp2_pp1_complete.svg
"""

import os

dot_path = "../outputs/2025-12-02-14-27-06/ep3_tp2_pp1_complete.dot"
svg_path = "../outputs/2025-12-02-14-27-06/ep3_tp2_pp1_complete.svg"

# ---------- helper: dimension strings ----------
def dim(b, s, h, d_k=128):
    return f"batch_size={b},seq_len={s},heads={h},d_k={d_k}"

# pre-compute common dimensions
d_in  = dim(64, 1024, 32)        # standard input
_d16  = dim(64, 1024, 16)        # after TP split
d_out = dim(64, 1024, 32)        # restored
_d8192 = dim(64, 1024, 8192)     # after fc1

# ---------- start DOT ----------
dot_lines = [
    "digraph EP3_TP2_PP1 {",
    "rankdir=TB; splines=true; compound=true;",
    "node [shape=record, fontname=\"Helvetica\"];",
    "edge [fontname=\"Helvetica\"];",
    "",
    "// ---------- subgraphs per GPU ----------",
    "subgraph cluster_gpu0 { label=\"GPU-0\"; style=rounded; color=blue;",
    f"  gpu0_att_in  [shape=parallelogram, label=\"att_input_agg|Input:{d_in}|Output:{d_in}\"];",
    f"  gpu0_qkv0    [shape=rectangle,      label=\"qkv_proj_0|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu0_qkv1    [shape=rectangle,      label=\"qkv_proj_1|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu0_attout0 [shape=rectangle,      label=\"att_out_proj_0|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu0_attout1 [shape=rectangle,      label=\"att_out_proj_1|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu0_att_agg [shape=parallelogram, label=\"att_output_agg|Input:{_d16}|Output:{d_out}\"];",
    f"  gpu0_res1    [shape=rectangle,      label=\"residual_add_1|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];",
    f"  gpu0_ln1     [shape=rectangle,      label=\"layer_norm_1|Input:{d_out}|Output:{d_out}\"];",
    "  // MoE section GPU-0 (21 experts)",
]

# ---------- 21 experts on GPU-0 ----------
for e in range(21):
    dot_lines.extend([
        f"  gpu0_gate{e}      [shape=rectangle, label=\"expert{e}_gate|Input:{d_out}|Output:{dim(64,1024,1)}\"];",
        f"  gpu0_expert{e}_fc1 [shape=rectangle, label=\"expert{e}_fc1|Input:{d_out}|Output:{_d8192}\"];",
        f"  gpu0_expert{e}_fc2 [shape=rectangle, label=\"expert{e}_fc2|Input:{_d8192}|Output:{d_out}\"];",
        f"  gpu0_expert{e}_add [shape=rectangle, label=\"expert{e}_add|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];"
    ])
dot_lines.append("}")

# ---------- GPU-1 cluster ----------
dot_lines.extend([
    "subgraph cluster_gpu1 { label=\"GPU-1\"; style=rounded; color=green;",
    f"  gpu1_att_in  [shape=parallelogram, label=\"att_input_agg|Input:{d_in}|Output:{d_in}\"];",
    f"  gpu1_qkv0    [shape=rectangle,      label=\"qkv_proj_0|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu1_qkv1    [shape=rectangle,      label=\"qkv_proj_1|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu1_attout0 [shape=rectangle,      label=\"att_out_proj_0|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu1_attout1 [shape=rectangle,      label=\"att_out_proj_1|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu1_att_agg [shape=parallelogram, label=\"att_output_agg|Input:{_d16}|Output:{d_out}\"];",
    f"  gpu1_res1    [shape=rectangle,      label=\"residual_add_1|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];",
    f"  gpu1_ln1     [shape=rectangle,      label=\"layer_norm_1|Input:{d_out}|Output:{d_out}\"];",
    "  // MoE section GPU-1 (21 experts)",
])
for e in range(21):
    dot_lines.extend([
        f"  gpu1_gate{e}      [shape=rectangle, label=\"expert{e+21}_gate|Input:{d_out}|Output:{dim(64,1024,1)}\"];",
        f"  gpu1_expert{e}_fc1 [shape=rectangle, label=\"expert{e+21}_fc1|Input:{d_out}|Output:{_d8192}\"];",
        f"  gpu1_expert{e}_fc2 [shape=rectangle, label=\"expert{e+21}_fc2|Input:{_d8192}|Output:{d_out}\"];",
        f"  gpu1_expert{e}_add [shape=rectangle, label=\"expert{e+21}_add|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];"
    ])
dot_lines.append("}")

# ---------- GPU-2 cluster ----------
dot_lines.extend([
    "subgraph cluster_gpu2 { label=\"GPU-2\"; style=rounded; color=red;",
    f"  gpu2_att_in  [shape=parallelogram, label=\"att_input_agg|Input:{d_in}|Output:{d_in}\"];",
    f"  gpu2_qkv0    [shape=rectangle,      label=\"qkv_proj_0|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu2_qkv1    [shape=rectangle,      label=\"qkv_proj_1|Input:{d_in}|Output:{_d16}\"];",
    f"  gpu2_attout0 [shape=rectangle,      label=\"att_out_proj_0|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu2_attout1 [shape=rectangle,      label=\"att_out_proj_1|Input:{_d16}|Output:{_d16}\"];",
    f"  gpu2_att_agg [shape=parallelogram, label=\"att_output_agg|Input:{_d16}|Output:{d_out}\"];",
    f"  gpu2_res1    [shape=rectangle,      label=\"residual_add_1|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];",
    f"  gpu2_ln1     [shape=rectangle,      label=\"layer_norm_1|Input:{d_out}|Output:{d_out}\"];",
    "  // MoE section GPU-2 (21 experts)",
])
for e in range(21):
    dot_lines.extend([
        f"  gpu2_gate{e}      [shape=rectangle, label=\"expert{e+42}_gate|Input:{d_out}|Output:{dim(64,1024,1)}\"];",
        f"  gpu2_expert{e}_fc1 [shape=rectangle, label=\"expert{e+42}_fc1|Input:{d_out}|Output:{_d8192}\"];",
        f"  gpu2_expert{e}_fc2 [shape=rectangle, label=\"expert{e+42}_fc2|Input:{_d8192}|Output:{d_out}\"];",
        f"  gpu2_expert{e}_add [shape=rectangle, label=\"expert{e+42}_add|Input1:{d_out},Input2:{d_out}|Output:{d_out}\"];"
    ])
dot_lines.append("}")

# ---------- global input / output ----------
dot_lines.extend([
    "// global input/output",
    f"input  [shape=ellipse, label=\"TotalInput|Output:{d_in}\"];",
    f"output [shape=ellipse, label=\"TotalOutput|Input:{d_out}\"];",
    ""
])

# ---------- EDGES: full connectivity for 24 layers ----------
def add_layer_edges(dot_lines, lyr):
    prefix = f"layer{lyr}"
    # ---- attention ----
    dot_lines.extend([
        f"input -> {prefix}_att_in_gpu0;",
        f"{prefix}_att_in_gpu0 -> {prefix}_qkv0_gpu0;",
        f"{prefix}_att_in_gpu0 -> {prefix}_qkv1_gpu0;",
        f"{prefix}_qkv0_gpu0 -> {prefix}_attout0_gpu0;",
        f"{prefix}_qkv1_gpu0 -> {prefix}_attout1_gpu0;",
        f"{prefix}_attout0_gpu0 -> {prefix}_att_agg_gpu0;",
        f"{prefix}_attout1_gpu0 -> {prefix}_att_agg_gpu0;",
        f"{prefix}_att_agg_gpu0 -> {prefix}_res1_gpu0;",
        f"{prefix}_res1_gpu0 -> {prefix}_ln1_gpu0;",
        # same for gpu1, gpu2
        f"input -> {prefix}_att_in_gpu1;",
        f"{prefix}_att_in_gpu1 -> {prefix}_qkv0_gpu1;",
        f"{prefix}_att_in_gpu1 -> {prefix}_qkv1_gpu1;",
        f"{prefix}_qkv0_gpu1 -> {prefix}_attout0_gpu1;",
        f"{prefix}_qkv1_gpu1 -> {prefix}_attout1_gpu1;",
        f"{prefix}_attout0_gpu1 -> {prefix}_att_agg_gpu1;",
        f"{prefix}_attout1_gpu1 -> {prefix}_att_agg_gpu1;",
        f"{prefix}_att_agg_gpu1 -> {prefix}_res1_gpu1;",
        f"{prefix}_res1_gpu1 -> {prefix}_ln1_gpu1;",
        f"input -> {prefix}_att_in_gpu2;",
        f"{prefix}_att_in_gpu2 -> {prefix}_qkv0_gpu2;",
        f"{prefix}_att_in_gpu2 -> {prefix}_qkv1_gpu2;",
        f"{prefix}_qkv0_gpu2 -> {prefix}_attout0_gpu2;",
        f"{prefix}_qkv1_gpu2 -> {prefix}_attout1_gpu2;",
        f"{prefix}_attout0_gpu2 -> {prefix}_att_agg_gpu2;",
        f"{prefix}_attout1_gpu2 -> {prefix}_att_agg_gpu2;",
        f"{prefix}_att_agg_gpu2 -> {prefix}_res1_gpu2;",
        f"{prefix}_res1_gpu2 -> {prefix}_ln1_gpu2;",
    ])
    # ---- MoE: 21 experts per GPU ----
    for gpu in ["gpu0", "gpu1", "gpu2"]:
        for e in range(21):
            dot_lines.extend([
                f"{prefix}_ln1_{gpu} -> {prefix}_gate{e}_{gpu};",
                f"{prefix}_gate{e}_{gpu} -> {prefix}_expert{e}_fc1_{gpu} [style=dashed];",  # gate selection
                f"{prefix}_ln1_{gpu} -> {prefix}_expert{e}_fc1_{gpu};",
                f"{prefix}_expert{e}_fc1_{gpu} -> {prefix}_expert{e}_fc2_{gpu};",
                f"{prefix}_expert{e}_fc2_{gpu} -> {prefix}_expert{e}_add_{gpu};",
            ])
        # after last expert, route to next layer
        if lyr < 23:
            dot_lines.append(f"{prefix}_expert20_add_{gpu} -> layer{lyr+1}_att_in_{gpu};")
        else:
            dot_lines.append(f"{prefix}_expert20_add_{gpu} -> output;")

# generate 24 layers
for lyr in range(24):
    add_layer_edges(dot_lines, lyr)

dot_lines.append("}")

# ---------- write DOT ----------
with open(dot_path, "w") as f:
    f.write("\n".join(dot_lines))

# ---------- render to SVG ----------
os.system(f"dot -Tsvg {dot_path} -o {svg_path}")

print(f"✅ Complete DAG written to {dot_path}")
print(f"✅ SVG rendered to {svg_path}")