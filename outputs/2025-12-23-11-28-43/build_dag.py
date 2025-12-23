#!/usr/bin/env python3
"""
Generate a complete operator-level DAG for the proposed cross-node expert-parallel
LLM deployment (16 layers, 64 experts, 16 GPUs, EP=16, TP=1, PP=1).

The DAG is drawn layer-by-layer; inside each layer we show:
  1. Token router (gate) – parallelogram, dashed edge to selected experts
  2. Expert split node – parallelogram
  3. One compute rectangle per expert (exact GPU id labelled)
  4. Expert aggregate node – parallelogram
  5. Attention operators (QKV proj, attn core, out proj) – rectangles
  6. Communication ellipses where tokens move GPU→GPU

Every node carries:
  INPUT  DIMENSION: batch=128, seq=128, features...
  OUTPUT DIMENSION: batch=128, seq=128, features...
"""

import os
import subprocess

# ---------- configuration taken from deployment_configuration.json ----------
L           = 16          # layers
E_PER_LAYER = 64          # experts per layer
GPUS        = 16          # GPUs
B           = 128         # batch size
S           = 128         # sequence length
H           = 1024        # hidden size (token dimension)
N_HEADS     = 16          # num attention heads
D_HEAD      = 64          # head dimension
FFN_HIDDEN  = 2048        # expert ffn hidden size
# ---------------------------------------------------------------------------

# GPU assignment table: gpu_id -> list of (layer, expert_id) it owns
# Built from the rotational mapping given in deployment_configuration.json
GPU_EXPERTS = {}
for gpu in range(GPUS):
    GPU_EXPERTS[gpu] = []
    for ly in range(L):
        # rotational assignment: expert = (gpu + ly*4) % 64
        expert = (gpu + ly * 4) % E_PER_LAYER
        GPU_EXPERTS[gpu].append((ly, expert))

# Utility to dimension strings (no brackets to avoid DOT parsing issues)
def dim_attn_input():
    return f"B={B}, S={S}, H={H}"
def dim_attn_output():
    return f"B={B}, S={S}, H={H}"
def dim_ffn_input():
    return f"B={B}, S={S}, H={H}"
def dim_ffn_hidden():
    return f"B={B}, S={S}, FFN={FFN_HIDDEN}"
def dim_ffn_output():
    return f"B={B}, S={S}, H={H}"

# Build DOT
dot_lines = [
    "digraph G {",
    "rankdir=TB;",
    "node [fontsize=10];",
    "// Styles",
    "compute [shape=rectangle, style=filled, fillcolor=lightblue];",
    "comm    [shape=ellipse,      style=filled, fillcolor=yellow];",
    "router  [shape=parallelogram, style=filled, fillcolor=lightgreen];",
    ""
]

def add_node(id_, label, shape, gpu=None, **kwargs):
    # strip brackets from label as well
    label_clean = label.replace("[", "").replace("]", "")
    attr = [f'label="{label_clean}"']
    if shape:
        attr.append(f'shape={shape}')
    if gpu is not None:
        attr.append(f'GPU={gpu}')
    for k, v in kwargs.items():
        # strip any remaining brackets from value
        v_clean = v.replace("[", "").replace("]", "")
        attr.append(f'{k}="{v_clean}"')
    dot_lines.append(f'{id_} [{",".join(attr)}];')

def add_edge(a, b, style=None, **kwargs):
    attr = []
    if style:
        attr.append(f'style={style}')
    for k, v in kwargs.items():
        attr.append(f'{k}="{v}"')
    if attr:
        dot_lines.append(f'{a} -> {b} [{",".join(attr)}];")
    else:
        dot_lines.append(f'{a} -> {b};')

# ---------- Input node ----------
add_node("input", f"INPUT\\n{dim_attn_input()}", "octagon", INPUT=dim_attn_input(), OUTPUT=dim_attn_output())
prev = "input"

# ---------- Loop over layers ----------
for ly in range(L):
    layer_prefix = f"L{ly}"
    # ---- Attention sub-graph ----
    # QKV proj (single GPU since TP=1)
    qkv_id = f"{layer_prefix}_qkv"
    add_node(qkv_id, f"QKV Proj\\n{dim_attn_input()}->{dim_attn_input()}", "rectangle", gpu=0,
             INPUT=dim_attn_input(), OUTPUT=dim_attn_input())
    add_edge(prev, qkv_id)
    prev = qkv_id

    # Attention core
    attn_id = f"{layer_prefix}_attn"
    add_node(attn_id, f"Attention Core\\n{dim_attn_input()}->{dim_attn_output()}", "rectangle", gpu=0,
             INPUT=dim_attn_input(), OUTPUT=dim_attn_output())
    add_edge(prev, attn_id)
    prev = attn_id

    # Out proj
    outproj_id = f"{layer_prefix}_outproj"
    add_node(outproj_id, f"Out Proj\\n{dim_attn_input()}->{dim_attn_output()}", "rectangle", gpu=0,
             INPUT=dim_attn_input(), OUTPUT=dim_attn_output())
    add_edge(prev, outproj_id)
    prev = outproj_id

    # ---- MoE sub-graph ----
    # Router (gate) – parallelogram
    gate_id = f"{layer_prefix}_gate"
    add_node(gate_id, f"Gate (Top-K)\\n{dim_ffn_input()}", "parallelogram", gpu=0,
             INPUT=dim_ffn_input(), OUTPUT=dim_ffn_input())
    add_edge(prev, gate_id)

    # Split node – parallelogram
    split_id = f"{layer_prefix}_split"
    add_node(split_id, f"Token Split\\n{dim_ffn_input()}", "parallelogram", gpu=0,
             INPUT=dim_ffn_input(), OUTPUT=dim_ffn_input())
    add_edge(gate_id, split_id)

    # Expert compute nodes – rectangles, placed on their GPU
    expert_nodes = []
    for expert in range(E_PER_LAYER):
        # find which GPU owns this (ly, expert)
        owner_gpu = None
        for g, assigns in GPU_EXPERTS.items():
            if (ly, expert) in assigns:
                owner_gpu = g
                break
        ex_id = f"{layer_prefix}_expert{expert}"
        add_node(ex_id, f"Expert {expert}\\n{dim_ffn_input()}->{dim_ffn_output()}", "rectangle", gpu=owner_gpu,
                 INPUT=dim_ffn_input(), OUTPUT=dim_ffn_output())
        # dashed edge from gate to expert (selection)
        add_edge(gate_id, ex_id, style="dashed")
        # solid edge from split to expert
        add_edge(split_id, ex_id)
        expert_nodes.append(ex_id)

    # Aggregate node – parallelogram
    agg_id = f"{layer_prefix}_agg"
    add_node(agg_id, f"Expert Agg\\n{dim_ffn_output()}", "parallelogram", gpu=0,
             INPUT=dim_ffn_output(), OUTPUT=dim_ffn_output())
    for ex in expert_nodes:
        add_edge(ex, agg_id)

    # All-reduce communication (experts → agg) shown as comm ellipse
    comm_id = f"{layer_prefix}_expert_ar"
    add_node(comm_id, f"AllReduce\\n{dim_ffn_output()}", "ellipse", gpu=0,
             INPUT=dim_ffn_output(), OUTPUT=dim_ffn_output())
    add_edge(agg_id, comm_id)
    prev = comm_id

# ---------- Output node ----------
add_node("output", f"OUTPUT\\n{dim_attn_output()}", "octagon", INPUT=dim_attn_input(), OUTPUT=dim_attn_output())
add_edge(prev, "output")

dot_lines.append("}")

# ---------- write DOT ----------
dot_path = os.path.join("../outputs/2025-12-23-11-28-43", "llm_ep16_model.dot")
with open(dot_path, "w") as f:
    f.write("\n".join(dot_lines))

# ---------- generate SVG ----------
svg_path = os.path.join("../outputs/2025-12-23-11-28-43", "llm_ep16_model.svg")
subprocess.check_call(["dot", "-Tsvg", dot_path, "-o", svg_path])

print("DOT saved to:", dot_path)
print("SVG saved to:", svg_path)