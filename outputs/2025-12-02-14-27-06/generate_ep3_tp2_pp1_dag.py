#!/usr/bin/env python3
"""
Generate complete EP3_TP2_PP1 DAG (24 layers) with full operator detail,
GPU card boundaries, communication, and exact dimension tracking.
"""
import os
os.makedirs("../outputs/2025-12-02-14-27-06", exist_ok=True)

dot_path = "../outputs/2025-12-02-14-27-06/ep3_tp2_pp1_complete.dot"
svg_path = "../outputs/2025-12-02-14-27-06/ep3_tp2_pp1_complete.svg"

dot_lines = []
gpus = [0, 1, 2]
L = 24
B = 64
S = 1024
H = 32
d_k = 128
d_model = 4096
ffn_hidden = 16384
experts_total = 63
experts_per_gpu = 21  # 63/3
tp_size = 2

def gpu_color(g):
    return ["lightblue", "lightgreen", "lightyellow"][g]

# Helper to add node
def add_node(name, shape, gpu, in_dim, out_dim, style="solid"):
    color = gpu_color(gpu)
    dot_lines.append(
        f'"{name}" [shape={shape}, style=filled, fillcolor={color}, label="{name}\\nINPUT: {in_dim}\\nOUTPUT: {out_dim}", style={style}];'
    )

# Helper to add edge
def add_edge(a, b):
    dot_lines.append(f'"{a}" -> "{b}";')

# Begin digraph
dot_lines.append("digraph EP3_TP2_PP1 {")
dot_lines.append('rankdir=TB;')
dot_lines.append('node [fontsize=10];')
dot_lines.append('edge [fontsize=9];')

# Global input node
add_node("INPUT", "ellipse", 0, f"batch_size={B}, seq_len={S}, d_model={d_model}", f"batch_size={B}, seq_len={S}, d_model={d_model}")
prev_layer_output = "INPUT"

for layer in range(L):
    prefix = f"L{layer}"
    # ---------- Attention phase ----------
    # Input split for TP=2 (column parallel QKV)
    for g in [0, 1]:
        add_node(f"{prefix}_SplitQKV_g{g}", "ellipse", g,
                 f"batch_size={B}, seq_len={S}, d_model={d_model}",
                 f"batch_size={B}, seq_len={S}, d_model={d_model}")
        add_edge(prev_layer_output, f"{prefix}_SplitQKV_g{g}")

    # QKV proj column-parallel on GPU0/1
    for g in [0, 1]:
        add_node(f"{prefix}_QKV_g{g}", "rect", g,
                 f"batch_size={B}, seq_len={S}, d_model={d_model}",
                 f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}, tp=2")
        add_edge(f"{prefix}_SplitQKV_g{g}", f"{prefix}_QKV_g{g}")

    # All-reduce QKV across TP pair
    add_node(f"{prefix}_AllReduceQKV", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}, tp=2",
             f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}")
    add_edge(f"{prefix}_QKV_g0", f"{prefix}_AllReduceQKV")
    add_edge(f"{prefix}_QKV_g1", f"{prefix}_AllReduceQKV")

    # Attention scores + softmax (single node, done on GPU0)
    add_node(f"{prefix}_AttnScores", "rect", 0,
             f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}",
             f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}")
    add_edge(f"{prefix}_AllReduceQKV", f"{prefix}_AttnScores")

    # Attention output proj row-parallel on GPU0/1
    for g in [0, 1]:
        add_node(f"{prefix}_AttnOut_g{g}", "rect", g,
                 f"batch_size={B}, seq_len={S}, heads={H}, d_k={d_k}",
                 f"batch_size={B}, seq_len={S}, d_model={d_model//2}")
        add_edge(f"{prefix}_AttnScores", f"{prefix}_AttnOut_g{g}")

    # All-reduce attention output
    add_node(f"{prefix}_AllReduceAttnOut", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, d_model={d_model//2}, tp=2",
             f"batch_size={B}, seq_len={S}, d_model={d_model}")
    add_edge(f"{prefix}_AttnOut_g0", f"{prefix}_AllReduceAttnOut")
    add_edge(f"{prefix}_AttnOut_g1", f"{prefix}_AllReduceAttnOut")

    # Residual add 1 (needs two inputs: prev_layer_output and attention output)
    add_node(f"{prefix}_ResAdd1", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, d_model={d_model} (x2)",
             f"batch_size={B}, seq_len={S}, d_model={d_model}")
    add_edge(prev_layer_output, f"{prefix}_ResAdd1")
    add_edge(f"{prefix}_AllReduceAttnOut", f"{prefix}_ResAdd1")
    after_attn = f"{prefix}_ResAdd1"

    # ---------- Expert FFN phase ----------
    # Gate routing (on GPU0)
    add_node(f"{prefix}_Gate", "rect", 0,
             f"batch_size={B}, seq_len={S}, d_model={d_model}",
             f"batch_size={B}, seq_len={S}, num_experts={experts_total}")
    add_edge(after_attn, f"{prefix}_Gate")

    # Expert dispatch (comm ellipse)
    add_node(f"{prefix}_Dispatch", "ellipse", 0,
             f"batch_size={B}, seq_len={S}, num_experts={experts_total}",
             f"batch_size={B}, seq_len={S}, num_experts={experts_per_gpu}")
    add_edge(f"{prefix}_Gate", f"{prefix}_Dispatch")

    # Per-GPU expert processing (21 experts each)
    for g in gpus:
        add_node(f"{prefix}_Experts_g{g}", "rect", g,
                 f"batch_size={B}, seq_len={S}, num_experts={experts_per_gpu}, d_model={d_model}",
                 f"batch_size={B}, seq_len={S}, num_experts={experts_per_gpu}, d_model={d_model}")
        add_edge(f"{prefix}_Dispatch", f"{prefix}_Experts_g{g}")

    # Expert aggregate back
    add_node(f"{prefix}_Aggregate", "ellipse", 0,
             f"batch_size={B}, seq_len={S}, num_experts={experts_per_gpu}, gpu=0,1,2",
             f"batch_size={B}, seq_len={S}, d_model={d_model}")
    for g in gpus:
        add_edge(f"{prefix}_Experts_g{g}", f"{prefix}_Aggregate")

    # Now column-row TP on MLP within each expert output (still on each GPU)
    # But we already have d_model vectors per token, so we apply TP=2 column-row on the single MLP inside each expert
    # We model this as an extra TP step on GPU0/1 for the aggregated expert output
    for g in [0, 1]:
        add_node(f"{prefix}_MLP_fc1_g{g}", "rect", g,
                 f"batch_size={B}, seq_len={S}, d_model={d_model}",
                 f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden//2}")
        add_edge(f"{prefix}_Aggregate", f"{prefix}_MLP_fc1_g{g}")

    # All-reduce after fc1 (optional, but follows column-row strategy)
    add_node(f"{prefix}_AllReduceFC1", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden//2}, tp=2",
             f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden}")
    add_edge(f"{prefix}_MLP_fc1_g0", f"{prefix}_AllReduceFC1")
    add_edge(f"{prefix}_MLP_fc1_g1", f"{prefix}_AllReduceFC1")

    # GELU (on GPU0)
    add_node(f"{prefix}_GELU", "rect", 0,
             f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden}",
             f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden}")
    add_edge(f"{prefix}_AllReduceFC1", f"{prefix}_GELU")

    # fc2 row-parallel on GPU0/1
    for g in [0, 1]:
        add_node(f"{prefix}_MLP_fc2_g{g}", "rect", g,
                 f"batch_size={B}, seq_len={S}, ffn_hidden={ffn_hidden}",
                 f"batch_size={B}, seq_len={S}, d_model={d_model//2}")
        add_edge(f"{prefix}_GELU", f"{prefix}_MLP_fc2_g{g}")

    # All-reduce final MLP output
    add_node(f"{prefix}_AllReduceFC2", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, d_model={d_model//2}, tp=2",
             f"batch_size={B}, seq_len={S}, d_model={d_model}")
    add_edge(f"{prefix}_MLP_fc2_g0", f"{prefix}_AllReduceFC2")
    add_edge(f"{prefix}_MLP_fc2_g1", f"{prefix}_AllReduceFC2")

    # Residual add 2 (two inputs: after_attn and MLP output)
    add_node(f"{prefix}_ResAdd2", "parallelogram", 0,
             f"batch_size={B}, seq_len={S}, d_model={d_model} (x2)",
             f"batch_size={B}, seq_len={S}, d_model={d_model}")
    add_edge(after_attn, f"{prefix}_ResAdd2")
    add_edge(f"{prefix}_AllReduceFC2", f"{prefix}_ResAdd2")
    prev_layer_output = f"{prefix}_ResAdd2"

# Global output node
add_node("OUTPUT", "ellipse", 0,
         f"batch_size={B}, seq_len={S}, d_model={d_model}",
         f"batch_size={B}, seq_len={S}, d_model={d_model}")
add_edge(prev_layer_output, "OUTPUT")

dot_lines.append("}")

# Write DOT
with open(dot_path, "w") as f:
    f.write("\n".join(dot_lines))

# Generate SVG
os.system(f"dot -Tsvg {dot_path} -o {svg_path}")

print("DOT saved to:", dot_path)
print("SVG saved to:", svg_path)