#!/usr/bin/env python3
import os, subprocess
os.makedirs("../outputs/2025-12-04-19-42-08", exist_ok=True)

dot_lines = ['digraph MoE_EP8_TP4_PP4_Layer {', 'rankdir=TB;', 'bgcolor=white;', 'node [fontsize=10, margin=0.04];']

def comp(id, gpu, op, inp, out):
    label = f"{op}\\nGPU {gpu}\\nIn: {inp}\\nOut: {out}"
    dot_lines.append(f'{id} [shape=rectangle, label="{label}"];')

def comm(id, label_, inp, out):
    label = f"{label_}\\nIn: {inp}\\nOut: {out}"
    dot_lines.append(f'{id} [shape=ellipse, label="{label}"];')

def route(id, label_, inp, out):
    label = f"{label_}\\nIn: {inp}\\nOut: {out}"
    dot_lines.append(f'{id} [shape=parallelogram, label="{label}"];')

comp("inp", "0-127", "LayerInput", "[b,s,h=1024]", "[b,s,h=1024]")

for gpu in range(4):
    comp(f"ln_{gpu}", gpu, "LayerNorm", "[b,s,h=1024]", "[b,s,h=1024]")
    dot_lines.append(f"inp -> ln_{gpu};")

for gpu in range(4):
    comp(f"qkv_{gpu}", gpu, "QKV_Proj(col)", "[b,s,h=1024]", "[b,s,3h/4=768]")
    dot_lines.append(f"ln_{gpu} -> qkv_{gpu};")

comm("ag_qkv", "AllGather(QKV)", "[b,s,3h/4=768]", "[b,s,3h=3072]")
for gpu in range(4):
    dot_lines.append(f"qkv_{gpu} -> ag_qkv;")
    dot_lines.append(f"ag_qkv -> {gpu}_qkv_full;")
    comp(f"{gpu}_qkv_full", gpu, "QKV_full", "[b,s,3h=3072]", "[b,s,3h=3072]")

for gpu in range(4):
    comp(f"attn_{gpu}", gpu, "Attention(heads=16/4=4)", "[b,s,3h=3072]", "[b,s,h=1024]")
    dot_lines.append(f"{gpu}_qkv_full -> attn_{gpu};")

for gpu in range(4):
    comp(f"attn_out_{gpu}", gpu, "AttnOut(row)", "[b,s,h=1024]", "[b,s,h/4=256]")
    dot_lines.append(f"attn_{gpu} -> attn_out_{gpu};")

comm("ar_attn", "AllReduce(AttnOut)", "[b,s,h/4=256]", "[b,s,h=1024]")
for gpu in range(4):
    dot_lines.append(f"attn_out_{gpu} -> ar_attn;")
    dot_lines.append(f"ar_attn -> {gpu}_attn_resid;")
    comp(f"{gpu}_attn_resid", gpu, "Attn+Residual", "[b,s,h=1024]", "[b,s,h=1024]")
    dot_lines.append(f"inp -> {gpu}_attn_resid [style=dashed, color=gray];")

for gpu in range(4):
    comp(f"ln_moe_{gpu}", gpu, "LayerNorm", "[b,s,h=1024]", "[b,s,h=1024]")
    dot_lines.append(f"{gpu}_attn_resid -> ln_moe_{gpu};")

for gpu in range(4):
    comp(f"gate_{gpu}", gpu, "Gate(h->64E)", "[b,s,h=1024]", "[b,s,64E=64*64]")
    dot_lines.append(f"ln_moe_{gpu} -> gate_{gpu};")

for gpu in range(4):
    comp(f"top2_{gpu}", gpu, "Top2ExpertSelect", "[b,s,64E]", "[b,s,2] (indices+weight)")
    dot_lines.append(f"gate_{gpu} -> top2_{gpu};")

comm("a2a_disp", "AllToAll(Dispatch)", "[b,s,h=1024] + indices", "[b,s,h=1024] (routed)")
for gpu in range(4):
    dot_lines.append(f"ln_moe_{gpu} -> a2a_disp;")
    dot_lines.append(f"top2_{gpu} -> a2a_disp [style=dashed];")
    dot_lines.append(f"a2a_disp -> {gpu}_disp;")
    comp(f"{gpu}_disp", gpu, "DispatchedTokens", "[b,s,h=1024]", "[b,s_per_expert,h=1024]")

for expert in range(8):
    gpu = expert // 2
    comp(f"exp{expert}_1", gpu, f"Expert{expert}MLP1(col)", "[b,s_e,h=1024]", "[b,s_e,ffn/4=1024]")
    dot_lines.append(f"{gpu}_disp -> exp{expert}_1;")
    comp(f"exp{expert}_2", gpu, f"Expert{expert}MLP2(row)", "[b,s_e,ffn/4=1024]", "[b,s_e,h/4=256]")
    dot_lines.append(f"exp{expert}_1 -> exp{expert}_2;")

for gpu in range(4):
    comm(f"ar_exp_{gpu}", "AllReduce(ExpertOut)", "[b,s_e,h/4=256]", "[b,s_e,h=1024]")
    for expert in range(8):
        if expert // 2 == gpu:
            dot_lines.append(f"exp{expert}_2 -> ar_exp_{gpu};")
    dot_lines.append(f"ar_exp_{gpu} -> {gpu}_exp_aggr;")
    route(f"{gpu}_exp_aggr", "AggregateExpertOut", "[b,s_e,h=1024]", "[b,s,h=1024]")

comm("a2a_comb", "AllToAll(Combine)", "[b,s,h=1024]", "[b,s,h=1024]")
for gpu in range(4):
    dot_lines.append(f"{gpu}_exp_aggr -> a2a_comb;")
    dot_lines.append(f"a2a_comb -> {gpu}_comb;")
    comp(f"{gpu}_comb", gpu, "CombinedTokens", "[b,s,h=1024]", "[b,s,h=1024]")

for gpu in range(4):
    comp(f"out_{gpu}", gpu, "MoE+Residual", "[b,s,h=1024]", "[b,s,h=1024]")
    dot_lines.append(f"{gpu}_attn_resid -> out_{gpu} [style=dashed, color=gray];")
    dot_lines.append(f"{gpu}_comb -> out_{gpu};")

route("output", "LayerOutput", "[b,s,h=1024]", "[b,s,h=1024]")
for gpu in range(4):
    dot_lines.append(f"out_{gpu} -> output;")

dot_lines.append("}")

dot = "\\n".join(dot_lines)
with open("../outputs/2025-12-04-19-42-08/moe_ep8_tp4_pp4_layer.dot", "w") as f:
    f.write(dot)

subprocess.run(["dot", "-Tsvg", "../outputs/2025-12-04-19-42-08/moe_ep8_tp4_pp4_layer.dot", "-o", "../outputs/2025-12-04-19-42-08/moe_ep8_tp4_pp4_layer.svg"], check=True)

print("DAG generated at:")
print("- DOT: ../outputs/2025-12-04-19-42-08/moe_ep8_tp4_pp4_layer.dot")
print("- SVG: ../outputs/2025-12-04-19-42-08/moe_ep8_tp4_pp4_layer.svg")