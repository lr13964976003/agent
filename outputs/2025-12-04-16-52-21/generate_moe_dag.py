#!/usr/bin/env python3

import os

def generate_moe_dag():
    """Generate complete MoE DAG with 64 experts across 64 GPUs"""
    
    dot_content = """digraph MoE_EP64_Layer {
    rankdir=TB;
    bgcolor=white;
    node [shape=record, fontname="Helvetica", fontsize=10];
    edge [fontname="Helvetica", fontsize=9];

    /* ---------- Input from previous layer (all GPUs hold full tensor) ---------- */
    Input [shape=box, label="Input\nGPU:all\nInput:[batch_size=?,seq_len=4096,hidden=4096]\nOutput:[batch_size=?,seq_len=4096,hidden=4096]"];

    /* ---------- Attention block (identical on every GPU, data-parallel) ---------- */
    subgraph cluster_attn {
        label="Attention (identical on all GPUs)";
        style=dashed;

        LN1 [shape=box, label="LayerNorm\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];
        QKV [shape=box, label="Linear QKV\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,12288]"];
        Reshape [shape=box, label="Reshape→[B,S,32,128]\nGPU:all\nInput:[B,S,12288]\nOutput:[B,S,32,128]"];
        Attn [shape=box, label="Attention(32 heads)\nGPU:all\nInput:[B,S,32,128]\nOutput:[B,S,32,128]"];
        Proj [shape=box, label="Linear Proj\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];
        Add1 [shape=box, label="Add\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];
        LN2 [shape=box, label="LayerNorm\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];
    }

    /* ---------- Gate network (identical on every GPU) ---------- */
    Gate [shape=box, label="Gate Linear(4096→64)\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,64]"];
    Softmax [shape=box, label="Softmax(top-k=2)\nGPU:all\nInput:[B,S,64]\nOutput:[B,S,64]"];

    /* ---------- Expert scatter (routing) – dashed ellipse ---------- */
    Scatter [shape=ellipse, style=dashed, label="All-to-All Scatter\nGPU:all→target\nInput:[B,S,4096]\nOutput:[tokens_per_expert,4096]"];

    /* ---------- Expert MLPs (one rectangle per GPU, only local expert) ---------- */
    subgraph cluster_experts {
        label="Experts (one per GPU)";
        style=solid;
"""

    # Generate all 64 expert nodes
    for i in range(64):
        dot_content += f'        Exp{i} [shape=box, label="Expert-{i} MLP\nGPU:{i}\nInput:[T{i},4096]\nOutput:[T{i},4096]"];\n'

    dot_content += """
    }

    /* ---------- Expert gather (aggregation) – dashed ellipse ---------- */
    Gather [shape=ellipse, style=dashed, label="All-to-All Gather\nGPU:target→all\nInput:[tokens_per_expert,4096]\nOutput:[B,S,4096]"];

    /* ---------- Weighted sum (identical on every GPU) ---------- */
    WeightedSum [shape=parallelogram, label="Weighted Sum\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];
    Add2 [shape=box, label="Add\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];

    /* ---------- Output to next layer ---------- */
    Output [shape=box, label="Output\nGPU:all\nInput:[B,S,4096]\nOutput:[B,S,4096]"];

    /* ---------- Edges ---------- */
    Input -> LN1;
    LN1 -> QKV;
    QKV -> Reshape;
    Reshape -> Attn;
    Attn -> Proj;
    Proj -> Add1;
    Add1 -> LN2;

    LN2 -> Gate;
    Gate -> Softmax;
    LN2 -> Scatter;
    Softmax -> Scatter;   /* gate controls scatter */

"""

    # Add edges from Scatter to all experts
    for i in range(64):
        dot_content += f'    Scatter -> Exp{i};\n'

    # Add edges from all experts to Gather
    for i in range(64):
        dot_content += f'    Exp{i} -> Gather;\n'

    dot_content += """
    Gather -> WeightedSum;
    Softmax -> WeightedSum;   /* provide weights */
    WeightedSum -> Add2;
    Add2 -> Output;
}"""

    return dot_content

def main():
    # Generate the DOT content
    dot_content = generate_moe_dag()
    
    # Write DOT file
    dot_file_path = "../outputs/2025-12-04-16-52-21/moe_ep64_complete.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dot_content)
    
    print(f"Generated DOT file: {dot_file_path}")
    
    # Generate SVG using graphviz
    svg_file_path = "../outputs/2025-12-04-16-52-21/moe_ep64_complete.svg"
    os.system(f"dot -Tsvg {dot_file_path} -o {svg_file_path}")
    print(f"Generated SVG file: {svg_file_path}")
    
    # Return paths for JSON output
    return {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path
    }

if __name__ == "__main__":
    result = main()
    print(f"Generated files: {result}")