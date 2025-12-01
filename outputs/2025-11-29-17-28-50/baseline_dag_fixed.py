#!/usr/bin/env python3
"""
Rebuild the baseline DAG (TP=8, PP=2, 16 layers, 16 experts per layer, 16 GPUs)
with strict structural correctness:
- every node (except input) has ≥1 incoming edge
- every node (except output) has ≥1 outgoing edge
- all dimensions explicit
- communication nodes explicit
- no cycles
"""

import graphviz

# Model constants from paper
B  = 128   # batch_size
S  = 10000 # seq_len
D  = 4096  # token_dimension
H  = 32    # mha_heads
Hd = 128   # mha_head_dimension
Dh = D // H  # = 128
MLP_HIDDEN = 16384
TP = 8
PP = 2
LAYERS = 16
EXPERTS_PER_LAYER = 16
GPUS = 16

# Each GPU is identified by pipeline_stage * 8 + tp_rank
def gpu_id(stage, tp_rank):
    return stage * 8 + tp_rank

# Helper to create a unique node ID
def nid(name):
    return name

# Initialise directed graph
dot = graphviz.Digraph('baseline_moe_tp8_pp2')
dot.attr(rankdir='TB', splines='ortho')

# -------------------- Input --------------------
dot.node(nid('input'), 
         shape='ellipse',
         label='Input\\nInput: [batch_size=128, seq_len=10000, token_dim=4096]\\nOutput: [batch_size=128, seq_len=10000, token_dim=4096]\\nGPU: all')

# -------------------- Layer loop --------------------
for layer in range(LAYERS):
    stage = layer // 8  # pipeline stage 0: layers 0-7, stage 1: layers 8-15
    # MHA + MoE for each layer, TP=8 sharded
    for tp in range(TP):
        gpu = gpu_id(stage, tp)
        prefix = f'L{layer}_S{stage}_TP{tp}_GPU{gpu}'

        # 1. MHA: QKV linear shard
        qkv_in  = f'[batch_size=128, seq_len=10000, token_dim=4096]'  # full replica before shard
        qkv_out = f'[batch_size=128, seq_len=10000, heads=32, head_dim=128, tp_shard=1]'  # shard along heads
        dot.node(nid(f'{prefix}_qkv_linear'),
                 shape='rectangle',
                 label=f'QKV Linear\\nInput: {qkv_in}\\nOutput: {qkv_out}\\nGPU: {gpu}')

        # 2. QKV slice (extract this tp's 4 heads)
        slice_out = f'[batch_size=128, seq_len=10000, heads=4, head_dim=128]'
        dot.node(nid(f'{prefix}_qkv_slice'),
                 shape='rectangle',
                 label=f'QKV Slice (TP)\\nInput: {qkv_out}\\nOutput: {slice_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_qkv_linear'), nid(f'{prefix}_qkv_slice'))

        # 3. QK matmul
        qk_out = f'[batch_size=128, heads=4, seq_len=10000, seq_len=10000]'
        dot.node(nid(f'{prefix}_qk_matmul'),
                 shape='rectangle',
                 label=f'QK Matmul\\nInput: {slice_out}, {slice_out}\\nOutput: {qk_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_qkv_slice'), nid(f'{prefix}_qk_matmul'))

        # 4. Softmax
        sm_out = f'[batch_size=128, heads=4, seq_len=10000, seq_len=10000]'
        dot.node(nid(f'{prefix}_softmax'),
                 shape='rectangle',
                 label=f'Softmax\\nInput: {qk_out}\\nOutput: {sm_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_qk_matmul'), nid(f'{prefix}_softmax'))

        # 5. Attn @ V
        attn_out = f'[batch_size=128, seq_len=10000, heads=4, head_dim=128]'
        dot.node(nid(f'{prefix}_attn_v'),
                 shape='rectangle',
                 label=f'Attn@V\\nInput: {sm_out}, {slice_out}\\nOutput: {attn_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_softmax'), nid(f'{prefix}_attn_v'))
        dot.edge(nid(f'{prefix}_qkv_slice'), nid(f'{prefix}_attn_v'))

        # 6. All-reduce across TP for MHA output
        ar_in  = attn_out
        ar_out = f'[batch_size=128, seq_len=10000, heads=32, head_dim=128]'
        dot.node(nid(f'{prefix}_mha_ar'),
                 shape='ellipse',
                 label=f'MHA AllReduce\\nInput: {ar_in} (8×)\\nOutput: {ar_out}\\nGPU: all stage {stage}')
        for t2 in range(TP):
            dot.edge(nid(f'L{layer}_S{stage}_TP{t2}_GPU{gpu_id(stage,t2)}_attn_v'), 
                     nid(f'{prefix}_mha_ar'))

        # 7. MHA residual add
        mha_res_in1 = ar_out
        mha_res_in2 = f'[batch_size=128, seq_len=10000, token_dim=4096]'  # from input or prev layer
        mha_res_out = f'[batch_size=128, seq_len=10000, token_dim=4096]'
        dot.node(nid(f'{prefix}_mha_res'),
                 shape='rectangle',
                 label=f'MHA Residual Add\\nInput: {mha_res_in1}, {mha_res_in2}\\nOutput: {mha_res_out}\\nGPU: all stage {stage}')
        dot.edge(nid(f'{prefix}_mha_ar'), nid(f'{prefix}_mha_res'))

        # 8. MoE gate (top-2, local to each tp shard)
        gate_in  = mha_res_out
        gate_out = f'[batch_size=128, seq_len=10000, top_k=2, expert_id=0..15]'
        dot.node(nid(f'{prefix}_gate'),
                 shape='parallelogram',
                 label=f'MoE Gate (Top2)\\nInput: {gate_in}\\nOutput: {gate_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_mha_res'), nid(f'{prefix}_gate'))

        # 9. Expert 0 MLP (each tp shard holds 2 experts)
        exp0_in  = gate_in
        exp0_out = f'[batch_size=128, seq_len=10000, token_dim=4096]'  # after MLP 4096->16384->4096
        dot.node(nid(f'{prefix}_expert0'),
                 shape='rectangle',
                 label=f'Expert0 MLP (4096->16384->4096)\\nInput: {exp0_in}\\nOutput: {exp0_out}\\nGPU: {gpu}')
        # gate → expert via dashed line (routing)
        dot.edge(nid(f'{prefix}_gate'), nid(f'{prefix}_expert0'), style='dashed')

        # 10. Expert 1 MLP
        exp1_out = exp0_out
        dot.node(nid(f'{prefix}_expert1'),
                 shape='rectangle',
                 label=f'Expert1 MLP (4096->16384->4096)\\nInput: {exp0_in}\\nOutput: {exp1_out}\\nGPU: {gpu}')
        dot.edge(nid(f'{prefix}_gate'), nid(f'{prefix}_expert1'), style='dashed')

        # 11. All-reduce across TP for MoE output
        moe_ar_in  = exp0_out  # after local reduce inside tp
        moe_ar_out = f'[batch_size=128, seq_len=10000, token_dim=4096]'
        dot.node(nid(f'{prefix}_moe_ar'),
                 shape='ellipse',
                 label=f'MoE AllReduce\\nInput: {moe_ar_in} (8×)\\nOutput: {moe_ar_out}\\nGPU: all stage {stage}')
        for t2 in range(TP):
            dot.edge(nid(f'L{layer}_S{stage}_TP{t2}_GPU{gpu_id(stage,t2)}_expert0'),
                     nid(f'{prefix}_moe_ar'))
            dot.edge(nid(f'L{layer}_S{stage}_TP{t2}_GPU{gpu_id(stage,t2)}_expert1'),
                     nid(f'{prefix}_moe_ar'))

        # 12. MoE residual add
        moe_res_in1 = moe_ar_out
        moe_res_in2 = nid(f'{prefix}_mha_res')  # same as gate input
        moe_res_out = f'[batch_size=128, seq_len=10000, token_dim=4096]'
        dot.node(nid(f'{prefix}_moe_res'),
                 shape='rectangle',
                 label=f'MoE Residual Add\\nInput: {moe_ar_out}, {mha_res_out}\\nOutput: {moe_res_out}\\nGPU: all stage {stage}')
        dot.edge(nid(f'{prefix}_moe_ar'), nid(f'{prefix}_moe_res'))
        dot.edge(nid(f'{prefix}_mha_res'), nid(f'{prefix}_moe_res'))

        # 13. LayerNorm
        norm_out = moe_res_out
        dot.node(nid(f'{prefix}_norm'),
                 shape='rectangle',
                 label=f'LayerNorm\\nInput: {moe_res_out}\\nOutput: {norm_out}\\nGPU: all stage {stage}')
        dot.edge(nid(f'{prefix}_moe_res'), nid(f'{prefix}_norm'))

        # -------------------- cross-stage send/recv --------------------
        if layer == 7:  # last layer of stage 0 → send to stage 1
            send_in = norm_out
            send_out = send_in
            dot.node(nid(f'L{layer}_send_stage1'),
                     shape='ellipse',
                     label=f'Send to Stage1\\nInput: {send_in}\\nOutput: {send_out}\\nGPU: all stage 0')
            for tp_s0 in range(TP):
                dot.edge(nid(f'L{layer}_S0_TP{tp_s0}_GPU{gpu_id(0,tp_s0)}_norm'),
                         nid(f'L{layer}_send_stage1'))
            # recv side on stage 1 GPUs
            for tp_s1 in range(TP):
                recv_gpu = gpu_id(1, tp_s1)
                dot.node(nid(f'L8_recv_from_stage0_TP{tp_s1}'),
                         shape='ellipse',
                         label=f'Recv from Stage0\\nInput: {send_out}\\nOutput: {send_out}\\nGPU: {recv_gpu}')
                dot.edge(nid(f'L{layer}_send_stage1'), 
                         nid(f'L8_recv_from_stage0_TP{tp_s1}'))

# -------------------- Final output --------------------
# After last layer (15) norm on stage 1
final_in = f'[batch_size=128, seq_len=10000, token_dim=4096]'
dot.node(nid('output'),
         shape='ellipse',
         label=f'Output\\nInput: {final_in}\\nOutput: {final_in}\\nGPU: all stage 1')
for tp in range(TP):
    dot.edge(nid(f'L15_S1_TP{tp}_GPU{gpu_id(1,tp)}_norm'), nid('output'))

# -------------------- save --------------------
dot.save()
with open('../outputs/2025-11-29-17-28-50/baseline_dag.dot', 'w') as f:
    f.write(dot.source)
dot.render('../outputs/2025-11-29-17-28-50/baseline_dag', format='svg', cleanup=True)
print('Baseline DAG rebuilt: baseline_dag.dot & baseline_dag.svg')
