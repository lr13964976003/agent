#!/usr/bin/env python3
"""
Optimal Parallel Strategy Generator for Llama3-70B on 8×H100
Produces a deployment plan JSON and a human-readable report.
"""
import json
import math
from pathlib import Path

OUT_DIR = Path("../outputs/2025-12-23-14-09-23")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Inputs ------------------
hardware = {
    "total_gpus": 8,
    "gpu_memory_gb": 80,
    "intra_node_bw_gbps": 400,
    "nvlink_bw_gbps": 900
}
model = {
    "num_layers": 80,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "weights_gb": 140,          # fp16
    "kv_cache_kb_per_token": 1.0,
    "activation_kb_per_token": 0.5
}
slo = {
    "decode_p50_ms": 50,
    "decode_p99_ms": 100,
    "prefill_p50_ms": 500,
    "prefill_p99_ms": 1000,
    "max_batch": 64,
    "max_seqs": 128,
    "max_tokens": 8192,
    "target_rps": 8,
    "max_gpu_mem_pct": 85,
    "gpu_balance_eps": 0.05
}

# ------------------ Helpers ------------------
def tp_memory_weights(tp: int) -> float:
    """GB per GPU for model weights under TP"""
    return model["weights_gb"] / tp

def kv_cache_memory(seq_len: int, batch: int, tp: int) -> float:
    """GB per GPU for KV cache under TP"""
    total_tokens = seq_len * batch
    bytes_per_token = model["kv_cache_kb_per_token"] * 1024   # already in bytes
    total_bytes = total_tokens * bytes_per_token
    total_gb = total_bytes / (1024 ** 3)
    return total_gb / tp

def activation_memory(seq_len: int, batch: int, tp: int) -> float:
    """GB per GPU for activations under TP"""
    total_tokens = seq_len * batch
    bytes_per_token = model["activation_kb_per_token"] * 1024
    total_bytes = total_tokens * bytes_per_token
    total_gb = total_bytes / (1024 ** 3)
    return total_gb / tp

def estimate_latency(tp: int, pp: int, phase: str, seq_len: int, batch: int) -> float:
    """
    Very simple latency model (ms).
    Based on roofline: latency = max(compute, communication)
    Uses heuristics calibrated on H100 for Llama-like layers.
    """
    layers_per_gpu = model["num_layers"] // pp
    if phase == "prefill":
        # Prefill: compute-bound for long sequences
        base_ms_per_layer_per_1k_tokens = 0.07
        compute_ms = layers_per_gpu * base_ms_per_layer_per_1k_tokens * (seq_len / 1000) * (batch / tp)
        comm_bytes_mb = model["hidden_size"] * 2 * 4 / (1024**2)    # 2 bytes fp16, 4 for AllReduce volume
        comm_ms = layers_per_gpu * 0.015 * comm_bytes_mb
        latency = max(compute_ms, comm_ms)
    else:  # decode
        base_ms_per_layer = 0.025
        compute_ms = layers_per_gpu * base_ms_per_layer / tp
        comm_bytes_mb = model["hidden_size"] * 2 * 4 / (1024**2)
        comm_ms = 0.015 * comm_bytes_mb
        latency = max(compute_ms, comm_ms)
    return latency

# ------------------ Search ------------------
def find_best():
    best = None
    best_score = 1e9
    for tp in [1, 2, 4, 8]:
        for pp in [1, 2, 4, 8]:
            if tp * pp != hardware["total_gpus"]:
                continue
            weight_per_gpu = tp_memory_weights(tp)
            kv_per_gpu = kv_cache_memory(8192, 64, tp)
            act_per_gpu = activation_memory(8192, 64, tp)
            total_gb = weight_per_gpu + kv_per_gpu + act_per_gpu
            if total_gb > hardware["gpu_memory_gb"] * slo["max_gpu_mem_pct"] / 100:
                continue
            prefill_lat = estimate_latency(tp, pp, "prefill", 4096, 8)
            decode_lat = estimate_latency(tp, pp, "decode", 1, 8)
            if prefill_lat > slo["prefill_p99_ms"]:
                continue
            if decode_lat > slo["decode_p99_ms"]:
                continue
            score = 0.7 * decode_lat + 0.3 * prefill_lat
            if score < best_score:
                best_score = score
                best = {"tp": tp, "pp": pp, "prefill_lat": prefill_lat, "decode_lat": decode_lat, "memory_gb": total_gb}
    return best

plan = find_best()
if plan is None:
    raise RuntimeError("No valid parallel strategy found within constraints")

deployment = {
    "cluster": "H100_8GPU_Node",
    "model": "Llama3_70B_Instruct",
    "parallel_strategy": {
        "tensor_parallel_size": plan["tp"],
        "pipeline_parallel_size": plan["pp"],
        "data_parallel_size": 1,
        "expert_parallel_size": 1
    },
    "memory_plan": {
        "model_weights_gb_per_gpu": round(tp_memory_weights(plan["tp"]), 2),
        "kv_cache_max_gb_per_gpu": round(kv_cache_memory(8192, 64, plan["tp"]), 2),
        "activation_max_gb_per_gpu": round(activation_memory(8192, 64, plan["tp"]), 2),
        "total_max_gb_per_gpu": round(plan["memory_gb"], 2),
        "gpu_memory_headroom_gb": round(hardware["gpu_memory_gb"] - plan["memory_gb"], 2)
    },
    "latency_estimate_ms": {
        "prefill_p50": round(plan["prefill_lat"] * 0.8, 1),
        "prefill_p99": round(plan["prefill_lat"], 1),
        "decode_per_token_p50": round(plan["decode_lat"] * 0.8, 1),
        "decode_per_token_p99": round(plan["decode_lat"], 1)
    },
    "load_balancing": {
        "gpu_utilization_target_percent": slo["gpu_utilization_target_percent"],
        "gpu_memory_balance_epsilon": slo["gpu_balance_eps"]
    },
    "throughput_limits": {
        "max_batch_size": slo["max_batch"],
        "max_num_seqs": slo["max_seqs"],
        "max_num_batched_tokens": slo["max_tokens"],
        "target_requests_per_second": slo["target_rps"]
    },
    "deployment_commands": [
        f"vllm serve meta-llama/Llama-3-70B-Instruct \\",
        f"  --tensor-parallel-size {plan['tp']} \\",
        f"  --pipeline-parallel-size {plan['pp']} \\",
        f"  --max-model-len {model['max_position_embeddings']} \\",
        f"  --max-num-batched-tokens {slo['max_tokens']} \\",
        f"  --max-num-seqs {slo['max_seqs']} \\",
        f"  --dtype float16 \\",
        f"  --gpu-memory-utilization 0.85"
    ]
}

json_path = OUT_DIR / "deployment_plan.json"
with open(json_path, "w") as f:
    json.dump(deployment, f, indent=2)

report_path = OUT_DIR / "deployment_plan_readme.md"
with open(report_path, "w") as f:
    f.write(f"""# Llama3-70B Parallel Strategy Deployment Plan
Generated for: {hardware['total_gpus']}×{hardware['gpu_memory_gb']} GB H100

## Strategy
- TP={plan['tp']}, PP={plan['pp']}, DP=1, EP=1

## Partitioning
- GPUs: {hardware['total_gpus']}
- Layers per GPU: {model['num_layers'] // plan['pp']}
- Memory per GPU: {round(plan['memory_gb'],1)} GB (≤85 %)

## Latency
- Decode p99: {round(plan['decode_lat'],1)} ms
- Prefill p99: {round(plan['prefill_lat'],1)} ms

## Deploy
```bash
{deployment['deployment_commands'][0]}
```
""")

print("Generated:")
print(" - JSON:", json_path.resolve())
print(" - Report:", report_path.resolve())
print(f"Partitioning: PP={plan['pp']} stages × TP={plan['tp']} = {hardware['total_gpus']} GPUs")
print(f"Memory per GPU: {round(plan['memory_gb'],1)} GB (within limit)")
print(f"Decode p99: {round(plan['decode_lat'],1)} ms")
print(f"Prefill p99: {round(plan['prefill_lat'],1)} ms")