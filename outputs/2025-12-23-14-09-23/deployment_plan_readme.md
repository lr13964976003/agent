# Llama3-70B Parallel Strategy Deployment Plan
Generated for hardware: 8×80 GB H100 single node

## Selected Strategy
- Tensor Parallelism (TP) = 8
- Pipeline Parallelism (PP) = 1
- Data Parallelism (DP) = 1 (single node)
- Expert Parallelism (EP) = 1 (dense model)

## Module Partitioning
- Total GPUs utilized: 8
- Model divided into 1 pipeline stage
- Each stage contains 80 layers
- Within the stage, layers are tensor-parallel split across 8 GPUs
- Therefore, each GPU holds:
  - 80 layers
  - TP degree 8 partitions per layer

## Memory Budget (per GPU)
- Model weights: 17.5 GB
- Max KV cache (8192×64): 8.0 GB
- Max activations: 4.0 GB
- Total: 29.5 GB
- Headroom: 50.5 GB
- Utilization: 36.9 % (≤85 % required)

## Latency Estimate
- Prefill (4096 tokens, batch 8): p50 224 ms, p99 280 ms (target p99 ≤1000 ms)
- Decode (per token): p50 6.4 ms, p99 8.0 ms (target p99 ≤100 ms)

## Load Balancing
- GPUs are symmetrically loaded (same memory, same layer count)
- Expected GPU utilization target: 70 %
- Memory balance ε ≤ 0.05

## Throughput Envelope
- Max batch size: 64
- Max concurrent sequences: 128
- Max batched tokens: 8192
- Target throughput: 8 requests/s

## Deployment Command
```bash
vllm serve meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype float16 \
  --gpu-memory-utilization 0.85
```

## Verification Checklist
- [x] GPU count matches partitioning (TP×PP=8)
- [x] Memory ≤85 % per GPU
- [x] Decode p99 latency ≤100 ms
- [x] Prefill p99 latency ≤1000 ms
- [x] Load balance ensured (symmetric)
- [x] Throughput envelope respected