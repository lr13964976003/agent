# Qwen3-235B MoE Parallel Strategy Deployment Plan
# Generated: 2026-01-04-09-49-07

## Model Configuration
- Model: Qwen3-235B
- Parameters: 235B
- Layers: 94
- Experts per layer: 128
- Top-K gate: 8
- Precision: FP8

## Optimal Parallel Strategy
- Tensor Parallel (TP): 8
- Pipeline Parallel (PP): 1
- Expert Parallel (EP): 8
- Sequence Parallel (SP): 1
- Data Parallel (DP): 1

## Resource Allocation
- Total GPUs: 8
- Memory per GPU: 43.84GB
- Memory utilization: 68.5%

## Performance Metrics
- Target TTFT: 30s
- Calculated prefill time: 0.38s
- Throughput: 12,324 tokens/s

## Module Division
- Layers per stage: 94
- Experts per GPU: 16.0
- Attention heads per TP: 8
- Sequence partition: 1 way

## Deployment Notes
- EP dominates GPU allocation following MoE inference rule
- Each GPU hosts 16 experts out of 128 total
- TP=8 maximizes compute parallelism for prefill phase
- PP=1 eliminates pipeline bubbles and latency overhead
- SP=1 avoids unnecessary sequence parallelism complexity
- Balanced load across all 8 GPUs
- Meets TTFT requirement with 29.62s margin
- Optimized for throughput while minimizing GPU usage

## Validation Results
- Memory budget: ✓ PASS (68.5% utilization)
- TTFT requirement: ✓ PASS (0.38s << 30s)
- Hardware compatibility: ✓ PASS
- Load balancing: ✓ OPTIMAL
- GPU count efficiency: ✓ MINIMAL (8 GPUs)