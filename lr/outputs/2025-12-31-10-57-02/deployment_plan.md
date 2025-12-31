# Qwen3-235B MoE Parallel Strategy Deployment Plan
# Generated: 2025-12-31-10-57-02

## Model Configuration
- Model: Qwen3-235B
- Parameters: 235B
- Layers: 94
- Experts per layer: 128
- Top-K gate: 8
- Precision: FP8

## Optimal Parallel Strategy
- Tensor Parallel (TP): 4
- Pipeline Parallel (PP): 2
- Expert Parallel (EP): 8
- Sequence Parallel (SP): 4

## Resource Allocation
- Total GPUs: 8
- Memory per GPU: 47.80GB
- Memory utilization: 74.7%

## Performance Metrics
- Prefill time: 0.87s
- Target TTFT: 30s
- Throughput: 3.0M tokens/s

## Module Division
- Layers per stage: 47.0
- Experts per GPU: 16.0
- Attention heads per TP: 16.0
- Sequence partition: 4 ways

## Deployment Notes
- EP dominates GPU allocation (MoE inference rule)
- Each GPU hosts 16.0 experts
- Balanced load across all GPUs
- Meets TTFT requirement with margin
- Maximizes throughput while minimizing GPU usage
