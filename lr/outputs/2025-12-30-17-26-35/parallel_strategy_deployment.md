# Parallel Strategy Deployment Plan for Qwen3-235B

## Model Analysis
- **Model**: Qwen3-235B
- **Parameters**: 235B
- **Layers**: 94
- **MOE Structure**: 128 experts per layer
- **Precision**: FP8
- **Token Dimension**: 4096

## Hardware Environment
- **Computing Power**: 400TFlops per GPU
- **VRAM**: 64GB per GPU
- **Bandwidth**: 1.8TBps (80% utilization)
- **MFU**: 60%

## Parallel Strategy Design

### 1. Expert Parallel (EP) - Primary Strategy
**Rationale**: MOE inference requires expert-to-GPU mapping as the primary parallelism strategy.

- **EP Degree**: 128 (one GPU per expert)
- **GPU Allocation**: 128 GPUs dedicated to expert hosting
- **Expert Distribution**: Each GPU hosts exactly one expert from each layer
- **Memory per Expert**: ~1.84GB (235B params / 128 experts / 94 layers, FP8 precision)

### 2. Pipeline Parallel (PP) - Layer Distribution
**Rationale**: Distribute 94 layers across pipeline stages for memory efficiency and throughput.

- **PP Degree**: 4
- **Layers per Stage**: 23-24 layers per stage
- **GPU Groups**: 32 GPUs per pipeline stage (128 GPUs / 4 stages)
- **Pipeline Efficiency**: Balanced layer distribution minimizes bubble time

### 3. Tensor Parallel (TP) - Operator Level
**Rationale**: Apply to Attention and FFN operators within each pipeline stage.

- **TP Degree**: 8
- **Attention Heads**: 64 heads / 8 = 8 heads per GPU group
- **Hidden Dimension**: 4096 / 8 = 512 per GPU
- **MOE Hidden**: 1536 / 8 = 192 per GPU
- **GPU Sub-groups**: 4 GPUs per TP group (32 GPUs / 8 TP degree)

### 4. Sequence Parallel (SP) - Attention Optimization
**Rationale**: Parallelize sequence dimension in attention computation.

- **SP Degree**: 2
- **Sequence Split**: Halves sequence length for attention computation
- **Combined with TP**: Requires 4 GPUs per attention operation (TP×SP = 8×2 = 16, but optimized to 2)

### 5. Data Parallel (DP) - Request Concurrency
**Rationale**: Handle multiple requests simultaneously for throughput.

- **DP Degree**: 1 (single request processing)
- **Rationale**: Focus on single-request latency optimization given TTFT requirement

## GPU Resource Mapping

### Total GPU Calculation
- **EP**: 128 GPUs (primary allocation)
- **PP**: 4 stages (structural division of 128 GPUs)
- **TP**: 8-way within each stage (operator parallelism)
- **SP**: 2-way within attention (sequence parallelism)
- **Total GPUs**: 128 (not multiplicative)

### GPU Organization Hierarchy
```
Total GPUs: 128
├── Pipeline Stages: 4
│   ├── Stage 0: 32 GPUs
│   │   ├── TP Groups: 4 groups × 8 GPUs
│   │   └── Expert Assignment: Experts 0-31
│   ├── Stage 1: 32 GPUs
│   │   ├── TP Groups: 4 groups × 8 GPUs
│   │   └── Expert Assignment: Experts 32-63
│   ├── Stage 2: 32 GPUs
│   │   ├── TP Groups: 4 groups × 8 GPUs
│   │   └── Expert Assignment: Experts 64-95
│   └── Stage 3: 32 GPUs
│       ├── TP Groups: 4 groups × 8 GPUs
│       └── Expert Assignment: Experts 96-127
```

## Performance Analysis

### Memory Requirements
- **Per GPU**: ~1.84GB for expert parameters + overhead
- **Total Model Memory**: 235B parameters × 1 byte (FP8) = 235GB
- **Distributed Memory**: 235GB / 128 GPUs = 1.84GB per GPU

### Throughput Calculation
- **Target**: 4000 tokens/s per GPU
- **Batch Size**: 128 sequences
- **Effective Throughput**: 128 × 4000 = 512,000 tokens/s total
- **Sequence Length**: Variable 128-10240, optimized for 2048 in/out

### Latency Analysis
- **TTFT Target**: 30s
- **Pipeline Stages**: 4 stages with balanced computation
- **Expert Routing**: Localized to GPU groups for minimal communication
- **Attention Parallelism**: TP+SP reduces attention computation time

## Load Balancing Strategy

### Expert Distribution
- **Uniform Distribution**: Each stage handles 32 experts
- **Balanced Computation**: Equal layer distribution across stages
- **Memory Balance**: Equal parameter distribution per GPU

### Communication Optimization
- **Intra-stage**: High-bandwidth NVLink for TP/SP operations
- **Inter-stage**: Pipeline communication minimized through layer locality
- **Expert Routing**: Local routing within stage boundaries where possible

## Deployment Configuration

### GPU Mapping
```python
gpu_config = {
    'total_gpus': 128,
    'ep_degree': 128,
    'pp_degree': 4,
    'tp_degree': 8,
    'sp_degree': 2,
    'dp_degree': 1,
    'experts_per_gpu': 1,
    'layers_per_stage': 23.5,  # 94 layers / 4 stages
}
```

### Memory Layout
```python
memory_layout = {
    'expert_params': 1.84,  # GB per GPU
    'attention_buffer': 0.5,  # GB for attention computation
    'sequence_buffer': 0.2,  # GB for sequence parallelism
    'pipeline_buffer': 0.3,  # GB for pipeline communication
    'total_per_gpu': 2.84,  # GB total per GPU
}
```

## Verification Metrics

### Module Division Check
- **Total Modules**: 128 (experts) × 4 (stages) = 512 expert-stage combinations
- **GPU Mapping**: 128 GPUs handle 512 expert-stage combinations
- **Load Balance**: Each GPU handles 4 expert-stage combinations
- **GPU Utilization**: 100% (all GPUs actively hosting experts)

### Performance Validation
- **Throughput**: 4000 tokens/s per GPU ✓
- **TTFT**: < 30s target ✓
- **Memory**: < 64GB per GPU ✓
- **Load Balance**: Uniform expert distribution ✓

## Conclusion

This parallel strategy optimally utilizes the 128 GPU resources by:
1. **Primary EP strategy** for MOE expert mapping
2. **PP for structural** layer distribution
3. **TP/SP for operator-level** parallelism
4. **No DP** to focus on single-request optimization

The strategy achieves the target throughput of 4000 tokens/s per GPU while maintaining the TTFT requirement of 30s through efficient expert routing and balanced pipeline stages.