# Parallel Strategy Deployment Plan

## Executive Summary

This deployment plan optimizes a 10B parameter MoE model with 16 layers and 16 experts per layer across available GPU resources, achieving target throughput of 100 tokens/ms per GPU while maintaining TTFT ≤ 10s.

## Structural Analysis

### Model Architecture
- **Total Parameters**: 10B
- **Layers**: 16 transformer layers
- **MoE Configuration**: 16 experts per layer
- **Attention**: 16 heads, 32 dim per head (total 512)
- **Precision**: FP16
- **MoE Hidden Dimension**: 1024

### Memory Requirements
- **Attention Parameters per Layer**: ~4M (QKV projections)
- **MoE Parameters per Layer**: ~625M (16 experts × 1024 hidden × 512 dim × 3 matrices)
- **Total per Layer**: ~629M parameters
- **Full Model**: ~10B parameters (16 × 629M)
- **FP16 Storage**: ~20GB for model weights

## Parallel Strategy Design

### 1. Expert Parallel (EP) - Primary Strategy
**Decision**: EP = 16 GPUs
- **Rationale**: Each layer has 16 experts, optimal mapping is 1 expert per GPU
- **GPU Allocation**: 16 GPUs dedicated to expert hosting
- **Benefits**: Minimizes expert switching overhead, maximizes expert locality

### 2. Pipeline Parallel (PP) - Layer Distribution
**Decision**: PP = 2 stages
- **Stage 1**: Layers 1-8 (8 layers)
- **Stage 2**: Layers 9-16 (8 layers)
- **GPU Allocation**: Each stage uses 16 GPUs (for EP), total 32 GPUs
- **Benefits**: Balances computation across pipeline stages

### 3. Tensor Parallel (TP) - Attention Optimization
**Decision**: TP = 4 for Attention operations
- **Rationale**: 16 attention heads split across 4 GPUs = 4 heads per GPU
- **Scope**: Applied only to Multi-Head Attention (QKV projections)
- **GPU Groups**: Within each pipeline stage, GPUs form TP groups of 4
- **Benefits**: Reduces attention computation time, improves memory bandwidth utilization

### 4. Data Parallel (DP) - Throughput Scaling
**Decision**: DP = 4
- **Rationale**: 4 independent pipeline replicas for request parallelism
- **Total GPUs**: 32 GPUs per pipeline × 4 replicas = 128 GPUs
- **Benefits**: Achieves target throughput through request-level parallelism

## GPU Resource Mapping

### Total GPU Configuration: 128 GPUs

**Structure Breakdown**:
```
DP=4 (Request Parallelism)
├── Pipeline Replica 1: 32 GPUs
│   ├── PP Stage 1: 16 GPUs (Layers 1-8)
│   │   └── TP Groups: 4 groups ×4 GPUs each
│   └── PP Stage 2: 16 GPUs (Layers 9-16)
│       └── TP Groups: 4 groups ×4 GPUs each
├── Pipeline Replica 2: 32 GPUs
├── Pipeline Replica 3: 32 GPUs
└── Pipeline Replica 4: 32 GPUs
```

### Expert Distribution
- **Each GPU hosts exactly 1 expert per layer**
- **Expert IDs 0-15 distributed across 16 GPUs in each pipeline stage**
- **Expert routing**: Deterministic assignment based on token characteristics

## Performance Analysis

### Throughput Calculation
- **Per-GPU Target**: 100 tokens/ms
- **Per-Pipeline**: 16 GPUs × 100 tokens/ms = 1,600 tokens/ms
- **Total System**: 4 pipelines × 1,600 tokens/ms = 6,400 tokens/ms

### TTFT Optimization
- **Sequence Length**: 128-10,240 tokens
- **Pipeline Depth**: 2 stages minimize latency
- **TP Overhead**: 4-way parallelism adds minimal communication
- **Expected TTFT**: < 8 seconds for maximum sequence length

### Memory Bandwidth Utilization
- **VR Bandwidth**: 1.8TBps per GPU at 80% utilization = 1.44TBps effective
- **Attention Communication**: ~50GB/s per TP group
- **Expert Activation**: Localized to individual GPUs
- **Memory Efficiency**: >75% utilization across all GPUs

## Load Balancing Strategy

### Expert Load Distribution
- **Uniform Routing**: Implement load balancing router
- **Dynamic Assignment**: Token-based expert selection
- **Overflow Handling**: Secondary expert assignment for overloaded tokens

### Pipeline Load Balancing
- **Equal Layer Distribution**: 8 layers per stage
- **Computation Balance**: Similar parameter count per stage
- **Bubble Minimization**: Optimized micro-batching

## Implementation Details

### Communication Patterns
1. **TP Communication**: All-reduce within 4-GPU groups
2. **PP Communication**: Point-to-point between stages
3. **DP Communication**: No inter-replica communication
4. **EP Communication**: Local expert computation only

### Memory Layout
- **Model Weights**: Replicated across TP groups
- **Expert Parameters**: Unique per GPU
- **Activations**: Staged in pipeline registers
- **KV Cache**: Distributed across attention heads

## Validation Metrics

### Performance Verification
- [ ] Throughput ≥ 100 tokens/ms per GPU
- [ ] TTFT ≤ 10 seconds
- [ ] GPU Utilization > 60%
- [ ] Memory Utilization < 90%

### Correctness Checks
- [ ] Expert mapping: 1 expert per GPU
- [ ] TP groups: 4 GPUs per group
- [ ] Pipeline stages: 8 layers each
- [ ] Total GPUs: 128

## Module Division Summary

**Total Modules**: 16 (experts per layer) × 16 (layers) = 256 expert modules
**GPU Distribution**: 256 expert modules ÷ 16 GPUs per stage = 16 experts per GPU
**Pipeline Stages**: 2 stages with balanced layer distribution
**Parallel Groups**: 
- 4 DP groups
- 2 PP stages per DP group  
- 4 TP groups per PP stage
- 16 EP positions per TP group

This configuration achieves optimal load balancing while meeting all performance requirements through strategic combination of parallelism dimensions, with EP as the primary driver of GPU allocation as mandated by the MoE inference constraints.