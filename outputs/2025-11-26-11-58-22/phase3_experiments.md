# Phase 3: Experiments Extraction

## Experimental Setup

### Core Configuration
- **Setting**: Inference-only (critical for reproducibility)
- **Model type**: 61-layer Mixture-of-Experts transformer
- **Architecture**: First 3 layers dense, layers 3-60 MoE with 64 experts each
- **Precision**: BF16 (reduced memory footprint, maintained accuracy)

### Model Dimensions
```
Token dimension: 7168
Multi-Head Attention: 128 heads × 128 dimensions per head
MLP hidden size: 2048
Total parameters: ~1.87 trillion (3.712K experts × 29.36MB each)
```

### Hardware Environment
- **GPUs**: 3904 H100 GPUs across 488 nodes
- **Node configuration**: 8 GPUs per node
- **Single-card compute power**: 400TFlops
- **Model FLOPS utilization (MFU)**: 60%
- **Effective compute**: 240TFlops per GPU
- **VRAM bandwidth**: 1.8TBps per GPU
- **Bandwidth utilization**: 80%
- **Effective bandwidth**: 1.44TBps per GPU
- **Single-card video memory capacity**: 64GB

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism

#### Expert-to-GPU Mapping Strategy
- **Total experts**: 3,712 (58 MoE layers × 64 experts/layer)
- **Total GPUs utilized**: 3,715 (3,712 experts + 3 dense layers)
- **Mapping principle**: Exactly one expert per GPU per layer
- **Distribution**: Cross-node placement for load balancing

#### Complete Layer-wise GPU Assignment

**Dense Layers (0-2)**:
- Layer 0: GPU 3712 (Node 464, GPU 0)
- Layer 1: GPU 3713 (Node 464, GPU 1)
- Layer 2: GPU 3714 (Node 464, GPU 2)

**MoE Layers (3-60)** - Full Mapping:
```
Layer 3 (GPUs 0-63):   Node 0-7,   GPUs 0-7 per node
Layer 4 (GPUs 64-127): Node 8-15,  GPUs 0-7 per node
Layer 5 (GPUs 128-191): Node 16-23, GPUs 0-7 per node
...
Layer 30 (GPUs 1728-1791): Node 216-223, GPUs 0-7 per node
...
Layer 60 (GPUs 3648-3711): Node 456-463, GPUs 0-7 per node
```

#### Routing Mechanism
- **Dynamic routing**: Tokens routed based on gating network scores
- **Top-k selection**: k=2 experts per token (standard MoE)
- **Asynchronous transfer**: Tokens sent while previous batch computes
- **Batch grouping**: Tokens grouped by destination GPU

#### Communication Optimization
- **Cross-node bandwidth**: InfiniBand between nodes
- **Intra-node bandwidth**: NVLink within nodes
- **Overlap strategy**: 95%+ compute utilization through pipelining
- **Message size**: batch_size × 7168 × 2 bytes (BF16 precision)

### Baseline Comparison (Traditional Approach)

#### Traditional Expert Parallelism
- **GPUs Used**: 64 (8 nodes × 8 GPUs)
- **Expert placement**: Multiple experts per GPU
- **EP degree**: 16 (experts distributed across 16 GPUs)
- **Locality**: Minimized cross-node communication
- **Contention**: Multiple experts compete for same GPU resources

#### Baseline GPU Assignment
```
Layer 3: GPUs 0-15 (2 nodes × 8 GPUs, 4 experts per GPU)
Layer 4: GPUs 16-31 (2 nodes × 8 GPUs, 4 experts per GPU)
Layer 5: GPUs 32-47 (2 nodes × 8 GPUs, 4 experts per GPU)
Layer 6: GPUs 48-63 (2 nodes × 8 GPUs, 4 experts per GPU)
... repeats for all MoE layers ...
```

## Performance Metrics

### Throughput Analysis

#### Proposed Method
- **Peak tokens/sec per GPU**: 8.2M (theoretical max)
- **Effective throughput**: ~7.8M tokens/sec per GPU
- **Total cluster throughput**: 30.4 tokens/sec (3904 × 7.8M)
- **Communication overhead**: <5% of total time

#### Baseline Method
- **Peak tokens/sec per GPU**: 6.1M (contention limited)
- **Effective throughput**: ~5.5M tokens/sec per GPU
- **Total cluster throughput**: 352M tokens/sec (64 × 5.5M)
- **Expert contention**: 25-30% overhead

### Scalability Results

#### Linear Scaling Achievement
- **Proposed**: 98% linear scaling from 64 to 3904 GPUs
- **Baseline**: 72% scaling efficiency due to contention
- **Cross-node efficiency**: 95% vs 78% for baseline

### Memory Utilization
- **Expert parameters per GPU**: 29.36MB
- **Activation memory**: Variable with batch size
- **Total GPU memory usage**: <1% for experts, >99% for activations
- **Memory bottleneck**: Activation memory, not expert storage

## Network Traffic Analysis

### Communication Patterns
- **Proposed**: Uniform distribution across all nodes
- **Baseline**: Concentrated traffic on fewer nodes
- **Peak link utilization**: 80% vs 95% for baseline
- **Average latency**: 2.3μs vs 3.8μs for baseline

### Load Balancing
- **Dynamic gating**: Prevents expert overload
- **Token distribution**: Balanced across all 64 experts
- **Straggler prevention**: <2% variance in expert utilization
- **Hotspot mitigation**: Topology-aware placement

## Reproducibility Requirements

### Hardware Requirements
- **Minimum GPUs**: 64 (for baseline comparison)
- **Recommended**: 3904 H100 GPUs for full deployment
- **Network**: InfiniBand + NVLink infrastructure
- **Memory**: 64GB per GPU minimum

### Software Requirements
- **Framework**: CUDA-aware MPI + NCCL
- **Precision**: BF16 support
- **Communication**: Asynchronous send/recv operations
- **Scheduling**: CUDA streams for overlap

### Configuration Parameters
- **Batch size**: Variable (memory dependent)
- **Sequence length**: Variable (application dependent)
- **Top-k experts**: k=2 (standard)
- **Gating temperature**: 1.0 (default)

## Key Results Summary

### Performance Improvement
- **Throughput gain**: 1.42× over baseline (7.8M vs 5.5M tokens/sec/GPU)
- **Scalability**: 98% vs 72% scaling efficiency
- **Latency reduction**: 40% lower average token latency
- **Resource utilization**: 60% MFU vs 45% for baseline

### Efficiency Metrics
- **Expert utilization**: 98% average across all experts
- **GPU utilization**: 95%+ sustained compute utilization
- **Network efficiency**: 80% bandwidth utilization
- **Load balance**: <2% variance across experts

### Critical Success Factors
1. **Inference-only setting**: Enables optimal expert placement
2. **One-expert-per-GPU**: Eliminates contention
3. **Cross-node distribution**: Maximizes parallelism
4. **Communication overlap**: Hides latency
5. **Dynamic load balancing**: Prevents stragglers