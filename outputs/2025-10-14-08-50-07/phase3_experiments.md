# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### Input Configuration
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192 dimensions
- **Total Tokens per Batch**: 10,240,000 tokens

### Multi-Head Attention (MHA) Details
- **Number of Heads**: 16
- **Dimension per Head**: 512
- **Total MHA Dimension**: 16 × 512 = 8,192 (matches token dimension)

### MLP Expert Details
- **Hidden Size**: 32,768 dimensions
- **Architecture**: Standard transformer FFN (Feed-Forward Network)
- **Activation Function**: GELU (implied from transformer architecture)

## Baseline Deployment Configuration

### Parallel Strategy
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)
- **Total GPUs**: 16 H100 GPUs

### GPU Allocation Details
- **Per-GPU Components**:
  - 1/8 of tensor-parallel shard for all layers
  - 8 experts per layer per GPU (16 experts/layer ÷ 2 pipeline stages = 8 experts per stage, all on one GPU)
  - Pipeline stage spans 8 GPUs each
- **Expert Placement**: Multiple experts colocated on same GPU
- **Processing Pattern**: Sequential through pipeline stages

## Proposed Deployment Configuration

### Parallel Strategy
- **Expert Parallelism (EP)**: 16-way (full expert distribution)
- **Tensor Parallelism (TP)**: 1-way (not used within experts)
- **Pipeline Parallelism (PP)**: 1-way (not used)
- **Total GPUs**: 16 H100 GPUs

### GPU Allocation Details
- **Per-GPU Components**:
  - Exactly one expert per layer per GPU
  - No tensor parallelism within experts
  - Full expert-level parallelism achieved
- **Expert Placement**: One expert per GPU across all nodes
- **Processing Pattern**: All 16 experts per layer compute in parallel

## Performance Results

### Throughput Metrics
| Method | GPUs Used | TPS (Tokens/s) | TPOT (ms) | Relative Improvement |
|--------|-----------|----------------|-----------|---------------------|
| Baseline (TP=8, PP=2) | 16 | 120,000 | 8.3 | 1.0× (baseline) |
| Proposed Cross-Node EP | 16 | 450,000 | 2.2 | 3.75× throughput, 3.8× latency |

### Detailed Analysis
- **Throughput Gain**: 450,000 ÷ 120,000 = 3.75× improvement
- **Latency Reduction**: 8.3 ÷ 2.2 = 3.77× improvement (≈3.8× as reported)
- **GPU Utilization**: 100% compute utilization per GPU (one expert per GPU)
- **Network Overhead**: Mitigated through asynchronous routing and token batching

## Experimental Environment
- **Hardware**: H100 GPUs (exact model not specified)
- **Network**: High-bandwidth interconnect (NVLink/InfiniBand implied)
- **Software**: CUDA, NCCL/MPI for communication
- **Setting**: Inference-only (no training results provided)

## Memory Requirements
### Per Expert Memory
- **Weight Matrix**: 8,192 × 32,768 × 2 bytes (FP16) = 512 MB
- **Activation Memory**: Variable based on batch size and sequence length
- **Total per GPU**: 512 MB (weights) + activations for one expert

### Total System Memory
- **Baseline**: 16 GPUs × (512 MB × 8 experts + TP overhead) ≈ 65 GB+ 
- **Proposed**: 16 GPUs × 512 MB = 8 GB (weights only)