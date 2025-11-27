# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 61-layer Mixture-of-Experts (MoE) transformer
- **Layer Distribution**:
  - Dense layers: 1-3 (feed-forward networks)
  - MoE layers: 4-61 (expert MLPs)
- **Expert Specifications**:
  - Expert Type: MLP (Multi-Layer Perceptron)
  - Input dimension: 7168 (token embedding size)
  - Hidden dimension: 18432 (MLP intermediate size)
  - Output dimension: 7168 (back to model dimension)
  - Activation function: GELU

### Multi-Head Latent Attention (MLA) Configuration
- **Total attention heads**: 128
- **Head dimension**: 56 (per head)
- **Total attention dimension**: 128 × 56 = 7168
- **Latent projection mechanism**: Compresses KV from 7168 to smaller latent dimension
- **KV cache optimization**: Reduces memory by ~4x compared to standard attention

### Precision and Compute
- **Numerical precision**: BF16 (Brain Float 16-bit)
  - Provides 8-bit exponent and 7-bit mantissa
  - Dynamic range similar to FP32, precision similar to FP16
- **Computational efficiency**: Maintains accuracy while doubling throughput vs FP32

### Variable Parameters
- **Batch size**: Dynamically adjusted based on:
  - Available GPU memory
  - Sequence length
  - Number of active experts
  - Typical range: 512-4096 tokens per micro-batch
- **Sequence length**: Variable (up to 8192 tokens)
  - Short sequences: 128-512 tokens (higher batch size)
  - Long sequences: 2048-8192 tokens (lower batch size)

## Hardware Environment

### GPU Specifications
- **Model**: NVIDIA H100 Tensor Core GPUs
- **Memory**: 64GB HBM3 per GPU
- **Compute**: 400 TFLOPS (BF16 precision)
- **Memory Bandwidth**: 1.8 TB/s (HBM3)
- **Interconnect**: 
  - NVLink 4.0: 900 GB/s GPU-to-GPU
  - NVSwitch: 7.2 TB/s all-reduce bandwidth
  - InfiniBand HDR: 200 Gbps node-to-node

### Resource Utilization
- **MFU (Model FLOPS Utilization)**: 60%
  - Accounts for memory bandwidth limitations
  - Communication overhead
  - Load imbalance
- **Memory Bandwidth Utilization**: 80%
  - Sustained memory access efficiency
  - Accounts for cache misses and bank conflicts
- **Network Bandwidth Utilization**: Variable based on EP degree

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism

#### GPU Allocation Strategy
- **One-GPU-Per-Expert Rule**: 
  - Each MoE layer e_i maps to GPU g_j
  - No GPU hosts more than one expert from same layer
  - Replication allowed across layers for E > G scenarios

#### Routing Implementation
- **Dynamic Token Routing**:
  - Gating network produces scores for all E experts
  - Top-K (typically K=2) experts selected per token
  - Tokens dispatched to GPUs holding selected experts
- **Asynchronous Communication**:
  - Sender threads: Package tokens by destination GPU
  - Receiver threads: Post buffers for incoming tokens
  - Overlap factor: 80% (communication hides behind computation)

#### Load Distribution
- **Expert Load Balancing**:
  - Monitor tokens/sec per expert
  - Adjust gating weights: Δw = α(μ_target - μ_actual)
  - α = 0.01, rebalancing interval = 100ms

### Baseline Comparison (Conventional Approach)

#### Traditional Expert Placement
- **Multiple Experts per GPU**: 
  - Typical: 4-8 experts per GPU
  - Limits EP degree to reduce communication
  - Creates memory contention
- **Intra-node Focus**:
  - Minimize cross-node traffic
  - Simpler routing (no network latency)
  - Higher per-GPU memory pressure

#### Baseline Configurations
- **EP=4**: 4 experts per parallel group
- **EP=8**: 8 experts per parallel group  
- **EP=16**: 16 experts per parallel group (minimum for "large EP")

## Performance Metrics

### Throughput Measurements
- **Primary Metric**: Tokens/second
- **Secondary Metrics**:
  - Latency per token (ms)
  - GPU utilization (%)
  - Network bandwidth utilization (%)
  - Memory usage (GB per GPU)

### Scaling Characteristics
- **Strong Scaling**:
  - Fixed model size, increasing GPUs
  - Target: Near-linear speedup up to network limits
- **Weak Scaling**:
  - Fixed model per GPU, increasing replicas
  - Target: Constant throughput per GPU

### Memory Footprint Analysis
- **Per-GPU Memory Usage**:
  - Expert weights: 18432 × 7168 × 2 = 264 MB per expert
  - Activation cache: Variable (2-8 GB typical)
  - KV cache: batch × seq × 128 × 56 × 2 = 14.3 kB per token
  - Communication buffers: 64 MB per peer connection
  - Total: 10-50 GB per GPU (within 64GB limit)

## Experimental Results Summary

### Key Findings
- **Throughput**: 3.2x higher than conventional EP=8 baseline
- **Latency**: 45% reduction in average token latency
- **Scalability**: Linear speedup up to 128 GPUs (EP=128)
- **Network Efficiency**: 85% bandwidth utilization maintained
- **Load Balance**: <5% variance in expert utilization

### Bottleneck Analysis
- **Communication**: Dominant factor at EP > 64
- **Memory**: KV cache becomes limiting at long sequences
- **Compute**: GPU utilization drops at very high EP due to small expert sizes

## Validation Environment

### Test Configurations
- **Small Scale**: 16 GPUs, EP=16
- **Medium Scale**: 64 GPUs, EP=64  
- **Large Scale**: 128 GPUs, EP=128
- **Variable Expert Count**: E=16, 32, 64, 128 experts per layer

### Workload Characteristics
- **Synthetic Data**: Random tokens (uniform distribution)
- **Real Data**: OpenWebText samples
- **Sequence Lengths**: 128, 512, 1024, 2048, 4096 tokens
- **Batch Sizes**: 512, 1024, 2048, 4096 tokens per micro-batch

### Profiling Tools
- **NVIDIA Nsight Systems**: GPU kernel profiling
- **NCCL Tests**: Communication bandwidth measurement
- **Custom Metrics**: Token routing distribution, expert utilization
- **PyTorch Profiler**: End-to-end performance analysis