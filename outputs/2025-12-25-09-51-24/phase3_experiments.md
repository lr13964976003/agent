# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (Half precision)
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10000 tokens per sequence
- **Token dimension**: 8192
- **Multi-Head Attention (MHA)**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32768

### Hardware Environment
- **GPUs**: H100 (NVIDIA Hopper architecture)
- **Total GPUs used**: 16
- **Setting**: Inference-only evaluation

### Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per individual token

## Parallel Deployment Details

### Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100s
- **Tensor Parallelism (TP)**: 8-way splitting
- **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages × 8 GPUs = 16 total)
  - Experts are colocated: typically 8 experts per layer per GPU
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource sharing**: Multiple experts share compute resources on each GPU

### Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 16 H100s
- **Expert Parallelism (EP)**: 16 (qualifies as "large EP")
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Routing mechanism**: 
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously to minimize idle time
- **Parallelism**: All 16 experts per layer compute in parallel

## Results

### Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput improvement**: 450,000 ÷ 120,000 = 3.75× higher TPS
- **Latency improvement**: 8.3 ÷ 2.2 = 3.77× lower TPOT
- **Round numbers**: ~3.75× higher throughput, ~3.8× lower latency

### Key Observations
1. **Expert dedication**: One expert per GPU eliminates intra-GPU contention
2. **Parallel efficiency**: All 16 experts process simultaneously
3. **Communication overlap**: Asynchronous routing prevents waiting
4. **Scalability**: Near-linear scaling achieved with EP ≥ 16

## Discussion

### Baseline Limitations
- **Intra-GPU contention**: Multiple experts share GPU resources
- **Pipeline stalls**: Sequential processing creates bottlenecks
- **Underutilization**: GPUs not fully utilized due to resource sharing

### Proposed Method Advantages
- **Full utilization**: Each GPU dedicated to single expert
- **No contention**: Experts process in isolation
- **Maximal parallelism**: All experts compute concurrently
- **Balanced load**: Topology-aware placement prevents hotspots

### Network Considerations
- **Communication cost**: Amortized across large batch sizes (1024 sequences)
- **Bandwidth utilization**: Modern HPC interconnects (NVLink, InfiniBand) sustain high throughput
- **Latency hiding**: Asynchronous transfers overlap with computation

## Experimental Validation
- **Setting**: Controlled inference-only comparison
- **Hardware consistency**: Same 16 H100 GPUs for both methods
- **Model consistency**: Identical 4-layer, 16-expert MoE architecture
- **Measurement accuracy**: TPS and TPOT measured under identical load conditions

## Scalability Implications
- **Large EP regime**: EP ≥ 16 proves effective for high-performance MoE deployment
- **Resource requirements**: Requires abundant GPU resources (one per expert)
- **Future scaling**: Method applicable to models with thousands of experts
- **Training extension**: Approach can be adapted for training scenarios