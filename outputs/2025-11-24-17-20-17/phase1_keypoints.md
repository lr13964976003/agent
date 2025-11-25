# Phase 1: Key Points Extraction - Helix Paper

## Key Points from Original Paper

### Core Problem
- Transformer models with Multi-Head Attention (MHA) need efficient distributed deployment
- Traditional head-wise partitioning limits scalability when devices > heads
- Need better load balancing and reduced communication overhead

### Proposed Solution
- **Two-Level Attention Partitioning**: Combine head-level and intra-head dimension-level partitioning
- Create m×n partitions from n head groups × m dimension slices
- Enables deployment on m×n devices (16 devices in experiments)

### Technical Details
- Input: X ∈ ℝ^(B×L×D) where B=128, L=10000, D=4096
- 32 heads, 128 dimensions per head (h=32, d=128)
- Partitioning: n=4 head groups × m=4 dimension slices = 16 partitions
- Each partition: 8 heads × 32 dimensions

### Key Metrics
- 31.7% throughput improvement (1.2M → 1.58M tokens/sec)
- 37.1% communication overhead reduction (0.35ms → 0.22ms per token)
- 16 NVIDIA H100 GPUs tested
- 4-layer Dense Transformer model used

### Implementation Features
- Hierarchical aggregation: concatenate dimension slices within groups, then concatenate groups
- Memory efficient: 1/16 parameters per device
- FP16 precision
- Balanced workload distribution across 16 GPUs

### Deployment Advantages
- Scalability beyond head count limitations
- Better hardware utilization
- Reduced memory footprint per device
- Localized communication patterns

## Dimensions and Parameters Summary
- Total model: 4 layers, hidden_size=4096, heads=32, head_dim=128
- Sequence length: 10000
- Batch size: 128
- MLP hidden: 16384
- Parameter count: 1,048,576 parameters per device (1/16 of total)
- Activations: 40,960,000 elements per device