# Model Deployment DAGs Summary

## Project Overview
Based on the research paper "Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models", we have generated complete deployment DAGs for both the proposed method and baseline configuration.

## Model Specifications
- **Model**: 16-layer MoE with 16 experts per layer
- **Token dimension**: 4096
- **Sequence length**: 10,000 tokens
- **Batch size**: 128 sequences
- **Precision**: BF16
- **Total GPUs**: 16 H100 GPUs

## Generated DAGs

### 1. Proposed Method (Large EP)
**Configuration**: Expert Parallelism (EP) = 16, 1 expert per GPU
- **Strategy**: One expert per GPU across all 16 GPUs
- **Expert distribution**: 16 experts × 16 layers = 256 total experts
- **GPU mapping**: Expert 0-15 on GPU 0-15 respectively
- **Throughput**: 450,000 TPS
- **Latency**: 2.2ms TPOT

### 2. Baseline Method 
**Configuration**: TP=8, PP=2, 8 experts per GPU
- **Strategy**: Tensor parallelism within GPU groups + Pipeline parallelism across stages
- **Expert distribution**: 8 experts per GPU × 16 GPUs = 128 experts total (replicated)
- **GPU mapping**: 
  - Stage 0: Layers 0-7 on GPUs 0-7
  - Stage 1: Layers 8-15 on GPUs 8-15
- **Throughput**: 120,000 TPS
- **Latency**: 8.3ms TPOT

## Key Engineering Details

### Expert Placement Verification
- **Proposed**: 16 experts per layer perfectly matches 16 GPUs (1:1 mapping)
- **Baseline**: 16 experts distributed as 8 per GPU, requiring 2 GPUs per layer

### Dimension Analysis
- All tensor dimensions are preserved throughout the pipeline
- Local dimensions properly map to global dimensions
- No dimensional information loss during expert routing and aggregation

### Communication Patterns
- **Proposed**: All-to-all communication for expert routing across 16 GPUs
- **Baseline**: Hierarchical communication within pipeline stages

### Load Balancing
- Uniform expert distribution ensures balanced GPU utilization
- Token routing ensures fair expert selection across all GPUs
- Residual connections maintain computational symmetry

## File Locations
All DAGs have been generated and saved in Graphviz DOT format and SVG images.