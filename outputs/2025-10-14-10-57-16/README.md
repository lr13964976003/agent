# Improved LLM Deployment DAGs

This directory contains the improved deployment strategy for a 16-layer large language model using tensor parallelism combined with pipeline parallelism.

## Generated Files

### 1. Main DAG Files
- **`improved_tensor_parallel_dag.dot`** - Complete Graphviz DAG showing tensor parallelism + pipeline parallelism
- **`improved_tensor_parallel_dag.svg`** - SVG visualization of the complete deployment

### 2. Overview Files
- **`deployment_overview.svg`** - High-level overview of the 2-stage pipeline with 8-way tensor parallelism
- **`layer_detailed_breakdown.svg`** - Detailed breakdown of operations within a single layer

### 3. Configuration Files
- **`deployment_config.json`** - Complete deployment configuration with GPU allocation, tensor parallel operations, and performance characteristics

### 4. Generation Scripts
- **`generate_visualizations.py`** - Python script to generate all visualizations

## Deployment Strategy Summary

### Architecture
- **Total GPUs**: 16
- **Pipeline Stages**: 2 (8 layers each)
- **Tensor Parallelism**: 8-way within each stage
- **GPU Allocation**: 
  - Stage 0: GPUs 0-7 (Layers 0-7)
  - Stage 1: GPUs 8-15 (Layers 8-15)

### Key Improvements Over Baseline
1. **Parallelization**: 8-way tensor parallelism vs sequential layer-wise
2. **Load Balancing**: Equal work distribution across all 16 GPUs
3. **Memory Optimization**: SRAM/L2 cache utilization per GPU
4. **Communication Patterns**: Optimized all-reduce operations within stages

### Model Specifications
- **Layers**: 16 transformer layers
- **Hidden Size**: 8192
- **Attention Heads**: 16 (512 dim per head)
- **FFN Hidden Size**: 32768
- **Batch Size**: 1024
- **Sequence Length**: 10000

### Tensor Parallel Operations
- **QKV Linear**: Column parallel across 8 GPUs
- **Attention**: Parallel across heads (2 heads per GPU)
- **Attention Projection**: Row parallel with all-reduce
- **MLP**: Column + Row parallel with all-reduce

### Verification Results
- ✅ No cycles detected in DAG
- ✅ All nodes properly connected
- ✅ Input/output dimensions preserved
- ✅ GPU load balancing achieved
- ✅ Complete model structure maintained

## Usage

To regenerate visualizations:
```bash
python3 generate_visualizations.py
```

To view the deployment:
1. Open `deployment_overview.svg` for high-level view
2. Open `layer_detailed_breakdown.svg` for layer operations
3. Open `improved_tensor_parallel_dag.svg` for complete DAG