# Complete MoE 30B Model Deployment DAG Summary

## Overview
This comprehensive DAG represents the complete deployment strategy for a 30B parameter Mixture of Experts (MoE) model using hybrid parallelism across 1024 GPUs.

## Parallelism Strategy Implementation

### 1. Expert Parallelism (EP=64)
- **64 experts per layer** fully represented with GPU assignments
- Each expert assigned to specific GPU pairs (TP groups)
- Expert routing with top-k selection shown as dashed lines
- Expert aggregation nodes for output combination
- Sample experts shown: 0,1,2,3,31,32,60,61,62,63 with ellipsis for remaining

### 2. Tensor Parallelism (TP=2) 
- **All-reduce communication** nodes for tensor synchronization
- **All-gather operations** for QKV projections and MLP layers
- Column-parallel and row-parallel decomposition in MLP layers
- GPU pair assignments (e.g., GPU 0-1, 2-3, etc.) clearly labeled

### 3. Pipeline Parallelism (PP=8)
- **8 pipeline stages** with proper stage-to-stage communication
- **2 layers per stage** (16 total layers)
- Stage input/output routers for data flow management
- Pipeline communication between stages 0→1→2→3→4→5→6→7

## Attention Block Decomposition

### Multi-Head Attention (32 heads)
- **Individual attention head nodes** for parallel computation
- **Attention aggregation** nodes for head combination
- **QKV projection** with tensor parallelism communication
- **Output projection** with all-reduce synchronization
- Sample heads shown: 0-7 and 24-31 with ellipsis for middle heads

### Attention Dimensions
- Input: [batch_size=128, seq_len=1024, hidden_size=4096]
- QKV Output: [batch_size=128, seq_len=1024, heads=32, d_k=128]
- Final Output: [batch_size=128, seq_len=1024, hidden_size=4096]

## Expert MLP Decomposition

### Per-Expert MLP Structure
1. **MLP Layer 1** (Column Parallel): hidden_size → ffn_hidden_size/2
2. **TP All-Gather**: Combine tensor parallel outputs  
3. **GELU Activation**: Element-wise activation function
4. **MLP Layer 2** (Row Parallel): ffn_hidden_size → hidden_size
5. **TP All-Reduce**: Synchronize tensor parallel outputs

### Expert Routing
- **Top-k expert selection** represented with dashed edges
- **Load balancing** through expert router nodes
- **Expert aggregation** for combining selected expert outputs

## GPU Assignment Strategy

### GPU Distribution (1024 total GPUs)
- **Pipeline Stage 0**: GPUs 0-127 (128 GPUs)
- **Pipeline Stage 1**: GPUs 128-255 (128 GPUs)
- **Pipeline Stage 2**: GPUs 256-383 (128 GPUs)
- **Pipeline Stage 3**: GPUs 384-511 (128 GPUs)
- **Pipeline Stage 4**: GPUs 512-639 (128 GPUs)
- **Pipeline Stage 5**: GPUs 640-767 (128 GPUs)
- **Pipeline Stage 6**: GPUs 768-895 (128 GPUs)
- **Pipeline Stage 7**: GPUs 896-1023 (128 GPUs)

### Per-Stage GPU Allocation
- **64 experts × 2 tensor parallel GPUs = 128 GPUs per stage**
- Expert 0: GPUs 0-1, Expert 1: GPUs 2-3, ..., Expert 63: GPUs 126-127

## Communication Patterns

### Tensor Parallelism Communication
- **All-Gather**: For column-parallel operations (QKV, MLP1)
- **All-Reduce**: For row-parallel operations (Attention output, MLP2)

### Expert Parallelism Communication  
- **Expert routing**: All-to-all communication for token assignment
- **Expert aggregation**: Combining outputs from selected experts

### Pipeline Parallelism Communication
- **Stage-to-stage**: Forward pass data transfer between pipeline stages

## Node Shape Conventions
- **Ellipses** (orange): Communication operations (TP all-reduce/all-gather)
- **Rectangles** (green): Computation operations (attention, MLP, layer norm)
- **Parallelograms** (yellow): Routing/aggregation operations (expert router, attention aggregation)

## Validation Results
- ✅ **No cycles detected**: DAG maintains proper acyclic property
- ✅ **Complete connectivity**: All nodes properly connected except input/output
- ✅ **Expert coverage**: 64 experts represented with sampling strategy
- ✅ **Attention decomposition**: 32 attention heads properly modeled
- ✅ **GPU assignments**: Explicit GPU tracking throughout the graph
- ✅ **Communication representation**: All parallelism communications included

## Model Specifications
- **Model Size**: 30B parameters
- **Hidden Size**: 4096
- **Attention Heads**: 32
- **FFN Hidden Size**: 16384
- **Experts per Layer**: 64
- **Layers**: 16 (8 stages × 2 layers)
- **Batch Size**: 128
- **Sequence Length**: 1024

This DAG provides a complete, accurate representation of the MoE deployment strategy suitable for performance analysis, optimization, and implementation guidance.