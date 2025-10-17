# MA Separation: Concise Version

## Abstract
Large language models with Mixture of Experts (MoE) architectures face significant challenges in parallel execution due to the temporal mismatch between attention mechanisms and expert computations. While MoE layers benefit from parallel expert execution across multiple GPUs, attention mechanisms typically operate sequentially, creating a computational bottleneck. We propose MA Separation, a novel parallel strategy that replicates attention computation across multiple cards to match the execution time of parallel MoE operations. Our approach enables synchronized co-execution of attention and MoE layers, maximizing GPU utilization and throughput. Experimental results on a 4-layer MoE model with 16 experts per layer across 16 GPUs demonstrate significant improvements: MA Separation achieves 34.2% reduction in Time per Output Token (TPOT) and 52.8% increase in Tokens per Second (TPS) compared to traditional tensor parallelism (TP=8) and pipeline parallelism (PP=2) baselines. This work presents a promising direction for scaling large MoE models by addressing the fundamental imbalance between attention and expert computation patterns.

**Keywords:** Mixture of Experts, Attention Mechanism, Parallel Computing, Large Language Models, GPU Computing

## 1. Core Problem and Solution

### Problem
- **Temporal Mismatch**: MoE layers can run experts in parallel across GPUs, but attention mechanisms operate sequentially
- **Inefficient GPU Utilization**: Attention becomes bottleneck while expert resources remain idle
- **Traditional Limitations**: Tensor parallelism (TP=8) and pipeline parallelism (PP=2) don't address attention-MoE temporal imbalance

### Solution: MA Separation
- **Novel Strategy**: Replicates attention computation across 12 GPUs to match execution time of 4-GPU MoE operations
- **GPU Allocation**: 12 GPUs for attention, 4 GPUs for MoE (3:1 ratio)
- **Synchronization**: Enables co-execution where attention and expert computations complete simultaneously

## 2. Model Architecture

### Model Specifications
- **Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32 heads
- **MoE Experts**: 16 experts per layer
- **Expert Hidden Dimension**: 16384 (4× hidden dimension)
- **Top-K Routing**: K=2 experts per token
- **Sequence Length**: 2048 tokens

### Hardware Configuration
- **Total GPUs**: 16 × NVIDIA A100 80GB
- **Architecture**: 4 nodes × 4 GPUs per node
- **Attention GPUs**: 12 GPUs (nodes 0-2)
- **MoE GPUs**: 4 GPUs (node 3)
- **Interconnect**: NVLink 3.0 (600 GB/s) + InfiniBand HDR

## 3. Methodology

### 3.1 Attention Parallelization (12 GPUs)

**Stage 1: QKV Projection**
- **Head Distribution**: 32 heads across 12 GPUs → 2.67 heads per GPU
- **Computation**: Each GPU computes Q/K/V for assigned heads
- **Input Broadcast**: Hidden states (4096-dim) replicated to all 12 GPUs

**Stage 2: Attention Computation**
- **Parallel Processing**: Each GPU computes attention for 2-3 heads
- **Communication**: All-reduce for key/value exchange
- **Output**: Attention outputs per GPU (size: batch_size × seq_len × head_dim×heads_per_gpu)

**Stage 3: Output Aggregation**
- **Hierarchy**: Intra-node (3 nodes × 4 GPUs) then inter-node
- **Final Output**: Aggregated attention output broadcast to 4 MoE GPUs

### 3.2 MoE Parallelization (4 GPUs)

**Expert Distribution**
- **Experts per GPU**: 16 experts / 4 GPUs = 4 experts per GPU
- **GPU 12**: Experts [0,1,2,3]
- **GPU 13**: Experts [4,5,6,7]
- **GPU 14**: Experts [8,9,10,11]
- **GPU 15**: Experts [12,13,14,15]

**Routing Process**
1. **Gate Computation**: Compute routing probabilities for all 16 experts
2. **Top-2 Selection**: Select 2 highest-scoring experts per token
3. **Token Distribution**: Route tokens to selected experts across 4 GPUs
4. **Expert Computation**: Parallel processing by 4-8 active experts per GPU

### 3.3 Synchronization Mechanism

**Timing Control**
- **Prediction Model**: Lightweight model predicts T_attention and T_moe
- **Parameters**: Sequence length, hidden size, active experts, GPU load
- **Balance Algorithm**: Adjusts head distribution to maintain T_attention ≈ T_moe

**CUDA Synchronization**
```
cudaEventRecord(attention_complete, attention_stream)
cudaEventRecord(moe_complete, moe_stream)
cudaStreamWaitEvent(next_layer, attention_complete)
cudaStreamWaitEvent(next_layer, moe_complete)
```

## 4. Experimental Results

### Performance Comparison

| Metric | Baseline (TP=8, PP=2) | MA Separation | Improvement |
|--------|----------------------|---------------|-------------|
| **TPOT (ms/token)** | 2.76 | 1.82 | **34.2% reduction** |
| **TPS (tokens/s)** | 8,696 | 13,289 | **52.8% increase** |
| **GPU Utilization** | 71.2% | 89.7% | **25.9% increase** |
| **Memory Efficiency** | 74.1% | 85.4% | **15.2% increase** |

### Detailed Experimental Configuration

**Training Setup**
- **Dataset**: C4 corpus, 2048 token sequences
- **Batch Size**: 1024 sequences = 2M tokens
- **Training Steps**: 50,000
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **Learning Rate**: 1e-4 with 5,000 step warmup

**Memory Usage (per GPU)**
- **Model Parameters**: 23.1 GB
- **Activations**: 18.7 GB
- **Optimizer States**: 46.2 GB
- **Total**: 123.7 GB (out of 80GB → uses gradient checkpointing)

### Scalability Results
- **Scaling Efficiency**: 87% at 16 GPUs
- **Linear Scaling**: Maintained up to 16 GPUs
- **Break-even**: Outperforms baseline starting from 8 GPUs
- **Communication Overhead**: 18.8% (vs 16.0% baseline)

### Inference Performance by Sequence Length

| Sequence Length | TPOT Improvement |
|-----------------|------------------|
| 512 | 27.6% |
| 1024 | 34.2% |
| 2048 | 35.9% |
| 4096 | 39.9% |

## 5. Deployment Configuration Summary

### Device Mapping
- **GPUs 0-11**: Attention computation (12 GPUs)
- **GPUs 12-15**: MoE computation (4 GPUs)
- **Expert Distribution**: 4 experts per MoE GPU
- **Head Distribution**: 32 attention heads across 12 GPUs
- **Synchronization**: CUDA events for precise timing control

### Critical Parameters for DAG Generation
- **Layer Count**: 4 transformer layers
- **Hidden Size**: 4096
- **Attention Heads**: 32 (2.67 heads per GPU)
- **MoE Experts**: 16 (4 per MoE GPU)
- **Sequence Length**: 2048
- **Batch Size**: 1024 sequences
- **Communication**: Hierarchical all-reduce for attention, all-to-all for MoE routing