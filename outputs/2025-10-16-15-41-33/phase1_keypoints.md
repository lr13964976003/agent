# Phase 1: Keypoints Extraction - FA Pool Paper

## Main Problem
- Attention mechanisms in transformers have O(n²) complexity with sequence length n
- Traditional static parallel strategies (TP, PP) have suboptimal resource utilization
- Fixed resource allocation leads to underutilization for short sequences or bottlenecks for long sequences

## Proposed Solution: FA Pool
- **FA Pool (Flash Attention Pool)**: A dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds
- **Core Innovation**: When sequences exceed threshold, additional GPUs form computation pool for parallel attention calculations
- **Key Features**:
  - Combines Flash Attention memory efficiency with dynamic resource allocation
  - Preserves model coherence through base layer + attention pool architecture
  - Minimizes communication overhead through intelligent workload distribution

## Technical Architecture
- **Base Layer**: 8 GPUs maintaining core model components (embedding, positional encoding, output layers, FFN)
- **Attention Pool**: Up to 32 additional GPUs dynamically allocated for attention computation
- **Resource Manager**: Monitors sequence length and manages GPU allocation/deallocation
- **Sequence Threshold**: 4096 tokens (empirically determined)

## Experimental Model Details
- **Architecture**: 4-layer Dense transformer model
- **Parameters**: ~13B parameters
- **Dimensions**:
  - Hidden dimension: 4096
  - Attention heads: 32
  - Feed-forward dimension: 16384
  - Batch size: 1024

## Baseline Configuration
- **Static Baseline**: TP=8, PP=2 (16 GPUs total)
- **Comparison Metrics**: TPOT (Time Per Output Token) and TPS (Tokens Per Second)

## Key Results
- **Maximum Improvements**:
  - 3.2x improvement in TPOT for 16K+ tokens
  - 2.8x improvement in TPS for 16K+ tokens
- **Scaling Characteristics**: Near-linear scaling up to 16K tokens
- **Resource Utilization**: 85-92% GPU utilization in attention pool vs 45-60% in baseline
- **Communication Overhead**: <15% of total computation time

## Critical Technical Details
- **Parallel Strategy**: Block-wise parallelization within attention pool
- **Block Size**: b = ceil(n/p) where n=sequence length, p=pool GPUs
- **Memory Requirements**: 65GB per base GPU, 45GB per pool GPU
- **Communication**: KV cache sharing + hierarchical reduction + asynchronous execution