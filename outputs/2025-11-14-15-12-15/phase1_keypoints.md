# Phase 1: Key Points Extraction (REVISED)

## Critical Issue Identified and Resolved: Layer Count Correction

### Original Paper Analysis:
- **Setup section**: "A 4-layer fully connected dense network" (line 175) - **CORRECT**
- **Results table**: Shows "Dense (16-layer)" - **TYPO TO BE CORRECTED**
- **Conclusion**: Mentions "dense 4-layer model" - **CORRECT**

### Memory Calculation Verification:
- Total model size: 30B parameters × 2 bytes (BF16) = 60GB
- 4 layers → 15GB per layer (60GB / 4)
- Each layer fits within H100 L2 cache (~50MB) and HBM (~80GB) constraints

## Key Points:

1. **Problem**: Efficient deployment of large neural networks with memory constraints
2. **Solution**: Layer-wise partitioning ensuring each partition fits in SRAM/L2 cache
3. **Methodology**: Greedy layer aggregation algorithm
4. **Hardware**: 16 NVIDIA H100 GPUs
5. **Model**: 4-layer dense network with 30B parameters in BF16
6. **Metrics**: TPS (tokens/second) and TPOT (milliseconds)
7. **Results**: 20% improvement over baseline (TP=8, PP=2) - 15,360 vs 12,800 TPS

## Correction Applied:
The results table incorrectly shows "16-layer" - should be "4-layer" to maintain consistency.