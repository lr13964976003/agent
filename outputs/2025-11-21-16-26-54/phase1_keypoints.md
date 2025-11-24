# Phase One: Keypoints Extraction

## Core Problem
- Transformers have quadratic attention complexity and heavy memory requirements
- Multi-Head Attention (MHA) becomes a bottleneck due to communication-intensive operations
- Challenges intensify when scaling to trillions of parameters or handling extremely long sequences (>16k tokens)

## Proposed Solution
- Novel parallelization strategy combining **Ring Attention** with **sequence parallelism**
- Leverages ring topology for distributed MHA computation
- Reduces peak communication bandwidth and memory footprint simultaneously

## Key Technical Contributions
1. **Ring Attention**: Replaces global all-to-all communication with sequential peer-to-peer exchanges in a ring topology
2. **Sequence Parallelism**: Splits input sequences across devices, reducing memory by factor of P (number of devices)
3. **Combined Strategy**: Integrates both approaches to achieve communication-efficiency and memory-friendliness

## Performance Claims
- **20-25% higher TPS** (Tokens Per Second) compared to baseline
- **24-27% higher TPOT** (Time Per Output Token) improvement
- Particularly effective for sequences >16k tokens
- Benefits grow with both sequence length (L) and number of devices (P)

## Experimental Validation
- Tested on 16×H100 GPUs with inference-only setting
- Dense Transformer: 4 layers, batch size 128, sequence length 100k tokens
- Baseline: TP=8, PP=2 without sequence parallelism or ring-based attention
- Results: 20.8% TPS improvement and 17.6% TPOT reduction for dense model

## Technical Specifications
- **Precision**: BF16
- **Batch Size**: 128 (fixed)
- **Sequence Length**: 100,000 tokens (fixed)
- **Attention Heads**: 32 (fixed)
- **Head Dimension**: 128 (fixed)
- **MLP Hidden Size**: 32,768 (fixed)