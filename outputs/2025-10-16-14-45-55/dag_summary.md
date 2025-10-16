
# MA Separation Deployment DAGs Summary

## Generated Files:
1. baseline_dag.dot / baseline_dag.svg - Complete baseline model with TP=8, PP=2
2. ma_separation_dag.dot / ma_separation_dag.svg - Complete MA Separation model
3. layer_0_ma.dot / layer_0_ma.svg - Detailed Layer 0 MA Separation
4. layer_0_baseline.dot / layer_0_baseline.svg - Detailed Layer 0 Baseline
5. layer_1_ma.dot / layer_1_ma.svg - Detailed Layer 1 MA Separation
6. layer_1_baseline.dot / layer_1_baseline.svg - Detailed Layer 1 Baseline
7. layer_2_ma.dot / layer_2_ma.svg - Detailed Layer 2 MA Separation
8. layer_2_baseline.dot / layer_2_baseline.svg - Detailed Layer 2 Baseline
9. layer_3_ma.dot / layer_3_ma.svg - Detailed Layer 3 MA Separation
10. layer_3_baseline.dot / layer_3_baseline.svg - Detailed Layer 3 Baseline

## Key Differences:
- **Baseline**: Uses TP=8 across all 8 GPUs per pipeline stage, PP=2 across 16 GPUs
- **MA Separation**: Separates attention (GPUs 0-7) from MoE experts (GPUs 8-15)
- **Attention Parallelism**: Baseline uses tensor parallelism, MA Separation replicates attention across GPUs
- **Expert Distribution**: MA Separation maps 2 experts per GPU on MoE GPUs

## Engineering Details:
- All tensor dimensions are perfectly aligned
- Communication paths explicitly shown with parallelogram nodes
- Residual connections properly represented with multiple inputs
- Expert routing shown with dashed lines from gate to experts
- GPU assignments clearly labeled for each operation
