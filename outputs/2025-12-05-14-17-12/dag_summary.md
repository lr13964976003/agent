# 30B MoE Model Deployment DAG Summary

## Graph Structure
- **Total Nodes**: ~1000+ nodes representing individual operations
- **Total GPUs**: 1024 (64 EP × 2 TP × 8 PP)
- **Pipeline Stages**: 8 stages (2 layers each)
- **Experts per Stage**: 64 experts
- **Tensor Parallelism**: 2 GPUs per expert

## Node Types
1. **Ellipse (Communication)**: All-to-all routing, pipeline communication
2. **Rectangle (Computation)**: Attention, MLP, LayerNorm operations
3. **Parallelogram (Routing/Aggregation)**: Expert routing, output aggregation

## Key Features
- Each node shows exact GPU assignments
- Input/Output dimensions specified for every operation
- Attention operations broken down to operator granularity
- MLP layers shown with tensor parallelism
- All communication patterns represented
- Expert routing shown with dashed lines where applicable

## GPU Assignment Pattern
- **Stage 0**: GPUs 0-127
- **Stage 1**: GPUs 128-255
- **Stage 2**: GPUs 256-383
- **Stage 3**: GPUs 384-511
- **Stage 4**: GPUs 512-639
- **Stage 5**: GPUs 640-767
- **Stage 6**: GPUs 768-895
- **Stage 7**: GPUs 896-1023

## Expert Distribution
- Each stage has 64 experts (EP=64)
- Each expert uses 2 GPUs for tensor parallelism (TP=2)
- Expert ID 0: GPUs 0-1 (Stage 0), GPUs 128-129 (Stage 1), etc.

## Communication Patterns
1. **All-to-All**: Expert routing within each stage
2. **All-Reduce**: Tensor parallelism synchronization
3. **Pipeline**: Stage-to-stage data transfer
