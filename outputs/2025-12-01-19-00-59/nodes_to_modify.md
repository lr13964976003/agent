# Nodes Requiring Modification

## Critical Hardware Incompatibility Issue

**Problem**: The deployment strategy requires 32 GPUs, but the current system only has 1 GPU available.

**Affected Nodes**:

### 1. parallel_strategy.expert_parallelism_degree
- **Current Value**: 32
- **Issue**: Requires 32 GPUs but only 1 available
- **Required Change**: Reduce to match available hardware (1)

### 2. module_division.total_parts
- **Current Value**: 32
- **Issue**: Assumes 32-way division for 32 GPUs
- **Required Change**: Adjust to work with single GPU constraints

### 3. expert_distribution Mapping
- **Current State**: 32 separate GPU groups (0-31)
- **Issue**: Maps experts to non-existent GPUs
- **Required Change**: Consolidate to single GPU distribution

### 4. performance_projection
- **Current Projection**: 31,537 tokens/sec with 32 GPUs
- **Issue**: Unrealistic for single GPU system
- **Required Change**: Recalculate for actual hardware capabilities

### 5. gpu_groups Configuration
- **Current**: 32 individual GPU groups
- **Issue**: Only GPU 0 exists
- **Required Change**: Single GPU group containing only GPU 0

## Recommended Solution

The parallel strategy needs to be re-optimized for single GPU deployment:
- Use EP1_TP1 (1-way Expert Parallelism, 1-way Tensor Parallelism)
- Consolidate all experts onto the single available GPU
- Recalculate performance projections based on single GPU capabilities
- Ensure memory utilization stays within single GPU limits

## Validation Required

Before deployment, verify:
1. Single GPU memory capacity can handle consolidated expert load
2. Performance projections align with single GPU benchmarks
3. No communication overhead assumptions from multi-GPU setup