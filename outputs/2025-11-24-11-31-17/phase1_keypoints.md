# Phase 1: Key Points Extraction

## Key Contributions

### 1. Large Cross-Node Expert Parallelism (Large EP)
- **Definition**: Expert Parallelism (EP) ≥ 16, with at most one expert per GPU
- **Core Innovation**: Maximizes computational parallelism by distributing experts across GPUs instead of colocating multiple experts on same device
- **Paradigm Shift**: Focuses on maximizing compute concurrency rather than reducing communication

### 2. Expert Placement Strategy
- **Principle**: One expert per GPU deployment
- **Distribution**: Cross-node expert placement with topology awareness
- **Constraint**: Each GPU hosts exactly one expert per layer when possible

### 3. Performance Achievement
- **Throughput**: 3.75× improvement (120K → 450K tokens/second)
- **Latency**: 3.8× reduction (8.3ms → 2.2ms TPOT)
- **Configuration**: 16-layer MoE, 16 experts/layer on 16 H100 GPUs

### 4. Technical Components
- **Token Routing**: Asynchronous cross-node token transfer
- **Load Balancing**: Dynamic gating probability adjustment
- **Communication Overlap**: CUDA streams/NCCL for compute-communication overlap
- **Pipeline Scheduling**: Fine-grained pipeline across MoE layers

### 5. Model Specifications
- **Architecture**: 16-layer MoE transformer
- **Experts**: 16 experts per layer (MLP-based)
- **Precision**: BF16
- **Token Dimension**: 4096
- **Hidden Size**: 16384 (MLP)
- **Sequence Length**: 10000 tokens
- **Batch Size**: 128 sequences

### 6. Baseline Comparison
- **Baseline**: TP=8, PP=2 with 16 GPUs (multiple experts per GPU)
- **Proposed**: Large EP with one expert per GPU
- **Resource**: Both use 16 H100 GPUs for fair comparison