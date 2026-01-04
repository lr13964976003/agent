# Model Parameters and Architecture

## Model Architecture
- **Model Type**: Mixture-of-Experts (MoE) Transformer
- **Layer Count**: 16 layers (Note: Abstract mentions 4 layers, but experiments section specifies 16 layers)
- **Expert Configuration**: 16 experts per layer
- **Expert Type**: Each expert is an MLP (Multi-Layer Perceptron)

## Attention Mechanism
- **Number of Heads**: 32 attention heads
- **Head Dimension**: 128 dimensions per head
- **Total Attention Dimension**: 32 × 128 = 4096 dimensions
- **Attention Type**: Multi-Head Attention (MHA)

## Model Dimensions
- **Token Dimension**: 4096
- **MLP Hidden Size**: 16384
- **Precision**: BF16 (Brain Floating Point 16-bit)
- **Activation Function**: Not specified (typically GELU or ReLU in MoE models)

## Expert Configuration Details
- **Experts per Layer**: 16
- **Expert Architecture**: Standard MLP within transformer block
- **Expert Placement**: One expert per GPU (when E ≤ G)
- **Expert Replication**: When E > G, experts replicated across GPUs

## Gating Mechanism
- **Routing Strategy**: Top-K gating scores (K value not specified)
- **Load Balancing**: Dynamic adjustment of gating probabilities
- **Token Distribution**: Balanced across experts to prevent overloading

## Model Parallelism Integration
- **Tensor Parallelism (TP)**: Can be applied within expert if needed for large models
- **Data Parallelism (DP)**: Applied across replicas of MoE network
- **Expert Parallelism (EP)**: Primary parallelism at EP ≥ 16
- **Pipeline Parallelism (PP)**: Mentioned in baseline comparison

## Memory Requirements per Expert
- **Model Parameters**: Not explicitly stated in paper
- **Activation Memory**: 4096 (token dim) × batch size × sequence length
- **Intermediate States**: 16384 (MLP hidden) for expert computation
- **Total per GPU**: Sufficient for single expert with BF16 precision