# Phase 2: Methodology Extraction - FA Pool Paper

## 3.1 System Architecture Components

### Base Layer (Primary Computational Layer)
- **Components**: Model's core components including embedding, positional encoding, and output layers
- **GPUs**: 8 GPUs (static allocation)
- **Responsibility**: Maintains model coherence and handles feed-forward network computations

### Attention Pool
- **Type**: Dynamically allocated set of GPUs
- **Purpose**: Dedicated to attention computation
- **Maximum Size**: Up to 32 additional GPUs
- **Activation Condition**: When sequence length exceeds 4096 tokens

### FFN Layer
- **Location**: Remains on the base layer (8 GPUs)
- **Function**: Feed-forward network computations
- **Architecture**: Each layer has one FFN with dimension 16384

### Resource Manager
- **Function**: Monitors sequence length and manages GPU allocation/deallocation
- **Trigger**: Sequence length threshold detection
- **Action**: Activates/deactivates attention pool GPUs

## 3.2 Dynamic Resource Allocation Strategy

### Operational Mechanism
1. **Sequence Length Monitoring**: Continuous real-time monitoring during inference
2. **Threshold Detection**: Compare against 4096-token threshold
3. **Resource Activation**: Activate additional GPUs when threshold exceeded
4. **Workload Distribution**: Partition attention computation across pool GPUs
5. **Result Aggregation**: Collect and synchronize results using hierarchical reduction
6. **Resource Deactivation**: Release pool resources when sequence drops below threshold

### Threshold Determination Formula
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
```
Where Overhead accounts for communication and synchronization costs

## 3.3 Attention Parallelization Strategy

### Block-wise Parallelization Algorithm
```
Input: Query Q, Key K, Value V, sequence length n, number of pool GPUs p
Output: Attention output O

1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool:
   - Extract block: Q_i = Q[i*b:(i+1)*b], K_i = K[i*b:(i+1)*b], V_i = V[i*b:(i+1)*b]
   - Compute local attention: O_i = FlashAttention(Q_i, K, V)
3. Synchronize and aggregate results: O = concat(O_0, O_1, ..., O_p-1)
4. Return final output O
```

### Key Parameters
- **Sequence Length (n)**: Variable input length
- **Pool GPUs (p)**: 1 to 32 depending on sequence length
- **Block Size (b)**: ceil(n/p) tokens per GPU

## 3.4 Communication Optimization Techniques

### KV Cache Sharing
- **Strategy**: Keys and values replicated across pool GPUs
- **Benefit**: Avoids communication during attention computation
- **Memory Trade-off**: Increased memory usage for reduced communication

### Asynchronous Execution
- **Mechanism**: Attention computation overlaps with FFN operations on base layer
- **Implementation**: Parallel processing pipelines between base and pool layers

### Hierarchical Reduction
- **Pattern**: Tree-based reduction for result aggregation
- **Steps**: Minimized communication steps through hierarchical structure
- **Efficiency**: <15% communication overhead even with 32 GPUs

## 3.5 Model Layer Specifications

### Transformer Layer Structure (4 layers total)
- **Multi-head Attention**: 32 attention heads
- **Hidden Dimension**: 4096 (d_model)
- **Feed-forward Dimension**: 16384 (4 * d_model)
- **Attention Head Dimension**: 128 (4096/32)

### Activation Functions
- **Attention**: Multi-head scaled dot-product attention with Flash Attention optimization
- **FFN**: GELU activation function
- **Normalization**: Pre-norm with RMSNorm

## 3.6 Memory Architecture

### Base Layer Memory
- **Per GPU**: 65GB (model parameters + activations)
- **Content**: Model weights, embeddings, FFN computations, output layers

### Attention Pool Memory
- **Per GPU**: 45GB (reduced due to block-wise computation)
- **Content**: Attention weights, KV cache blocks, temporary attention matrices
- **Scaling**: Memory usage scales with sequence length and pool size

## 3.7 Resource Allocation Patterns

### Dynamic Allocation Logic
- **Below 4096 tokens**: No attention pool activation, use base 8 GPUs
- **4096-8192 tokens**: Activate 8-16 attention pool GPUs
- **8192-16384 tokens**: Activate 16-24 attention pool GPUs
- **Above 16384 tokens**: Activate 24-32 attention pool GPUs

### Performance Plateau
- **Optimal Pool Size**: 24 GPUs (gains plateau beyond this)
- **Efficiency Factor**: Resource utilization averages 85-92%