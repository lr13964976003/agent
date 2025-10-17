# Phase 2: Methodology Extraction

## MA Separation Detailed Methodology

### 3.1 Problem Formulation (Mathematical Foundation)
```
T_attention = O(n²d)  // Sequential attention complexity
T_moe = f(expert_parallelism)  // Parallel expert execution

Temporal Mismatch: T_attention > T_moe when experts distributed
Goal: T_attention ≈ T_moe through attention parallelization
```

### 3.2.1 Attention Parallelization Strategy (Detailed)

**GPU Allocation for 16-GPU Setup:**
- Attention GPUs: 12 devices (GPUs 0-11)
- MoE GPUs: 4 devices (GPUs 12-15)

**Stage 1: Query-Key-Value Projection Parallelization**
```python
# Configuration parameters
num_attention_gpus = 12
num_heads = 32
hidden_dim = 4096
head_dim = hidden_dim // num_heads  # 128

# Head distribution across 12 GPUs
heads_per_gpu = num_heads // num_attention_gpus  # 2.67 heads/gpu
actual_distribution = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]  # 32 total

# GPU mapping
GPU_0: heads 0-2   (3 heads)
GPU_1: heads 3-5   (3 heads)
GPU_2: heads 6-8   (3 heads)
GPU_3: heads 9-11  (3 heads)
GPU_4: heads 12-14 (3 heads)
GPU_5: heads 15-17 (3 heads)
GPU_6: heads 18-20 (3 heads)
GPU_7: heads 21-23 (3 heads)
GPU_8: heads 24-25 (2 heads)
GPU_9: heads 26-27 (2 heads)
GPU_10: heads 28-29 (2 heads)
GPU_11: heads 30-31 (2 heads)
```

**Stage 2: Attention Score Computation**
```python
# All-reduce operation for attention outputs
# Each GPU computes partial attention for assigned heads
# Final aggregation via hierarchical all-reduce

# Communication pattern:
# Intra-node (4 GPUs per node): Reduce-scatter within node
# Inter-node (4 nodes): All-reduce across nodes

# GPU grouping for communication:
Node_0: GPUs [0,1,2,3]    
Node_1: GPUs [4,5,6,7]
Node_2: GPUs [8,9,10,11]
Node_3: GPUs [12,13,14,15]  # MoE GPUs
```

### 3.2.2 MoE Parallelization Strategy (Detailed)

**Expert Distribution Matrix:**
```python
# 16 experts distributed across 4 MoE GPUs
experts_per_gpu = 16 // 4  # 4 experts per GPU

GPU_12: experts [0,1,2,3]
GPU_13: experts [4,5,6,7]  
GPU_14: experts [8,9,10,11]
GPU_15: experts [12,13,14,15]

# Expert specifications:
- Expert_hidden_dim = 16384
- Top_K = 2
- Capacity_factor = 1.0
- Expert_dropout = 0.1
```

**Routing Computation:**
```python
# Gating network output processing
# From attention GPUs to MoE GPUs

# Communication flow:
1. Attention output (batch_size, seq_len, 4096) computed on 12 attention GPUs
2. All-reduce aggregation across attention GPUs
3. Broadcast to 4 MoE GPUs
4. Expert routing on MoE GPUs
5. Expert computation on assigned GPUs
6. All-to-all communication for expert outputs
```

### 3.3 Synchronization Mechanism (Technical Details)

**Time Prediction Model:**
```python
# Neural network predictor
input_features = [
    'sequence_length',      # 2048
    'hidden_dimension',     # 4096
    'num_active_experts',   # 2 * batch_size
    'gpu_utilization',      # Current load
    'memory_bandwidth'      # 2.0 TB/s for A100
]

# 3-layer MLP predictor
predictor_layers = [
    Linear(5, 64),
    ReLU(),
    Linear(64, 32),
    ReLU(), 
    Linear(32, 2)  # [T_attention, T_moe]
]

threshold = 0.05  # 5% execution time difference
```

**CUDA Synchronization:**
```cuda
// CUDA stream configuration
cudaStream_t attention_stream[12];    // 12 attention GPUs
cudaStream_t moe_stream[4];          // 4 MoE GPUs
cudaEvent_t attention_complete[12];
cudaEvent_t moe_complete[4];

// Synchronization pattern
for i in 0..11:
    cudaEventRecord(attention_complete[i], attention_stream[i]);
    
for j in 0..3:
    cudaEventRecord(moe_complete[j], moe_stream[j]);

// Barrier for next layer
for all streams:
    cudaStreamWaitEvent(next_layer_stream, attention_complete[i]);
    cudaStreamWaitEvent(next_layer_stream, moe_complete[j]);
```

### 3.4 Communication Optimization (Technical)

**Communication Volume Analysis:**
```python
# Per layer communication
attention_all_reduce = 2 * hidden_dim * batch_size * seq_len  # 2 * 4096 * 1024 * 2048 = 16.8 GB

# Hierarchical breakdown:
intra_node_reduce = attention_all_reduce / 4  # 4.2 GB per node
inter_node_reduce = attention_all_reduce / 4  # 4.2 GB across nodes

# MoE all-to-all
moe_all_to_all = batch_size * seq_len * hidden_dim * top_k  # 1024 * 2048 * 4096 * 2 = 16.8 GB
```

**Gradient Compression:**
```python
# 8-bit quantization for gradients
quantization_scale = max(abs(grad)) / 127
quantized_grad = int8(grad / quantization_scale)

# Top-K sparsification
top_k_percent = 0.1  # Keep top 10% of gradients
k = int(numel(grad) * top_k_percent)
```

### 4. Experimental Configuration (Methodology Details)

**Hardware Topology:**
```
4 nodes × 4 GPUs each
├── Node 0: GPUs [0,1,2,3]    (Attention)
├── Node 1: GPUs [4,5,6,7]    (Attention)  
├── Node 2: GPUs [8,9,10,11]  (Attention)
└── Node 3: GPUs [12,13,14,15] (MoE)

Interconnect:
- NVLink 3.0: 600 GB/s intra-node
- InfiniBand HDR: 200 Gb/s inter-node
```

**Memory Layout:**
```python
# Per GPU memory allocation (GB)
model_params = 23.1
activations = 18.7
gradients = 23.1
optimizer_states = 46.2
communication_buffers = 12.6
total_per_gpu = 123.7 GB  # A100 80GB × 1.55 oversubscription with offloading
```

**CUDA Kernel Optimizations:**
```cpp
// Custom attention kernel with fused operations
__global__ void attention_kernel(
    float* Q, float* K, float* V,
    float* output,
    int batch_size, int seq_len, int num_heads, int head_dim,
    int* head_mapping  // Maps threads to attention heads
);

// Expert routing kernel
__global__ void expert_routing_kernel(
    float* gate_scores,
    int* expert_assignments,
    int* expert_counts,
    int batch_size, int seq_len, int num_experts, int top_k
);
```

### 5. Evaluation Metrics Methodology

**TPOT Measurement:**
```python
# Time per output token calculation
def measure_tpot(model, input_sequence):
    start_time = cudaEventRecord()
    output_tokens = []
    
    for token in generate_tokens(model, input_sequence):
        end_time = cudaEventRecords()
        output_tokens.append((end_time - start_time) / num_tokens)
        start_time = end_time
        
    return np.mean(output_tokens)  # milliseconds per token
```

**TPS Measurement:**
```python
def measure_tps(dataloader, model):
    total_tokens = 0
    start_time = time.time()
    
    for batch in dataloader:
        tokens = batch['input_ids'].numel()
        model(batch)
        total_tokens += tokens
        
    end_time = time.time()
    return total_tokens / (end_time - start_time)  # tokens per second
```