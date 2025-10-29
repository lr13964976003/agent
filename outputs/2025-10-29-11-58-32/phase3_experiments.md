# Helix Parallelism: Experiments Extraction (Phase 3)

## Experimental Setup

### 1. Hardware Configuration
- **Platform**: NVIDIA GB200 NVL72 with FP4 precision
- **Simulator**: In-house high-fidelity GB200 hardware simulator
- **Scale**: 1-64 GPUs within single GB200 node
- **Precision**: FP4 for weights, KV states, and arithmetic operations

### 2. Models Evaluated

#### 2.1 Llama-405B (Dense Model)
- **Architecture**: Dense 405B parameter model
- **Attention**: GQA with 128 query heads, 8 KV heads
- **Context**: Simulated 1 million token KV cache
- **Head dimensions**: Hsz = 128, H = 16,384
- **FFN dimension**: F = 65,536

#### 2.2 DeepSeek-R1 (MoE Model)
- **Architecture**: 671B parameter MoE model
- **Attention**: MLA attention (single KV head shared across 128 query heads)
- **Context**: Simulated 1 million token KV cache
- **Expert structure**: Multiple expert modules with expert parallelism

### 3. Baseline Comparisons
- **Search space**: >100,000 configurations across TP, EP, PP, KVP
- **Metrics**: Throughput/GPU vs. interactivity (tokens/sec/user reciprocal of TTL)
- **Optimization**: Exhaustive search for optimal configurations under TTL constraints
- **Batch scalability**: Maximum concurrent users under fixed TTL budget

### 4. Performance Results

#### 4.1 DeepSeek-R1 Results
- **TTL improvement**: Up to 1.5× reduction in token-to-token latency
- **Batch scaling**: 32× more concurrent users vs. baseline
- **Throughput**: 32× higher tokens/sec/GPU under same latency budget
- **Pareto frontier**: Significantly pushed outward for throughput-latency trade-off

#### 4.2 Llama-405B Results
- **Interactivity**: 1.13× improvement in maximum achievable interactivity
- **Throughput**: 4× higher throughput and batch capacity vs. TP sharding
- **KV duplication**: Eliminated by lifting TP's KV duplication ceiling via KVP
- **FFN parallelism**: Increased without reintroducing cache duplication

#### 4.3 Comparison with Medha
- **Medha limitation**: Tying TP between FFNs and attention not suitable for MLA
- **Medha communication**: Exposes all communication overheads (no overlap)
- **Helix advantage**: HOP-B optimization provides significant benefits

### 5. Ablation Study: HOP-B Impact

#### 5.1 HOP-B OFF Mode
- **Execution**: Sequential communication and computation
- **GPU stalls**: Idle periods during communication
- **Performance drop**: 
  - DeepSeek-R1: ~1% degradation (communication ~1% of total latency)
  - Llama-405B: ~12% drop in tokens/sec/user

#### 5.2 HOP-B ON Mode
- **Recovery**: Most lost TTL recovered through overlap
- **Critical factor**: Communication becomes larger fraction of TTL as context grows
- **Scalability**: Increasingly important with longer contexts

### 6. Configuration Parameters

#### 6.1 Optimal Parameter Ranges
- **TPA (Attention TP)**: 1-8 (constrained by K = 8 KV heads)
- **TPF (FFN TP)**: 1-64 (flexible, can exceed K)
- **KVP (KV Parallelism)**: 1-32
- **EP (Expert Parallelism)**: 1-8 (for MoE models)
- **N (Total GPUs)**: N = KVP × TPA ≤ 64

#### 6.2 Memory Distribution
- **KV cache per GPU**: S/KVP tokens
- **Weight shards**: Proportional to 1/TPF for FFN, 1/TPA for attention
- **Communication volume**: B × H per token (independent of sequence length)

### 7. Scaling Characteristics

#### 7.1 Context Length Scaling
- **Linear**: KV cache reads scale linearly with S (sequence length)
- **Sublinear**: With KVP, per-GPU traffic becomes sublinear
- **Practical limit**: Multi-million tokens achievable with proper KVP scaling

#### 7.2 Batch Size Scaling
- **Traditional limit**: Memory bandwidth bottleneck caps batch size
- **Helix advantage**: Enables 4-32× larger batches under TTL constraints
- **Efficiency**: Better GPU utilization across attention and FFN phases

### 8. Benchmark Methodology

#### 8.1 Simulation Approach
- **Accuracy**: High-fidelity GB200 modeling including communication latencies
- **Normalization**: All results normalized to baseline for trend analysis
- **Validation**: Cross-validation with production DeepSeek-R1 configurations

#### 8.2 Pareto Frontier Construction
- **Process**: Systematic variation of partitioning strategies and batch sizes
- **Optimization**: Maximum throughput for given TTL constraint
- **Representation**: Unified Pareto curve combining all optimal configurations

### 9. Failure Modes Identified
- **TP > K**: Leads to KV duplication and memory inefficiency
- **Communication exposure**: Without HOP-B, communication stalls reduce throughput
- **Expert imbalance**: In MoE, poor expert routing affects scaling
- **Memory hotspots**: Without staged KV concatenation, unbalanced memory growth

### 10. Future Validation Points
- **Sparse attention**: Compatibility with NSA (Natively Sparse Attention) mechanisms
- **Larger scales**: >64 GPU configurations across multiple nodes
- **Dynamic adaptation**: Runtime reconfiguration based on workload characteristics
- **Precision study**: Impact of FP4 vs. other precision formats on accuracy and performance