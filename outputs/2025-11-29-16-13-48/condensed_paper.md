# Layer-wise Deployment Strategy for Large Neural Network Models

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

### 1. Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

Our method includes an analytical procedure to estimate the memory footprint of each partition and determine the optimal splitting scheme that fits the on-chip memory constraints. This approach facilitates scalable deployment of large models across multiple devices without sacrificing memory locality and efficiency.

### 2. Methodology

#### a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, the goal is to partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:

* The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card.
* The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order.
* The number of partitions $k$ is minimized or balanced to maximize hardware utilization.

Formally, for each partition $P_i$, the size $S(P_i)$ satisfies:

$$
S(P_i) = sum_{l_j in P_i} text{size}(l_j) leq C
$$

where $text{size}(l_j)$ is the estimated memory footprint of layer $l_j$.

The memory footprint of each layer includes:

* **Weights**: The parameter tensors stored for the layer.
* **Activations**: Intermediate outputs needed during inference or training.
* **Temporary Buffers**: Workspace memory required by operators during computation.

To accurately estimate $text{size}(l_j)$, we calculate:

$$
text{size}(l_j) = text{weight_size}(l_j) + text{activation_size}(l_j) + text{buffer_size}(l_j)
$$

Our method applies a greedy or dynamic programming algorithm to determine layer partitions. Starting from the first layer $l_1$:

1. Initialize an empty partition $P_i$.
2. Iteratively add subsequent layers $l_j$ to $P_i$, accumulating $S(P_i)$.
3. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$.
4. Start a new partition $P_{i+1}$ beginning from layer $l_j$.
5. Repeat until all layers are assigned.

After partitioning, each group $P_i$ is deployed on a separate accelerator card with the following steps:

* Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache.
* Execute the layers sequentially on the assigned card.
* Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication.

### 3. Experiments

We evaluate our proposed layer-wise deployment method for large models in the inference stage. The hardware platform consists of 16 NVIDIA H100 GPUs. We use a dense model: a 4-layer fully connected dense network.

Both models use BF16 precision and the model's weight size is 30B. We calculate the memory usage for weights, activations, and buffers for each layer, and determine the number of layers allocated to each card according to the allocation method.

Both models are tested with a batch size of 128 and a sequence length of 10000. The number of head is fixed at 32, the dimension of each head is fixed at 128, the hidden size of MLP is fixed at 16384. The baseline comparison is a standard tensor parallelism (TP) and pipeline parallelism (PP) setup, specifically TP=8 and PP=2, which fully utilizes the 16 GPUs (8 × 2 = 16).

We measure performance with two key metrics:

* **Tokens Per Second (TPS):** The number of output tokens generated per second.
* **Time Per Output Token (TPOT):** The average time to produce a single output token, in milliseconds.

| Model                   | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
| ----------------------- | --------------------- | ---- | -------------- | --------------- |
| Dense (4-layer)         | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078           |
| Dense (4-layer)         | Proposed Layer-wise   | 16   | 15,360         | 0.065           |

For the dense model, our proposed deployment method achieves a **20% increase in TPS** and a corresponding **17% reduction in TPOT** compared to the baseline. This improvement results from more efficient on-chip memory utilization, reducing memory access latency.

The baseline TP=8, PP=2 approach is effective but does not consider on-chip memory constraints explicitly, leading to more off-chip memory accesses and communication delays.

### 4. Conclusion

In this paper, we proposed a novel layer-wise deployment strategy for large neural network models that explicitly partitions the model layers across multiple accelerator cards with the constraint that each partition fits entirely within the SRAM or L2 cache of the target hardware. This approach significantly reduces off-chip memory accesses and improves inference efficiency.

Experimental results on a dense 4-layer model demonstrate that our method achieves substantial performance gains over the baseline tensor and pipeline parallelism setup. Specifically, we observed up to 20% improvement in throughput (TPS) and a corresponding reduction in latency (TPOT).

Future work includes extending this method to training workloads, exploring adaptive partitioning strategies under varying batch sizes, and applying the approach to even larger, more complex models to further validate its scalability and effectiveness.