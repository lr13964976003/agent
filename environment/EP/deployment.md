```
## **Environment Condition**

* **GPUs**: Ample GPU resources, no limits.
* **Single-card computing power**: 400TFlops.
* **MFU utilization**: 60%.
* **VRAM Bandwidth**: 1.8TBps.
* **bandwidth utilization**: 80%.
* **Single-card video memory capacity**: 64GB.


---


## **Model Configuration**

* **weights**: 10B
* **Layers**: 16-layer, Each layer includes Multi-head attention + Mixture of experts, Each layer has 16 experts.
* **Precision**: FP16
* **Batch size**: Each batch consists of 128 sequences.
* **Sequence Length**: 128 ~ 10240 tokens per sequences in each batch.
* **Token Dimension**: The dimension of each token is 512.
* **Dimension of MHA**: The number of heads is 16 and the dimension of each heads is 32
* **Hidden size of MOE**: The hidden is of MOE is 1024

---


## **Current Parallelism Strategy**
- **Expert Parallelism (EP)**: 2-way - Each expert assigned to separate GPU
- **Tensor Parallelism (TP)**: 2-way - Intra-layer parallelism for attention and MLP
- **Pipeline Parallelism (PP)**: 2-way - Layer distribution across pipeline stages
- **Data Parallelism (DP)**: 2-way - Batch processing parallelism



```
