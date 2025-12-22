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

* **weights**: 30B
* **Layers**: 16-layer, Each layer includes Multi-head attention + Mixture of experts, Each layer has 64 experts.
* **Precision**: FP16
* **Batch size**: Each batch consists of 128 sequences.
* **Sequence Length**: 128 ~ 10240 tokens per sequences
* **Token Dimension**: The dimension of each token is 1024.
* **Dimension of MHA**: The number of heads is 16 and the dimension of each heads is 64
* **Hidden size of MOE**: The hidden is of MOE is 2048

---


## **Current Parallelism Strategy**
- **Expert Parallelism (EP)**: 8-way - Each expert assigned to separate GPU
- **Tensor Parallelism (TP)**: 8-way - Intra-layer parallelism for attention and MLP
- **Pipeline Parallelism (PP)**: 8-way - Layer distribution across pipeline stages
- **Data Parallelism (DP)**: 8-way - Batch processing parallelism



```
