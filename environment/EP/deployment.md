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

* **Layers**: 16-layer, Multi-head attention + Mixture of experts, Each layer has 64 experts
* **Precision**: FP8
* **Batch size**: Each batch consists of 128 sequences.
* **Sequence Length**: 1024 tokens per sequence.
* **Token Dimension**: The dimension of each token is 1024.
* **Dimension of MHA**: The number of heads is 16 and the dimension of each heads is 64
* **Hidden size of MOE**: The hidden is of MOE is 2048


```
