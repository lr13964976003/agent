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


## **Basic Performance Requirements**
- **The time to the first token(TTFT)**: 10s
- **Time per output token(TPOT)**:  None
- **Throughput per GPU**: 100tokens/ms


```
