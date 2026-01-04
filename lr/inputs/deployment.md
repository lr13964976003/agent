```
#Prefill

## **Environment Condition**

* **NPUs**: Ample NPU resources, no limits.
* **Single-card computing power**: 400TFlops.
* **MFU utilization**: 60%.
* **VRAM Bandwidth**: 1.8TBps.
* **bandwidth utilization**: 80%.
* **Single-card video memory capacity**: 64GB.

---

## **Model Configuration**

* **name**: Qwen3-235B
* **weights**: 235B
* **Layers**: 94-layer, Each layer includes GQA + MOE, each layer has 128 experts.
* **Precision**: FP8
* **Token Dimension**: The dimension of each token is 4096
* **Dimension of Attention**: The number of heads is 64 and the dimension of each heads is 64
* **Hidden size of MOE**: The hidden is of FFN is 1536
* **Top-K Gate**: 8
* **Vocabulary size**: 151936
* **GQA kv heads**: 4

---

## **Input Data**

* **Batch size**: Each batch consists of 128 sequences.
* **Sequence Length**: The sequence length for each batch is variable, ranging from [128, 10240].
* **Sequence In**: 2048
* **Sequence Out**: 2048

---

## **Basic Performance Requirements**

- **The time to the first token(TTFT)**: 30s

```

