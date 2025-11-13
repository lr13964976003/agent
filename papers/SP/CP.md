<p>CONTEXT PARALLELISM FOR SCALABLE MILLION-TOKEN INFERENCE</p>
<p>5
2
0
2</p>
<p>r
p
A
1
2</p>
<p>]</p>
<p>C
D
.
s
c
[</p>
<p>3
v
3
8
7
1
0
.
1
1
4
2
:
v
i
X
r
a</p>
<p>Amy (Jie) Yang 1 Jingyi Yang 1 Aya Ibrahim 1 Xinfeng Xie 1 Bangsheng Tang 1 Grigory Sizov 1
Jeremy Reizenstein 1 Jongsoo Park 1 Jianyu Huang 1</p>
<p>ABSTRACT
We present context parallelism for long-context large language model inference, which achieves near-linear scaling
for long-context prefill latency with up to 128 H100 GPUs across 16 nodes. Particularly, our method achieves
1M context prefill with Llama3 405B model in 77s (93% parallelization efficiency, 63% FLOPS utilization) and
128K context prefill in 3.8s. We develop two lossless exact ring attention variants: pass-KV and pass-Q
to cover a wide range of use cases with the state-of-the-art performance: full prefill, persistent KV prefill and
decode. Benchmarks on H100 GPU hosts inter-connected with RDMA and TCP both show similar scalability
for long-context prefill, demonstrating that our method scales well using common commercial data center with
medium-to-low inter-host bandwidth.</p>
<p>1</p>
<p>INTRODUCTION</p>
<p>Contemporary large language models (LLMs), such as
Llama (Touvron et al., 2023a;b; Llama Team, 2024), Gem-
ini (Gemini Team, 2023; 2024), GPT-4 (Achiam et al.,
2023), require significant computational resources for infer-
ence, especially with long context lengths: OpenAI GPT-4o
128K context length (ope), Anthropic’s Claude with 200K
context length (ant), Google’s Gemini 1.5 Pro with 1M con-
text length (goo). With a single H100 GPU host (8 GPUs),
it can take 60 seconds to serve 128K context length1 or
1200 seconds to serve 1M context length for Llama3 405B
model. Context parallelism (CP) is a system optimization
technique that improves the latency and scalability of LLM
inference, particularly for long contexts. Without modifying
the underlying dense attention algorithms, CP offers several
advantages for long-context LLM inference:</p>
<p>• Compute parallelization: CP distributes computation
across multiple GPUs in order to reduce latency, in
contrast with pipeline parallelization (PP) (Huang et al.,
2019) that improves throughput but not latency.</p>
<p>• Communication message size reduction: Compared
to tensor parallelism (TP) (Shoeybi et al., 2019), CP
demands less communication bandwidth in multi-host</p>
<p>1Meta Platforms, Inc., Menlo Park, California, USA. Corre-</p>
<p>spondence to: Jianyu Huang <a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;&#106;&#105;&#97;&#110;&#121;&#117;&#104;&#117;&#97;&#110;&#103;&#64;&#109;&#101;&#116;&#97;&#46;&#99;&#111;&#109;">&#106;&#105;&#97;&#110;&#121;&#117;&#104;&#117;&#97;&#110;&#103;&#64;&#109;&#101;&#116;&#97;&#46;&#99;&#111;&#109;</a>.</p>
<p>Proceedings of the 8 th MLSys Conference, Santa Clara, CA, USA,
2025. Copyright 2025 by the author(s).</p>
<p>1Google’s Gemini 1.5 Pro (Sep 2024) has a latency of 20.43
seconds for time to first token on 100K context length, from
https://artificialanalysis.ai/models/gemini-
1-5-pro/prompt-options/single/100k.</p>
<p>environments, by maintaining a communication size
that is orders of magnitude smaller than TP, especially
for inter-node communication.</p>
<p>• KV cache distribution: Key and value (KV) embed-
dings grow linearly with context length. CP distributes
the storage of KV embeddings across multiple GPUs,
enabling larger batch sizes with the addition of more
CP ranks.</p>
<p>To the best of our knowledge, this is the first paper to dis-
close the system implementation details on applying context
parallelism in inference scenario. Our main contribution
lies in the adaptation and optimization of ring attention (Liu
et al., 2023) for efficient LLM inference with long con-
text lengths. While the previous work primarily focuses on
leveraging ring attention to enhance training throughput for
long sequences, this paper identifies and addresses unique
challenges posed by inference:</p>
<p>• Support for multi-turn prefill and decoding: We
recognize the importance of multi-turn conversations,
a common characteristic of online LLM applications.
Unlike prior research focused on training, we intro-
duce novel strategies on load-balanced sharding for
persistent KV cache and parallelization algorithms that
leverage sharded KV cache across multi-turn prefill
and decode. These mechanisms are crucial for main-
taining conversation history during inference.</p>
<p>• Optimization for latency: Latency is critical for user
experience in real-time inference. To optimize latency
in multi-turn conversations, we developed pass-KV
and pass-Q ring attention variants and heuristics to</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>dynamically select the ring attention algorithms for the
lowest latency under varying context lengths and KV
cache hit rates.</p>
<p>• Compute and memory load balancing: To maintain
balanced load among CP ranks across batched requests
with varying input lengths, we introduce load-balanced
sharding of both input tokens and KV cache entries.
Previous work targets training typically with uniform
sequence length. We proposed innovative algorithms
to ensure even distribution of compute and KV cache
memory across CP ranks, contributing to improved
overall performance and scalability.</p>
<p>In essence, our work extends context parallelism to effi-
ciently address the challenges and requirements of serving
millions of tokens in LLM inference. We introduce novel al-
gorithms and heuristics for optimizing ring attention, demon-
strating their effectiveness in reducing latency, improving
KV cache utilization, and enabling scalable distributed infer-
ence for long-context LLMs. Since our method focuses on
system-level optimizations, it can be seamlessly integrated
with architectural innovations or algorithmic enhancements
to further amplify performance gains.</p>
<p>2 BACKGROUND</p>
<p>2.1 Large Language Models (LLM)</p>
<p>Since the introduction in the seminal work (Vaswani, 2017),
the transformer model architecture has become the funda-
mental building block for modern language models. Re-
cently, language models have increased exponentially in
complexity (measured in number of parameters). Examples:
BERT was trained with 0.34B parameters in 2018 (Devlin,
2018), 1.5B parameter GPT-2 was released in 2019 (Rad-
ford et al., 2019), and 175B parameter GPT-3 was released
one year later in 2020 (Brown, 2020), and the latest Llama
3.1 model pushed to 405B parameters (Llama Team, 2024).</p>
<p>Besides the parameter number, the context length is another
important indicator of LLM’s capabilities. In general, a
longer context window indicates better capability to handle
a large body of input texts, audios, images, and videos. Mod-
ern LLMs support 128K to more than 1M context lengths
(ope; ant; goo).</p>
<p>2.2 Challenges with Serving Long Context LLM</p>
<p>In this work, we mainly address the challenges with ex-
tremely large (128K-1M) context lengths.</p>
<p>• Compute: While an W -parameter Transformer model
requires 2 · W matrix multiplication FLOPs for each
token during inference or forward pass (Kaplan et al.,
2020), the pairwise attention architecture found in</p>
<p>mainstream transformers (Vaswani, 2017) incurs a
quadratic cost in FLOPs w.r.t. context lengths, which
would be dominating in long context cases. Several
approximate and sparse methods were proposed, in-
cluding focusing attention on a subset of tokens, and
employing a combination of local and global attention
strategies. Techniques such as window attention (Liu
et al., 2021), local attention (Xiong et al., 2021), Lin-
former (Wang et al., 2020), and semi-local sparse at-
tention (Jiang et al., 2024; Beltagy et al., 2020) are
examples of such innovations that help manage the
computational cost.</p>
<p>• Memory: Memory usage for LLMs, particularly the
KV cache (Pope et al., 2023), scales linearly with
the context length. Model compression techniques
such as KV cache quantization are crucial for bend-
ing the growth curve:
lower precision formats like
3-bit, INT4/8 or FP8 can achieve a 2× to 4× reduc-
tion in memory requirements compared to using 16-
bit (Hooper et al., 2024; Lin et al., 2024). Grouped
Query Attention (GQA) (Ainslie et al., 2023) and
MQA (Shazeer, 2019) were widely adopted to reduce
memory usage by reducing the number of KV heads
by 8× to 64×. Additionally, strategies like paged at-
tention (Kwon et al., 2023) have been developed to
provide efficient page-like memory management for
large numbers of tokens.</p>
<p>2.3 Prior works on Long Context LLM</p>
<p>The following are the main directions to achieve efficient
long context window LLM inference:</p>
<p>• New model architectures:</p>
<p>introduce long context
window comprehension components at pretraining
stage (Munkhdalai et al., 2024).</p>
<p>• Post-training changes: modify a pretrained model
with shorter context window to support longer or even
infinite context windows (Xiao et al., 2023).</p>
<p>• System-level optimizations: preserve the model ar-
chitecture, instead improve the scalability of existing
dense attention algorithms to leverage more compute
resources (Li et al., 2021; Brandon et al., 2023; Liu
et al., 2023; Wu et al., 2024; Li et al., 2023; Jacobs
et al., 2023; Fang &amp; Zhao, 2024).</p>
<p>Our work falls into the third category, and can be used in
conjunction with methods from the other two categories
with minor or no modifications. Our method accelerates
future algorithmic research or real-world LLM applications
for long-context LLM serving, and also provides the flex-
ibility to trade off model inference latency with hardware</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>Notation
NH
NKV
DH
D
Q, K, V
B
W
T
P
e
C
BW
bid
j |P i
T i
j
Li
N
NT P
k|KV s
Os
k</p>
<p>Qs</p>
<p>k |bids
k</p>
<p>T P 8, T P 16</p>
<p>CPN</p>
<p>T T F T</p>
<p>T T IT</p>
<p>Table 1. Notation table.</p>
<p>Description
Number of query heads (or attention heads)
Number of key/value heads
Head dimension in Transformer
Model dimension in Transformer: DH · NH
Query, Key, Value tensors
Batch size (# of input sequences)
Model parameter size
Input sequence length
Previously cached KV length
Element data type size
Peak compute
Peak comm bandwidth
Decode batch id</p>
<h1>of {new tokens | cached KV tokens} in i-th</h1>
<p>sequence sharded to CP rank j
KV length in i-th sequence: maxN −1
Number of hosts/nodes/CP ranks
TP group size (# of GPUs in a TP group)
{Query | key and value tensors | batch id }
on rank k, originally allocated to rank s
Attention output from Qk and KV s
Tensor parallel sharding over 8 GPUs
on one node or 16 GPUs on two nodes
Context parallel sharding on N nodes
with TP8 for each node (same as CPN +T P 8)
Time-to-first-token: latency for prefilling
the whole input tokens
Time-to-incremental-token: latency for
decoding each output token</p>
<p>j=0 (P i</p>
<p>j + T i
j )</p>
<p>capacity depending on the latency requirements of specific
applications.</p>
<p>3 CONTEXT PARALLEL INFERENCE</p>
<p>3.1 Notations</p>
<p>The notations used in this paper are summarized in Table 1.</p>
<p>3.2 Model Parallelization</p>
<p>Large language models are commonly parallelized across
multiple GPUs using a combination of various parallelism
paradigms: Tensor Parallelism (TP) (Shoeybi et al., 2019;
Korthikanti et al., 2023) partitions the weights of fully con-
nected layers (i.e., linear layers) by alternating the shard-
ing in row and column dimensions. Pipeline Parallelism
(PP) (Narayanan et al., 2021) shards layers into different
pipeline stages, and splits input tensors along the batch
size dimension into micro-batches to orchestrate a pipeline
schedule to optimize the system throughput.
Instead of
sharding model weights, Context Parallelism (CP) (Li
et al., 2021) distributes input tokens to multiple GPUs along
the sequence length dimension. CP ranks communicate
QKV tensors for attention, which is the only computation</p>
<p>Table 2. Communication and memory cost comparison between
tensor parallel (TP) and context parallel (CP) for full prefill. T :
sequence length, DH : head dimension, NH : # of attention heads,
NKV : # of key/value heads, NT P : TP group size, W: model
parameter size. Total comm cost shows the communication cost
per transformer block.</p>
<p>COLLECTIVE
COMM PER 2 LINEAR
COMM PER ATTN
TOTAL COMM
PARAMETER SIZE</p>
<p>TP
ALLREDUCE
T · NH · DH
0</p>
<p>CP
SENDRECV
0
T · NKV · DH
2 · (T · NH · DH ) T · NKV · DH</p>
<p>W
NT P</p>
<p>W</p>
<p>with dependency between tokens in the same sequence.</p>
<p>Both TP and CP reduce latency when scaled to multiple
nodes. Compared with TP, CP provides an alternative design
choice for trade-offs between memory consumption and sys-
tem performance. As detailed in Table 2, CP communicates
token embeddings on attention layers while TP communi-
cates on linear layers. CP has less communication traffic
for two reasons: (1) Contemporary LLMs have more lin-
ear layers than attention layers: each canonical transformer
block has four linear layers and one attention layer. (2) CP
may communicate KV tensors instead of Q tensors, which
leads to much less communication for models with GQA
(Ainslie et al., 2023). For Llama3 405B model with 128
query heads and 8 KV heads (NKV = 8 vs. NH = 128),
communicating KV heads has 16× smaller message sizes
than communicating query heads (Llama Team, 2024). CP’s
communication cost advantage over TP results in significant
latency improvements for multi-node inference, as inter-
connect bandwidth between nodes are several times lower
than intra-node bandwidth (Section 4.2.2). Although CP
offers lower communication costs, it incurs higher memory
consumption because its lack of model weight sharding.</p>
<p>In this paper, we design and implement an efficient LLM
inference system with CP to unblock such a trade-off when
scaling out the number of GPUs. In practice, we set TP size
into a number (usually 8) to fit the model into GPU memory,
and we leverage CP to efficiently scale out into multiple
nodes as it saves communication traffic.</p>
<p>3.3</p>
<p>Inference Prefill and Decode Attention</p>
<p>We characterize large language model online inference for
multi-turn messaging into three stages: full prefill, partial
prefill, and decode. When user initiates the conversation
with an initial prompt, the entire prompt goes through full
prefill, where we compute full causal attention between
tokens. Projected key and value tensors from multi-head
(MHA) or grouped query attention (GQA) (Ainslie et al.,
2023) are saved in GPU HBM as KV cache. After the initial</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>full prefill, the model then starts generating a response with
auto-regressive decoding, where a new token attends to
previously cached KV tensors and outputs response tokens
one at a time. KV values generated during decoding stage
are also saved in KV cache. After the server returns a
response, the user may give a follow-up prompt, which
will go through partial prefill (or persistent KV prefill),
where tokens within the new prompt attend to themselves as
well as all cached tokens in the previous prompt and model
response. This process may repeat multiple times in real
world applications, which requires persistency of KV cache
between prompts from the same user.</p>
<p>3.4 Computation and Communication Modeling</p>
<p>Each of the three stages of multi-turn online LLM inference
carries different performance characteristics.</p>
<p>Assume we have an input sequence with length T , with
previously cached KV length P , and a generic GQA model
with NH query heads, NKV key and value heads and model
dimension D. We have the following shapes for query (Q),
key (K), and value (V) embeddings:</p>
<p>shape(Q) = [T, NH ,</p>
<p>D
NH</p>
<p>]</p>
<p>shape(K) = shape(V ) = [(T + P ), NKV ,</p>
<p>D
NH</p>
<p>]</p>
<p>When Q and KV have the same lengths, passing KV around
in ring attention incurs smaller traffic than passing Q, and
the communication can be fully overlapped with attention
computation (Li et al., 2021). LLM training guarantees this
property len(Q) = len(K) = len(V ) = T , or equiva-
lently, P = 0. This is not necessarily true for inference as
len(Q), len(K), and len(V ) depend on user behaviors and
KV cache configurations.</p>
<p>For inference, with high persistent KV hit rate, the ring
attention algorithm that always passes KV around may not
provide the best performance, as:</p>
<p>• Attention computation is much faster with fewer Q
than cached KV. Communication cost will be exposed
on critical path if not fully overlap with computation.</p>
<p>• When Q is significantly smaller than the cached KV,
communicating the full persistent KV would be signifi-
cantly more costly than communicating Q.</p>
<p>To achieve better inference performance for full prefill, per-
sistent KV prefill, and decode, we extend ring attention with
an option to pass Q instead of KV, when passing Q leads to
less communication cost. Specifically, Q embeddings have
smaller size than KV embeddings if:</p>
<p>Table 3. GQA attention complexity for full prefill and partial prefill
(e: number of bytes per element).</p>
<p>FLOPS
Q BYTES
KV BYTES</p>
<p>FULL PREFILL
4T 2D
T De
2T D NKV
NH</p>
<p>e</p>
<p>PARTIAL PREFILL
4T D(T + P )
T De
2(P + T )D NKV
NH</p>
<p>e</p>
<p>T
T + P</p>
<p>≤ 2 ·</p>
<p>NKV
NH</p>
<p>(1)</p>
<p>Note that the right hand side (RHS) is constant given a pre-
trained model. Therefore we can use the RHS as a constant
threshold to switch between passing KV embeddings and
Q embeddings dynamically depending on T
T +P , or the KV
cache miss rate ( 1− KV cache hit rate).</p>
<p>Specifically, for full prefill where P = 0, communicating
KV embeddings results in a smaller message size for GQA
models with NH &gt; 2 × NKV . For decoding where T =
1, communicating Q embedding almost always results in
smaller communication sizes. Consequently, we leverage
ring pass-KV for full prefill, and ring pass-Q for decode
and partial prefill with high KV cache hit rate.</p>
<p>To understand whether communication can be reliably over-
lapped with attention computation with varying persistent
KV hit rates, we approximate the attention computation
and QKV communication latency using a simple roof-line
model (Table 3).</p>
<p>Let’s assume a system with peak compute of C, bandwidth
of BW for QKV communication, new token length T , and
cached token length P . We focus the analysis on prefill with
low persistent KV hit rate, which is compute-bound and
the culprit of long (e.g. 60s) prefill latency for inference.
In the following analysis, we aim to identify values of P
and T such that the communication latency is smaller than
the computation latency. In simplified terms: F LOP S
≥
min(Qbytes,KVbytes)
BW</p>
<p>C</p>
<p>.</p>
<p>For low-to-medium KV cache hit rate prefill, we will not be
bound by ring pass-KV communication if:</p>
<p>4 · T · D(T + P )
C</p>
<p>≥</p>
<p>2 · (T + P ) · D · e · NKV
NH
BW</p>
<p>To extend to multi-host distributed inference, we would fur-
ther partition each CP rank with TP over intra-node GPUs,
and add additional CP nodes to increase parallelization on
context dimension. For CP over N nodes, we would be
able to hide ring pass-KV communication latency under
attention computation if:</p>
<p>4 · T · D(T + P )
N · C</p>
<p>≥</p>
<p>2 · (T + P ) · D · e · NKV
NH
BW</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>T ≥ N ·</p>
<p>C · NKV · e
2 · NH · BW</p>
<p>(2)</p>
<p>Note that the threshold for T , the length of new tokens is a
static threshold with respect to a given model and hardware,
which is independent of KV cache hit on the previously
cached KV length P .</p>
<p>Similarly, in a distributed inference setting with CP over
N nodes, we will not be bottlenecked by ring pass-Q
communication if:</p>
<p>4 · T · D(T + P )
N · C</p>
<p>≥</p>
<p>T · D · e
BW</p>
<p>(T + P ) ≥ N ·</p>
<p>e · C
4 · BW</p>
<p>(3)</p>
<p>Note that RHS is also static with respect to one particular
system. As we have discussed, we will leverage pass-Q
when the number of new tokens to prefill T is significantly
smaller than the number of cached tokens P . In this case,
whether we will be able to completely overlap the latency for
communicating Q is determined by the total context length
(T + P ). Sufficiently large total context length would allow
us to overlap the pass-Q communication regardless of KV
cache hit rate.</p>
<p>Algorithm 1 Pass-KV vs. Pass-Q Partial Prefill Heuristics
T</p>
<p>if T ≥ N C·NKV ·e</p>
<p>2·NH ·BW or</p>
<p>T +P ≥ 2 NKV
NH</p>
<p>then</p>
<p>pass-KV</p>
<p>else</p>
<p>pass-Q</p>
<p>end if</p>
<p>To summarize, we adaptively switch between pass-KV
and pass-Q for inference partial prefill following the
heuristics in Algorithm 12. It’s worth noting that the full
prefill can be considered as a special case where P = 0,
while decoding can be viewed as a special case where T =
1. We can calculate the static thresholds for this heuristics
once based on the system and model spec, and use the
heuristics to choose which options to use dynamically for
the optimal performance in a wide combination of total
context length and KV cache hit thresholds.</p>
<p>3.5 Ring Pass-KV, Pass-Q Prefill</p>
<p>We implemented both pass-KV and pass-Q ring atten-
tion to minimize the communication latency with different
context lengths and KV cache hit rate. In this section, we
delve into the implementation details for achieving effective</p>
<p>2In practice, the achieved BW and C are lower than the theo-
retical hardware peaks. We start with these peak values and then
fine-tune the thresholds based on empirical data.</p>
<p>Figure 1. Load-balanced CP sharding with fused inputs in full
prefill with 2 CP ranks (CP2). We have 2 input sequences: S1,
S2. Each is partitioned evenly into 4 chunks: Qi / Ki, where
i = 1, 2, 3, 4.</p>
<p>load balancing and communication overhead management,
which are critical to the the scalability of distributed context
parallel inference.</p>
<p>3.5.1 Load Balanced Sharding</p>
<p>In causal attention each token attends to all tokens before it
in the same sequence. Naively partitioning all tokens evenly
over CP ranks in the order of the original sequence results
in imbalanced compute over different CP ranks. Prior work
leverages order permutation and uneven partition to achieve
load balance for causal attention (Cho et al., 2024; Brandon
et al., 2023). To support maximum context length provided
by the pretrained model without OOM on any particular CP
rank with heavier load, we aim for load-balancing for both
attention compute and KV cache capacity. To shard an input
sequence into N CP ranks, we partition the sequence evenly
into 2 × N chunks: C0, C1, ..., C2×N −1, and have each CP
rank i take two chunks: (Ci, C2×N −i−1).</p>
<p>For fused variable length inputs in full prefill, we partition
each individual sequence in the same way and pad the input
sequence length if needed (Figure 1).</p>
<p>For partial prefill with new tokens (total length: T ) and
cached tokens (total length: P ), we apply the load-balanced
sharding in the dimension of the new tokens regardless of
cached tokens (Figure 2).</p>
<p>3.5.2 Ring Pass-KV Algorithm</p>
<p>In Llama3 training (Llama Team, 2024), the all-gather based
pass-KV algorithm is utilized, which initially performs an
all-gather on the key and value tensors, followed by com-
puting the attention output for the local query tensor chunk.</p>
<pre><code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Context Parallelism for Scalable Million-Token Inference
</code></pre>
<p>Algorithm 2 Fused Varseq Ring Pass-KV Partial Prefill</p>
<p>for i = 0 to B − 1 do</p>
<p>Li ← max0≤j&lt;N (P i</p>
<p>j + T i
j )</p>
<p>end for
// On CP rank k
k ← concatB−1
i=0 (pad(P i
KV k
Qk ← concatB−1
i=0 (T i
k)
p ← (k − 1) mod N
for j = 0 to N − 1 do</p>
<p>k + T i</p>
<p>k, Li))</p>
<p>Figure 2. Load-balanced CP sharding with fused inputs partial pre-
fill with 2 CP ranks (CP2). We have 2 input sequences: S1, S2.
Load-balanced sharding is applied to the new token Qi dimension
(4 chunks), regardless of how cached token dimension Ki is parti-
tioned in partial prefill.</p>
<p>The all-gather communication latency becomes a bottleneck
in the critical path, complicating the overlap of operations
during inference, especially with variant sequence lengths
in a batch and partial prefill used in multi-turn chat. Con-
versely, the ring-based pass-KV approach, while reducing
the computation in smaller granularity, facilitates the over-
lapping of SendRecv with attention computations within the
ring loop.</p>
<p>We further make a modification to the ring pass-KV al-
gorithm (Liu et al., 2023) to better suit the partial prefill
use case in multi-turn chats. Here an invariant we need
to maintain for the ring algorithm is passing equal-sized
messages between CP ranks to adhere to collective commu-
nication interfaces. CP ranks hold different numbers of KV
embeddings as a result of multi-turn chat. Padding and de-
coding introduce slight variations in KV embedding length
per rank even though our load-balanced sharding distributes
KV embeddings evenly.</p>
<p>Assume we have N CP ranks CP0, CP1, ..., CPN −1 with
cached KV lengths of P0, ..., PN −1, and partial prefill new
tokens of length T . We pass KV embeddings of length
max0≤i&lt;N (Pi) + ⌈T /N ⌉ around CP ranks in a ring (Fig-
ure 3), where ⌈T /N ⌉ indicates the lengths of load-balanced
sharding (Section 3.5.1) of T tokens over N ranks.</p>
<p>For fused variable sequence lengths (Varseq) partial prefill
of B sequences in one batch, assume we have sequences
S0(P 0, T 0), ..., SB−1(P B−1, T B−1). The i-th sequence
Si has P i cached KV embeddings, T i new prefill tokens,
with P i
j new tokens sharded to CP
rank j. We have Algorithm 2 for a ring pass-KV partial
prefill with fused inputs for CP over N hosts. KV s
k indicates
key and value embeddings received from rank k which is
originally allocated to rank s.</p>
<p>j cached tokens and T i</p>
<p>In the ring algorithm, Qk, Q embeddings sharded to rank</p>
<p>k to next rank</p>
<p>s ← (k − j) mod N
Rank k sends KV s
Rank k receives KV s
Compute Os
KV s
end for
Compute Ok ← mergeN −1</p>
<p>k ← GQA(Qk, KV s
k )
k ← KV s
p</p>
<p>s=0 (Os
k)</p>
<p>p from previous rank</p>
<p>k, need to attend to all key and value embeddings sharded
to all ranks: KV0, KV1, ..., KVN −1. The attention com-
pute between Qk and KVj is overlapped with SendRecv for
KVj−1 from a neighbor rank. We pass KVj in a ring N − 1
times and each rank executes N partial attention compute.</p>
<p>At the end of the ring algorithm loop, each CP rank k will
have the attention output of Os
k with s = 0, 1, ..., N − 1,
where Os
k denotes the attention output from Qk and KV s
(key and value embeddings originally sharded to rank s, see
bottom of Figure 3). We then apply a merge attention opera-
tor (Juravsky et al., 2024) to get the result of Qk interacted
with all KV embeddings across CP ranks (See Appendix B,
Equation (4)).</p>
<p>3.5.3 Ring Pass-Q Algorithm</p>
<p>Passing Q embeddings around while keeping K and V em-
beddings stationary will have partial attention results scat-
tered across CP ranks. We need to have another round
of collective communication over CP process group to re-
store the partial outputs to the original source rank. Fol-
lowing the notations of ring pass-KV algorithm in Section
3.5.2, we have Algorithm 3 for ring pass-Q attention (Fig-
ure 4). Similarly, Qs
k indicates a Q embedding received
from rank k which was initially allocated to rank s. Note
that with pass-Q we have the guarantee that all CP ranks
have the same embedding lengths for query as a result of
load-balanced sharding (Section 3.5.1).</p>
<p>All2All for partial attention outputs is on the critical path and
therefore introduces an additional communication overhead
apart from the communication for passing query embedding.
The analysis for overlapping query embedding and attention
in Equation (2) and (3) only applies to the ring communica-
tion. The heuristics in Algorithm 1 for switching between
pass-KV and pass-Q doesn’t take All2All latency into</p>
<pre><code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Context Parallelism for Scalable Million-Token Inference
</code></pre>
<p>Figure 3. Ring Pass-KV Attention with 4 CP ranks (CP4).</p>
<p>account3.</p>
<p>3.6 Ring Pass-Q Decode</p>
<p>With multi-turn prefill and decode, key and value embed-
dings of the decode tokens are also stored in the KV cache.
As decoding generates one response token at a time for each
sequence, each decode batch contains exactly one token
for each sequence in the batch. If context-parallel decode
consistently shards the decoding tokens of a sequence to one
specific rank, the rank that handles both decode and prefill
will encounter load imbalance issues: it will have longest
KV cache and out-of-memory (OOM) before other ranks
reach their KV cache capacity.</p>
<p>To ensure we utilize full KV cache capacity from all CP
ranks, we implemented batched ring pass-Q decode where
we offset by 1 index for each decode iterations and shard
batched decode evenly with round-robin. With exactly 1
token per sequence for decode, we pass Q rather than K and
V embeddings to minimize communication size (Equation
1). Algorithm 4 summarizes our CP decode algorithm with
the same notations used for prefill algorithms.</p>
<p>Similar to ring pass-Q prefill, we need to permute the
partial attention output order and communicate scattered
partial attention outputs back to the original source ranks.</p>
<p>3We present a refined algorithm in Appendix C and provide a</p>
<p>detailed time breakdown for validations in Table 5.</p>
<p>Figure 4. Ring Pass-Q Attention with 4 CP ranks (CP4).</p>
<p>Algorithm 3 Fused Varseq Ring Pass-Q Partial Prefill</p>
<p>// On CP rank k with KVk
Qk ← concatB−1
i=0 (T i
k)
p ← (k − 1) mod N
for j = 0 to N − 1 do</p>
<p>k to next rank</p>
<p>s ← (k − j) mod N
Rank k sends Qs
Rank k receives Qs
Compute Ok
Qs
k ← Qs
p
end for
s }N −1
Permute {Ok
Compute Ok ← mergeN −1</p>
<p>s ← GQA(Qs</p>
<p>s=0 (Os
k)</p>
<p>p from previous rank
k, KVk)</p>
<p>s=0 and All2All to recover {Os</p>
<p>k}N −1
s=0 .</p>
<p>4 EXPERIMENTS</p>
<p>4.1 Experiment Setup</p>
<p>We used Llama3 405B model with row-wise quantized FP8
weights (Llama Team, 2024) for feed forward layers after
GQA. Llama3 405B is a dense transformer model with
126 transformer layers, 16384 model dimension, 128 query
heads, and 8 key and value heads (Table 9).</p>
<p>We ran our performance benchmarks on the Grand Teton
platform (Meta Engineering, 2022), where each host has 8
Nvidia H100 GPUs fully connected with NVLink (“host”
and “node” are interchangeable in the subsequent text).
Each H100 GPU is equipped with 96GB HBM2e with 2.4
TB/sec peak memory bandwidth. We tested on two subtypes
of Grand Teton platforms: Grand Teton Training (GTT) and</p>
<p>CP rank0Q0K0V0CP rank1Q1K1V1CP rank3Q3K3V3CP rank2Q2K2V2CP rank0Q0K1V1CP rank1Q1K2V2CP rank3Q3K0V0CP rank2Q2K3V3Step 1Step 2Attn(Q0, KV0)Attn(Q1, KV1)Attn(Q3, KV3)Attn(Q2, KV2)SendRecv(KV1)SendRecv(KV0)SendRecv(KV3)SendRecv(KV2)Attn(Q0, KV1)Attn(Q3, KV3)Attn(Q2, KV3)Attn(Q1, KV1)SendRecv(KV2)Attn(Q0, KV0)Attn(Q3, KV0)Attn(Q2, KV2)Attn(Q1, KV2)SendRecv(KV1)SendRecv(KV0)SendRecv(KV3)CP rank0CP rank1CP rank3CP rank2Attn(Q0, KV1)Attn(Q3, KV3)Attn(Q2, KV3)Attn(Q1, KV1)Attn(Q0, KV0)Attn(Q3, KV0)Attn(Q2, KV2)Attn(Q1, KV2)Q0K0V0Attn(Q0, KV2)Attn(Q0, KV3)Q1K1V1Attn(Q1, KV3)Attn(Q1, KV0)Q2K2V2Attn(Q2, KV0)Attn(Q2, KV1)Q3K3V3Attn(Q3, KV1)Attn(Q3, KV2)mergeattnmergeattnmergeattnmergeattnCP rank0Q0K0V0CP rank1Q1K1V1CP rank3Q3K3V3CP rank2Q2K2V2CP rank0Q1K0V0CP rank1Q2K1V1CP rank3Q0K3V3CP rank2Q3K2V2Step 1Step 2Attn(Q0, KV0)Attn(Q1, KV1)Attn(Q3, KV3)Attn(Q2, KV2)SendRecv(Q1)SendRecv(Q0)SendRecv(Q3)SendRecv(Q2)Attn(Q1, KV0)Attn(Q3, KV3)Attn(Q3, KV2)Attn(Q1, KV1)SendRecv(Q2)Attn(Q0, KV0)Attn(Q0, KV3)Attn(Q2, KV2)Attn(Q2, KV1)SendRecv(Q1)SendRecv(Q0)SendRecv(Q3)CP rank0Q0K0V0CP rank1Q1K1V1CP rank3Q3K3V3CP rank2Q2K2V2Attn(Q0, KV0)Attn(Q1, KV1)Attn(Q3, KV3)Attn(Q2, KV2)Attn(Q1, KV0)Attn(Q2, KV0)Attn(Q3, KV0)Attn(Q2, KV1)Attn(Q3, KV1)Attn(Q0, KV1)Attn(Q0, KV3)Attn(Q1, KV3)Attn(Q2, KV3)Attn(Q3, KV2)Attn(Q0, KV2)Attn(Q1, KV2)permutepermutepermutepermuteCP rank0Q0K0V0CP rank1Q1K1V1CP rank3Q3K3V3CP rank2Q2K2V2Attn(Q0, KV0)Attn(Q1, KV1)Attn(Q3, KV3)Attn(Q2, KV2)Attn(Q1, KV0)Attn(Q2, KV0)Attn(Q3, KV0)Attn(Q2, KV1)Attn(Q3, KV1)Attn(Q0, KV1)Attn(Q0, KV3)Attn(Q1, KV3)Attn(Q2, KV3)Attn(Q3, KV2)Attn(Q0, KV2)Attn(Q1, KV2)all2allContext Parallelism for Scalable Million-Token Inference</p>
<p>Algorithm 4 Batched Ring Pass-Q Decode</p>
<p>4.2 Context Parallel Prefill Scaling</p>
<p>// On CP rank k with KVk, query Qk, batch ids bidk
p ← (k − 1) mod N
for j = 0 to N − 1 do</p>
<p>k to next rank</p>
<p>p from previous rank
k, KVk[bids</p>
<p>k])</p>
<p>p, bids
s ← GQA(Qs</p>
<p>s ← (k − j) mod N
k, bids
Rank k sends Qs
Rank k receives Qs
Compute Ok
k ← Qs
Qs
p
k ← bids
bids
p
end for
s }N −1
Permute {Ok
Compute Ok ← mergeN −1</p>
<p>s=0 and All2All to recover {Os</p>
<p>k}N −1
s=0 .</p>
<p>s=0 (Os
k)</p>
<p>Figure 5. Context parallel across nodes and tensor parallel within
nodes, with 2 CP ranks (CP2).</p>
<p>Grand Teton Inference (GTI). GTT hosts are inter-connected
with backend RDMA network with 400 Gb/s per GPU, and
GTI hosts are inter-connected with frontend network over
TCP/IP with 100 Gb/s per GPU.</p>
<p>With row-wise FP8 quantization4, the entire 405B model
fits into one node with TP8 (tensor parallelism across 8 par-
titions) partitioning. Each GPU holds 1 KV head and 16
Q heads, and feed forward layers are partitioned with alter-
nating column and row parallelism (Shoeybi et al., 2019).
Flash Attention 3 (Shah et al., 2024) is adopted for attention
kernels in prefill, while Flash Decoding (fla) with number
of K/V splits 256 is used during decoding.</p>
<p>We tested full prefill, partial prefill, and decode performance
with context parallelism over 1-16 nodes. Within each CP
node the model is partitioned with TP8 over 8 GPUs. We
form one CP communication group per KV head, with each
CP group consisting of N GPUs (one GPU in each node)
holding the same KV head in their respective tensor parallel
groups. Ring communication around CP ranks is imple-
mented an 8-way SendRecv (Figure 5).</p>
<p>4.2.1 Latency Reduction with Fixed Context Length</p>
<p>Llama3 405B model supports a maximum of 128K context
window, which is equivalent to 300-400 pages of books. We
used max batch size 1 and tested how the full prefill latency
for context lengths 2K to 128K vary with respect to the
addition of more CP nodes.</p>
<p>Figure 6(a) shows the full prefill latency of pass-KV full
prefill on GTI and GTT for 1-8 CP nodes. With sufficiently
large context lengths, the latency for passing key and value
embeddings are overlapped with attention compute, and
we get proportional latency reduction with more CP nodes:
latency for the same input length is halved as we double the
number of CP nodes. Specifically, with CP8 on GTT, an
FP8 Llama3 405B model can process a 128K token prefill
in 5.85 seconds.</p>
<p>For GTI systems with much lower inter-host bandwidth over
frontend TCP/IP network, we observe the same scalability
with up to 4 nodes. Inspecting the GPU trace from GTI, we
found the achieved bandwidth for inter-host communication
is roughly 3GB/s per rank, which is still enough to over-
lap the pass-KV communication with attention compute,
demonstrating the robustness of pass-KV algorithm even
with low inter-connect bandwidth.</p>
<p>4.2.2 Comparing with Multi-Node Tensor-Parallel</p>
<p>To compare with context-parallel performance, we bench-
marked tensor-parallel over multiple nodes on GTT with up
to 8 nodes. Llama3 405B model has 8 KV heads. To effec-
tively parallelize 8 KV heads across more than 8 GPUs, we
replicate each KV head over NT P /NKV GPUs where NT P
is the total number of GPUs in the tensor parallel group
and NKV is the number of KV heads. Query heads are
distributed evenly to all GPUs with NH /NT P query heads
per GPU. Computation is still fully parallelized over NT P
GPUs.</p>
<p>We calculate scaling ratio for a paralellization across N
nodes as as τ1/τN , where τN is the latency for N nodes
to process a 128K context prefill. Better parallelization
algorithms would have scaling ratios closer to N .</p>
<p>Figure 7 illustrates the scaling ratios for multi-node tensor
parallelism compared to context parallelism across 1 to 8
GTT nodes. Tensor-parallel becomes more bottlenecked by
inter-host communication with the growth of capacity, as
AllReduce latency increased significantly with the addition
of more nodes. While the latency is different by roughly
15% between CP2 and TP16 on 2 nodes, the difference
drastically increases to 100% when scaled to 8 nodes.</p>
<p>4https://github.com/pytorch/FBGEMM/tree/</p>
<p>main/fbgemm gpu/experimental/gen ai</p>
<p>This evaluation is performed on H100 hosts which exhibit
significantly lower inter-host bandwidth compared to intra-</p>
<p>xx6S2, chunk3, CP0xxxxrank0: KV0rank1: KV1rank2: KV2rank3: KV3rank4: KV4rank5: KV5rank6: KV6rank7: KV7rank8: KV0rank9: KV1rank10: KV2rank11: KV3rank12: KV4rank13: KV5rank14: KV6rank15: KV7CP node 1CP node 2cp group 1: (0,8)cp group 2: (1,9)...cp group 8: (7,15)Figure5. Context parallel across nodes and tensor parallel within nodesContext Parallelism for Scalable Million-Token Inference</p>
<p>(a) GTT Latency for CP with 1, 2, 4, 8 nodes.</p>
<p>Figure 7. Scaling ratio (latency with one node over latency with N
nodes) of context parallel vs. multi-node tensor parallel.</p>
<p>(b) GTI Latency for CP with 1, 2, 4 nodes</p>
<p>Figure 6. Llama3 405B pass-KV full prefill latency.</p>
<p>host badwidth. For future GB200 (nvg) with NVLink con-
necting multiple hosts, tensor parallelism can still benefits
with reasonable scalability.</p>
<p>4.2.3 Scaling Context Length with Fixed Capacity</p>
<p>By partitioning the KV cache across CP ranks, we also
enhance the KV cache capacity as more CP nodes are added.
To demonstrate scalability in terms of both capacity and
latency, we run up to 1M context prefill over 8 and 16 GTT
nodes. This corresponds to approximately 1 hour of video
content. With a 16-node setup, we achieve an exact prefill
in 77 seconds for a 1M context length and 3.8 seconds for a
128K context length (Figure 8). The quadratic increase in
attention latency with context length begins to dominate the
overall time to first token (TTFT) latency, resulting in more
than 2× increase in TTFT with a 2× increase in context
length for ≥ 512K token prefill.</p>
<p>We calculate the FLOPS utilization of a 1M context length
on 16 nodes in Appendix A. The achieved FLOPS is 502
TF/sec per H100, compared to a standalone Flash Attention
v3 benchmark performance of 540 TF/sec for 8K context
length (1M over 128 H100 GPUs) on a single GPU, resulting</p>
<p>Figure 8. TTFT of 128K-1M context with 8 and 16 CP ranks (CP8
and CP16).</p>
<p>in a 93% parallelization efficiency. Considering the peak
FLOPS on the specialized H100 configurations, we achieve
approximately 63% FLOPS utilization.</p>
<p>4.2.4 Pass-KV vs. Pass-Q Partial (Persistent KV) Prefill</p>
<p>The persistent KV cache provides substantial advantages in
long-context LLM inference by minimizing repeated com-
putational overhead in multi-turn conversations. In Table 4,
experiments with a 128K context length on 4 GTT nodes
demonstrated that, in both pass-KV and pass-Q imple-
mentations, TTFT latency is linearly proportional to the
persistent KV cache miss rate ( T</p>
<p>T +P ).</p>
<p>Figure 9 compares pass-KV and pass-Q in terms of the
KV cache miss rate. When the KV cache miss rate is less
than 5%, pass-Q exhibits better latency; however, when
the miss rate exceeds 5%, pass-KV achieves lower latency.</p>
<p>The tipping point between pass-Q and pass-KV occurs
at T = 6400 (5% KV cache miss rate). Table 5 details
the time breakdown for cache miss rates slightly below</p>
<p>Context Length (#tokens)Average Latency (s)01020304050250005000075000100000125000CP1CP2CP4CP8Context Length (#tokens)Average Latency (s)01020304050250005000075000100000125000CP1CP2CP4Number of Nodes (N)Scaling Ratio024682468TPCP - passKV Perfect ScalingContext Length (#tokens)TTFT (s)050100150128K256K512K1MCP8CP16Context Parallelism for Scalable Million-Token Inference</p>
<p>Table 4. TTFT (in ms) for pass-KV vs. pass-Q varying P and
T with P + T = 128000, on 4 CP ranks (CP4). P : length of
existing tokens in the KV cache, T : length of new tokens.</p>
<p>Table 5. Time breakdown (in µs) on pass-KV vs. pass-Q ring
attention at cache miss rate of 2.5% and 10% with P + T =
128000, on 4 CP ranks (CP4).</p>
<p>P
126720
124800
123840
121600
115200
102400
89600
76800
64000
51200
38400
25600
12800
0</p>
<p>T
1280
3200
4160
6400
12800
25600
38400
51200
64000
76800
89600
102400
115200
128000</p>
<p>MISS RATE
1.00%
2.50%
3.25%
5.00%
10.00%
20.00%
30.00%
40.00%
50.00%
60.00%
70.00%
80.00%
90.00%
100.00%</p>
<p>pass-KV
1023.39
1110.18
1298.92
1305.56
2080.67
3353.02
4629.23
5745.08
6845.21
7890.35
8697.27
10105.78
11136.4
11462.15</p>
<p>pass-Q
898.71
1046.43
1280.1
1302.01
2205.27
3617.02
4922.52
6217.83
7367.99
8468.66
9666.62
10652.39
11571.62
12360.57</p>
<p>Miss Rate pass-KV/Q SendRecv ATTN All2All</p>
<p>2.5%</p>
<p>10%</p>
<p>pass-KV
pass-Q
pass-KV
pass-Q</p>
<p>627
166
631
544</p>
<p>414
414
1608
1608</p>
<p>N/A
424
N/A
1023</p>
<p>Table 6. TTFT / TTIT (in ms) comparisons between TP8 and CP2
with different context lengths at batch size 1.</p>
<p>Context length</p>
<p>8K
32K
128K</p>
<p>TP8</p>
<p>CP2+TP8</p>
<p>TTFT
1740
7658
42010</p>
<p>TTIT
44.51
44.64
46.26</p>
<p>TTFT
999
4015
21042</p>
<p>TTIT
65.61
65.66
66.63</p>
<p>• At 10% KV cache miss rate, pass-KV remains the
choice since the number of new tokens T is sufficiently
large, satisfying Equation 2 (with SendRecv hidden
under ATTN in Table 5).</p>
<p>• Around 5% cache miss rate (e.g., T = 6400), the
differences between pass-KV and pass-Q is less
than 1%, allowing for either option to be selected.</p>
<p>• When cache miss rate falls below 3.25%, pass-KV
communication becomes exposed, leading to the se-
lection of pass-Q. Specifically, at a 2.5% cache
miss rate, the sum of the exposed communication in
pass-KV ring loop is larger than All2All exposed in
pass-Q (Equation 5, Appendix C), resulting in the
selection of pass-Q .</p>
<p>Figure 9. pass-KV / pass-Q speed ratio of 128K context with
persistent KV cache miss rate, varying P and T with P + T =
128000, on 4 CP ranks (CP4).</p>
<p>and above this configuration (2.5% and 10% miss rate).
SendRecv and ATTN represent the SendRecv time and the
partial attention compute time (in µs) for each iteration of
the ring algorithm loop, which is repeated N − 1 times.
The All2All time refers to the communication required in
the merge attention step at the end of pass-Q algorithm.
Note that for T = 3200, the sum of exposed pass-KV
communication ((N − 1)· (SendRecv - ATTN)) is longer
than pass-Q All2All, resulting in better performance for
pass-Q compared to pass-KV.</p>
<p>We further validate the analytical model in Algorithm 1 for
predicting the selection of pass-KV vs. pass-Q from
Table 4.</p>
<p>• When the KV cache miss rate exceeds 12.5% (= 2 ·
in Equation 1), pass-KV is always selected,</p>
<p>NKV
NH
meeting the 2nd condition in Algorithm 1.</p>
<p>4.3 Decode Performance</p>
<p>Inference decode generates one output token at a time, re-
sulting in a small amount of computation workloads and
communication traffic. To avoid host kernel launch bot-
tlenecks for these small kernels, we run both CP and TP
decode with CUDA Graphs (Nvidia Blog, 2019).</p>
<p>Context Length Scalability: We benchmarked CP decod-
ing performance with 2 nodes on GTT (using ring pass-Q
decode algorithm in Section 3.6), and compare with TP8
decoding performance on 1 node using a single batch de-
code with various context lengths. As shown in Table 6, the
TTIT of both TP8 and CP2 does not increase too much: For
both TP8 and CP2, the computation and communication for
linear layers stay the same while the latency of attention
kernels increases with a longer context length.</p>
<p>Parallelism Scalability: We benchmarked different paral-
lelization configurations up to four CP nodes to observe the</p>
<p>1%5%10%100%KV cache miss rate (%, T/(T+P)) in log scale0.850.900.951.001.051.10pass-KV vs. pass-Q performance ratioContext Parallelism for Scalable Million-Token Inference</p>
<p>prefill and decode (Qin et al., 2024; Zhong et al., 2024).
For standalone deployment where prefill and decode are
both on the same set of hosts, CP drastically improves the
prefill latency, at the expense of decode latency regression
(Removing batch padding and better overlap of computation
and communication can help to minimize this regression).</p>
<p>5 CONCLUSION</p>
<p>In conclusion, our work highlights the effectiveness of con-
text parallelism and ring attention variants in improving
the efficiency of LLM inference for long-context scenarios.
By leveraging up to 128 GPUs, we achieved near-linear
scaling and significantly reduced latency, completing tasks
with impressive speed and efficiency. Our implementation
of the lossless exact pass-KV and pass-Q ring attention
variants has been critical in supporting various full prefill,
partial prefill, and decoding scenarios. The runtime heuris-
tic adaptively selects pass-KV or pass-Q based on KV
cache hit rate, optimizing their application for the most
suitable scenarios.</p>
<p>As we keep improving LLM’s capacity to understand in-
creasingly longer and more complex context, one can expect
diminishing utility with exact attention over all historical to-
kens. More efficient algorithms for retrieving a small subset
of information from a much larger context to answer sim-
ple probe questions will be increasingly important. While
context parallel is an efficient exact algorithm for scaling
exact attention with more capacity, combining its processing
power with an approximate retrieval algorithm for ultra-long
context may be the best way to bound the processing latency
for context window growth at and beyond 1M.</p>
<p>6 ACKNOWLEDGMENTS</p>
<p>We express our gratitude to our outstanding colleagues for
their significant contributions to various aspects of LLM
inference. Special thanks go to Geonhwa Jeong, Jaewon
Lee, Jason Park, Vlad Mihailescu, Zheng Yan, and Daniel
Haziza for their invaluable efforts related to this paper. Our
thanks also go to Chunqiang Tang for his early feedback
and proofreading of the draft. Furthermore, we appreciate
the leadership and support provided by Chunqiang Tang,
Maxim Naumov, Amit Nagpal, Tony Liu, Changkyu Kim,
Anca Agape, and Stephen Chen.</p>
<p>Table 7. TTFT / TTIT (in ms) comparisons between TP8, CP2,
TP16, CP4, TP32 with 128K context length at batch size 1.</p>
<p>CP1+TP8
CP2+TP8
TP16
CP4+TP8
TP32</p>
<p>TTFT
42010
21042
29917
10950
19841</p>
<p>TTIT
46.26
60.23
39.52
71.31
47.3</p>
<p>Table 8. Attention scaling with the number of CP hosts (Time in
µs).</p>
<p>Effective context length
Individual attention op
Attn (whole ring loop)
SendRecv
All2All
Whole pass-Q</p>
<p>Context length 128K, batch size 1</p>
<p>TP8
128K
38.9
38.9
0
0
38.9</p>
<p>CP2+TP8
64K
22.0
43.2
32.3
81.1
157.7</p>
<p>CP4+TP8
32K
14.7
60.8
105.7
79.9
238.6</p>
<p>Context length 32K, batch size 4</p>
<p>Effective context length
Individual attention op
Attn (whole ring loop)
SendRecv
All2All
Whole pass-Q</p>
<p>32K
60.1
60.1
0
0
60.1</p>
<p>16K
13.9
24.5
33.3
66.8
136.6</p>
<p>8K
9.6
41.3
104.9
72.2
180.6</p>
<p>scalability of both TP and CP. Table 7 shows that TTIT tends
to be longer for both scaling TP and scaling CP. TTIT for
scaling TP increases to 47 ms while TTIT for scaling CP
increases to 71 ms. Both TP and CP have poor scalability
for decoding when adding more hosts (e.g., using 4 nodes
can result in worse TTIT than using a single node). For
TP, lower computation latency on linear layers is offset by
increased communication latency increased.</p>
<p>For CP, as we increase the number of hosts, the effective
length seen by each attention kernel decreases, so each in-
dividual attention op becomes faster (Table 8). However
TTIT still degrates compared to CP=1, and the reason for
that is two-fold: (1) Current implementation pads the num-
ber of queries to make it divisible by the number of ranks,
which for B=1 means the total number of processed queries
increases with CP. (2) The communication latency - sending
Q chunks to the next rank at each iteration of the loop and
All2All-exchanging partial attention outputs after the loop -
also grows with the number of hosts. As a result, the total
pass-Q attention latency and TTIT increase with CP.</p>
<p>In summary, context parallel is best suited for improving
prefill performance and can be best leveraged with a serv-
ing system that decouples the parallelization scheme for</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>REFERENCES</p>
<p>Claude with</p>
<p>200K context</p>
<p>length.</p>
<p>https:</p>
<p>//support.anthropic.com/en/articles/
8606395-how-large-is-the-anthropic-
api-s-context-window. Accessed: 2024-10-30.</p>
<p>Flash-decoding for long-context</p>
<p>https:
//pytorch.org/blog/flash-decoding/. Ac-
cessed: 2024-10-30.</p>
<p>inference.</p>
<p>Google’s Gemini 1.5 Pro with 1M context</p>
<p>length.</p>
<p>https://blog.google/technology/ai/
google-gemini-next-generation-model-
february-2024. Accessed: 2024-10-30.</p>
<p>Dao, T., Fu, D., Ermon, S., Rudra, A., and R´e, C. Flashat-
tention: Fast and memory-efficient exact attention with
io-awareness.
In Koyejo, S., Mohamed, S., Agarwal,
A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances
in Neural Information Processing Systems, volume 35,
Inc., 2022.
pp. 16344–16359. Curran Associates,
https://proceedings.neurips.cc/
URL
paper files/paper/2022/file/
67d57c32e20fd0a7a302cb81d36e40d5-
Paper-Conference.pdf.</p>
<p>Devlin, J. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. arXiv preprint
arXiv:1810.04805, 2018.</p>
<p>Nvidia GB200 NVL72. https://www.nvidia.com/
en-us/data-center/gb200-nvl72/. Accessed:
2024-10-30.</p>
<p>Fang, J. and Zhao, S. Usp: A unified sequence parallelism
approach for long context generative ai. arXiv preprint
arXiv:2405.07719, 2024.</p>
<p>GPT-4o with 128K context</p>
<p>https://
platform.openai.com/docs/models. Accessed:
2024-10-30.</p>
<p>length.</p>
<p>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.</p>
<p>Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y.,
Lebr´on, F., and Sanghai, S. GQA: Training generalized
multi-query transformer models from multi-head check-
points. arXiv preprint arXiv:2305.13245, 2023.</p>
<p>Beltagy,</p>
<p>I., Peters, M. E., and Cohan, A.
Long-
former: The long-document transformer. arXiv preprint
arXiv:2004.05150, 2020.</p>
<p>Brandon, W., Nrusimha, A., Qian, K., Ankner, Z., Jin, T.,
Song, Z., and Ragan-Kelley, J. Striped attention: Faster
ring attention for causal transformers. arXiv preprint
arXiv:2311.09431, 2023.</p>
<p>Brown, T. B. Language models are few-shot learners. arXiv</p>
<p>preprint arXiv:2005.14165, 2020.</p>
<p>Cho, M., Rastegari, M., and Naik, D. Kv-runahead: Scal-
able causal llm inference by parallel key-value cache
generation. arXiv preprint arXiv:2405.05329, 2024.</p>
<p>Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra,
G., Roberts, A., Barham, P., Chung, H. W., Sutton, C.,
Gehrmann, S., et al. Palm: Scaling language modeling
with pathways. Journal of Machine Learning Research,
24(240):1–113, 2023.</p>
<p>Dao, T.</p>
<p>Flashattention-2: Faster attention with bet-
ter parallelism and work partitioning. arXiv preprint
arXiv:2307.08691, 2023.</p>
<p>Gemini Team. Gemini: a family of highly capable multi-
modal models. arXiv preprint arXiv:2312.11805, 2023.</p>
<p>Gemini Team. Gemini 1.5: Unlocking multimodal under-
standing across millions of tokens of context. arXiv
preprint arXiv:2403.05530, 2024.</p>
<p>Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney,
M. W., Shao, Y. S., Keutzer, K., and Gholami, A.
Kvquant: Towards 10 million context length llm in-
arXiv preprint
ference with kv cache quantization.
arXiv:2401.18079, 2024.</p>
<p>Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, D., Chen,
M., Lee, H., Ngiam, J., Le, Q. V., Wu, Y., et al. Gpipe:
Efficient training of giant neural networks using pipeline
parallelism. Advances in neural information processing
systems, 32, 2019.</p>
<p>Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song,
S. L., Rajbhandari, S., and He, Y. Deepspeed ulysses:
System optimizations for enabling training of extreme
arXiv preprint
long sequence transformer models.
arXiv:2309.14509, 2023.</p>
<p>Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han,
Z., Abdi, A. H., Li, D., Lin, C.-Y., et al. Minference 1.0:
Accelerating pre-filling for long-context llms via dynamic
sparse attention. arXiv preprint arXiv:2407.02490, 2024.</p>
<p>Juravsky, J., Brown, B., Ehrlich, R., Fu, D. Y., R´e, C., and
Mirhoseini, A. Hydragen: High-throughput llm inference
with shared prefixes. arXiv preprint arXiv:2402.05099,
2024.</p>
<p>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
Amodei, D. Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361, 2020.</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>Korthikanti, V. A., Casper, J., Lym, S., McAfee, L., Ander-
sch, M., Shoeybi, M., and Catanzaro, B. Reducing acti-
vation recomputation in large transformer models. Pro-
ceedings of Machine Learning and Systems, 5:341–353,
2023.</p>
<p>Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu,
C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient
memory management for large language model serving
In Proceedings of the 29th Sym-
with pagedattention.
posium on Operating Systems Principles, pp. 611–626,
2023.</p>
<p>Li, D., Shao, R., Xie, A., Xing, E. P., Ma, X., Stoica, I.,
Gonzalez, J. E., and Zhang, H. Distflashattn: Distributed
memory-efficient attention for long-context llms training.
arXiv preprint arXiv:2310.03294, 2023.</p>
<p>Li, S., Xue, F., Baranwal, C., Li, Y., and You, Y. Sequence
parallelism: Long sequence training from system perspec-
tive. arXiv preprint arXiv:2105.13120, 2021.</p>
<p>Lin, Y., Tang, H., Yang, S., Zhang, Z., Xiao, G., Gan, C.,
and Han, S. Qserve: W4a8kv4 quantization and sys-
tem co-design for efficient llm serving. arXiv preprint
arXiv:2405.04532, 2024.</p>
<p>Liu, H., Zaharia, M., and Abbeel, P. Ring attention with
blockwise transformers for near-infinite context. arXiv
preprint arXiv:2310.01889, 2023.</p>
<p>Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin,
S., and Guo, B. Swin transformer: Hierarchical vision
transformer using shifted windows. In Proceedings of the
IEEE/CVF international conference on computer vision,
pp. 10012–10022, 2021.</p>
<p>Llama Team. The Llama 3 herd of models. 2024. URL</p>
<p>https://arxiv.org/abs/2407.21783.</p>
<p>Meta Engineering. OCP summit 2022: Open hardware for
ai infrastructure. https://engineering.fb.com/
2022/10/18/open-source/ocp-summit-
2022-grand-teton/, 2022. Accessed: 2024-10-30.</p>
<p>Milakov, M. and Gimelshein, N. Online normalizer cal-
culation for softmax. arXiv preprint arXiv:1805.02867,
2018.</p>
<p>Munkhdalai, T., Faruqui, M., and Gopal, S. Leave no con-
text behind: Efficient infinite context transformers with
infini-attention. arXiv preprint arXiv:2404.07143, 2024.</p>
<p>Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Pat-
wary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P.,
Bernauer, J., Catanzaro, B., et al. Efficient large-scale
language model training on gpu clusters using megatron-
lm. In Proceedings of the International Conference for</p>
<p>High Performance Computing, Networking, Storage and
Analysis, pp. 1–15, 2021.</p>
<p>Nvidia Blog.</p>
<p>Getting started with CUDA Graphs.</p>
<p>https://developer.nvidia.com/blog/
cuda-graphs/, 2019. Accessed: 2024-10-30.</p>
<p>Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury,
J., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently
scaling transformer inference. Proceedings of Machine
Learning and Systems, 5:606–624, 2023.</p>
<p>Qin, R., Li, Z., He, W., Zhang, M., Wu, Y., Zheng, W.,
and Xu, X. Mooncake: A kvcache-centric disaggre-
gated architecture for llm serving. URL https://arxiv.
org/abs/2407.00079, 2024.</p>
<p>Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,
Sutskever, I., et al. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.</p>
<p>Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani,
P., and Dao, T. Flashattention-3: Fast and accurate atten-
tion with asynchrony and low-precision. arXiv preprint
arXiv:2407.08608, 2024.</p>
<p>Shazeer, N. Fast transformer decoding: One write-head is
all you need. arXiv preprint arXiv:1911.02150, 2019.</p>
<p>Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper,
J., and Catanzaro, B. Megatron-lm: Training multi-
billion parameter language models using model paral-
lelism. arXiv preprint arXiv:1909.08053, 2019.</p>
<p>Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models. arXiv preprint arXiv:2302.13971, 2023a.</p>
<p>Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models. arXiv preprint arXiv:2307.09288,
2023b.</p>
<p>Vaswani, A. Attention is all you need. Advances in Neural</p>
<p>Information Processing Systems, 2017.</p>
<p>Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H.
Linformer: Self-attention with linear complexity. arXiv
preprint arXiv:2006.04768, 2020.</p>
<p>Wu, B., Liu, S., Zhong, Y., Sun, P., Liu, X., and Jin, X.
Loongserve: Efficiently serving long-context large lan-
guage models with elastic sequence parallelism. In Pro-
ceedings of the ACM SIGOPS 30th Symposium on Oper-
ating Systems Principles, pp. 640–654, 2024.</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Ef-
ficient streaming language models with attention sinks.
arXiv preprint arXiv:2309.17453, 2023.</p>
<p>Xiong, W., O˘guz, B., Gupta, A., Chen, X., Liskovich, D.,
Levy, O., Yih, W.-t., and Mehdad, Y. Simple local atten-</p>
<p>tions remain competitive for long-context tasks. arXiv
preprint arXiv:2112.07210, 2021.</p>
<p>Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin,
X., and Zhang, H. Distserve: Disaggregating prefill and
decoding for goodput-optimized large language model
serving, 2024.</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>Table 9. Llama3 405B model configurations.
Parameter
Layers (#layers)
Model Dimension (D)
FFN Dimension
Attention Heads (NH )
Key/Value Heads (NKV )
Parameter Size (W )</p>
<p>Value
126
16,384
53,248
128
8
405 B</p>
<p>A MFU CALCULATION FOR 1M CONTEXT</p>
<p>LENGTH</p>
<p>We calculate the effective Model FLOPS utilization
(MFU) (Chowdhery et al., 2023) in this section. The Llama3
405B model configurations are listed in Table 9. The total
FLOPS are dominant by GEMM and Attention parts:</p>
<p>Total FLOPS = GEMM FLOPS + ATTN FLOPS.</p>
<p>B MERGE ATTENTION</p>
<p>The idea of merging attention outputs from different
keys/values originates from Online Softmax (Milakov &amp;
Gimelshein, 2018). Later this idea was reused in Flash At-
tention (Dao et al., 2022; Dao, 2023). Here we derive the
equation to merge the partial attention outputs from different
CP ranks.</p>
<p>scaled dot production attention operates on
The
query/key/value tensors Q/K/V .
For simplicity, we
don’t consider various mask like causal masks (no batching
or multiple attention heads either). There is one Q/K/V
corresponding to each sequence position. Q/K/V at a
given sequence position is a vector in the embedding space.
The attention output is defined as</p>
<p>O = Attn(Q, K, V ) = softmax</p>
<p>(cid:18) QK T
√
d</p>
<p>(cid:19)</p>
<p>V,</p>
<p>• For GEMM, an W -parameter Transformer model re-
quires 2 · W matrix multiplication FLOPs for each
token during inference:</p>
<p>GEMM FLOPS = 2 × W × T × B.</p>
<p>• For Attention, the FLOPS is quadratic with respect to</p>
<p>the context length T :</p>
<p>ATTN FLOPS = 1/2 × 4 × B × T 2 × D × #layers,</p>
<p>where softmax is applied row-wise.</p>
<p>Assuming the size of row is R,</p>
<p>O =</p>
<p>(cid:80)R−1</p>
<p>i=0 expQ·KT
i /
expLSE</p>
<p>√</p>
<p>d · Vi</p>
<p>,</p>
<p>where log-sum-exp LSE is defined as:</p>
<p>LSE = log</p>
<p>expQ·KT
i /</p>
<p>√</p>
<p>d.</p>
<p>R−1
(cid:88)</p>
<p>i=0</p>
<p>where 1/2 is from the causal mask, 4 is from 2 batch
matmul and 2 FLOPS for multiplication and addition.</p>
<p>In Section 3.5.2, we calculate the attention output and LSE
on each CP rank k:</p>
<p>With input sequence length T = 1M , batch size B = 1, the
parameter size W = 405B, we can get GEMM FLOPS =
2 × 405B × 1M = 8.1 × 1017. With the model dimension
D = 16384, and number of layers #layers = 126, we
can derive ATTN FLOPS = 1/2 × 1M 2 × 16384 × 126 =
4.1 × 1018. Attention FLOPS is more dominant compared
to GEMM FLOPS. The total FLOPS is 4.9 × 1018. With 77
seconds for 1M context length using 128 H100 GPUs, each
H100 achieves 4.9 × 1018/77/128 = 502 TF/sec. Note
that with the standalone Flash Attention v3 causal attention
benchmark using 8K context length on a single H100 (1M
context length sharded across 128 H100 GPUs), we achieve
540 TF/sec. One caveat for the evaluation is that GTT/GTI
(Section 4.1) are configured with power limited H100 GPUs
(500 Watt) with lower memory bandwidth (96 GB HBM2e
with 2.4 TB/sec instead of 80 GB HBM3 with 3.35 TB/sec),
where the BF16 peak for each H100 is 800 TF/sec, instead
of 989 TF/sec for H100 HBM3 with 700 Watt.</p>
<p>LSEs</p>
<p>k, Os</p>
<p>k = Attn(Qk, KV s),</p>
<p>with s = 0, 1, ..., N − 1 on CP rank k.</p>
<p>Similar to blocked softmax computation in Flash Atten-
tion (Dao et al., 2022; Dao, 2023) and the derivation process
in (Juravsky et al., 2024), we can get</p>
<p>Ok =</p>
<p>(cid:80)N −1</p>
<p>s=0 (Os
(cid:80)N −1</p>
<p>k × expLSEs</p>
<p>k−LSEmax</p>
<p>k</p>
<p>s=0 expLSEs</p>
<p>k−LSEmax</p>
<p>k</p>
<p>)</p>
<p>,</p>
<p>(4)</p>
<p>k</p>
<p>= maxN −1</p>
<p>where LSEmax</p>
<p>s=0 LSEs
k.
In this way5, we can combine attention output computed on
different chunks of K/V for the same query to get attention
on the whole K/V.</p>
<p>5Merge attention implementation is open sourced at
https://facebookresearch.github.io/xformers/
components/ops.html#module-xformers.ops.fmha.</p>
<p>Context Parallelism for Scalable Million-Token Inference</p>
<p>Compared to (1), this shows that considering All2All de-
creases the KV cache miss rate threshold for selecting
pass-Q .</p>
<p>Algorithm 5 is the adjusted heuristic algorithm to select
between pass-KV and pass-Q , considering All2All used
in merge attention in pass-Q .</p>
<p>Figure 10. A heuristic model using empirical data points. Green:
prefer pass-KV , Red: prefer pass-Q</p>
<p>C ANALYTICAL MODEL SELECTION</p>
<p>CONSIDERING ALL2ALL</p>
<p>pass-Q merge attention requires an All2All (Section 3.5.3),
whereas in pass-KV merge attention only needs to merge
the partial attn results on local node (Section 3.5.2). When
pass-KV communication is exposed, we want to compare
the total of exposed pass-KV’s communication time to
the pass-Q’s all2all, which is the time to send partial
attention output and partial attention softmax log-sum-exp
(LSE) (Appendix B):</p>
<p>Latency(All2All) = (N − 1) ·</p>
<p>(D + 1) · T · e
BW</p>
<p>This means pass-Q has better prefill latency only if:</p>
<p>(N −1)·</p>
<p>(cid:32) 2(T + P )D · e · NKV
NH
BW</p>
<p>−</p>
<p>4 · T · D · (T + P )
N · C</p>
<p>(cid:33)</p>
<p>≥ (N − 1) ·</p>
<p>(D + 1) · T · e
BW</p>
<p>.</p>
<p>Assuming D ≈ D + 1, through algebraic rearrangement,
we get:</p>
<p>2 ·</p>
<p>NKV
NH</p>
<p>−</p>
<p>4T · BW
N · C · e</p>
<p>≥</p>
<p>T
T + P</p>
<p>(5)</p>
<p>Algorithm 5 Pass-KV vs. Pass-Q Partial Prefill Heuristics
T</p>
<p>if T ≥ N C·NKV ·e</p>
<p>2·NH ·BW or</p>
<p>T +P ≥ 2 · NKV
NH</p>
<p>− 4T ·BW
N ·C·e</p>
<p>then</p>
<p>pass-KV</p>
<p>else</p>
<p>pass-Q</p>
<p>end if</p>
<p>D HEURISTIC BASED ON EMPIRICAL DATA</p>
<p>For practical uses, we further establish a simplified heuristic
to choose between pass-KV and pass-Q based on em-
prical data points. Particularly we collected data points for
various combinations of T and T /(T + P ), and establish
an empirical formula:</p>
<p>h(T, P ) = α · log(T ) + β · log</p>
<p>(cid:18) T</p>
<p>(cid:19)</p>
<p>T + P</p>
<ul>
<li>γ</li>
</ul>
<p>We prefer pass-KV when h evaluates to a positive value
and prefer pass-Q otherwise. We fit empirical data points
to this formula with parameters: α = −1.059, β = 1.145
and γ = 12.112, as show in Figure 10. One way to inter-
pret the heuristic is that, for each particular T , there is a
threshold for T /(T + P ) based on which we should switch
from pass-Q to pass-KV for best performances, and the
threshold increases as T increases.</p>
<p>Note that we do not expect the linear model to perfectly
capture all cases, so some misclassifications are present
due to variances and other factors, but the general trend is
obvious. We inspected the misclassified data points, and
they turned out to be the ones where the differences between
the two strategies were relatively small (&lt; 1%). In practice
we can run this heuristic at the beginning of each round and
get the best of both worlds.</p>
<p>02468log(T)14121086420log(T/(T+P))</p>