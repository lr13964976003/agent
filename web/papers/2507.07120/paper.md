<p>5
2
0
2</p>
<p>l
u
J</p>
<p>7</p>
<p>]</p>
<p>C
D
.
s
c
[</p>
<p>1
v
0
2
1
7
0
.
7
0
5
2
:
v
i
X
r
a</p>
<p>Helix Parallelism: Rethinking Sharding Strategies for
Interactive Multi-Million-Token LLM Decoding</p>
<p>Nidhi Bhatia</p>
<p>Ankit More</p>
<p>Ritika Borkar</p>
<p>Tiyasa Mitra</p>
<p>Ramon Matas</p>
<p>Ritchie Zhao Max Golub</p>
<p>Dheevatsa Mudigere</p>
<p>Brian Pharris</p>
<p>Bita Rouhani</p>
<p>NVIDIA Corporation</p>
<p>Abstract</p>
<p>As LLMs scale to multi-million-token KV histories, real-time autoregressive decod-
ing under tight Token-to-Token Latency (TTL) constraints faces growing pressure.
Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights
and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost
of FFN weight reads, it does not scale well for attention. When TP width exceeds
the number of KV heads, it leads to inefficient KV duplication, limits parallelism,
and constrains batch size. Simultaneously, DRAM reads for long KV histories
scale linearly with batch size, further capping efficiency.
We introduce Helix Parallelism, a hybrid execution strategy that applies KV par-
allelism during attention to shard KV caches across GPUs, then reuses the same
GPUs for TP in dense LLMs or TP×Expert Parallel (EP) in MoEs during FFN
computation. To preserve exact attention behavior, Helix includes a lightweight
communication step. To minimize the exposed communication cost, we intro-
duce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead
through batchwise overlap, preserving low TTL while improving GPU efficiency.
Compared to conventional parallelism approaches, Helix reduces TTL by up to
1.5x at fixed batch sizes and supports up to 32× larger batches under the same
latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto
on Blackwell and making real-time inference with ultra-long-sequence practical.</p>
<p>1</p>
<p>Introduction</p>
<p>Large Language Models (LLMs) are increasingly expected to handle ultra-long histories [1] [2],
precomputed context spanning millions of tokens1, while still delivering millisecond-level Token-to-
Token Latency (TTL) for interactive applications. Long contexts allow models to maintain narrative
coherence, capture long-range dependencies, and support complex reasoning and planning. At the
same time, interactivity demands that each token be generated almost instantly, as users expect
real-time responsiveness from AI assistants, copilots, and autonomous agents.</p>
<p>Decoding under these dual pressures exposes two fundamental bottlenecks. First, KV cache reads
during self-attention become increasingly expensive. The KV cache size grows linearly with both
context length and batch size, rapidly overwhelming DRAM capacity and bandwidth. To fit multi-
million-token caches, systems are often forced to reduce batch sizes, but read time still remains high
with longer histories, driving TTL beyond acceptable limits. Second, FFN weight reads during</p>
<p>1Here, context refers to the sequence of previously generated tokens, whose intermediate key and value</p>
<p>representations are stored as KV cache and accessed at every decoding step.</p>
<p>Preprint. Under review.</p>
<p>autoregressive decoding contribute heavily to latency. Generating every new token requires loading
large feed-forward network (FFN) weights from DRAM. With small batch sizes, this cost cannot be
amortized, making weight reads another dominant factor in overall decoding time.</p>
<p>Modern attention variants such as Grouped-Query Attention (GQA) [3], Multi-Query Attention
(MQA) [4], and Multi-Head Latent Attention (MLA) [5] reduce KV-cache pressure by merging
multiple keys and values into shared representations, i.e., collapsing Q original query heads into K
KV heads where K &lt; Q. A typical value of K is 8 or less in modern LLMs. For MLA, there is just
a single latent representation of both K and V for all Q heads.</p>
<p>Tensor Parallelism (TP) [6] shards both FFN weights and attention heads evenly over T P GPUs,
so that each device only holds and reads its portion of the weights and KV cache. When T P ≤ K,
TP naturally splits the KV cache without duplication. However, once T P &gt; K (common in large
models needing high parallelism) each additional shard must store a full copy of the KV cache to
serve its assigned query heads, despite splitting computation. Beyond this point, increasing T P
neither shrinks per GPU KV cache size nor speeds up KV cache reads, imposing a hard ceiling on
attention time (see Figure 1) and forcing smaller batch sizes under tight TTLs. Furthermore, since TP
must be capped at K shards (the number of KV heads), only those K GPUs can be used to shard
the FFN, which accounts for roughly two-thirds of model parameters. Reading these large weight
matrices on just K devices not only monopolizes memory that could host additional KV caches, but
also makes FFN weight loads the primary latency bottleneck, further limiting how many concurrent
batches can be maintained in real-time decoding.</p>
<p>To tackle KV cache scaling, recent work like Medha [7] shards the KV cache across an auto-scaled
pool of N GPUs using KV Parallelism (KVP), so each device stores only a fraction of the multi-
million-token history. This approach significantly reduces both per GPU cache size and read latency
during self-attention. However, Medha and similar methods then gather the attention outputs onto a
fixed group of TP GPUs (e.g., 8) for all subsequent FFN computations. In effect, while KVP fans
out computation across N GPUs for attention, it does not repurpose those same GPUs to further
accelerate FFN execution. As a result, FFN weight loads remain a latency bottleneck, and hardware
resources become increasingly underutilized as N grows.</p>
<p>This paper introduces Helix Parallelism and Helix HOP-B for interactive multi-million-token LLM
decoding. Helix parallelism is a hybrid sharding strategy where we disaggregate the mapping of
attention and FFNs in a temporal pipeline to address both KV cache and FFN bottlenecks. Specifically:</p>
<ol>
<li>
<p>Attention phase: Applies KVP to shard the KV cache along the sequence dimension across
KVP GPUs, combined with TP across KV heads when T P ≤ K, resulting in a total of N =
KV P × T P GPUs handling attention with no cache duplication.</p>
</li>
<li>
<p>FFN phase: Immediately reconfigures the same N GPUs, now running either dense TP or
combined TP × Expert Parallelism (in MoE models), to shard FFN weight matrices and accelerate
weight reads. This reconfiguration also allows TP widths for FFN to exceed the number of KV
heads without reintroducing cache duplication during the attention phase.</p>
</li>
</ol>
<p>HOP-B is an overlap optimization that mitigates communication latencies introduced by Helix
Parallelism by masking them behind computation, effectively maintaining low TTL.</p>
<p>By aligning the parallelism scheme to each stage’s distinct computational demands, Helix reduces
TTL by up to 50% at fixed batch sizes, and expands batch size by up to 32× under the same
latency budget. Such a hybrid optimization is particularly important on modern GPU systems like
Blackwell [8], which feature large NVLink domains. Helix parallelism is fully compatible with
modern LLM architectures, including MLA and GQA attention, as well as MoEs.</p>
<p>2 Helix parallelism</p>
<p>Helix Parallelism is based on a key insight: achieving real-time decoding with multi-million-token
KV histories requires decoupling the mapping of attention and FFNs. Figure 1 provides a high-level
intuition for why this separation is essential. Helix introduces a temporal pipeline within each layer,
allowing the same set of GPUs to be reused across attention and FFN computation, while applying
different parallelism strategies for each. In the remainder of this section, we describe the core building
blocks of Helix Parallelism in detail.</p>
<p>2</p>
<p>Figure 1: Roofline analysis for KV cache and Linear weight reads, assuming a Dense LLM
with batch B=8, Query heads Q=128, KV heads K=8, head size Hsz=128, and FFN dimension
F =65536 running on GB200 NVL72. Both weights and KV cache, are stored and fetched in FP4.
Communication overhead from TP and KVP is not included; these plots show only the change in
GPU DRAM-read latency as TP width and KVP width vary. Details can be found in Appendix A
(Left) DRAM read latency vs. TP width. Benefits plateau beyond T P =K due to full KV duplication,
highlighting the need for KV sharding in Helix. (Middle) DRAM read time vs. KV length S.
Self-attention cost scales linearly with S, eventually dominating latency. (Right) DRAM read time vs.
KVP width. Helix applies KVP in attention to reduce per-GPU memory traffic and achieve sublinear
scaling, enabling multi-million-token inference. The same GPUs are then re-provisioned for TP or
EP in FFNs to minimize latency.</p>
<p>2.1 Attention partitioning</p>
<p>2.1.1 KV partitioning (KVP)</p>
<p>Figure 2: Overview of different attention sharding strategies. Here we are using GQA as an
example with Query heads Q=4 and KV heads K=2. Each block processes one token with context
length S. (Left) No TP: all Q and KV heads are co-located on a single GPU; no duplication. (Middle-
Left) TP=2: Query heads are split across 2 GPUs; KV heads are still partitioned cleanly since
T P ≤ K. (Middle-Right) TP=4: More shards than KV heads; GPUs must duplicate KV cache
to serve their assigned query heads, reintroducing DRAM capacity and bandwidth inefficiencies.
(Right) Helix (TP=2, KVP=2): Helix shards the KV cache across sequence length (S) using KVP, so
each KVP rank holds only a slice (S/2). By capping TP at K and assigning the remaining GPUs to
KVP, Helix avoids duplication and forms a 2D layout: TP splits heads, KVP splits the sequence.</p>
<p>During the attention phase, Helix configures all available GPUs into a pool of N = KVP ×
TPA (TPA ≤ K), then shards the KV cache along the sequence dimension across the KV P
GPUs, eliminating full-cache replication and cutting DRAM footprint and bandwidth demands.
To avoid an expensive pre-attention All-Gather of queries across the KVP GPUs, Helix has each
KVP GPU independently compute the full QKV projections. Concretely, every GPU takes the full
input batch [B, H] and multiplies it by the QKV weight matrices WQ ∈ RH×(H/T P A), WK ∈
RH×(⌈K/T P A⌉·Hsz), and WV ∈ RH×(⌈K/T P A⌉·Hsz) to produce the full QKV projections. This
means each of the KV P GPUs holds its own full set of query heads, and the corresponding key/value</p>
<p>3</p>
<p>projections, so it can run FlashAttention [9] on its KV shard in isolation. Each GPU emits a partial
attention output and a log-sum-exp scalar per token; a single All-to-All over the query-head axis then
exchanges these fragments, and each GPU rescales and sums them to reconstruct the exact softmax-
normalized attention in one communication round, with no extra synchronization or normalization
passes [10]. This All-to-All exchange also realigns the data partitions for subsequent TP-based
execution. Figure 2 provides a high-level overview of different sharding schemes in attention and
their corresponding layout.</p>
<p>2.1.2 Optimized all-to-all communication</p>
<p>The communication volume in Helix parallelism is independent of the KV-sequence length S and
scales only with the number of query tokens in a batch B and the hidden dimension H. This constant
per-token overhead makes Helix parallelism highly scalable, allowing efficient decode-time attention
even with multi-million-token KV caches.</p>
<p>2.1.3 Batch-wise communication–computation overlap</p>
<p>To minimize exposed communication time, we introduce and deploy Helix HOP-B (Helix Overlap
Pipeline – Batch-wise), a fine-grained pipelining strategy that overlaps All-to-All communication
with ongoing attention computation across the batch dimension. As illustrated in Figure 3, once the
attention output for the first query token is computed, Helix immediately initiates All-to-All for that
token while concurrently processing attention for the next. This overlap maintains high hardware
utilization and effectively hides communication latency, further reducing TTL for real-time inference
decoding.</p>
<p>Figure 3: KVP All-to-All exposed time: (Top without HOP-B) All 8 requests execute in lockstep,
each consumes 16 time units of attention before initiating 9.6 units of communication, for a total
span of 25.6 units with no overlap; (Bottom with HOP-B) Requests are pipelined so that while one
request’s communication is ongoing, the next request begins its attention compute immediately. Here
each request’s compute (2 units) and communication (1.2 units) are drawn to scale as discrete blocks.
A dashed gray line and "TTL Saving" arrow highlight the reduction in overall TTL from 25.6 units in
the baseline down to 17 units when HOP-B is enabled.</p>
<p>4</p>
<p>2.2 FFN partitioning</p>
<p>After self-attention, Helix reuses the same pool of N = KVP × TPA GPUs, originally configured for
KV partitioning and, where applicable, TP, to execute the FFN in a layout optimized for the model
type, whether dense or MoE. Figure 4, provide a detailed end-to-end view of Helix parallelism for a
single transformer layer. For simplicity, layer-normalization and residual-addition are omitted from
the diagram.2</p>
<p>Figure 4: Helix’s layout per-GPU workflow in attention and FFN stages. Helix reuses the same
pool of N GPUs by configuring N = KV P × T P A (T P A ≤ K) −→ N = T P F × EP
on a per-layer basis. (Top) During attention, each of the KV P GPUs independently projects the
full batch [B, H] into QKV, runs FlashAttention [9] on its KV shard to produce partial outputs and
log-sum-exp scalars, and then participates in a single All-to-All over the query-head axis; each GPU
rescales and sums the received fragments into the exact softmax-normalized tensor, followed by a
TP All-Reduce to form the final [B, H] attention output. (Bottom) For FFNs, Helix follows two
modes depending on model type: for Dense FFNs EP = 1, it retains all N GPUs in tensor-parallel
T P F = N to compute [B, H] → [B, F/N ] → [B, H] followed by a TP All-Reduce; for MoE
FFNs EP &gt; 1, it repartitions the N GPUs into a T P F × EP grid, routes tokens to the appropriate
experts, applies TP to FC layers within each expert group, performs an intra-expert All-Reduce
followed by an inter-expert All-Gather, and concludes with a local reduction to yield [B, H]. Helix
switches between these configurations seamlessly, enabling zero-downtime pipelining and better
GPU utilization.</p>
<p>At the end of local attention, each KVP GPU has computed partial attention outputs over its KV shard
and respective query heads and the entire batch B. These are exchanged via a single All-to-All
(cid:1) partial results to every other
along the query-head dimension: each GPU sends its B × (cid:0)</p>
<p>H
KV P ×T P A</p>
<p>2While we depict a two-stage FFN, FC1 (H × F ) and FC2 (F × H), modern designs often employ gated
variants (e.g. an additional H × F gating matrix) or fuse these into combined projections (e.g. H × 2F and
2F × H).</p>
<p>5</p>
<p>H</p>
<p>GPU in KVP domain and receives the corresponding slices in return. After rescaling and summing
with the per-shard log-sum-exp statistics, each GPU holds the fully normalized attention outputs
for its assigned
KV P ×T P A hidden-dimension slice, but for the entire batch, effectively forming a
TP group of size T P A × KV P = N . These normalized outputs feed into the post-attention linear
projection, which runs in tensor-parallel fashion across the same N GPUs. Each TP rank holds a
shard of the projection weight matrix of shape H
N × H, computes its local matrix-multiply on the
full batch inputs of shape B × H
N , and then participates in an All-Reduce over the N GPUs. This
All-Reduce aggregates the B × H partial projections into the full B × H output, which is then passed
through layer normalization and into the FFN block.</p>
<p>The post-attention re-provisioning, from KVP to TP for the post-attention linear projection, is identical
in both dense and MoE models. After the post-attention linear projection and All-Reduce over the
N GPUs on its output, the full pool of N GPUs is re-provisioned for the FFNs (see Figure 4):</p>
<p>• Dense FFNs (EP = 1): Keep T P F = N , all GPUs collaborate on amortizing weight-read costs</p>
<p>and maintaining low latency on small batches.</p>
<p>• MoE FFNs, (EP &gt; 1): Repartition into an optimal T P F × EP grid, choosing T P F versus EP</p>
<p>to best match expert size and quantity.</p>
<p>By chaining KVP × TPA for attention, a TP = N post-attention linear projection, and a flexible
TPF × EP FFN layout, Helix ensures all N GPUs remain fully utilized in a zero-downtime pipeline,
maximizing scalability and throughput across attention, projection, and FFN stages.</p>
<p>2.3 Distributed KV concatenation strategy</p>
<p>During decoding, each newly generated token is broadcast to all KVP GPUs so that every device
has access to the current query. However, Helix staggers the KV cache updates across KVP ranks to
maintain balanced memory growth and ensure each GPU appends KV entries uniformly. Specifically,
the system appends KV pairs for a fixed number of decode steps (e.g., 16 tokens) to the shard on KVP
Rank 0, then switches to KVP Rank 1 for the next 16 tokens, and so on, cycling through all N KVP
ranks in round-robin fashion. This staged KV concatenation guarantees that all KVP GPUs contribute
to KV storage regardless of batch size or sequence length, avoiding hot spots and distributing KV
cache growth evenly across the pool.</p>
<p>3 Evaluation</p>
<p>In this section, we evaluate Helix Parallelism against the best known prior LLM sharding methods
on NVIDIA’s latest GB200 NVL72 hardware [8] with FP4 precision [11]. Rather than focusing
on isolated configurations, we characterize the full Pareto frontier between throughput per gpu
(tokens/sec/gpu) vs. interactivity (tokens/sec/user). Here, user interactivity is measured as reciprocal
of decoding TTL (token-to-token latency), representing the rate at which new tokens are generated
for a single user. Throughput per GPU is quantified as the total number of tokens generated per
second per GPU, reflecting system-wide efficiency.</p>
<p>To provide a more comprehensive view, we also introduce batch scalability, the maximum number
of concurrent user requests that can be sustained under a fixed TTL budget. This metric reflects a
system’s ability to maintain real-time responsiveness at scale.</p>
<p>3.1 Experimental setup</p>
<p>To evaluate the performance of different sharding strategies independent of potential feature gaps in a
given software framework, we opt to leverage an in-house high-fidelity simulator modeling the latest
GB200 hardware. The simulator accounts for both compute and communication costs, including
latency from inter-GPU NVLink transfers, DRAM bandwidth constraints, and FLOP throughput.
All performance numbers are normalized to that of the baseline to focus on the trends as opposed to
specific performance claims.</p>
<p>We evaluate Helix Parallelism on two large-scale LLMs representative of dense and MoE architectures:
(i) Llama-405B [12], a dense 405B parameter model with 128 query heads and 8 grouped KV heads
(i.e., GQA attention). (ii) DeepSeek-R1 [13], a 671B parameter MoE model with MLA attention.</p>
<p>6</p>
<p>In MLA, key and value projections are absorbed into a latent space during decoding and are not
explicitly materialized, effectively resulting in a single KV head shared across all 128 query heads.
Although these models do not yet natively support million-token contexts, we simulate decode-time
inference with KV-cache sequence lengths of one million tokens and beyond. This allows us to
analyze system-level bottlenecks and assess the potential of Helix for real-time applications at this
scale.</p>
<p>Our baseline search space covers the best-known partitioning strategies, tensor parallelism (TP),
pipeline parallelism (PP), expert parallelism (EP), and vanilla KV partitioning (KVP), alongside a
full sweep over batch sizes. Here, EP denotes data-parallel attention coupled with expert-parallel
FFNs, as adopted in production DeepSeek-R1 [13]. Vanilla KVP refers to the original Medha-style
sharding approach, first introduced in [7].</p>
<p>Each point on the Pareto frontier corresponds to a unique combination of model partitioning and batch
size. For any given TTL constraint, we report the configuration that maximizes system throughput,
forming a unified Pareto curve. Throughout this section, "Baseline" refers to the best of the baseline
search space above, while "Helix" denotes the use of temporal pipelining with decoupled attention
and FFN sharding paired with HOP-B optimization.</p>
<p>We constrain our search for optimal model mappings to configurations using 1-64 GPUs, fitting
within a single GB200 node. All model weights, KV states, and arithmetic operations are assumed to
use FP4 [11], reflecting emerging trends in low-precision LLM inference deployments.</p>
<p>3.2 Results</p>
<p>Figure 5 and Figure 6 provide the throughput vs.
interactivity Pareto frontier for DeepSeek-R1
and Llama-405B, respectively. To derive these Pareto frontiers, we exhaustively simulated over
100,000 configurations, systematically varying model partitioning strategies (TP, EP, PP, KVP), batch
sizes, and GPU counts across different LLM architectures and inference serving techniques.3 As
demonstrated, Helix significantly improves the throughput-latency trade-off by pushing the Pareto
frontier outward, enabling both higher system throughput and lower per-user latency simultaneously.</p>
<p>Figure 5: Pareto frontier of serving DeepSeek-R1 with 1-Million context length on GB200
.</p>
<p>For DeepSeek-R1, Helix improves user interactivity by up to 1.5×, enabling lower token latency
while simultaneously scaling batch capacity and throughput. Specifically, Helix supports up to 32×
more concurrent users (i.e., achieves 32× higher Tokens/s/GPU) compared to the baseline. This is
enabled by Helix’s ability to shard both KV caches and FFN weights across all available devices,
reducing DRAM pressure and increasing compute efficiency. Note that Medha’s approach of tying
TP between FFNs and attention is not well-suited for modern networks with MLA attention, i.e.,</p>
<p>3For clarity, the Pareto frontiers shown in the figures represent only the optimal configurations that achieve</p>
<p>the best throughput-latency trade-offs.</p>
<p>7</p>
<p>T P &gt; 1 causes duplication of KV cache. In addition, Medha [7] does not provide any results on
MoE models. As such, a direct comparison is not applicable for the DeepSeek-R1 network.</p>
<p>For Llama-405B, Helix yields a 1.13× improvement in maximum achievable interactivity and
a 4× higher throughput and batch capacity compared to TP sharding. These gains come from
(1) lifting TP’s KV-duplication ceiling via KVP, and (2) further increasing FFN parallelism without
introducing cache duplication. The comparison with Medha in Figure 6 reinforces the benefits of
removing the constraint of tying TP width between attention and FFNs. While both Helix and our
baseline TP implementation include communication-computation overlap, Medha systems expose all
communication overheads, which further underscores the importance of HOP-B.</p>
<p>Figure 6: Pareto frontier of serving Llama-405B with 1-Million context length on GB200
.</p>
<p>3.3 Ablation study: impact of HOP-B</p>
<p>We isolate Helix’s batch-wise overlap strategy (HOP-B) by turning it off during attention. In "HOP-B
OFF" mode, communication and computation execute strictly sequentially, incurring idle GPU stalls
that reduce Tokens/s/User by up to 12% at a fixed Tokens/s/GPU (Figure 7 (right)). Re-enabling
HOP-B, pipelines each request’s communication with the next request’s attention compute, closing
these gaps and recovering most of the lost TTL (Figure 3).</p>
<p>Figure 7: Pareto frontier of HOP-B ON vs. HOP-B OFF with 1-Million context length on GB200.</p>
<p>In Figure 7, DeepSeek-R1 (left) suffers only ~1% degradation when HOP-B is OFF, its all-to-all
exchange accounts for just ~1% of end-to-end decode latency, with latent projections, shared-expert
computation, and multi-expert GEMMs dominating, whereas Llama-405B (right) incurs a ~12% drop</p>
<p>8</p>
<p>in Tokens/s/User without HOP-B. This stark contrast highlights that communication–computation
overlap becomes increasingly critical as communication forms a larger fraction of TTL.</p>
<p>4 Related work</p>
<p>A large body of prior work has focused on sequence parallelism techniques, primarily in the context
of training and prefill stages [14–17]. These methods often partition the sequence dimension to
improve memory efficiency and parallelism during the non-autoregressive phases of model execution.
While effective for prefill and training, these strategies are not well-optimized for the decoding phase,
where strict TTL constraints and the need for causal access to growing KV caches introduce unique
system-level bottlenecks, particularly when dealing with KV histories spanning millions of tokens.</p>
<p>Another line of related work explores KV Parallelism and the practice of tying TP across attention
and FFN layers to simplify execution (e.g., [7]). This tight coupling, however, limits scalability
especially on new model architectures such as DeepSeek-R1 which feature MLA and MoE FFNs.</p>
<p>To the best of our knowledge, Helix is the first parallelism framework explicitly designed to address
decoding bottlenecks in modern LLM architectures with increasing context lengths. By decoupling
sharding strategies for attention and FFNs and introducing a temporal execution pipeline, Helix
better aligns GPU utilization with the computational characteristics of each stage, enabling low
latency decoding at multi-million-token context lengths. Helix is compatible with diverse attention
mechanisms, including GQA and MLA, and is co-designed with Blackwell’s latest capabilities to
leverage features such as its large NVLink domain.</p>
<p>5 Discussion</p>
<p>Although the primarily discussion in this paper focuses on long-context inference through Helix
Parallelism, it is important to note that Helix is a general-purpose inference optimization technique
that extends beyond long-sequence scenarios. Its core principle, separating attention and FFN
computations and distributing them independently, applies broadly across context lengths.</p>
<p>In the short-context regime (e.g., context lengths &lt; 4K), Helix simplifies to data-parallel attention and
tensor-parallel FFN, a pattern already widely used in modern LLM inference serving frameworks[18,
19]. Helix captures a growing trend in handling inference across diverse context lengths - offering a
coherent abstraction for both mainstream and emerging long-context workloads.</p>
<p>6 Future work</p>
<p>In this paper, we introduced and evaluated Helix parallelism and illustrated its effectiveness in
optimizing decoding performance with multi-million-token KV histories across a variety of latest
dense attention architectures and MoE models. One natural extension is to support sparse attention
mechanisms such as Natively Sparse Attention (NSA) [20], which reduce KV read bandwidth but
not overall memory capacity requirements. Given Helix’s modular design, we expect its principles to
translate naturally to these emerging architectures.</p>
<p>7 Conclusion</p>
<p>Helix Parallelism represents a paradigm shift in decoding efficiency for ultra-long-context LLMs
operating under tight latency constraints. By decoupling parallelism as well as scaling strategies
for attention and FFN layers through a temporal pipeline, Helix overcomes key bottlenecks in
long-context decoding, boosting system interactivity and efficiency. Helix is fully compatible with
modern LLM architectures, including diverse attention mechanisms (e.g., GQA, MLA) and sparse
Mixture-of-Experts. Its design also aligns seamlessly with emerging GPU platforms such as the
Blackwell system, ensuring readiness for future hardware and model trends.</p>
<p>9</p>
<p>References</p>
<p>[1] Gemini Team Google. Gemini 1.5: Unlocking multimodal understanding across millions of</p>
<p>tokens of context, Dec 2024.</p>
<p>[2] Meta Llama 4 Team. The Llama 4 herd: The beginning of a new era of natively multimodal AI</p>
<p>innovation, Apr 2025.</p>
<p>[3] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and
Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints, 2023.</p>
<p>[4] Noam Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. Blog post, 2019.</p>
<p>https://noam.github.io/2019/09/18/fast-transformer-decoding.html.</p>
<p>[5] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao,
Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji,
Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui
Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie
Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu,
Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua
Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang,
Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu,
Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng
Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang,
Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng
Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong
Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun,
Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan
Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li,
Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong,
Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji
Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma,
Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang,
Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu
Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. DeepSeek-V2: A Strong,
Economical, and Efficient Mixture-of-Experts Language Model, 2024.</p>
<p>[6] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism, 2020.</p>
<p>[7] Amey Agrawal, Haoran Qiu, Junda Chen, Íñigo Goiri, Ramachandran Ramjee, Chaojie Zhang,
Alexey Tumanov, and Esha Choukse. Medha: Efficiently Serving Multi-Million Context Length
LLM Inference Requests Without Approximations, 2025.</p>
<p>[8] NVIDIA. Nvidia blackwell architecture technical brief, 2024. NVIDIA Technical Documenta-</p>
<p>tion.</p>
<p>[9] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao.</p>
<p>Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024.</p>
<p>[10] Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. Flash-Decoding for long-context
inference. https://crfm.stanford.edu/2023/10/12/flashdecoding.html, 2023.</p>
<p>[11] Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Sum-
mer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, Stosic Dusan,
Venmugil Elango, Maximilian Golub, Alexander Heinecke, Phil James-Roxby, Dharmesh Jani,
Gaurav Kolhe, Martin Langhammer, Ada Li, Levi Melnick, Maral Mesmakhosroshahi, Andres
Rodriguez, Michael Schulte, Rasoul Shafipour, Lei Shao, Michael Siu, Pradeep Dubey, Paulius
Micikevicius, Maxim Naumov, Colin Verrilli, Ralph Wittig, Doug Burger, and Eric Chung.
Microscaling data formats for deep learning, 2023.</p>
<p>10</p>
<p>[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ah-
mad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem
Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson,
Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux,
Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret,
Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,
Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco
Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind
Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah
Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan
Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason
Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya
Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton,
Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Va-
suden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield,
Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal
Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz
Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke
de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin
Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam-
badur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi,
Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne,
Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal
Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao
Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert
Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre,
Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hos-
seini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov,
Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale,
Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane
Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha,
Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet,
Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin
Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan,
Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine
Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert,
Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain,
Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay
Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit
Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu,
Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco,
Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe,
Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang,
Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock,
Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker,
Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester
Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon
Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine,
Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin
Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers,
Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank
Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee,
Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan
Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison</p>
<p>11</p>
<p>Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj,
Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman,
James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff
Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin,
Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh
Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun
Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh,
Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro
Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt,
Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew
Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao
Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel
Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat,
Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White,
Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich
Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem
Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager,
Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang,
Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra,
Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ
Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh,
Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao
Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang,
Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen
Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng,
Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez,
Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim
Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez,
Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu
Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Con-
stable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman,
Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin
Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3
herd of models, 2024.</p>
<p>[13] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu,
Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian
Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang,
Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo,
Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong
Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li,
Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian,
Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du,
R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu
Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu,
Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng
Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng,
Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang,
X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen,
Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang,
Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi
Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei,
Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng
Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying
He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,</p>
<p>12</p>
<p>Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha,
Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,
Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang,
Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong
Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu,
Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report,
2025.</p>
<p>[14] Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, and Xin Jin. Loongserve:
Efficiently serving long-context large language models with elastic sequence parallelism, 2024.</p>
<p>[15] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long</p>
<p>context generative ai, 2024.</p>
<p>[16] Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Xuezhe Ma, Ion Stoica, Joseph E. Gonzalez,
and Hao Zhang. Distflashattn: Distributed memory-efficient attention for long-context llms
training, 2024.</p>
<p>[17] Amy Yang, Jingyi Yang, Aya Ibrahim, Xinfeng Xie, Bangsheng Tang, Grigory Sizov, Jeremy
Reizenstein, Jongsoo Park, and Jianyu Huang. Context parallelism for scalable million-token
inference, 2025.</p>
<p>[18] Nvidia tensorrt-llm. https://github.com/NVIDIA/TensorRT-LLM.</p>
<p>[19] Data parallelism attention for deepseek models.</p>
<p>https://lmsys.org/blog/
2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models.</p>
<p>[20] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda
Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng
Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable
sparse attention, 2025.</p>
<p>13</p>
<p>A Roofline analysis</p>
<p>This section shows the formulas used to derive KV cache and FFN weight read times shown in
Figure 1.</p>
<p>B : Batch Size
Q : Q heads
K : KV heads
Hsz : Attention Head Size
S : KV Sequence length
H : Hidden dimension = Q × Hsz
F : Intermediate Hidden dimension
T P A : T P width f or Attention
T P F : T P width f or F F N
KV P : KV P width
bytesparam : Bytes per parameter
M emBW : GP U M emory bandwidth</p>
<p>Time to read KV cache per LLM layer is given by below equation. It assumes individual K and V
heads to represent attention variants like GQA.</p>
<p>B × 2 × ⌈ K</p>
<p>T P A ⌉ × Hsz × S</p>
<p>KV P × bytesparam</p>
<p>Time to read weights per LLM layer is given by below equation. It assumes SwiGLU activation in
FFN block.</p>
<p>M emBW</p>
<p>((2 × H × Q</p>
<p>T P A × Hsz) + (2 × H × ⌈ K</p>
<p>T P A ⌉ × Hsz) + (3 × H×F</p>
<p>T P F )) × bytesparam</p>
<p>Figure 1 assumes MemBW = 8000 GB/s.</p>
<p>M emBW</p>
<p>14</p>
