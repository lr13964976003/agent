Training-free and Adaptive Sparse Attention for Efficient Long Video Generation
Yifei Xia1,2Suhan Ling1,2Fangcheng Fu1Yujie Wang1,2
Huixia Li2Xuefeng Xiao2Bin Cui1
1Peking University2ByteDance
{yifeixia, lingsuhan}@stu.pku.edu.cn {ccchengff, alfredwang, bin.cui}@pku.edu.cn
{lihuixia, xiaoxuefeng.ailab}@bytedance.com
CogVideoX1.5-5B, 161 frames, 720p 
 HunyuanVideo, 129 frames, 720p 
Sparse VideoGen 
PSNR = 27.61 
Latency = 34 min MInference 
PSNR = 22.53 
Latency = 42 min AdaSpa (ours) 
PSNR = 29.07 
Latency = 30 minFull Attention 
Latency = 54 min Sparse VideoGen 
PSNR = 18.98 
Latency = 34 min MInference 
PSNR = 10.31 
Latency = 38 min AdaSpa (ours) 
PSNR = 23.25 
Latency = 31 minFull Attention 
Latency = 52 min 
Prompt: Clown fish swimming through the coral reef Prompt: A cute happy Corgi playing in park, sunset, pan right 
Figure 1. Comparison of the visualization effects of different sparse attention methods on HunyuanVideo [33] and CogVideoX1.5-5B [65].
Our method AdaSpa consistently achieves the best performance and the best speedup, and keep almost the same as original videos.
Abstract
Generating high-fidelity long videos with Diffusion Trans-
formers (DiTs) is often hindered by significant latency, pri-
marily due to the computational demands of attention mech-
anisms. For instance, generating an 8-second 720p video
(110K tokens) with HunyuanVideo takes about 600 PFLOPs,
with around 500 PFLOPs consumed by attention computa-
tions. To address this issue, we propose AdaSpa , the first Dy-
namic Pattern andOnline Precise Search sparse attention
method. Firstly, to realize the Dynamic Pattern, we introduce
a blockified pattern to efficiently capture the hierarchical
sparsity inherent in DiTs. This is based on our observation
that sparse characteristics of DiTs exhibit hierarchical and
blockified structures between and within different modalities.
This blockified approach significantly reduces the complex-
ity of attention computation while maintaining high fidelity
in the generated videos. Secondly, to enable Online Pre-
cise Search, we propose the Fused LSE-Cached Search with
Head-adaptive Hierarchical Block Sparse Attention. Thismethod is motivated by our finding that DiTs’ sparse pattern
and LSE vary w.r.t. inputs, layers, and heads, but remain
invariant across denoising steps. By leveraging this invari-
ance across denoising steps, it adapts to the dynamic nature
of DiTs and allows for precise, real-time identification of
sparse indices with minimal overhead. AdaSpa is imple-
mented as an adaptive, plug-and-play solution and can be
integrated seamlessly with existing DiTs, requiring neither
additional fine-tuning nor a dataset-dependent profiling. Ex-
tensive experiments validate that AdaSpa delivers substantial
acceleration across various models while preserving video
quality, establishing itself as a robust and scalable approach
to efficient video generation.
1. Introduction
Diffusion models [ 14,23,24,47,52] have emerged as a pow-
erful framework for generative tasks, achieving state-of-the-
art results across diverse modalities, including text-to-image
synthesis [ 5,7,17,34,35,46,48,49,51,64,68], realisticarXiv:2502.21079v1  [cs.CV]  28 Feb 20252 8 16 32
Video Length (s)0%20%40%60%80%100%Proportion
02K4K6K8K
PFLOPs
HunyuanVideo-Attn
HunyuanVIdeo-OthersCogVideoX-Attn
CogVideoX-OthersHunyuanVideo-FLOPs
CogVideoX-FLOPsFigure 2. The total FLOPs required and the proportion of attention
when generating 720p videos with different video lengths (16FPS).
video generation [ 25,27,31,33,37,55,61,65,72], and
3D content creation [ 6,9,26,28,43]. Recently, the intro-
duction of Diffusion Transformers (DiTs) [ 42], exemplified
by Sora [ 4], has set new benchmarks in video generation,
enabling the production of long, high-fidelity videos.
Despite these advances, generating high-quality videos
remains computationally expensive, especially for long
videos [ 8,16,22,54]. The attention mechanism [ 58] in
the Transformer architecture [ 58], with its O(n2)complex-
ity, is a major bottleneck, where ndenotes the sequence
length. For instance, generating an 8-second 720p video
with HunyuanVideo takes about 600 PFLOPs, with nearly
500 PFLOPs consumed by attention computations. This pro-
portion increases with higher resolution or longer duration
videos, as illustrated in Figure 2.
Although attention mechanisms are essential for sound
performance, they involve significant computational redun-
dancy [ 10]. Addressing this redundancy can greatly reduce
inference costs and accelerate video generation [ 62]. Sparse
attention mechanisms [ 1,3,10,15,18,19,21,30,38,39,44,
45,53,59,62,66,67,69], which exploit this redundancy,
have shown success in large language models (LLMs) by
reducing computational costs without compromising perfor-
mance.
Sparse attention typically characterizes this redundancy
assparse patterns (a.k.a. sparse masks ), indicating which
interactions between tokens can be omitted to reduce com-
putational load. The specific positions of the selected tokens
insparse patterns that are not omitted are called sparse in-
dices . Based on the flexibility of pattern recognition, existing
sparse patterns can be broadly categorized into the following
two types:
•Static Pattern [1,3,10,21,62,67] refers to the use of
predetermined sparse indices that are defined by prior
knowledge. This category can be further divided into two
types:
Fixed Pattern uses only one fixed sparse pattern based
on empirical experience. For instance, LM-Infinite [ 21]and StreamingLLM [ 63] (Figure 3a) consistently utilize
the sliding window [ 3] pattern. This approach is straight-
forward, generally requiring no pattern search, and only
necessitates the prior specification of hyperparameters.
Mixed Pattern involves determining several fixed patterns
based on experience and then selecting one or more of
these patterns during the execution of attention. Exam-
ples include BigBird [ 67] and Sparse VideoGen [ 62] (Fig-
ure 3b), which typically perform a rough online switching
mechanism to estimate and determine which pattern (or
combination of patterns) should be applied in each atten-
tion operation.
•Dynamic Pattern [19,30,38,44,45,53] features ad hoc
sparse indices that need to be decided in real time. Exam-
ples include DSA [ 38] and MInference [ 30] (Figure 3c).
It necessitates a search to determine which indices to use
for each attention operation. Due to the extensive time
consumption involved in searching, current Dynamic Pat-
tern methods typically rely on offline search and/or online
approximation search.
Offline Search methods involve performing offline
searches to determine the specific indices. A subset of
the target dataset is usually used in the offline search.
Online Approximate Search methods involve searching in
real-time, yet applying some form of approximation to
estimate sparse indices during the execution.
However, due to the dynamic complexity and data-
adaptive nature of DiT patterns, these methods face sig-
nificant limitations when applied to DiTs.
Firstly, the Static Pattern is not flexible enough to sum-
marize the sparse characteristics of DiTs. In particular, as
we will show in Section 3, the sparse patterns of DiTs are
extremely dynamic and irregular. Thus, static pattern meth-
ods fail to accurately capture the sparse indices and thereby
suffer from poor performance (as evaluated in Section 5).
Secondly, the existing Dynamic Pattern methods are un-
able to adaptively and accurately identify the sparse pat-
terns of DiTs. For one thing, our empirical observations
in Section 3 demonstrate that the sparsity of DiTs exhibits
considerable variation depending on the input, which makes
offline search in DiTs lack good portability and accuracy.
For another, it can be observed that the sparse indices in
DiTs are complex, with key areas being dispersed and not
concentrated and continuous, making it difficult to accu-
rately estimate sparse indices through approximation search.
Thus, directly applying current dynamic pattern methods
(e.g., MInference) to DiT also yields poor results (detailed
in Section 5).
Therefore, identifying and generalizing sparse patterns
suitable for DiTs, and implementing kernel-efficient methods
for precise pattern search and attention execution remains an
urgent problem to be solved.
Motivated by this, we propose Adaptive Sparse Atten-(a)  StreamingLLM 
(Fixed Pattern) (d) AdaSpa 
(Online Precise Search) (c) Minference 
(Offline Search + Online Approx. Search) (b) Sparse VideoGen 
(Mixed Pattern) Sink + Window Diagonal Column Block 
Attention Sink Diagonal 
First Frame Sink 
Text Sink Spatial Head 
Temporal Head Block + Text Sink 
Block ① Online 
Switch 
Static Patterns Dynamic Patterns Fixed sparse indices 
Column + Diagonal Column 
① Offline 
Search 
Target Pattern ② Online 
Approx. Search 
 ① Online 
Precise Search Suboptimal 
Dynamic 
sparse indices 
No Switch 
No Search 
Optimal 
Dynamic 
sparse indices 
Process Fixed sparse indices Figure 3. Different types of Sparse Pattern recognition methods. (a) StreamingLLM: using a static sink+sliding window pattern, need no
search or switch. (b) Sparse VideoGen: preparing two predefined Static Patterns, and using an online switching method to determine which
to use. (c) MInference: preparing several dynamic patterns, first do an offline search to determine the target pattern to use, then perform an
online approximate search to search suboptimal sparse indices of this pattern. (d) AdaSpa: our method proves that the most suitable pattern
for DiT is blockified pattern, and performs an online precise search to find the optimal sparse indices for blockified pattern.
tion (AdaSpa) , the first Dynamic Pattern + Online Precise
Search (Figure 3d) method for high-fidelity sparse atten-
tion. It is a training-free and data-free method designed to
accelerate video generation in DiTs while preserving gen-
eration quality. It outperforms all other SOTA methods in
both Static and Dynamic Patterns, as shown in Figure 1. Our
contributions are summarized as follows:
•Comprehensive Analysis of Attention Sparsity in DiTs.
We present an in-depth analysis of sparse characteristics
in attention mechanisms for DiTs, examining the special
sparse characteristics of DiTs to reveal optimal sparsity
strategies and provide new insights for future research.
Based on extensive observations and summaries, we found
that the sparse characteristics of DiTs have two traits: 1)
Hierarchical andBlockified , 2)Invariant in steps, Adap-
tive in prompts and heads .
•First Dynamic Patterns and Precise Online Search
Sparse Attention Solution without Training and Pro-
filing. We propose AdaSpa, a novel sparse attention ac-
celeration framework that is both training-free and data-
free. As shown in Figure 3d, AdaSpa is the first effective
method that combines Dynamic Pattern and Online Precise
Search, proposing an efficient pipeline for online sparse
pattern search and fine-grained sparse attention computa-
tion. Leveraging the invariant characteristics across denois-
ing steps, AdaSpa is equipped with Fused LSE-Cached
Online Search, which reduces online search time to under
5% of full attention generation time using our optimizedkernel, significantly reducing the additional time for search
while ensuring accurate search. Additionally, in order to
better adapt to the sparse characteristics of DiT, we pro-
pose a Head-Adaptive Hierarchical Block Sparse method
for AdaSpa to address the head-adaptive sparsity feature
of DiTs.
•Implementation and Evaluation. AdaSpa provides
a plug-and-play adaspa _attention _handler that seam-
lessly integrates with DiTs, requiring no fine-tuning or data
profiling. It is orthogonal to other acceleration techniques
like parallelization, quantization and cache reuse. Exten-
sive experiments validate AdaSpa’s consistent speedups
across models with negligible quality loss.
2. Preliminaries
2.1. Diffusion Transformers and 3D Full Attention
Diffusion Transformers (DiTs) [ 42] refine predictions with a
diffusion process, handling multimodal data like video and
text through an attention mechanism that captures spatial,
temporal, and cross-modal dependencies. DiTs traditionally
use Spatial-Temporal Attention [ 37,72], applying spatial
attention with each video frame, temporal attention across
all frames, and cross-attention to connect video and text, as
shown in Figure 4. This separation limits frame continuity
and fusion.
Figure 4d illustrates the 3D Full Attention mechanism [ 27,
33,65] in DiTs. It integrates video and text tokens into2-2
5-22-5
5-51-1
4-11-4
4-4523-33-43-5
1-01-11-5
2-02-15-534
15012 345 67
012
3
4
50
1
20-00-10-2
1-01-11-2
2-02-12-2
40
352
1403
0-0
3-00-3
3-30
1
2
3
4
567
0-6
1-6
2-6
3-6
4-6
5-60-7
1-7
2-7
3-7
4-7
5-70
1
2
3
4
5
6
701234567
0-0
1-0
2-0
3-0
4-0
5-0
6-0
7-00-1
1-1
2-1
3-1
4-1
5-1
6-1
7-10-2
1-2
2-2
3-2
4-2
5-2
6-2
7-20-3
1-3
2-3
3-3
4-3
5-3
6-3
7-30-4
1-4
2-4
3-4
4-4
5-4
6-4
7-40-5
1-5
2-5
3-5
4-5
5-5
6-5
7-50-6
1-6
2-6
3-6
4-6
5-6
6-6
7-60-7
1-7
2-7
3-7
4-7
5-7
6-7
7-7Video Frame1 Video Frame2 Text
QueryKey
(a) Spatial Attention
Key
Query
(b) Temporal AttentionKey
QueryKey
Query
(d) 3D Full Attention (c) Cross AttentionFigure 4. Different Attention Mechanisms in DiTs.
a unified sequence and applies self-attention across them.
Operating in the latent space, DiTs process video frames
that have been pre-encoded. Let fbe the number of latent
frames, h×wthe spatial resolution of each frame, and tthe
text token length, with f·h·w≫t. The total sequence
length, L, can be represented as:
L=f·h·w+t. (1)
This unified approach enhances modality fusion and boosts
overall performance.
Despite the increased computational cost of 3D Full At-
tention, it marks the future of DiTs, offering superior multi-
modal learning compared to Spatial-Temporal Attention.
2.2. FlashAttention
In the self-attention mechanism [ 58], tokens are pro-
jected into the query, key, and value matrices Q,K,V∈
RH×L×D, where His the number of attention heads, Lis
the input length, and Dis the hidden dimension of each head.
The attention weights matrix Wattn∈RL×Lis computed as:
Wattn=softmaxQK⊤
√
D
, (2)
which quantifies token-to-token interactions across the se-
quence. To maintain numerical stability during the expo-
nentiation, the Log-Sum-Exp (LSE) [ 2] trick is commonly
employed. Let Z=QK⊤
√
dand denote by zjthej-th compo-
nent of a row z. Then, LSE can be written as:
LSE(z) = logX
jexp(zj)
= max
jzj+ logX
jexp 
zj−max
kzk
.(3)
Using this, the safe Softmax can be expressed as:
Softmax safe(zj) = exp 
zj−LSE(z)
, (4)and the entire dense attention distribution in a numerically
stable form is:
Wattn=Softmax safe(QK⊤
√
D). (5)
This operation, however, requires constructing an L×L
attention matrix, leading to O(L2)time and memory com-
plexity, which becomes prohibitive for long sequences.
FlashAttention [ 12,13,50] addresses this issue by per-
forming attention in a blockwise manner. Instead of storing
the full attention matrix, FlashAttention processes smaller
chunks sequentially. In FlashAttention, attention is com-
puted for smaller blocks of tokens, and the key idea is to
perform attention on these blocks without constructing the
entire attention matrix at once. Specifically, for each block,
the attention is computed as:
W(b)
attn=online_softmax(QbK⊤
b)√
D
, (6)
where QbandKbrepresent the query and key matrices
for block b, where L≫b, and online_softmax [ 41] is a
blockwise equivalent version of the safe softmax. The result
is then multiplied by the value matrix for the block, Vb, to
obtain the final attention output:
Ab=W(b)
attnVb. (7)
This block-wise computation significantly reduces the mem-
ory footprint to O(Lb), as only a subset of the full attention
matrix is processed at any given time. FlashAttention is
particularly effective for large-scale transformers and long-
sequence tasks, such as 3D Full Attention.
2.3. Sparse Attention and Sparse Patterns
Attention mechanisms exhibit inherent sparsity [ 10], en-
abling computational acceleration by limiting interactions to
a subset of key-value pairs. Sparse attention reduces com-
plexity by ignoring interactions where the attention weight
W(i,j)
attn is small. This principle forms the basis of sparse
attention.
Formally, sparse attention is defined by a masking func-
tionM∈ {0,1}L×L, which Mij= 1indicates that token
iattends to token j, andMij= 0removes the interaction.
This masking function Missparse pattern , the indices set of
Mij= 1issparse indices , and the proportion of Mij= 0
is called sparsity . The sparse attention operation is defined
as:
Aattn=softmax(QK⊤)⊙M√
D
V, (8)
where⊙denotes element-wise multiplication. The effective-
ness of a sparse pattern is evaluated using Recall [57], whichmeasures how well the sparse pattern preserves the original
dense attention behavior:
Recall =P
(i,j)∈sparse indicesW(i,j)
attn
P
i,jW(i,j)
attn, (9)
Higher Recall indicates better retention of the original atten-
tion structure.
3. Sparse Pattern Characteristic in DiTs
In this section, we present the key observations of the sparse
characteristics and opportunities in DiTs that motivate our
work.
Observation 1: DiTs exhibit Hierarchical Structure of
sparse pattern within and between different Modality, mak-
ing continous patterns unsuitable. As introduced in Sec-
tion 2, DiTs leverage 3D attention to model spatial and tem-
poral dependencies across video frames while integrating
text tokens for joint attention. Given an input sequence, it
comprises video tokens and text tokens, with a total length
ofL=f·h·w+t(Equation 1). Thus, the attention weights
matrix, Wattn∈RL×L, has a hierarchical organization of
text and video tokens. Particularly, as depicted in Figure 5,
it can be decomposed as follows:
Wattn=Wvideo-video Wvideo-text
Wtext-video Wtext-text
, (10)
where:
•Video-video attention ,Wvideo-video ∈R(f·h·w)×(f·h·w),
captures spatial and temporal interactions among video
tokens.
•Text-video and text-text attention ,Wtext-text ∈Rt×t,
Wtext-video ∈Rt×(f·h·w)andWvideo-text ∈R(f·h·w)×t,
model interactions involving text tokens, which often serve
as a global text sink for attention.
Moreover, within Wvideo-video , attention weights are fur-
ther structured into f×fframe regions :
Wvideo-video =
R1,1R1,2···R1,f
R2,1R2,2···R2,f
............
Rf,1Rf,2···Rf,f
, (11)
where Ri,j∈R(h×w)×(h×w)represents interactions be-
tween the i-th and j-th video frames. As shown in Figure 5,
there are clear boundaries between the frames.
This hierarchical characteristic makes continuous sparse
patterns ineffective, as the sparsity structure is no longer
globally uninterruptible. In a continuous sparse pattern,
nonzero elements extend continuously across the entire ma-
trix, such as colpatterns, where specific columns remain
active in all rows, or diag patterns, where nonzero valuesform a diagonal path from one side to the other. However,
due to the hierarchical structure of certain attention weight,
their sparse patterns become fragmented rather than main-
taining such continuity, making it impossible to describe
them using continuous sparse patterns. Nevertheless, while
the overall structure lacks continuity, we observe that within
each frame region, the sparsity pattern remains locally struc-
tured and can often be well characterized using continuous
patterns like colordiag.
This insight motivates a frame region -wise search strategy
to capture localized continuous structures and reconstruct
the overall sparsity pattern. However, as shown in Figure 5,
attention distribution varies significantly across different
frame regions , nonzero weights tend to concentrate in only a
fewframe regions rather than being evenly distributed. This
imbalance reduces the effectiveness of the frame region-wise
approach, as it fails to provide a globally optimized sparse
representation.
Solution 1: Using the blockified pattern to describe the
sparse features of DiT. Although continuous patterns like
col or diag do not work well, we find that the sparse pattern
evolves into a blockified structure globally. For example, as
shown in Figure 5a, within each frame region , the sparsity
follows a colpattern. However, due to weak inter-region
interactions, hierarchical sparsity disrupts interlinearly con-
tinuous colpatterns, leading to a blockified structure. As
observed in the figure, this blockified characteristic achieves
better Recall , indicating the blockified pattern a more suitable
pattern. Similarly, in Figure 5b, each frame region follows a
hybrid of diag andcolpatterns. Yet, due to significant varia-
tions in inter-frame interactions, the global attention weights
exhibit a combination of a sliding window pattern and a dis-
tinct random blockified structure, making it impossible to
describe with standard sparsity patterns. Another example is
shown in Figure 5c, where individual frame regions lack a
clear local pattern, while the global attention weights form
anoncontinuous-diag pattern combined with a bottom sink
effect. As seen in Figure 5c, this characteristic can also be
effectively modeled using a blockified representation with
the best Recall .
In summary, due to the hierarchical nature of the DiT
patterns, conventional continuous patterns fail to provide
an effective representation. Thus, adopting the blockified
pattern is the optimal choice for capturing the sparsity char-
acteristics of DiT, because it consistently achieves the best
recall, as shown in Figure 5.
Observation 2: DiTs’ sparse pattern vary w.r.t. inputs,
layers and heads, making offline search unsuitable. As
illustrated in Figure 6a, the sparse patterns in DiTs vary
depending on attention head, and layer, which is similar to
LLMs [30, 38].
Meanwhile, we observe that the sparse patterns of dif-
ferent prompts also vary significantly. In Figure 6c, weTopK Col Diag Diag+Col Weight
(a)
(b)
Block
Recall=0.54 Recall=0.52 Recall=0.16 Recall=0.46 Recall=0.47
Recall=0.93 Recall=0.90 Recall=0.18 Recall=0.79 Recall=0.79
Recall=1.0
 Recall=1.0
 Recall=0.12 Recall=0.96 Recall=0.93(c)
Figure 5. Typical attention weight maps from HunyuanVideo. Weight represents the visualization result of attention weights. Topk ,Block ,
Col,Diag ,Diag+Col represent the visualization results of sparse patterns under sparsity 0.9. The far right shows an enlarged view of the
attention weights selected from the bottom right corner with a size of (2∗h∗w+t)×(2∗h∗w+t), where a clear hierarchical effect
between frames can be observed. At the same time, there is a distinct boundary between the text modality and the pure video modality,
exhibiting varying degrees of text sink effect. (720p, 129 frames, block size of the block pattern is 32)
01020304050Step05101520
Head0102030405060
Layer
0.20.40.60.81.0Recall
(a) Sparse pattern’s Recall changes with
head and layer, but invariant step.
0 1 2 3 4 5 6 7 8 9
Step91011121314151617Value(b) LSE distribution changes with the varia-
tion of step.
prompt1
seed0prompt2
seed0prompt1
seed42prompt2
seed420.20.40.60.81.0Recall Distribution(c) Sparse pattern changes with inputs
Figure 6. (a) Visualization of recall changes with head, layer, and step. Under the condition of fixed sparsity = 0.9, the attention recall of
HunyuanVideo in the TopK paradigm changes with the variations of Head and Layer, but stay invariant with Step. (b) LSE distribution
among different steps. We used HunyuanVideo to generate a 720p 8s video and recorded the distribution of LSE at each layer. It is easy to
see that as the Step changes, the distribution of LSE remains almost unchanged. (c) Recall Distribution of Different Inputs. We used the best
sparse pattern obtained from prompt1-seed0 and applied it to different prompts and seeds. The recall decreases when the prompt or seed
changes, meaning different inputs do not share the same sparse pattern.
conducted the following experiment: we searched for the
optimal sparse pattern for a specific prompt with a fixed spar-
sityof 0.9. Subsequently, this pattern was directly applied
to other prompts. We selected various prompts and different
random seeds, and the results revealed that the sparse pat-
tern optimized for one input is not necessarily optimal for
other inputs. These observations reveal that the sparse pat-
terns of different prompts differ significantly, making offline
searches likely to fail.
Another conventional approach is online approximate
search [30]. However, due to the hierarchical structure anddispersed attention distribution described in Observation 1,
this method fails to accurately capture the correct sparse in-
dices, resulting in poor performance within DiT (as evaluated
in Section 5).
Therefore, DiT requires a precise online search ; how-
ever, its prohibitive computational cost makes it impractical,
which is why no prior methods have adopted it.
Solution 2: DiTs’ sparse pattern and LSE keep invari-
ant in diffusion steps, caching those invariables making
a fast precise online search feasible. DiTs perform an
iterative multi-step denoising process, and we observe animportant invariance: for a given layer and head, although
the specific values of the attention weights change dynami-
cally across denoising steps, the underlying sparse pattern
remains consistent throughout the process. Furthermore, we
statistically analyze the distribution of the LSE data calcu-
lated in FlashAttention at different steps within the same
layer. The results in Figure 6b show that the distribution of
LSE remains stable across denoising steps.
Those similarities between consecutive steps provide an
opportunity to explore the reuse of sparse patterns and LSE
to accelerate online search, as detailed in Section 4.
4. Methodology
Motivated by those observations, we propose AdaSpa , a
sparse attention mechanism featuring Dynamic Pattern and
Online Precise Search, to accelerate long video generation
with DiTs.
4.1. Problem Formulation
Section 3 demonstrates that the attention weights of DiTs
cannot be well represented using patterns such as colordiag
due to the discontinuities caused by hierarchical structures,
while the block pattern shows advantages. Thus, to facilitate
the online search of dynamic sparse masks, we formulate the
problem of how to find the optimal block sparse indices.
Definition of Blockified Sparse Attention. Block Sparse
Attention employs a block-wise attention method similar
to FlashAttention, with the distinction that Block Sparse
Attention ignores the computation of certain blocks based on
its sparse indices, thereby achieving a speedup. Concretely,
partition the length dimension LintoL/B chunks, where
Bis the block size of sparse attention, and define a block-
level sparse pattern MS∈ {0,1}L
B×L
B, where Sis the
set of sparse indices of M. By expanding MStogMS∈
{0,1}L×Land applying a large negative bias −c(1−gMS),
we can exclude the discarded blocks from the safe Softmax
computation:
Wattn(gMS) = Softmax safe
QK⊤
√
D−c 
1−gMS
,(12)
where cis sufficiently large.
Optimal sparse indices. The goal of block sparse atten-
tion is to retain as much of the attention weights as possible,
thus to achieve the best Recall .
We predefine Wsum_attn as the sum of attention weights
within each block of Wattn:
Wsum_attn =B−1X
i=0B−1X
j=0Wattn[B·p+i, B·q+j](13)
where p, q∈ {0,1, . . . ,L
B−1},Wsum_attn ∈RL
B×L
BFormally, at a given sparsity , the precise sparse indices
of block sparse attention can be expressed as:
S∗= arg min
SWattn−Wattn(gMS)
= arg max
SWattn(gMS)
= arg max
SWsum _attn(MS)
= arg max
k∈{1,...,(1−sparsity )(L
B)2}Wsum _attn[k](14)
This indicates that we can obtain the optimal sparse indices
by calculating Wsum _attnand utilizing topk. We only need
to calculate the block with index in S∗, while omitting other
blocks. Thus, under the given sparsity, the complexity can be
reduced from O(L2d)toO((1−sparsity )L2d), providing
a significant speedup.
4.2. Design of Adaptive Sparse Attention
We illustrate the overview of AdaSpa in Figure 7. As previ-
ously mentioned, in order to perform a precise search, it is
necessary to obtain the complete Wattn, which has a size of
O(L2). In the context of long video generation, this results
in significant time and memory overhead. Moreover, since
the mask for each attention operation must be determined in
real-time, this time consumption is not affordable [30].
To address this issue, we exploit the property of DiT’s
sparse pattern, which exhibits similarity in denoising steps,
and construct AdaSpa with a two-phase Fused LSE-Cached
Online Search and Head-adaptive Hierarchical Block Sparse
Attention.
Fused LSE-Cached Online Search. The first phase of
Fused LSE-Cached Online Search is a Fused online Search,
which is a two-pass search: the first pass computes the origi-
nal FlashAttention outputs and stores each row’s LSE, while
the second pass uses the previously generated LSE to com-
puteWsum _attnin a block-wise manner fused with FlashAt-
tention.
The second phase is an LSE-Cached online Search, which
only contains one pass. Due to the similarity of LSE in steps,
we directly use the LSE obtained from the Fused online
Search to calculate Wsum _attn, thereby saving one pass
of search time and further reducing the search time by half.
Algorithm 1 and 2 demonstrate the pseudocode of our precise
online search.
Head-adaptive Hierarchical Block Sparse Attention.
Figure 6a shows that not all attention heads share the same
sparsity characteristics. A single uniform sparsity across all
heads is often suboptimal because certain heads may func-
tion well with fewer retained blocks, while others require
more. However, if each head employs a totally distinct spar-
sity level, it will cause huge search time and lead to severe
kernel load imbalance that significant wastage of computa-
tional resources.Full Attention Fused Online 
SearchHead-adaptive
Hierarchical Block Sparse AttentionLSE-Cached 
Online SearchHead-adaptive
Hierarchical Block Sparse AttentionBlock mask Cached LSE
Warmup Block mask
Sparse AttentionStep
Layer...
...
......
...
...
...........................
...
...
.........Figure 7. Overview of AdaSpa. We define a warm-up step Tw={1,2, ..., t w}, and select k steps: Ts={t1
s, t2
s, ..., tk
s}to perform a precise
online search, with t1
key=tw. Initially, during steps 1totw−1, we use full attention. At step tw, we apply Fused Online Search to do full
attention and thereby compute block mask, which is then passed to the subsequent steps t1
key+ 1, t1
key+ 2, . . . , t2
key−1for Head-adaptive
Hierarchical Block Sparse Attention. Subsequently, for each ti
key, where i > 1, we leverage the Cached LSE from the previous t1
key
search to perform the LSE-Cached Online Search, thereby obtaining a new mask. This new mask is then passed to the subsequent steps
ti
key, ti
key+ 1, ti
key+ 2, . . . , ti+1
key−1for Head-adaptive Hierarchical Block Sparse Attention computation.
Algorithm 1 Fused Online Search
1:Input: Q, K, V ,
2:Output: LSE, Out, W sum_attn
3:Initialize: lse← −∞ ,row_max←1,acc←0
4:Load query block in parallel: q←Q[current block ]
5:// First Pass: Compute FlashAttention outputs and store
LSE.
6:for each key block k∈K, value block v∈Vdo
7: qk←Dot(q, k)
8: row_max←update (row_max, qk )
9: p←online_softmax (row_max, qk )
10: lse+ = Sum(p,−1)
11: acc←Dot(p, v, acc )
12:end for
13:LSE←Log(lse) +row_max
14:Out←acc
15:// Second Pass: Use cached LSE to compute Wsum _attn
and reduce time.
16:for each key block k∈Kdo
17: qk←Dot(q, k)
18: p←Log(qk−LSE )
19: p_sum =Sum(p)
20: Store p_sum to coresponding position in Wsum_attn
21:end for
22:Return: LSE ,Out,Wsum_attn
To utilize the head adaptive feature while mitigate
wastage of computational resources, we employ a hierar-
chical search and calculation strategy. Specifically, we start
by fixing a given sparsity and computing the Recall for eachAlgorithm 2 LSE-Cached Online Search
1:Input: Q, K, LSE ,
2:Output: Wsum_attn
3:Load query block in parallel: q←Q[current block ]
4:// Only one pass; use cached LSE to compute
Wsum _attn.
5:for each key block k∈Kdo
6: qk←Dot(q, k)
7: p←Log(qk−LSE )
8: p_sum =Sum(p)
9: Store p_sum to coresponding position in Wsum_attn
10:end for
11:Return: Wsum_attn
head. We then sort all heads according to their respective Re-
call. Letndenote the number of heads whose Recall exceeds
0.8, which is a well-known fine Recall to a sparse attention.
Next, we increase the sparsity of the nheads with the high-
estRecall to1+sparsity
2, and we decrease the sparsity of the n
heads with the lowest Recall to3×sparsity −1
2. This hierarchi-
cal head-adaptive procedure effectively reduces redundancy
among heads exhibiting higher Recall while improving the
precision of those with lower Recall . Consequently, we
achieve elevated accuracy without altering the average spar-
sity, thus realizing a head-adaptive mechanism.
4.3. Implementation
AdaSpa is implemented with over 2,000 lines of Python and
1000 lines of Triton [ 56] codes. It is provided as a plug-
and-play interface, as shown in Figure 8. Users can enable1  from adaspa import adaspa_attention_handler
2  # Suppose q, k, v each has shape: [batch_size, head_num, 
seq_len, head_dim]
3  # One can simply use AdaSpa by replacing origin attention 
with sparse attention from Adaspa:
4  q, k, v = get_qkv(hidden_states, qkv_weight)
    - out = original_attention(query=q, key=k, value=v)
5  + out = adaspa_attention_handler(query=q, key=k, value=v)
6  return outFigure 8. Minimal usage of AdaSpa .
AdaSpa with only a one-line change. We use sparsity=0.8,
block_size=64, Ts={10,30}as the default configuration.
We implement our Head-adaptive Hierarchical Block Sparse
Attention based on Block-Sparse-Attention [ 20]. Unless
otherwise noted, all other attention mechanisms employ
FlashAttention 2 [12].
In addition, we employ two optimization techniques for
better efficiency. (1) Text Sink. We manually select all the
indices of video-text, text-video, and text-text parts, which
can enhance video modality’s perception to text modality,
thereby achieving better results. (2) Row Wise. We find that
ensuring each query attends to roughly the same number of
keys can improve continuity in generated videos. Otherwise,
certain regions deemed “unimportant” might never be at-
tended to, producing artifacts. Hence, we enforce a per-row
uniform selection in our block sparse pattern .
5. Experiments
Models. We experiment with two state-of-the-art open-
source models, namely HunyuanVideo (13B) [ 33] and
CogVideoX1.5-5B [65]. We generate 720p, 8-second
videos for HunyuanVideo, 720p and 10-second videos for
CogVideoX1.5-5B, with 50 steps for both of these models.
Baselines. We compare AdaSpa with Sparse VideoGen [62]
(static pattern) and Minference [30] (dynamic pattern). In
addition, we also consider two variants of AdaSpa to assess
the effectiveness of the proposed methods: (1) AdaSpa (w/o
head adaptive) , with uses the same sparsity for all heads,
and (2) AdaSpa (w/o lse cache) , which does not leverage
the LSE-Cached method. For all methods, the first 10 steps
generate with full attention for warmup.
Datasets. For all the experiments, we use the default dataset
from VBench [ 29] for testing. Specially, for CogVideoX1.5-
5B, we use VBench dataset after applying prompt optimiza-
tion, following the guidelines provided by CogVideoX [ 65].
Metrics. To evaluate the performance of our video genera-tion model, we employ several widely recognized metrics
that assess both the quality and perceptual similarity of the
generated videos. Following previous works [ 32,36,71],
we utilize Peak Signal-to-Noise Ratio [ 11] (PSNR), Learned
Perceptual Image Patch Similarity [ 70] (LPIPS), and Struc-
tural Similarity Index Measure [ 60] (SSIM) to evaluate the
similarity of generated videos. As for video quality, we
introduce the VBench Score [ 29], which provides a more
comprehensive evaluation by considering both pixel-level
accuracy and perceptual consistency across frames. For
efficiency, we report latency and speedup, where both are
measured using a single A100 GPU-80GB.
5.1. Main Results
In Table 1, we present a comprehensive evaluation of
AdaSpa, comparing it with various baseline methods across
both quality and efficiency metrics.
We observe that AdaSpa consistently achieves the best
performance in both quality and efficiency across all experi-
ments. On HunyuanVideo, AdaSpa ranks first in most met-
rics and achieves the highest speedup of 1.78 ×. In contrast,
both Sparse VideoGen and MInference show suboptimal
results, with speedups of 1.58 ×and 1.27 ×, respectively. On
CogVideoX1.5-5B, AdaSpa delivers the best performance
across all quality metrics and achieves a speedup of 1.66 ×,
the highest among the evaluated methods.
MInference, due to its reliance on online approximate
search, struggles to accurately capture the precise sparse
indices, leading to the lowest accuracy. Moreover, because
of the dispersed characteristic of sparse patterns in DiT, the
patterns obtained through approximate search exhibit a lower
true sparsity, resulting in slower performance with speedups
of only 1.27 ×and 1.39 ×. Sparse VideoGen, which lever-
ages a static pattern that is specifically designed for DiT,
performs relatively well, as it can capture some optimal
sparse patterns for specific heads. However, due to its inabil-
ity to dynamically capture accurate sparse patterns for all
heads, it fails to outperform AdaSpa in all accuracy metrics.
For the two variants of AdaSpa, AdaSpa (w/o head
adaptive) shows worse performance in terms of quality
metrics, providing strong evidence of the effectiveness of
head-adaptive sparsity. Additionally, AdaSpa (w/o LSE
cache) generally performs worse or on par with AdaSpa
across most metrics. Due to slower search speeds, it only
achieves speedups of 1.71 ×and 1.60 ×on Hunyuan and
CogVideoX1.5-5B, respectively, both lower than AdaSpa’s
performance. This further corroborates the effectiveness
of LSE-Cached Search and our Head-adaptive Hierarchical
method in enhancing speedup and quality.
5.2. Ablation Study
Quality-Sparsity trade-off. In Figure 9, we compare the
quality metrics of AdaSpa with MInference and SparseMethodQuality Metrics Efficiency Metrics
VBench (%) ↑PSNR ↑SSIM ↑LPIPS ↓Latency (s) Speedup
HunyuanVideo 80.10 - - - 3213.76 1.00×
+ MInference 79.17 22.53 0.7435 0.3550 2532.80 1.27×
+ Sparse VideoGen 79.39 27.61 0.8683 0.1703 2035.59 1.58×
+ AdaSpa (w/o head adaptive) 79.64 28.51 0.8825 0.1574 1823.34 1.76×
+ AdaSpa (w/o lse cache) 80.16 28.97 0.8898 0.1481 1877.13 1.71×
+ AdaSpa ( ours ) 80.13 29.07 0.8905 0.1478 1810.23 1.78×
CogVideoX1.5 81.16 - - - 3135.24 1.00×
+ MInference 65.30 10.31 0.3113 0.6820 2258.35 1.39×
+ Sparse VideoGen 79.40 18.98 0.6465 0.3632 2061.42 1.52×
+ AdaSpa (w/o head adaptive) 81.54 22.99 0.8133 0.2203 1915.88 1.64×
+ AdaSpa (w/o lse cache) 81.73 23.14 0.8255 0.2091 1961.71 1.60×
+ AdaSpa ( ours ) 81.90 23.25 0.8267 0.2067 1888.14 1.66×
Table 1. Quantitative evaluation of quality and latency for AdaSpa and other methods.
0.7 0.8 0.9
Sparsity0.7000.7250.7500.7750.800VBench
0.7 0.8 0.9
Sparsity15202530PSNR
0.7 0.8 0.9
Sparsity0.60.70.80.9SSIM
0.7 0.8 0.9
Sparsity0.10.20.30.40.50.6LPIPS
ours Sparse VideoGen MInference
Figure 9. Quality-Sparsity trade off.
VideoGen at different sparsity levels. As observed in the
VBench metric, which measures video quality, AdaSpa con-
sistently maintains the highest video quality across all spar-
sity levels, with no significant degradation as sparsity in-
creases. In contrast, both Sparse VideoGen and MInfer-
ence experience a considerable drop in quality as sparsity
increases. This demonstrates that AdaSpa is capable of
preserving critical information as much as possible under
limited sparsity, thereby ensuring the reliability of video
quality.
Similarly, in the PSNR, SSIM, and LPIPS metrics, which
measure the similarity between the videos generated with
and without sparse attention, a consistent trend is observed:
as sparsity increases, the similarity for all video methods
declines. However, AdaSpa maintains significantly higher
similarity compared to other methods, with a gradual linear
decrease as sparsity increases. This is in stark contrast to the
abrupt decline observed in MInference.
Warmup. As mentioned in many previous works [ 32,40,
62], warmup can significantly enhance the similarity and sta-
bility of video generation. Therefore, we compared the video
quality and similarity of AdaSpa, MInference, and Sparse
VideoGen under different warmup setups in Figure 10. It
0 2 5 10
Warmup Steps0.00.20.40.60.81.0VBench
0 2 5 10
Warmup Steps0102030PSNRours Sparse VideoGen MInferenceFigure 10. The impact of different warmup steps for AdaSpa,
Sparse VideoGen, and MInference.
can be seen that as warmup decreases, the similarity of all
methods also decreases, which is consistent with the con-
clusions of previous works. However, we find that as the
warmup period increases, AdaSpa still achieves the best per-
formance across all setups. Additionally, the video quality
for all methods does not show significant improvement with
the increase in warmup, remaining almost unchanged. This
suggests that warmup has minimal impact on the quality of
video generation itself and primarily affects the similarity
between the generated video and the original video.Table 2. The impact of different Search Strategies for AdaSpa.
Search Strategy ( Ts)PSNR ↑ SSIM↑ LPIPS ↓
{10} 28.9629 0.8879 0.1509
{10, 30} 29.0749 0.8905 0.1478
{10, 20, 30} 28.9343 0.8894 0.1500
{10, 20, 30, 40} 28.9313 0.8898 0.1494
3.05x3.65x
2.79x4.01x
2.01x
Figure 11. Scaling test of AdaSpa.
Search Strategy. To verify the impact of our search strat-
egy on video generation, we evaluate AdaSpa on video qual-
ity and similarity with different search strategies, as shown
in Table 2. The results indicate that increasing the number of
searches might be beneficial for improving accuracy, yet to
a limited extent. When the number of searches reaches a cer-
tain threshold, further increasing the number of searches may
even lower the video generation quality. This sufficiently
demonstrates that the patterns between steps have a strong
similarity, and as the number of searches increases, the video
quality may actually decline. This suggests that the impact
of sparse attention has a certain transmissibility and may
affect subsequent steps, which will be further explored in
future work.
5.3. Scaling Study
To further assess the scalability of our method, we tested the
generation time for videos of different lengths under the con-
figuration of sparsity =0.9, block_size=64, and Ts={0,30}.
As shown in Figure 11, as the length of the generated video
increases, AdaSpa’s speedup continues to improve, ulti-
mately reaching a speedup of 4.01 ×when the video length
is 24 seconds. This demonstrates the excellent scalability of
our method.
6. Conclusion
In this work, we comprehensively analyze the sparse
characteristics in the attention mechanisms when generating
videos with DiTs. Based on the observations and analyses,
we develop AdaSpa, a brand new sparse attention approach
featuring dynamic pattern and online precise search, toaccelerate long video generation. Empirical results show
that AdaSpa achieves a 1.78 ×of efficiency improvement
while maintaining high quality in the generated videos.
References
[1]Shantanu Acharya, Fei Jia, and Boris Ginsburg. Star attention:
Efficient llm inference over long sequences. arXiv preprint
arXiv:2411.17116 , 2024. 2
[2]Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.
Neural machine translation by jointly learning to align and
translate, 2016. 4
[3]Iz Beltagy, Matthew E Peters, and Arman Cohan. Long-
former: The long-document transformer. arXiv preprint
arXiv:2004.05150 , 2020. 2
[4]Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei
Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric
Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.
Video generation models as world simulators. 2024. 2
[5]Chenjie Cao, Yunuo Cai, Qiaole Dong, Yikai Wang, and
Yanwei Fu. Leftrefill: Filling right canvas based on left
reference through generalized text-to-image diffusion model.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 7705–7715, 2024. 1
[6]Cheng Chen, Xiaofeng Yang, Fan Yang, Chengzeng Feng,
Zhoujie Fu, Chuan-Sheng Foo, Guosheng Lin, and Fayao Liu.
Sculpt3d: Multi-view consistent text-to-3d generation with
sparse 3d prior. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 10228–
10237, 2024. 2
[7]Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul,
Ping Luo, Hang Zhao, and Zhenguo Li. Pixart- δ: Fast and
controllable image generation with latent consistency models,
2024. 1
[8]Jingyuan Chen, Fuchen Long, Jie An, Zhaofan Qiu, Ting Yao,
Jiebo Luo, and Tao Mei. Ouroboros-diffusion: Exploring con-
sistent content generation in tuning-free long video diffusion.
arXiv preprint arXiv:2501.09019 , 2025. 2
[9]Yang Chen, Yingwei Pan, Haibo Yang, Ting Yao, and Tao Mei.
Vp3d: Unleashing 2d visual prompt for text-to-3d generation.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 4896–4905, 2024. 2
[10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.
Generating long sequences with sparse transformers. arXiv
preprint arXiv:1904.10509 , 2019. 2, 4
[11] OpenCV Contributors. Opencv: Open source computer vision
library, 2025. Accessed: 2025-02-26. 9
[12] Tri Dao. Flashattention-2: Faster attention with bet-
ter parallelism and work partitioning. arXiv preprint
arXiv:2307.08691 , 2023. 4, 9
[13] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christo-
pher Ré. Flashattention: Fast and memory-efficient exact
attention with io-awareness. Advances in Neural Information
Processing Systems , 35:16344–16359, 2022. 4
[14] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat gans on image synthesis. Advances in neural information
processing systems , 34:8780–8794, 2021. 1[15] Hangliang Ding, Dacheng Li, Runlong Su, Peiyuan Zhang,
Zhijie Deng, Ion Stoica, and Hao Zhang. Efficient-vdit: Effi-
cient video diffusion transformers with attention tile. arXiv
preprint arXiv:2502.06155 , 2025. 2
[16] Jiarui Fang, Jinzhe Pan, Jiannan Wang, Aoyu Li, and Xibo
Sun. Pipefusion: Patch-level pipeline parallelism for diffusion
transformers inference. arXiv preprint arXiv:2405.14430 ,
2024. 2
[17] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, and Tat-
Seng Chua. Dysen-vdm: Empowering dynamics-aware text-
to-video diffusion with llms. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 7641–7653, 2024. 1
[18] Tianyu Fu, Haofeng Huang, Xuefei Ning, Genghan Zhang,
Boju Chen, Tianqi Wu, Hongyi Wang, Zixiao Huang, Shiyao
Li, Shengen Yan, et al. Moa: Mixture of sparse attention for
automatic large language model compression. arXiv preprint
arXiv:2406.14909 , 2024. 2
[19] Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Hayden
Kwok-Hay So, Ting Cao, Fan Yang, and Mao Yang. Seerat-
tention: Learning intrinsic sparse attention in your llms. arXiv
preprint arXiv:2410.13276 , 2024. 2
[20] Junxian Guo, Haotian Tang, Shang Yang, Zhekai Zhang, Zhi-
jian Liu, and Song Han. Block Sparse Attention. https:
//github.com/mit- han- lab/Block- Sparse-
Attention , 2024. 9
[21] Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji,
and Sinong Wang. Lm-infinite: Simple on-the-fly length
generalization for large language models. arXiv preprint
arXiv:2308.16137 , 2023. 2
[22] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and
Qifeng Chen. Latent video diffusion models for high-fidelity
long video generation. arXiv preprint arXiv:2211.13221 ,
2022. 2
[23] Jonathan Ho and Tim Salimans. Classifier-free diffusion
guidance. arXiv preprint arXiv:2207.12598 , 2022. 1
[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models. Advances in neural information
processing systems , 33:6840–6851, 2020. 1
[25] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan,
Mohammad Norouzi, and David J Fleet. Video diffusion
models. Advances in Neural Information Processing Systems ,
35:8633–8646, 2022. 2
[26] Lukas Höllein, Aljaž Boži ˇc, Norman Müller, David Novotny,
Hung-Yu Tseng, Christian Richardt, Michael Zollhöfer, and
Matthias Nießner. Viewdiff: 3d-consistent image generation
with text-to-image models. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition , pages
5043–5052, 2024. 2
[27] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie
Tang. Cogvideo: Large-scale pretraining for text-to-video gen-
eration via transformers. arXiv preprint arXiv:2205.15868 ,
2022. 2, 3
[28] Tianyu Huang, Yihan Zeng, Zhilu Zhang, Wan Xu, Hang
Xu, Songcen Xu, Rynson WH Lau, and Wangmeng Zuo.
Dreamcontrol: Control-based text-to-3d generation with 3d
self-prior. In Proceedings of the IEEE/CVF conference oncomputer vision and pattern recognition , pages 5364–5373,
2024. 2
[29] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang
Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang
Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin
Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Com-
prehensive benchmark suite for video generative models. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , 2024. 9
[30] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui
Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi,
Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accel-
erating pre-filling for long-context llms via dynamic sparse
attention. arXiv preprint arXiv:2407.02490 , 2024. 2, 5, 6, 7,
9
[31] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si,
Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Video-
booth: Diffusion-based video generation with image prompts.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 6689–6700, 2024. 2
[32] Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu,
Menglin Jia, Chenyang Zhang, Michael S Ryoo, and Tian Xie.
Adaptive caching for faster video generation with diffusion
transformers. arXiv preprint arXiv:2411.02397 , 2024. 9, 10
[33] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai,
Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang,
et al. Hunyuanvideo: A systematic framework for large video
generative models. arXiv preprint arXiv:2412.03603 , 2024.
1, 2, 3, 9
[34] Black Forest Labs. Flux. https://github.com/
black-forest-labs/flux , 2024. 1
[35] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai,
Junjie Bai, Yangqing Jia, Kai Li, and Song Han. Distrifusion:
Distributed parallel inference for high-resolution diffusion
models. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages 7183–7193,
2024. 1
[36] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu
Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu,
and Song Han. Svdqunat: Absorbing outliers by low-
rank components for 4-bit diffusion models. arXiv preprint
arXiv:2411.05007 , 2024. 9
[37] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu,
Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Li-
uhan Chen, et al. Open-sora plan: Open-source large video
generation model. arXiv preprint arXiv:2412.00131 , 2024. 2,
3
[38] Liu Liu, Zheng Qu, Zhaodong Chen, Yufei Ding, and Yuan
Xie. Transformer acceleration with dynamic sparse attention.
arXiv preprint arXiv:2110.11299 , 2021. 2, 5
[39] Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang,
Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi
Wang, et al. Moba: Mixture of block attention for long-
context llms. arXiv preprint arXiv:2502.13189 , 2025. 2
[40] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache:
Accelerating diffusion models for free. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 15762–15772, 2024. 10[41] Maxim Milakov and Natalia Gimelshein. Online normalizer
calculation for softmax, 2018. 4
[42] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision , pages 4195–4205,
2023. 2, 3
[43] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall.
Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint
arXiv:2209.14988 , 2022. 2
[44] Yifan Pu, Zhuofan Xia, Jiayi Guo, Dongchen Han, Qixiu Li,
Duo Li, Yuhui Yuan, Ji Li, Yizeng Han, Shiji Song, et al.
Efficient diffusion transformer with step-wise dynamic atten-
tion mediators. In European Conference on Computer Vision ,
pages 424–441. Springer, 2025. 2
[45] Jiezhong Qiu, Hao Ma, Omer Levy, Scott Wen-tau Yih,
Sinong Wang, and Jie Tang. Blockwise self-attention for long
document understanding. arXiv preprint arXiv:1911.02972 ,
2019. 2
[46] Leigang Qu, Wenjie Wang, Yongqi Li, Hanwang Zhang,
Liqiang Nie, and Tat-Seng Chua. Discriminative probing
and tuning for text-to-image generation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 7434–7444, 2024. 1
[47] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Björn Ommer. High-resolution image
synthesis with latent diffusion models. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 10684–10695, 2022. 1
[48] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas
Blattmann, Patrick Esser, and Robin Rombach. Fast high-
resolution image synthesis with latent adversarial diffusion
distillation. In SIGGRAPH Asia 2024 Conference Papers ,
pages 1–11, 2024. 1
[49] Idan Schwartz, Vésteinn Snæbjarnarson, Hila Chefer, Serge
Belongie, Lior Wolf, and Sagie Benaim. Discriminative class
tokens for text-to-image diffusion models. In Proceedings of
the IEEE/CVF International Conference on Computer Vision ,
pages 22725–22735, 2023. 1
[50] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and ac-
curate attention with asynchrony and low-precision. Advances
in Neural Information Processing Systems , 37:68658–68685,
2025. 4
[51] Takahiro Shirakawa and Seiichi Uchida. Noisecollage: A
layout-aware text-to-image diffusion model based on noise
cropping and merging. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 8921–8930, 2024. 1
[52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising
diffusion implicit models. arXiv preprint arXiv:2010.02502 ,
2020. 1
[53] Xin Tan, Yuetao Chen, Yimin Jiang, Xing Chen, Kun Yan,
Nan Duan, Yibo Zhu, Daxin Jiang, and Hong Xu. Dsv:
Exploiting dynamic sparsity to accelerate large-scale video
dit training. arXiv preprint arXiv:2502.07590 , 2025. 2
[54] Zhenxiong Tan, Xingyi Yang, Songhua Liu, and Xinchao
Wang. Video-infinity: Distributed long video generation.
arXiv preprint arXiv:2406.16260 , 2024. 2[55] Genmo Team. Mochi 1. https://github.com/
genmoai/models , 2024. 2
[56] Philippe Tillet, H. T. Kung, and David Cox. Triton: an in-
termediate language and compiler for tiled neural network
computations. In Proceedings of the 3rd ACM SIGPLAN
International Workshop on Machine Learning and Program-
ming Languages , page 10–19, New York, NY , USA, 2019.
Association for Computing Machinery. 8
[57] Marcos Treviso, António Góis, Patrick Fernandes, Erick Fon-
seca, and André F. T. Martins. Predicting attention sparsity in
transformers, 2022. 4
[58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. Advances in neural
information processing systems , 30, 2017. 2, 4
[59] Jing Wang, Ao Ma, Jiasong Feng, Dawei Leng, Yuhui Yin,
and Xiaodan Liang. Qihoo-t2x: An efficient proxy-tokenized
diffusion transformer for text-to-any-task. arXiv preprint
arXiv:2409.04005 , 2024. 2
[60] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing , 13(4):
600–612, 2004. 9
[61] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian
Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu
Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning
of image diffusion models for text-to-video generation. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 7623–7633, 2023. 2
[62] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu,
Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang,
Dacheng Li, et al. Sparse videogen: Accelerating video
diffusion transformers with spatial-temporal sparsity. arXiv
preprint arXiv:2502.01776 , 2025. 2, 9, 10
[63] G Xiao, Y Tian, B Chen, S Han, and M Lewis. Efficient
streaming language models with attention sinks, 2023. URL
http://arxiv. org/abs/2309 , 17453, 2023. 2
[64] Shuchen Xue, Zhaoqiang Liu, Fei Chen, Shifeng Zhang,
Tianyang Hu, Enze Xie, and Zhenguo Li. Accelerating diffu-
sion sampling with optimized time steps. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 8292–8301, 2024. 1
[65] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu
Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiao-
han Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video
diffusion models with an expert transformer. arXiv preprint
arXiv:2408.06072 , 2024. 1, 2, 3, 9
[66] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang
Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang,
Zhiping Xiao, et al. Native sparse attention: Hardware-
aligned and natively trainable sparse attention. arXiv preprint
arXiv:2502.11089 , 2025. 2
[67] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey,
Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham,
Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Trans-
formers for longer sequences. Advances in neural information
processing systems , 33:17283–17297, 2020. 2[68] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding
conditional control to text-to-image diffusion models. In
Proceedings of the IEEE/CVF international conference on
computer vision , pages 3836–3847, 2023. 1
[69] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang
Ding, Ion Stoica, Zhenghong Liu, and Hao Zhang. Fast
video generation with sliding tile attention. arXiv preprint
arXiv:2502.04507 , 2025. 2
[70] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric, 2018. 9
[71] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-
time video generation with pyramid attention broadcast. arXiv
preprint arXiv:2408.12588 , 2024. 9
[72] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen,
Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang
You. Open-sora: Democratizing efficient video production
for all. arXiv preprint arXiv:2412.20404 , 2024. 2, 3