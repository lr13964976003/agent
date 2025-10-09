Compact Attention: Exploiting Structured
Spatio-Temporal Sparsity for Fast Video Generation
Qirui Li1†, Guangcong Zheng1†, Qi Zhao1, Jie Li1, Bin Dong2, Yiwu Yao2, Xi Li1
1College of Computer Science & Technology, Zhejiang University
2Huawei Technologies
{qirui.l, guangcongzheng}@zju.edu.cn
Abstract
The computational demands of self-attention mechanisms pose a critical challenge
for transformer-based video generation, particularly in synthesizing ultra-long
sequences. Current approaches, such as factorized attention and fixed sparse
patterns, fail to fully exploit the inherent spatio-temporal redundancies in video data.
Through systematic analysis of video diffusion transformers (DiT), we uncover
a key insight: Attention matrices exhibit structured, yet heterogeneous sparsity
patterns, where specialized heads dynamically attend to distinct spatiotemporal
regions (e.g., local pattern, cross-shaped pattern, or global pattern). Existing
sparse attention methods either impose rigid constraints or introduce significant
overhead, limiting their effectiveness. To address this, we propose Compact
Attention , a hardware-aware acceleration framework featuring three innovations:
1) Adaptive tiling strategies that approximate diverse spatial interaction patterns via
dynamic tile grouping, 2) Temporally varying windows that adjust sparsity levels
based on frame proximity, and 3) An automated configuration search algorithm
that optimizes sparse patterns while preserving critical attention pathways. Our
method achieves 1.6∼2.5×acceleration in attention computation on single-GPU
setups while maintaining comparable visual quality with full-attention baselines.
This work provides a principled approach to unlocking efficient long-form video
generation through structured sparsity exploitation. Project Page: https://
yo-ava.github.io/Compact-Attention.github.io/
1 Introduction
The rapid advancement of generative models has enabled high-quality video synthesis; however,
processing ultra-long sequences remains a critical bottleneck. For Transformer-based video generation
models, the quadratic complexity of self-attention mechanisms presents a fundamental challenge, as
modeling spatiotemporal dependencies requires handling extensive token sequences. For example,
in the Hunyuan-video architecture [19], generating a 128-frame 720p HD video entails processing
over 100K tokens, with attention computation consuming 68-72% of the total generation time.
This computational burden becomes prohibitive for long-form video generation, necessitating the
development of innovative acceleration strategies.
Recent studies [61, 47, 13, 46, 8, 60, 54] show that full attention matrices in video generation exhibit
significant sparsity, with complex distributions of attention weights and structured yet seemingly
irregular patterns (Fig. 1), indicating substantial untapped acceleration potential. The primary chal-
lenge lies in effectively leveraging these heterogeneous sparse patterns. Even if accurate predictions
can be made, the overhead associated with locating the sparse locations often offsets the potential
speed gains.
†These authors contributed equally to this work.
Preprint. Under review.arXiv:2508.12969v1  [cs.CV]  18 Aug 2025Temporal 
Dimension3) Compact Masks costs too much to obtain and apply.
2) Per-query -token attention distributions 
enables structured interpretation.
Time -Variant Time -Invariant
Spatial 
DimensionLocal Pattern Cross -shaped Pattern Global Pattern1) Sparsity within slash -separated domains is 
hard to leverage.
0 1Query token position 0 1Mask
Per-query -token attention distribution on each frame.
Attention 
Map
MaskAttention 
Map
MaskAttention 
MapAttention Map
Figure 1: Sparsity between slashes are hard to exploit for acceleration. Periodic, hierarchical attention
patterns are shown in complicated attention maps when a single query token is arranged properly.
Our analysis of video diffusion transformers (DiT) uncovers a critical phenomenon: the interaction
between 3D spatiotemporal token sequences may induce periodic, hierarchical attention patterns. As
shown in Fig. 1, specialized attention heads emerge with distinct functional roles: focusing on local
spatial regions, forming cross-shaped spatial interactions, and exhibiting a global or input-related
focus. Additionally, certain heads exhibit relationships with frames at specific relevant distances.
Some attention heads demonstrate temporal locality by suppressing distant frames, while others do
the opposite. These structured sparsity patterns present opportunities for efficient approximation,
however, existing methods fail to fully exploit these patterns.
Previous approaches, such as Minference[15], generalize sparse attention from language models by
using fixed patterns (e.g., diagonals, blocks), but when applied to video generation, they overlook
the unique 3D redundancies inherent in video data. Sparge Attention[60] improves spatial grouping
through Hilbert-order flattening and leverages locality, but it introduces additional overhead. SVG
[46] recognizes the unique periodic patterns in video data but does not account for the dynamic
sparsity exhibited by each attention head. The sliding window approach in STA[61] captures local
spatiotemporal correlations but limits attention to rigid cubic regions, thereby missing crucial cross-
frame interactions, sparsity related to relative temporal distance, and corresponding redundancies.
These limitations highlight the need for a video-specific sparse attention mechanism that can adap-
tively capture structured and dynamic spatiotemporal patterns. We summarize our key contributions
as follows:
•We reveal structured and hierarchical attention patterns in video diffusion transformers, uncovering
specialized spatiotemporal attention behaviors that motivate efficient sparse approximations.
•We propose Compact Attention , a training-free sparse attention framework that integrates an
offline configuration search strategy with an efficient attention computation mechanism, while
preserving the fidelity of generated videos.
•We validate our approach on the Wan2.1 and Hunyuan model, achieving up to 2.5×end-to-end
speedup with negligible degradation in generated video quality.
2 Related Works
Acceleration of diffusion models. Due to the high inference cost, accelerating diffusion models has
become a central research focus. Existing methods largely aim to reduce sampling steps and fall into
two main categories: improved sampling algorithms [39, 28, 25, 2, 26, 62] and distillation-based
approaches [32, 36, 55, 21, 51, 14, 56, 37, 17, 40]. Distillation methods compress multi-step diffusion
into a compact student model via teacher-student training, reducing inference steps. Beyond step
reduction, several works [30, 1, 45, 23, 31, 29, 16] explore cache mechanisms to eliminate redundant
2t
w hFlattened based on adjacent token groups
Flattened along dimension hierarchy [f, h, w]Query, Key, Value Sequence
58.8% attention achieves 95% recalltimestep
head59.9% attention achieves 95% recalltimeste p
headFigure 2: Heatmap of attention map and the k% values required to retain top- kfor 0.95 recall before
and after rearranging attention maps into 3D spatially adjacent groups.
computation. Notably, [31] proposes a learning-based caching (L2C) strategy, while [45] applies
block-level caching to reuse layer outputs across steps. DeepCache [30] further leverages temporal
redundancy by reusing high-level features and updating only low-level ones. In addition to these
approaches, attention-level optimization such as attention quantization and sparsification [64, 22, 58,
59]—offers complementary acceleration potential.
Sparse attention. Sparse attention reduces the quadratic complexity of self-attention by masking
computations to predefined sparse regions. In large language models, numerous studies [3, 57, 65,
52, 53, 50, 10, 4] have explored sparse attention designs. Some methods [49, 63, 48, 6, 33, 12, 9,
24, 41, 5] use fixed patterns targeting specific positions. For instance, LM Infinite [12], MoA [9],
and MAPSparse [24] adopt A-shaped patterns and Ltri-LLM [41] identifies triangular structures.
Recognizing the dynamic nature of attention, several works [34, 18, 44, 35, 42, 38, 20, 15, 11]
introduce input-adaptive sparse attention. MInference [15], for example, identifies three pattern
types—A-shape, Vertical-Slash, and Block-Sparse. In video generation, however, the inherent 3D
redundancy poses challenges for directly transferring LLM-based sparse attention methods. Recent
efforts [61, 47, 13, 46, 8, 60, 54] target video-specific sparsity. STA [61] adopts a sliding window for
local spatiotemporal attention but underutilizes the full 3D redundancy present in video data.
3 Stable Spatiotemporal Patterns Enable Offline Attention Mask
Precomputation
3.1 Tile-Based Sparsity for Efficient Blockwise Attention
Although attention exhibits significant sparsity, performing sparse predictions in a token-by-token
manner is impractical for real-world acceleration scenarios due to the substantial overhead introduced
by both prediction and execution processes. We observe that critical information within attention
maps tends to cluster in three-dimensional space. Considering low-level computational efficiency,
processing data in a block-wise manner simultaneously exploits the clustering characteristics of
sparsity, meets acceleration demands, and reduces memory consumption.
We conducted sparsity validation experiments on attention maps, analyzing various timesteps and
layers within T2V models Wan1.2 [43] and Hunyuan [19]. More results are shown in appendix D. By
comparing two flattening strategies—directly flattening the three-dimensional sequence (f, h, w )into
a one-dimensional sequence versus grouping and flattening based on spatially adjacent tiles in 3D
space[61] (as illustrated in Fig. 2)—We observe that the latter reduces the average number of blocks
required to retain the top- kvalues for 0.95 recall by 1.1% in the Wan2.1 model, increasing to 3.4%
in the Hunyuan model, while preserving a high overall sparsity rate and maintaining compatibility
with block-wise attention mechanisms.
3.2 Structured Spatiotemporal Patterns in Attention Maps
The attention maps derived from 1D sequences with 3D structural information (f, h, w) exhibit highly
complex morphological diversity. Existing sparse attention methods in video generation models have
identified clustering patterns such as slash-line, vertical-line, and block-shaped formations in these
3Figure 3: Three characteristic attention patterns observed in video transformers: local pattern (left),
cross-shaped pattern(middle) and global pattern(right) with the upper one showing temporal dynamics
while lower one being persistent along frames. Each Attention map is shown using query of the token
in the middle and keys from the other tokens.
attention maps, and have subsequently designed minimal pattern primitives (e.g., vertical stripes,
slash stripes, and blocks) to approximate these sparse motifs. However, a fine-grained analysis of
per-query-token attention distributions facilitates a structured interpretation of the intricate patterns
within the attention maps. By examining full attention maps at the query-specific token level, we
observe that diagonal patterns arise from systematic position-relative attention biases, which visually
manifest as distinct spatiotemporal modes. Through empirical analysis, we identify three dominant
spatial patterns and two temporal patterns that are commonly present in 3D full-attention video
generation models (as illustrated in Fig. 3).
Spatial Patterns:
•Local Patterns : Certain attention heads focus on compact neighborhoods around target positions,
forming spherical attention fields that are likely crucial for fine-grained detail synthesis.
•Cross-Shaped Patterns : Specialized attention heads exhibit directional sensitivity, creating contin-
uous attention corridors along the horizontal and vertical axes.
•Global Patterns : Some attention heads preserve full spatial connectivity irrespective of the relevant
distance. Additionally, input-dependent attention heads exhibit strong weight clustering around
salient objects, which are also observed as global patterns.
Temporal Patterns:
•Time-Variant Patterns : This pattern exhibits a strong correlation with temporal relative distance.
Some attention heads demonstrate progressive weight decay across frames, while others focus
more on frames at a specific distance, excluding local or nearby frames.
•Time-Invariant Patterns : These attention heads maintain frame-agnostic distributions, ensuring a
consistent focus across all timesteps regardless of the relative temporal distance.
3.3 Pattern Stability Enables Offline Acceleration
Our systematic analysis demonstrates that spatiotemporal attention patterns arise as inherent properties
of the model architecture, rather than being driven by input adaptations.
The attention patterns demonstrate significant stability across various layers and heads within a video
generation model. We define pattern classification using a recall-based threshold criterion: attention
mechanisms that cover more than 85% of the spatial extent are classified as global patterns, while
smaller, more concentrated regions correspond to local interaction modes.
•Local Patterns: focus around query positions (xt, yt)with axes-aligned constraints. ωandη
denote boundaries of patterns:
Rlocal=
(x, y)max|x−xt|
ω,|y−yt|
η
≤1
(1)
4LayerCalculate SizeCalculate size across different seed
Calculate size across different prompt(a) Attention pattern is stable
Pattern similarity within a certain range(94.4%)
Relative denoising step distanceSimilarity (b) Pattern is similar within a certain denoising step range
Figure 4: (a) A visualization of the layer-wise computation during denoising. For different prompts,
the computational demands are nearly identical. (b) Region similarity across denoising steps.
•Cross-shaped Pattern: Cross-shaped regions with complementary spatial constraints:
Rcross=(
(x, y)2_
k=1|x−xt|
ωk≤1∧|y−yt|
ηk≤1)
(2)
where (ω1−ω2)(η1−η2)<0enforces complementary axis dominance. ωkandηkdenote
boundaries of patterns.
Input/Seed Invariance. As demonstrated in Fig. 4a, sizes of pattern regions stay alike with average
similarity over 0.8 across varying text prompts and random initializations, measured by:
Sim(MA, MB) =∥MA⊙MB∥1
∥MA+MB−MA⊙MB∥1(3)
where MA, MBare binarized attention masks.
Temporal Robustness. As shown in Fig. 4b, attention configurations remain stable within a certain
range across denoising steps, enabling reliable offline pre-computation of attention masks optimized
per model-layer-head combination.
4 Compact Attention
4.1 Tile-based Deformable Sparse Pattern
Building upon the spatio-temporal patterns identified in Section 3, we propose an adaptive tile-based
strategy that effectively captures complex attention distributions while preserving hardware efficiency.
Our approach fundamentally reconsiders the interaction between sparse attention patterns and the
intrinsic structure of video data. Instead of relying on rigid, predefined attention windows, we allow
sparse configurations to dynamically adapt across both spatial and temporal dimensions.
The core innovation of our method is conducted based on a hierarchical grouping mechanism that
respects the dual nature of video data—temporal variation and spatial locality. By reorganizing tokens
into spacetime tiles—clusters of tokens that are adjacent in both spatial and temporal domains—we
construct computational units that align with the natural locality inherent in video content. This tile
abstraction serves as the foundational building block for constructing deformable attention patterns.
Frame-Group-wise Patterns : To capture temporal dynamics, we partition frames into distance-based
groups relative to the current frame being processed, with each group governed by its own sparse
configuration. Dual Attention Windows : Within each group, spatial attention masks are adaptively
composed from two complementary window shapes that approximate the observed attention patterns
(e.g., cross-shaped, local blocks). This design eliminates the need for explicit pattern classification
during inference.
5Shrink ModuleRecall ≥ τ
Compact Attention Kernel
𝑲𝒆𝒓𝒏𝒆𝒍 𝒊
𝑄𝑖 𝑉 𝐾
𝐾′ 𝑉′
𝐴𝑖Mask Configs
𝑂𝑖
𝑂Flag = 1
Off-line Auto -Search of Sparse Masks
Masks
⋯ ⋯
……
……
……Recall < τ
Mask
Configs∃Flag=1
Calculate
RecallSelect Mask 
( Minimum Cost )1 0
Recall < τ Flag∀Flag=0
1
1
0
1
Update Mask
Update FlagRecall ≥ τ 
Omitted 0
τRecall Threshold
λCost ThresholdMask 1
Shrink
Module
Shrink
Module
Shrink
Module
LegendFigure 5: Compact Attention: Pipeline
This deformable architecture achieves a three-fold synergy: (1) spatial adaptability through tile
combinations that emulate diverse attention modes, (2) temporal awareness via distance-stratified
configurations, and (3) hardware efficiency by preserving the computational regularity inherent in
tile-based processing.
4.2 Optimized Auto-Search of Sparse Masks
Our mask search framework addresses two fundamental challenges: (1) the prohibitive computational
overhead of online mask prediction, and (2) the stability of attention patterns across diverse inputs (as
discussed in Section 3). The key insight lies in decoupling pattern discovery from runtime execution
through an offline configuration pipeline that preserves spatiotemporal coherence. The whole pipeline
is shown in Fig. 5.
Guided by the spatial variation characteristics presented in Section 3, we formulate mask optimization
as a boundary contraction process along hierarchical dimensions. The process starts with full attention
coverage and iteratively tightens window boundaries across spatial dimensions, prioritizing regions
with lower recall contributions, as indicated by the recall loss per computational unit (cost). This
directional shrinkage operates independently across different frame groups.
The contraction process is governed by dual thresholds: a minimum recall threshold τ, which
preserves critical interactions, and a maximum cost threshold λ, which balances computational
reduction against accuracy loss. The mask shrinking process for a given frame group terminates
when either the recall drops below τor the cost of further shrinking ( ∆Recal/ ∆Cost ) exceeds λ,
ensuring an effective trade-off between quality and efficiency. To obtain the final configuration, we
merge configurations across prompts through union operations (see Section 6). This conservative
merging strategy guarantees that all potentially relevant attention regions are retained.
Capitalizing on the temporal stability of diffusion trajectories(see Fig. 4b), we implement mask reuse
across nconsecutive denoising steps. This configuration caching mechanism reduces the search
frequency by n×, while maintaining generated video quality.
6Table 1: Comparative Analysis of Sparse Attention Methods for Text-to-Video Models. Compact
Attention achieves faster high-quality video generation compared with methods with higher sparsity.
Model Method SparsityQuality Speed
SSIM ↑PSNR ↑MSE ↓Latency (s) Speedup
Wan2.1
(80K)Full Attention 0% - - - 1092.168 1.00x
Sparse VideoGen 32.08% 0.529 15.9564 1894.3672 1200.148 0.91x
SpargeAttn 32.27% 0.6102 20.5163 676.0723 1065.796 1.02x
Compact Attention(Ours) 33.99% 0.7754 23.7297 351.6015 663.824 1.65x
Compact Attention(Ours) 24.66% 0.8147 25.2664 254.1789 758.176 1.44x
Hunyuan
(127K)Full Attention 0% - - - 1370.658 1.00x
Sparse VideoGen 50.35% 0.7254 20.4297 822.8567 1117.767 1.23x
SpargeAttn 47.77% 0.7794 23.5889 369.3112 1148.628 1.19x
Compact Attention(Ours) 62.36% 0.9040 30.0822 105.1957 546.504 2.51x
Compact Attention(Ours) 52.90% 0.9452 34.5506 35.1307 750.201 1.83x
Table 2: Quantitative Comparison of Sparse Attention Methods in Wan2.1 and Hunyuan on VBench:
Visual Consistency, Aesthetic Quality and Text-Video Alignment.
Model Method SparsitySubject
ConsistencyBackground
ConsistencyAesthetic
QualityCLIPSIM CLIP-T
Wan2.1
(80K)Full Attention 0% 0.9681 0.9616 0.6486 0.2118 0.9985
Sparse VideoGen 32.08% 0.9547 0.9565 0.6380 0.2116 0.9987
SpargeAttn 32.27% 0.9357 0.9500 0.5320 0.2064 0.9982
Compact Attention (Ours) 33.99% 0.9659 0.9650 0.6480 0.2121 0.9985
Compact Attention (Ours) 24.66% 0.9674 0.9638 0.6459 0.2122 0.9986
Hunyuan
(127K)Full Attention 0% 0.9736 0.9735 0.6542 0.2181 0.9995
Sparse VideoGen 50.35% 0.9701 0.9722 0.6638 0.2014 0.9995
SpargeAttn 47.77% 0.9664 0.9731 0.5794 0.2112 0.9995
Compact Attention (Ours) 62.36% 0.9716 0.9693 0.6531 0.2184 0.9995
Compact Attention (Ours) 52.90% 0.9723 0.9735 0.6536 0.2184 0.9995
5 Experiments
5.1 Experimental Setup
Our evaluations are primarily conducted on the state-of-the-art video generation architecture
Wan2.1(14B) and Hunyuan on a single H800 GPU. We apply Compact Attention to generate outputs
consisting of 81 frames in Wan2.1 and 129 frames in Hunyuan at a resolution of 768 ×1280. To
evaluate the acceleration effect achieved through the exploitation of attention sparsity by Compact
Attention, we measured video quality using SSIM, PSNR, MSE and six quality metrics (Subject
Consistency, Background Consistency, Aesthetic Quality) in VBench, and CLIPSIM and CLIPTemp
(CLIP-T) [27] to measure the text-video alignment on Open-Sora benchmark. For computational
performance, we report both the attention sparsity rate and attention latency. Compact Attention is
implemented based on ThunderKittens, with reference to the STA framework.
Baselines: We evaluated Compact Attention against several state-of-the-art sparse attention ap-
proaches, including STA (spatio-temporal locality), Sparse VideoGen (static pattern), and Sparse
Attention (dynamic pattern). For performance comparison, we measured similarity relative to full
attention and quantified speedup using FlashAttention-2[7]. Additional implementation details can
be found in Appendix. C.
5.2 Acceleration Performance and Quality Preservation
Similarity Tab. 1 illustrates the sparsity efficiency of Compact Attention during end-to-end video
inference within the Wan2.1 and Hunyuan model framework. A comparative analysis with various
sparsity methods shows that Compact Attention achieves a superior acceleration ratio (2.51 ×speedup)
7Full Attention:  Sparsity: 0.00%
Sparsity: 58.37% PSNR = 21.63
Sparsity: 62.36% PSNR = 24.24
Full Attention ： Sparsity: 0.00%
Sparse VideoGen Sparsity: 50.35% PSNR = 18.33
Sparge Attention Sparsity: 47.7 7% PSNR = 18.89
PSNR = 22.95 Sparsity: 62.36% Compact Attention (Ours)Sliding Tile Attention (STA)
Compact Attention (Ours)
Details
Figure 6: Performance of different sparse attention methods on end-to-end video generation.
in hunyuan at a higher sparsity level (62.36%) while maintaining high-quality generation, as reflected
by a average PSNR of 30.0822 . This performance significantly surpasses baseline approaches: Sparse
VideoGen experiences substantial quality degradation (PSNR 20.4297 at 50.35% sparsity) due to
its uniform sparsity allocation across attention heads, whereas Sparse Attention, which employs
dynamic block sparsity based on cosine similarity thresholds, exhibits limited stability despite its
adaptive top- kselection strategy.
Quality Table 2 evaluates sparse attention methods using selected VBench metrics. In some
cases, text-video alignment is not meaningful to assess, as outputs from certain sparse attention
variants deviate significantly from the original content. While videos from SpargeAttn exhibit low
visual quality, both Sparse VideoGen andCompact Attention even outperform the full attention
baseline on some metrics.
Fig. 6 presents the visual performance of various sparse attention methods on Hunyuan. Due to strict
resolution constraints in STA, all methods are evaluated on 117-frame videos for consistency. While
STA significantly improves generation speed, it suffers from notable quality degradation in Wan2.1.
And in Hunyuan, our proposed Compact Attention achieves the miner impact on visual quality with
higher sparsity. Results are also shown in Fig. 11, which demonstrates the superior performance of
Compact Attention compared with Sliding Tile Attention (STA) in the Hunyuan video generation
framework.
8Table 3: Sparsity of different sparse pattern methods. Method of Sliding tile window uses a cubic
window as an attention mask. In our method, we propose frame-group-wise masks and dual window
to deal with time-variant heads and cross-shaped pattern seperately, achieving 9.8% more sparsity
when using params τ= 0.9andλ= 0.011in searching.
Pattern Cubic window + Frame-group-wise Patterns + Dual Windows
locality Patterns 0.726 0.758 0.766
cross Patterns 0.385 0.406 0.516
global Patterns 0.078 0.085 0.099
Time-Variant Patterns 0.441 0.472 0.567
Time-Invariant Patterns 0.306 0.317 0.385
Overall 0.361 0.370 0.459
Recall thresholdSparsity of Auto-search with Different Recall ThresholdSparsity
Figure 7: (a)Distribution of PSNR values across different parameter groups under diverse text prompts
and random seeds. (b)Sparsity trends under different recall thresholds for auto-searched attention
patterns on Wan2.1 and Hunyuan.
5.3 Ablation Studies
Sparse Pattern Effectiveness To validate the effectiveness of our proposed tile-based window with
temporal difference in exploiting more sparsity for 3D full attention, We conduct experiments on the
attention heads obtained during the inference phase of Wan2.1 and categorize them into different
groups to analyze the detailed improvements within each pattern. The results show that the Dual
Attention Windows increase the sparsity of cross patterns by approximately 10%, while the frame-
group-wise pattern, which enables variation across the temporal dimension, contributes an additional
improvement of around 3%.
6 Discussion: Robustness and Generalization
6.1 Stability Across Input Conditions
Fig. 7 presents the distribution of PSNR values obtained under varying input conditions, including
diverse text prompts and random initialization seeds, across different parameter configurations and
model variants. Notably, the configuration labeled Compact_Attention —corresponding to sparsity
parameters (τ= 0.9, λ= 0.011) —exhibits the highest median and mean PSNR values, along with a
relatively narrow interquartile range. This indicates that it consistently delivers high-quality outputs
with limited variance across stochastic or semantic input perturbations.
6.2 Sensitivity to Recall Threshold in Auto-search
As shown in Fig. 7, we analyze the sparsity patterns under varying recall thresholds (with fixed cost
thresholds: 0.011 for Wan and 0.04 for Hunyuan). The results reveal that: Hunyuan, as a smaller
model, consistently achieves higher attention sparsity than Wan. And with fixed cost threshold,
the sparsity converges to an upper bound determined by the cost constraint when recall threshold
9decreases. This suggests the need to carefully balance between acceleration (higher sparsity) and
generation quality (lower recall threshold) in parameter selection.
6.3 Sensitivity to Sparsification in Early Denoising Steps
Our analysis of the denoising process reveals that sparse attention is most sensitive during the early
stages, where high-noise inputs require structural initialization. Quantitative results show a 1.02dB
PSNR drop when full attention is applied only in the final 15 steps, compared to the first 15. This
highlights the importance of preserving full attention in the early timesteps to ensure quality, while
allowing sparsification in later stages for acceleration without compromising visual fidelity. As is
partly shown in Fig. 8, our empirical results suggest that maintaining full attention for the initial
15 denoising timesteps proves essential for preserving generation quality, whereas applying sparse
attention in the remaining steps achieves notable acceleration with minimal quality loss.
PSNR: 11.29
Latency: 640.97sPSNR: 13.44
Latency: 646.78sPSNR: 15.87
Latency: 655.72sPSNR: 19.17
Latency: 663.82 sPSNR: 22.49
Latency: 674.64 s
Full Attention
Latency: 1544.00sStart Apply ing Compact Attention from:
Denoising step 0 Denoising step 5 Denoising step 10 Denoising step 15 Denoising step 20
Figure 8: Effect of delaying sparse attention application: PSNR score and visual performance of
sparse attention versus full attention.
7 Conclusion
The high computational cost of video generation models necessitates efficient spatiotemporal attention
mechanisms that maintain generation quality. Through systematic analysis, we identify structured
sparsity patterns in video diffusion transformers, including (1) local, cross-shaped, and global patterns
in spatial dimensions, and (2) time-variant or time-invariant patterns in temporal dimensions. To
leverage these patterns, we propose Compact Attention, which introduces three key innovations:
(1) tile-based computation optimized for heterogeneous sparsity structures, (2) an automated mask
search algorithm with cross-prompt merging for adaptive pattern selection. Extensive experiments
on HD video generation show that our method achieves a 2.5 ×speedup compared to full attention
while preserving visual quality. This work presents a principled framework for co-designing sparse
operators based on empirical attention characteristics, with potential applications in multimodal
generation and real-time streaming systems.
References
[1] Shubham Agarwal et al. “Approximate caching for efficiently serving {Text-to-Image }diffu-
sion models”. In: 21st USENIX Symposium on Networked Systems Design and Implementation
(NSDI 24) . 2024, pp. 1173–1189.
[2] Fan Bao et al. “Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion
probabilistic models”. In: arXiv preprint arXiv:2201.06503 (2022).
[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. “Longformer: The long-document trans-
former”. In: arXiv preprint arXiv:2004.05150 (2020).
[4] Chaoxiang Cai et al. “Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large
Vision-Language Model”. In: arXiv preprint arXiv:2507.01351 (2025).
[5] Zefan Cai et al. “Pyramidkv: Dynamic kv cache compression based on pyramidal information
funneling”. In: arXiv preprint arXiv:2406.02069 (2024).
[6] Rewon Child et al. “Generating long sequences with sparse transformers”. In: arXiv preprint
arXiv:1904.10509 (2019).
[7] Tri Dao et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”.
In:Advances in Neural Information Processing Systems . 2022, pp. 16344–16359.
10[8] Hangliang Ding et al. “Efficient-vDiT: Efficient Video Diffusion Transformers With Attention
Tile”. In: arXiv preprint arXiv:2502.06155 (2025).
[9] Tianyu Fu et al. “Moa: Mixture of sparse attention for automatic large language model
compression”. In: arXiv preprint arXiv:2406.14909 (2024).
[10] Zichuan Fu et al. “Sliding Window Attention Training for Efficient Large Language Models”.
In:arXiv preprint arXiv:2502.18845 (2025).
[11] Yizhao Gao et al. “Seerattention: Learning intrinsic sparse attention in your llms”. In: arXiv
preprint arXiv:2410.13276 (2024).
[12] Chi Han et al. “Lm-infinite: Zero-shot extreme length generalization for large language
models”. In: arXiv preprint arXiv:2308.16137 (2023).
[13] Ali Hassani et al. “Generalized Neighborhood Attention: Multi-dimensional Sparse Attention
at the Speed of Light”. In: arXiv preprint arXiv:2504.16922 (2025).
[14] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. “Multistep consistency models”. In:
arXiv preprint arXiv:2403.06807 (2024).
[15] Huiqiang Jiang et al. “Minference 1.0: Accelerating pre-filling for long-context llms via
dynamic sparse attention”. In: Advances in Neural Information Processing Systems 37 (2024),
pp. 52481–52515.
[16] Kumara Kahatapitiya et al. “Adaptive caching for faster video generation with diffusion
transformers”. In: arXiv preprint arXiv:2411.02397 (2024).
[17] Dongjun Kim et al. “Consistency trajectory models: Learning probability flow ode trajectory
of diffusion”. In: arXiv preprint arXiv:2310.02279 (2023).
[18] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. “Reformer: The efficient transformer”.
In:arXiv preprint arXiv:2001.04451 (2020).
[19] Weijie Kong et al. “Hunyuanvideo: A systematic framework for large video generative models”.
In:arXiv preprint arXiv:2412.03603 (2024).
[20] Xunhao Lai et al. “FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient
Long-Sequence Inference”. In: arXiv preprint arXiv:2502.20766 (2025).
[21] Jiachen Li et al. “T2v-turbo: Breaking the quality bottleneck of video consistency model with
mixed reward feedback”. In: arXiv preprint arXiv:2405.18750 (2024).
[22] Muyang Li et al. “Svdqunat: Absorbing outliers by low-rank components for 4-bit diffusion
models”. In: arXiv preprint arXiv:2411.05007 (2024).
[23] Senmao Li et al. “Faster diffusion: Rethinking the role of unet encoder in diffusion models”.
In:CoRR (2023).
[24] Yucheng Li et al. “MAPSparse: Accelerating Pre-filling for Long-Context Visual Language
Models via Modality-Aware Permutation Sparse Attention”. In: ICLR 2025 Workshop on
Foundation Models in the Wild .
[25] Enshu Liu et al. “Oms-dpm: Optimizing the model schedule for diffusion probabilistic models”.
In:International Conference on Machine Learning . PMLR. 2023, pp. 21915–21936.
[26] Luping Liu et al. “Pseudo numerical methods for diffusion models on manifolds”. In: arXiv
preprint arXiv:2202.09778 (2022).
[27] Yaofang Liu et al. “Evalcrafter: Benchmarking and evaluating large video generation models”.
In:Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .
2024, pp. 22139–22149.
[28] Cheng Lu et al. “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in
around 10 steps”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 5775–
5787.
[29] Zhengyao Lv et al. “Fastercache: Training-free video diffusion model acceleration with high
quality”. In: arXiv preprint arXiv:2410.19355 (2024).
[30] Xinyin Ma, Gongfan Fang, and Xinchao Wang. “Deepcache: Accelerating diffusion models for
free”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition .
2024, pp. 15762–15772.
[31] Xinyin Ma et al. “Learning-to-cache: Accelerating diffusion transformer via layer caching”.
In:Advances in Neural Information Processing Systems 37 (2024), pp. 133282–133304.
[32] Chenlin Meng et al. “On distillation of guided diffusion models”. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition . 2023, pp. 14297–14306.
11[33] Jiezhong Qiu et al. “Blockwise self-attention for long document understanding”. In: arXiv
preprint arXiv:1911.02972 (2019).
[34] Luka Ribar et al. “Sparq attention: Bandwidth-efficient llm inference”. In: arXiv preprint
arXiv:2312.04985 (2023).
[35] Aurko Roy et al. “Efficient content-based sparse attention with routing transformers”. In:
Transactions of the Association for Computational Linguistics 9 (2021), pp. 53–68.
[36] Tim Salimans and Jonathan Ho. “Progressive distillation for fast sampling of diffusion models”.
In:arXiv preprint arXiv:2202.00512 (2022).
[37] Axel Sauer et al. “Adversarial diffusion distillation”. In: European Conference on Computer
Vision . Springer. 2024, pp. 87–103.
[38] Prajwal Singhania et al. “Loki: Low-rank keys for efficient sparse attention”. In: arXiv preprint
arXiv:2406.02542 (2024).
[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising diffusion implicit models”. In:
arXiv preprint arXiv:2010.02502 (2020).
[40] Yang Song et al. “Consistency models”. In: (2023).
[41] Hongyin Tang et al. “Ltri-LLM: Streaming Long Context Inference for LLMs with Training-
Free Dynamic Triangular Attention Pattern”. In: arXiv preprint arXiv:2412.04757 (2024).
[42] Yi Tay et al. “Sparse sinkhorn attention”. In: International conference on machine learning .
PMLR. 2020, pp. 9438–9447.
[43] Team Wan et al. “Wan: Open and advanced large-scale video generative models”. In: arXiv
preprint arXiv:2503.20314 (2025).
[44] Hanrui Wang, Zhekai Zhang, and Song Han. “Spatten: Efficient sparse attention architecture
with cascade token and head pruning”. In: 2021 IEEE International Symposium on High-
Performance Computer Architecture (HPCA) . IEEE. 2021, pp. 97–110.
[45] Felix Wimbauer et al. “Cache me if you can: Accelerating diffusion models through block
caching”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition . 2024, pp. 6211–6220.
[46] Haocheng Xi et al. “Sparse VideoGen: Accelerating Video Diffusion Transformers with
Spatial-Temporal Sparsity”. In: arXiv preprint arXiv:2502.01776 (2025).
[47] Yifei Xia et al. “Training-free and Adaptive Sparse Attention for Efficient Long Video Genera-
tion”. In: arXiv preprint arXiv:2502.21079 (2025).
[48] Chaojun Xiao et al. “Infllm: Training-free long-context extrapolation for llms with an efficient
context memory”. In: arXiv preprint arXiv:2402.04617 (2024).
[49] Guangxuan Xiao et al. “Duoattention: Efficient long-context llm inference with retrieval and
streaming heads”. In: arXiv preprint arXiv:2410.10819 (2024).
[50] Guangxuan Xiao et al. “Efficient streaming language models with attention sinks”. In: arXiv
preprint arXiv:2309.17453 (2023).
[51] Qingsong Xie et al. “Mlcm: Multistep consistency distillation of latent diffusion model”. In:
arXiv e-prints (2024), arXiv–2406.
[52] Shang Yang et al. “Lserve: Efficient long-sequence llm serving with unified sparse attention”.
In:arXiv preprint arXiv:2502.14866 (2025).
[53] Shuo Yang et al. “Post-training sparse attention with double sparsity”. In: arXiv preprint
arXiv:2408.07092 (2024).
[54] Xinhao Yang et al. “PARO: Hardware-Software Co-design with Pattern-aware Reorder-based
Attention Quantization in Video Generation Models”. In: Des. Automat. Conf . 2025.
[55] Tianwei Yin et al. “Improved distribution matching distillation for fast image synthesis”. In:
Advances in neural information processing systems 37 (2024), pp. 47455–47487.
[56] Tianwei Yin et al. “One-step diffusion with distribution matching distillation”. In: Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition . 2024, pp. 6613–6623.
[57] Manzil Zaheer et al. “Big bird: Transformers for longer sequences”. In: Advances in neural
information processing systems 33 (2020), pp. 17283–17297.
[58] Jintao Zhang et al. “Sageattention: Accurate 8-bit attention for plug-and-play inference accel-
eration”. In: arXiv preprint arXiv:2410.02367 (2024).
[59] Jintao Zhang et al. “SageAttention2 Technical Report: Accurate 4 Bit Attention for Plug-and-
play Inference Acceleration”. In: arXiv preprint arXiv:2411.10958 (2024).
12[60] Jintao Zhang et al. “Spargeattn: Accurate sparse attention accelerating any model inference”.
In:arXiv preprint arXiv:2502.18137 (2025).
[61] Peiyuan Zhang et al. “Fast Video Generation with Sliding Tile Attention”. In: arXiv preprint
arXiv:2502.04507 (2025).
[62] Qinsheng Zhang and Yongxin Chen. “Fast sampling of diffusion models with exponential
integrator”. In: arXiv preprint arXiv:2204.13902 (2022).
[63] Zhenyu Zhang et al. “H2o: Heavy-hitter oracle for efficient generative inference of large lan-
guage models”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 34661–
34710.
[64] Tianchen Zhao et al. “Vidit-q: Efficient and accurate quantization of diffusion transformers for
image and video generation”. In: arXiv preprint arXiv:2406.02540 (2024).
[65] Qianchao Zhu et al. “SampleAttention: Near-Lossless Acceleration of Long Context LLM
Inference with Adaptive Structured Sparse Attention”. In: arXiv preprint arXiv:2406.15486
(2024).
13A Limitations
The proposed auto-search strategy offers computational efficiency. However, through low recall and
cost thresholds, this design choice may potentially compromise the visual fidelity of generated video
content. Specifically, under more demanding scenarios, critical visual details could be omitted, leading
to suboptimal generation quality. Future work will explore adaptive thresholding and context-aware
search strategies to better balance efficiency and perceptual performance.
B Broader Impacts
Our work reduces computational barriers for deploying long-video generation models through
accelerated inference and lower memory costs, enabling broader access to high-quality video synthesis
for individual creators and small teams. This democratization could catalyze innovation in education,
digital art, and low-resource creative industries. Notably, our discovery of hierarchical attention
patterns—such as localized spatial focus(local pattern, cross-shaped pattern), temporally-varying
frame dependencies provides new insights into how video Transformers model spatiotemporal
relationships. These patterns reveal specialized roles of attention heads (e.g., handling short-term
motion or global context), improving model interpretability and offering a foundation for future
research. Such findings could inspire targeted architectural designs (e.g., hybrid sparse attention
modules) or curriculum learning strategies that align training with inherent spatiotemporal priors,
potentially advancing both efficiency and controllability in video generation systems.
C Baseline Implementation Details
STA STA is implemented based on FlashAttention-3 within the ThunderKittens framework and is
compatible exclusively with the Hopper architecture. We adopt the publicly released mask configura-
tion for Wan2.1 and Hunyuan from the official STA repository. Due to STA’s strict constraints on
video resolution, all experiments are conducted on 69-frame or 117-frame videos at a resolution of
768×1280 using the STA kernel for fair comparison.
Sparge Attention Sparge Attention provides an interface for sparse attention operations via its open-
source implementation. In our experiments, we integrate this interface into the diffusers library
and evaluate the method using its default hyperparameters ( simthreshd1=0.1 ,cdfthreshd=0.9 ,
pvthreshd=20 ). The observed average sparsity is comparable to that of other baseline methods.
Sparse VideoGen We conduct experiments using the official implementation of Sparse VideoGen
(SVG) from its open-source repository. The observed average sparsity is adjusted to be comparable
to that of other baseline methods.
D Sparsity Validation after rearranged based on adjacent 3D tiles
FlashAttention tiles the query, key, and value tensors along the token dimension into blocks Qi,Ki,
Viwith block sizes bq,bkrespectively, and computes each output block Oiincrementally using an
online softmax[7]. This design achieves lower memory consumption and faster execution, while
enabling attention acceleration through tile-level sparsity, thus avoiding the inefficiency of token-level
sparsity. As a result, many sparse attention methods adopt small blocks as the basic computation unit.
However, applying block-wise sparsity directly on attention over sequences obtained by flattening a
3D feature map (f, h, w) may be suboptimal, as it treats tokens within each block as equally important,
regardless of spatial relationships. In Fig. 9 and Fig. 10, We show that reordering tokens based
on 3D spatial locality prior to applying block-wise sparsity also improves attention sparsity while
maintaining acceleration benefits in hunyuan. This spatially-aware grouping yields a 1%reduction in
the average number of active blocks on the Wan2.1 (14B) model, and 3.4% on the Hunyuan model.
14timestep47.7% attention achieves 95% recall
headFigure 9: Flattening sequence on tiles
timestep
head51.1% attention achieves 95% recall Figure 10: Directly flattening sequence
E Additional Experiment Results
E.1 Comparative Analysis with Baseline Methods
Fig. 11 demonstrates the superior performance of Compact Attention compared with Sliding Tile
Attention (STA) in the Hunyuan video generation framework. Our method achieves enhanced Peak
Signal-to-Noise Ratio (PSNR) while operating at higher sparsity rates. Specifically, to accommodate
STA’s tile grouping requirements for attention sequence processing, we conducted comparative
evaluations using 117-frame video sequences. The visual comparisons reveal that Compact Attention
maintains better video quality preservation despite increased sparsity levels, confirming that our
approach more effectively identifies and retains critical attention computation components - a core
design principle of our architecture. Furthermore, we extended the comparison to include Sparge
Attention and Sparse VideoGen using the standard 129-frame sequences recommended for optimal
video generation performance as shown in Fig. 12. While these baseline methods employ dynamic
and static sparsification strategies respectively, neither sufficiently addresses the precise identification
of computationally critical attention regions. As evidenced by the quantitative metrics, Compact
Attention exhibits significant advantages in video quality retention metrics under comparable sparsity
conditions. This performance gap highlights our method’s improved capability in preserving essential
spatial-temporal attention patterns through systematic sparse computation. More generation cases are
shown in Fig. 13.
15Full AttentionSparsity :  62.36%
PSNR = 25.41 PSNR = 27.69Sliding Tile Attention (STA) Compact Attention (Ours)Sparsity :  58.37%Hunyuan
Difference Difference
MSE  = 117.66 MSE  = 197.76
PSNR = 25.30 PSNR = 27.27 MSE  = 123.40 MSE  = 193.39
Figure 11: Performance of Compact Attention and Sliding Tile Attention on end-to-end video
generation.
16Full Attention
PSNR = 31.67
MSE = 45.09Sparse VideoGen Compact Attention (Ours)Hunyuan
Sparge Attention
PSNR = 25.50
MSE = 185.46PSNR = 20.96
MSE = 568.00
PSNR = 29.08
MSE = 81.22
PSNR = 23.08
MSE = 321.50PSNR = 18.70
MSE = 883.94Figure 12: Performance of Compact Attention and baselines on end-to-end video generation.
17Full
Attention
Compact
Attention
(Ours )
Sparsity:  62.36%Hunyuan
Compact
Attention
(Ours )
Sparsity:  58.37%
Full
Attention
Compact
Attention
(Ours )
Sparsity:  62.36%
Compact
Attention
(Ours )
Sparsity:  58.37%
Full
Attention
Compact
Attention
(Ours )
Sparsity:  62.36%
Compact
Attention
(Ours )
Sparsity:  58.37%Figure 13: Performance of Compact Attention on end-to-end video generation.
18