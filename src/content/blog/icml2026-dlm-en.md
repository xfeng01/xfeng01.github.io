---
title: "ICML 2026 Recap: Diffusion Language Models"
description: "A personal recap of ICML 2026 diffusion language model papers, with trends, caveats, and a selective reading list."
date: 2026-07-23
lang: en
translationKey: ICML2026-Recap
tags:
  - DLM
---

This is a short recap of ICML 2026, mainly focusing on diffusion language models. There are roughly 100 related papers. What becomes clear is that the community is now paying more attention to some of the most fundamental questions: what arbitrary-order generation actually brings, where the quality bottleneck of parallel decoding lies, and whether diffusion can truly enter the existing LLM inference stack.


To summarize briefly:

> **The main theme of dLLMs in 2026 is no longer “whether diffusion can decode in parallel,” but “in parallel generation, in what order should tokens be determined?”**


## 1 Overview

| Metric | Value |
|---|---|
| Submissions | 23,918 |
| Accepted papers | 6,352 |
| Acceptance rate | 26.56% |
| Oral | 168 (2.6% of accepted papers) |
| Spotlight | 536 (8.4% of accepted papers) |
| **dLLM-related papers** | **Approximately 95–105 (~1.5% of accepted papers)** |
| Orals among dLLM papers | 4 |
| Spotlights among dLLM papers (including Orals) | 9 (5 Spotlights + 4 Orals) |



### Awards

This year was indeed very favorable to diffusion. The clearest signal is that [both Outstanding Papers](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/) were about diffusion:

- *The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models* (dLLM, and the paper this blog focuses on most)
- *High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions* (theory)
- The **Honorable Mentions** also included *A Random Matrix Perspective on the Consistency of Diffusion Models*; the other papers fell into memorization, RLVR honesty, and video generation
- **Outstanding Position Paper**: *Position: The Alignment Community is Unintentionally Building a Censor's Toolkit*
- **Test of Time**: *Asynchronous Methods for Deep Reinforcement Learning* (A3C, ICML 2016)

Of course, this does not mean that dLLMs have already become mainstream. The other Outstanding Paper is about more general diffusion sampling theory, and *The Flexibility Trap* remains the only one that is truly about dLLMs. Still, at least judging from the award signal, diffusion is no longer merely a peripheral direction that attracts attention by presenting itself as a “new paradigm.”




## 2 dLLM Orals (4 Papers)

### 1 [The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models](https://arxiv.org/abs/2601.15165)
**Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang** (Tsinghua + Taobao and Tmall Group) | **Outstanding Paper**

dLLMs break the left-to-right constraint of traditional LLMs and allow generation in arbitrary orders. Intuitively, this flexibility should only expand the solution space; this paper systematically questions whether that intuition actually holds in reasoning settings.

The authors observe that a model can exploit arbitrary order to bypass high-entropy tokens that are nevertheless critical to the reasoning path. In the short term, this makes the trajectory easier; in the long term, it may cause exploration to collapse prematurely. The corresponding method, JustGRPO, is simple: constrain exploration with a left-to-right trajectory during RL training, then restore parallel dLLM decoding at inference time.



### 2 [Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture](https://openreview.net/forum?id=sEYoG3tAXN)
**Shuchen Xue, Tianyu Xie, Tianyang Hu, Zijin Feng, Jiacheng Sun, Kenji Kawaguchi, Zhenguo Li, Zhi-Ming Ma** (Huawei Noah's Ark Lab + NUS + Chinese Academy of Sciences)


Comparisons between AR models (usually decoder-only) and MDMs (usually encoder-only) have long been confounded by architectural differences. This paper places MDMs in a decoder-only framework in order to: (1) fairly compare MDMs, viewed as Any-Order AR models, with standard AR models along the dimension of generation order; and (2) separately examine how architecture affects computational efficiency. The result is that, although a decoder-only MDM must cover a larger modeling space, temperature annealing can give it an approximately 25× sampling speedup at comparable perplexity.

> Many previous “dLLM vs. AR” experiments changed the formulation, attention mask, and backbone at the same time, making their conclusions difficult to attribute. This paper provides a cleaner comparison axis.

### 3 [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106)
**Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, Michael Kirchhof, Joao Monteiro, Victor Guilherme Turrisi da Costa, Jason Ramapuram, Marco Cuturi** (Apple + UvA + MIT)


This paper formulates masked diffusion sampling as an MDP, with the frozen dLLM serving as the environment. It then uses a single-layer transformer as a lightweight policy that maps token confidence to unmasking decisions. Under semi-AR (block) generation, it matches SOTA heuristics; in the full-diffusion setting, it clearly outperforms hand-designed strategies.

### 4 [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)
**Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, Jie Zhou** (Tencent WeChat AI + Peking University + Tsinghua University)

WeDLM makes dLLMs compatible with standard causal attention, allowing them to reuse mature prefix KV-cache infrastructure. Its core technique is Topological Reordering: determined tokens are moved into the physical prefix while retaining their original logical positions. This lets masked positions continue to see all known context without requiring the attention kernel itself to leave the standard causal mask.

It also introduces streaming parallel decoding, allowing high-confidence tokens to enter the prefix continuously instead of making each block stop and wait. This direction is practical because it no longer discusses parallelism only in theory; it measures wall-clock latency against a vLLM-served AR baseline. One caveat is that the gains depend heavily on output entropy, commit rate, and the specific serving setup. A 10× speedup on low-entropy tasks should not be interpreted as a fixed speedup on general reasoning tasks.



**Observation: although the four papers come from reasoning, formulation, policy learning, and systems, their central variable always comes back to “generation order.”** This is the clearest dLLM theme this year.



## 3 dLLM Spotlights (Excluding Orals)

| Title | Authors | Main idea |
|---|---|---|
| **Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization** | Yuchen Zhu, Wei Guo, Jaemoo Choi, Petr Molodyk, Bo Yuan, Molei Tao | Moves away from the policy-gradient framework and uses distribution matching for dLLM RL |
| **Unifying Masked Diffusion Models with Various Generation Orders and Beyond** | Chunsan Hong, Sanghyun Lee, Jong Chul Ye | A unified framework for MDM generation orders |
| **Balancing Understanding and Generation in Discrete Diffusion Models** | Yue Liu, Yuzhong Zhao, Zheyong Xie, Qixiang Ye et al. | The structural asymmetry in MDLMs between strong semantic understanding and weak generation |
| **Variational Learning for Insertion-based Generation** | Yangtian Zhang, Zhe Wang, Arthur Gretton, Zhitao Ying, David van Dijk, Michalis Titsias | A third path for non-monotonic generation: insertion rather than masking |
| **Training Diffusion Language Models for Black-Box Optimization** | Zipeng Sun, Can Chen, Ye Yuan, Haolun Wu, Christopher Pal et al. | Offline black-box optimization |



## 4 Distribution by Direction (Approximately 100 Papers)

| Direction | Share | Representative work |
|---|---|---|
| **Decoding order / unmasking / parallel decoding strategies** | **~35%** | Lookahead Unmasking, Locally Coherent Parallel Decoding, Plan for Speed (dilated scheduling), Set Diffusion, Demystifying MaskGIT Sampler, Scheduling Thoughts, From Bits to Rounds |
| **Inference efficiency / systems** (cache / quantization / sparse attention / distillation) | **~25%** | dLLM-Cache, DLLMQuant, LoSA, Mosaic (30× context), TEAM (MoE), d3LLM, FlashBlock, DFlash, Swordsman |
| **RL / post-training / reasoning** | ~15% | dTRPO, LightningRL, Stabilizing RL for DLMs, d2, Simple Policy Gradients, DiffuReason (MCTS) |
| **Architectures and AR–diffusion hybrids** | ~8% | WeDLM, Esoteric Language Models, DiffuMamba, Efficient-DLM, Break the Block, Residual Context DLM |
| **Theory / analysis / interpretability** | ~8% | Generalization Bounds for Discrete Diffusion, Breaking the Factorization Barrier, Tuning the Implicit Regularizer (k-parity), DLM-Scope (SAE), Is Your Diffusion Sampler Actually Correct? |
| **Multimodal / VLM / applications** | ~7% | Lavida-R1, VidLaDA, Discrete Diffusion VLA, Any-Diffusion, ST-Veto |
| **Safety / watermarking / unlearning** | ~2% | dgMARK, The Safety-Aware Denoiser for Text Diffusion Models, Adversarial RL for dLLM Unlearning |



## 5 Trend Analysis

### 1 Many Papers, but the Number of Top-Tier Papers Is Only “Slightly Above Average”

- Oral rate among dLLM papers ≈ 4% (2.6% overall)
- Spotlight rate among dLLM papers ≈ 10% (8.4% overall)

The field has already moved from a period of “new-paradigm dividends” into an **engineering / incremental phase**. Around 60% of the papers concentrate on decoding strategies and inference acceleration, and the homogeneity is already quite noticeable. dLLMs are slightly above the overall average among top-tier papers, but they have not formed an overwhelming concentration.

### 2 The Most Distinctive Work Focuses on “Falsification” or “Disentanglement”

Among the four Orals, the two most distinctive papers do not simply add another set of decoding tricks: *Flexibility Trap* questions the intuitive benefit of arbitrary order, while *Any-Order GPT* removes the confounding between formulation and architecture. Both papers are doing **subtraction**.

> My feeling is that the community is finally beginning to systematically examine its original selling points rather than simply amplifying them, and to provide insight into some of the most fundamental questions.

### 3 “Generation Order” Is Replacing “Speed” as the First-Principles Question

In 2025, the discussion was more about *parallel decoding = faster inference*. In 2026, the question has become *which tokens should be decoded together, and in what order?* Once multiple tokens are committed at the same time, structural mismatches can arise among the factorized proposal, the random masking distribution used during training, and the actual inference trajectory.

Order is therefore not merely an implementation detail of the sampler. It determines what context will be visible at the next step, as well as which dependencies are handled in parallel and which are converted into sequential conditioning. In other words, the decoding policy itself has become part of the model.


### 4 AR and Diffusion Are Converging, Not Competing

WeDLM (returning to causal attention in exchange for KV caching), Any-Order GPT (decoder-only MDM), Esoteric LM, and Set Diffusion (interpolating token orderings between AR and diffusion) are all pulling diffusion toward AR infrastructure.

> **At least on the deployment side, purely encoder-only, purely full-diffusion approaches are becoming less attractive.**

Of course, we cannot conclude from this that full diffusion has no value. For infilling, bidirectional constraints, or structured editing, it still has capabilities that do not come naturally to AR models. But if the goal is general-purpose LLM serving, practical constraints such as prefix caching, streaming, and variable-length generation are forcing diffusion to reabsorb AR structure.

### 5 RL on dLLMs Has Not Yet Converged

dTRPO / LightningRL / DMPO / StableDRL / Simple Policy Gradients: five mutually incompatible approaches were accepted at the same time. This means that **the question of a “GRPO equivalent for dLLMs” still has no winner.**

The core difficulties are:

- Training and evaluation usually rely on an ELBO rather than a direct factorization of the exact sequence log-likelihood as in AR models
- The trajectory is not uniquely defined
- Token content and unmasking / remasking decisions jointly constitute the action
- There is no standard approach to credit assignment across denoising steps

The responses are also quite different. DMPO turns directly to reward-tilted distribution matching and weighted denoising cross-entropy, reducing its dependence on rollout trajectories as much as possible. StableDRL retains the GRPO framework but redesigns clipping and normalization specifically to handle noisy importance ratios.



## 6 Selected Reading List


### Most Recommended

| # | Paper | Type | Why it is worth reading |
|---|---|---|---|
| 1 | [**The Flexibility Trap**](https://arxiv.org/abs/2601.15165) | Oral / Outstanding | A direct answer to the question of whether arbitrary order is beneficial |
| 2 | [**Any-Order GPT as Masked Diffusion Model**](https://openreview.net/forum?id=sEYoG3tAXN) | Oral | A methodological contribution that provides a clean decoder-only MDM baseline |
| 3 | [**Breaking the Factorization Barrier in Diffusion Language Models**](https://arxiv.org/abs/2603.00045)<br>*Ian Li, Zilei Shao, Benjie Wang, Rose Yu, Guy Van den Broeck, Anji Liu* | poster | It directly addresses the fundamental limitation of parallel generation: the conditional-independence assumption implicit in predicting multiple tokens simultaneously. Rather than “adding another planner,” it tackles the factorization barrier with tractable probabilistic modeling. Many decoding tricks operate beneath this barrier |
| 4 | [**Generalization Bounds for Discrete Diffusion: Statistical Advantage of Masking**](https://openreview.net/forum?id=Ofhq7nBVHu)<br>*Zixuan Zhang, Hengyu Fu, Zhuoran Yang, Mengdi Wang, Tuo Zhao, Minshuo Chen* | poster | The first systematic answer to “why masking rather than another corruption process?” The dLLM field has long lacked the why; most work focuses on the how |
| 5 | [**Fine-Tuning Masked Diffusion for Provable Self-Correction**](https://arxiv.org/abs/2510.01384)<br>*Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, Sitan Chen* | poster | It directly addresses the hardest structural limitation of monotone MDMs: **once a token is unmasked, the decision cannot be reversed**. At least four papers this year work on self-correction; the “provable” part is what matters here |

> The actual significance of Papers 3, 4, and 5 depends on how much of their theoretical assumptions and proofs survives when applied to real models. I would prioritize checking the theorem setting rather than just reading the headline.

### Next — Work That Actually Changes the Architecture

- [**Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding**](https://arxiv.org/abs/2607.01775) (Marianne Arriola & Volodymyr Kuleshov)
  It replaces fixed blocks with flexible-position, flexible-length token sets, addressing both fixed-length generation and KV-cache compatibility.
  **The continuity of the Kuleshov group's research line—MDLM → Block Diffusion → Set Diffusion → d2—is unusually strong. It is currently the single research trajectory most worth following.**

- [**Scaling Beyond Masked Diffusion Language Models**](https://arxiv.org/abs/2602.15014) + [**Esoteric Language Models**](https://arxiv.org/abs/2506.01928) (both first-authored by Subham Sekhar Sahoo)
  **One of the original MDLM authors is now saying that we need to “move beyond masked.” That signal itself is more important than the paper content.** Eso-LM is an AR + MDM hybrid with KV-cache support.

- [**WeDLM**](https://arxiv.org/abs/2512.22737) (Oral)
  Another route in the same direction: return directly to standard causal attention in exchange for infrastructure compatibility.

> These three lines are saying the same thing: **the mainstream direction in 2026 is “how to pull diffusion back toward AR infrastructure.”**

### Also Quite Good — The Only Three RL Papers with Substantive Content

| Paper | Main idea |
|---|---|
| **Enhancing Reasoning for dLLMs via Distribution Matching Policy Optimization** (Spotlight) | Uses distribution matching + weighted denoising cross-entropy to reduce dependence on rollout trajectories. The most aggressive idea of the three |
| **Stabilizing Reinforcement Learning for Diffusion Language Models** | Directly diagnoses the causes of severe GRPO instability on dLLMs. A problem-defining paper |
| **d2: Improved Techniques for Training Reasoning Diffusion Language Models** (Kuleshov group) | A consolidation of engineering techniques; use it as a recipe |

The remaining dTRPO / LightningRL / Simple Policy Gradients papers look more like their own combinations of tricks, and their methods have not converged with one another.

### Worth Reading If You Have Time

- **Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models**
  The subtext is quite sharp: **many sampling methods that claim “acceleration” may quietly change the target distribution, while benchmark scores conceal the fact.** This is a useful calibration tool for experimental comparisons.

- **Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models** (Qualcomm)
  A failure mode in MDLM long-context understanding: the denoising objective theoretically provides bidirectional context, but in practice the mask tokens themselves interfere with attention.

- **Tuning the Implicit Regularizer of Masked Diffusion Language Models: Insights from k-Parity** (Jianhao Huang & Baharan Mirzasoleiman)
  Uses the controlled k-parity problem to characterize the implicit regularization of MDMs. Small and clean.

- **Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner** (Cai Zhou et al.)
  Continuous and discrete diffusion evolving together. Worth scanning if you are interested in latent reasoning or world models.

- **Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models** (Minjoon Seo's group)
  Studies proximity bias in early decoding decisions and how those decisions shape the entire trajectory.

- **DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders**
  Brings the SAE interpretability toolkit from AR LLMs to dLLMs.



## 7 Summary

I think dLLMs are undergoing a healthy contraction in focus this year:

- Arbitrary order is not automatically an advantage;
- Parallel decoding is not only about how many tokens can be generated per step;
- Inference speed cannot be discussed independently of caching, kernels, and serving baselines.

At the moment, I think the more interesting question is: **how many dependencies should we leave to parallel prediction, and how many should we explicitly turn into a generation order?** After thinking about editing and remasking since last year, I have come to see this as the issue that deserves the most attention—a kind of convergence point for this direction.

