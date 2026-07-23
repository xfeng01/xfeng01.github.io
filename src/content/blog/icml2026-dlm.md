---
title: "ICML 2026 Recap: Diffusion Language Models"
description: "A personal recap of ICML 2026 diffusion language model papers, with trends, caveats, and a selective reading list."
date: 2026-07-23
lang: zh
translationKey: ICML2026-Recap
tags:
  - DLM
---

简单写一下对 ICML 2026 的一个简单recap，主要是关于 diffusion language models 方面的，大概有100篇。其实可以发现，现在整个 community 更加关心一些最基本的问题：arbitrary-order generation 到底带来了什么，parallel decoding 的质量瓶颈在哪里，以及 diffusion 能不能真正进入现有的 LLM inference stack。


简单总结一下：

> **2026 年 dLLM 的主线已经不是「diffusion 能不能并行」，而是「并行生成时，token 应该以什么顺序被确定」。**


## 1 数据总览

| 指标 | 数值 |
|---|---|
| 投稿量 | 23,918 |
| 接收量 | 6,352 |
| 录用率 | 26.56% |
| Oral | 168（占接收 2.6%） |
| Spotlight | 536（占接收 8.4%） |
| **dLLM 相关论文** | **约 95–105 篇（占接收 ~1.5%）** |
| dLLM 中的 Oral | 4 篇 |
| dLLM 中的 Spotlight（含 Oral） | 9 篇（5 Spotlight + 4 Oral） |



### 奖项层面

今年对 diffusion 确实非常友好。最明显的信号是，[两篇 Outstanding Paper](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/) 全部属于 diffusion：

- *The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models*（dLLM，也是这篇 blog 最关心的一篇）
- *High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions*（理论）
- **Honorable Mention** 中还有 *A Random Matrix Perspective on the Consistency of Diffusion Models*；另外几篇分别落在 memorization、RLVR honesty 和 video generation
- **Outstanding Position Paper**：*Position: The Alignment Community is Unintentionally Building a Censor's Toolkit*
- **Test of Time**：*Asynchronous Methods for Deep Reinforcement Learning*（A3C, ICML 2016）

当然，这不意味着 dLLM 已经成为主流。另一个 Outstanding Paper 是更一般的 diffusion sampling theory，真正属于 dLLM 的仍然只有 *The Flexibility Trap*。但至少从 award signal 来看，diffusion 今年不再只是一个靠「新范式」吸引注意力的边缘方向。




## 2 dLLM 的 Oral（4 篇）

### 1 [The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models](https://arxiv.org/abs/2601.15165)
**Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang**（清华 + 淘天）| **Outstanding Paper**

dLLM 打破了传统 LLM 的左到右约束，允许任意顺序生成。直觉上，这种灵活性应该只会扩大 solution space；本文系统性地质疑了这个直觉在 reasoning 场景下是否成立。

作者观察到，模型会利用 arbitrary order 绕过 high-entropy、但对推理路径很关键的 token。短期看，这让 trajectory 更容易；长期看，它可能让探索过早 collapse。对应的方法 JustGRPO 很简单：RL 训练时先用 left-to-right trajectory 约束探索，推理时再恢复 dLLM 的 parallel decoding。



### 2 [Any-Order GPT as Masked Diffusion Model: Decoupling Formulation and Architecture](https://openreview.net/forum?id=sEYoG3tAXN)
**Shuchen Xue, Tianyu Xie, Tianyang Hu, Zijin Feng, Jiacheng Sun, Kenji Kawaguchi, Zhenguo Li, Zhi-Ming Ma**（华为诺亚 + NUS + 中科院）


AR（通常 decoder-only）与 MDM（通常 encoder-only）的对比长期被架构差异混淆。本文把 MDM 放进 decoder-only 框架，从而：(1) 在「生成顺序」这一维度上公平对比 MDM（作为 Any-Order AR）与标准 AR；(2) 单独考察 architecture 对计算效率的影响。结果是，decoder-only MDM 虽然需要覆盖更大的建模空间，但配合 temperature annealing 可以在 perplexity 可比的情况下获得约 25× 的采样加速。

> 过去很多「dLLM vs AR」实验同时改变了 formulation、attention mask 和 backbone，结论很难归因。这篇给出了一个更干净的 comparison axis。

### 3 [Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106)
**Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, Michael Kirchhof, Joao Monteiro, Victor Guilherme Turrisi da Costa, Jason Ramapuram, Marco Cuturi**（Apple + UvA + MIT）


把 masked diffusion sampling 形式化为 MDP（冻结的 dLLM 作为 environment），再用单层 transformer 作为轻量 policy，把 token confidence 映射为 unmask 决策。在 semi-AR（block）生成下，它与 SOTA heuristic 持平；在 full-diffusion 设置下，则明显优于手工策略。

### 4 [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)
**Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, Jie Zhou**（腾讯微信 AI + 北大 + 清华）

让 dLLM 兼容标准 causal attention，从而复用成熟的 prefix KV-cache 基础设施。核心做法是 Topological Reordering：把已经确定的 token 移到 physical prefix，同时保留它们原来的 logical position。这样 masked positions 仍然可以看到全部已知 context，但 attention kernel 本身不需要离开标准 causal mask。

它还加入 streaming parallel decoding，让高置信 token 持续进入 prefix，而不是每个 block 都 stop-and-wait。这个方向很实用，因为它讨论的不再是理论上的 parallelism，而是相对 vLLM-served AR baseline 的 wall-clock latency。需要注意的是，收益会明显依赖输出 entropy、commit rate 和具体 serving setup；低熵任务上的 10× 不应该直接理解成通用 reasoning 上的固定加速比。



**观察：四篇虽然分别来自 reasoning、formulation、policy learning 和 systems，但核心变量都绕不开「生成顺序」。** 这是今年 dLLM 最清楚的一条主线。



## 3 dLLM 的 Spotlight（非 Oral）

| 标题 | 作者 | 要点 |
|---|---|---|
| **Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization** | Yuchen Zhu, Wei Guo, Jaemoo Choi, Petr Molodyk, Bo Yuan, Molei Tao | 放弃 policy gradient 框架，改用分布匹配做 dLLM 的 RL |
| **Unifying Masked Diffusion Models with Various Generation Orders and Beyond** | Chunsan Hong, Sanghyun Lee, Jong Chul Ye | MDM 生成顺序的统一框架 |
| **Balancing Understanding and Generation in Discrete Diffusion Models** | Yue Liu, Yuzhong Zhao, Zheyong Xie, Qixiang Ye 等 | MDLM 在语义理解强、生成弱上的结构性不对称 |
| **Variational Learning for Insertion-based Generation** | Yangtian Zhang, Zhe Wang, Arthur Gretton, Zhitao Ying, David van Dijk, Michalis Titsias | non-monotonic 生成的第三条路：不是 mask，是 insertion |
| **Training Diffusion Language Models for Black-Box Optimization** | Zipeng Sun, Can Chen, Ye Yuan, Haolun Wu, Christopher Pal 等 | 离线 black-box optimization 场景 |



## 4 方向分布统计（约 100 篇）

| 方向 | 占比 | 代表工作 |
|---|---|---|
| **解码顺序 / unmasking / 并行解码策略** | **~35%** | Lookahead Unmasking、Locally Coherent Parallel Decoding、Plan for Speed（dilated scheduling）、Set Diffusion、Demystifying MaskGIT Sampler、Scheduling Thoughts、From Bits to Rounds |
| **推理效率 / 系统**（cache / quant / sparse attn / 蒸馏） | **~25%** | dLLM-Cache、DLLMQuant、LoSA、Mosaic（30× context）、TEAM（MoE）、d3LLM、FlashBlock、DFlash、Swordsman |
| **RL / 后训练 / reasoning** | ~15% | dTRPO、LightningRL、Stabilizing RL for DLMs、d2、Simple Policy Gradients、DiffuReason（MCTS） |
| **架构与 AR–diffusion 混合** | ~8% | WeDLM、Esoteric Language Models、DiffuMamba、Efficient-DLM、Break the Block、Residual Context DLM |
| **理论 / 分析 / 可解释** | ~8% | Generalization Bounds for Discrete Diffusion、Breaking the Factorization Barrier、Tuning the Implicit Regularizer (k-parity)、DLM-Scope (SAE)、Is Your Diffusion Sampler Actually Correct? |
| **多模态 / VLM / 应用** | ~7% | Lavida-R1、VidLaDA、Discrete Diffusion VLA、Any-Diffusion、ST-Veto |
| **安全 / 水印 / unlearning** | ~2% | dgMARK、The Safety-Aware Denoiser for Text Diffusion Models、Adversarial RL for dLLM Unlearning |



## 5 趋势判断

### 1 数量多，但顶层Paper数量只是「略高于均值」

- dLLM 的 Oral 率 ≈ 4%（全场 2.6%）
- dLLM 的 Spotlight 率 ≈ 10%（全场 8.4%）

领域已经从「新范式红利期」进入了 **工程化 / 增量期**。约 60% 的论文集中在 decoding strategy 和 inference acceleration，同质化已经很明显。dLLM 在 top-tier paper 中略高于全场均值，但还没有形成压倒性的 concentration。

### 2 最有特色的工作在做「证伪」或「解耦」

四篇 Oral 里，最有特色的两篇不是再叠一套 decoding trick：*Flexibility Trap* 质疑 arbitrary order 的直觉收益，*Any-Order GPT* 拆掉 formulation / architecture 的混淆。两篇都在做**减法**。

> 我感觉是 Community 终于在系统性检查自己最初的 selling point，而不只是继续放大它，针对一些最 foundamental 的问题提供 insight 。

### 3 「生成顺序」取代「速度」成为第一性问题

2025 年更多的是 *parallel decoding = faster inference*，2026 年开始追问 *which tokens should be decoded together, and in what order?* 一旦多个 token 同时 commit，factorized proposal、训练时的 random masking distribution 和推理时的实际 trajectory 之间就可能出现结构性 mismatch。

所以 order 不只是 sampler 的 implementation detail。它在决定下一步能看到什么 context，也在决定哪些 dependency 被并行处理、哪些 dependency 被转化成 sequential conditioning。换句话说，decoding policy 本身已经变成 model 的一部分。


### 4 AR 与 diffusion 在收敛，不是在竞争

WeDLM（回归 causal attention 换 KV-cache）、Any-Order GPT（decoder-only MDM）、Esoteric LM、Set Diffusion（在 AR 与 diffusion 之间插值 token ordering）——都在把 diffusion 往 AR 基础设施上拉。

> **至少在 deployment 这一侧，纯 encoder-only、纯 full-diffusion 的吸引力正在下降。**

当然我们不能因此说 full diffusion 没有价值。对于 infilling、bidirectional constraint 或结构化编辑，它仍然有 AR 不自然具备的能力。但如果目标是通用 LLM serving，那么 prefix cache、streaming 和 variable-length generation 这些现实约束，正在迫使 diffusion 重新吸收 AR 的结构。

### 5 RL on dLLM 尚未收敛

dTRPO / LightningRL / DMPO / StableDRL / Simple Policy Gradients，五种互不兼容的思路同时被接收 → **「dLLM 上的 GRPO 等价物」这个问题还没有 winner。**

核心难点：

- 训练和评估时通常依赖 ELBO，而不是像 AR 那样直接分解 exact sequence log-likelihood
- trajectory 定义不唯一
- token content 与 unmask / remask decision 共同构成 action
- denoising steps 之间的 credit assignment 没有标准做法

各家的应对也很不一样：DMPO 直接转向 reward-tilted distribution matching 和 weighted denoising cross-entropy，尽量减少对 rollout trajectory 的依赖；StableDRL 则保留 GRPO 框架，但重新设计 clipping 和 normalization，专门处理 noisy importance ratio。



## 6 精选阅读清单


### 最推荐

| # | 论文 | 类型 | 为什么值得读 |
|---|---|---|---|
| 1 | [**The Flexibility Trap**](https://arxiv.org/abs/2601.15165) | Oral / Outstanding | arbitrary order 收益问题的直接回答 |
| 2 | [**Any-Order GPT as Masked Diffusion Model**](https://openreview.net/forum?id=sEYoG3tAXN) | Oral | 方法论级贡献；给出干净的 decoder-only MDM 基线 |
| 3 | [**Breaking the Factorization Barrier in Diffusion Language Models**](https://arxiv.org/abs/2603.00045)<br>*Ian Li, Zilei Shao, Benjie Wang, Rose Yu, Guy Van den Broeck, Anji Liu* | poster | 直指并行生成的根本限制——同时预测多 token 时隐含的条件独立假设。不是「再加个 planner」，而是用 tractable probabilistic modeling 正面处理 factorization barrier。很多 decoding trick 都在这个 barrier 之下打转 |
| 4 | [**Generalization Bounds for Discrete Diffusion: Statistical Advantage of Masking**](https://openreview.net/forum?id=Ofhq7nBVHu)<br>*Zixuan Zhang, Hengyu Fu, Zhuoran Yang, Mengdi Wang, Tuo Zhao, Minshuo Chen* | poster | 第一次系统回答「为什么是 masking，而不是别的 corruption」。dLLM 领域长期缺 why，绝大部分工作是 how |
| 5 | [**Fine-Tuning Masked Diffusion for Provable Self-Correction**](https://arxiv.org/abs/2510.01384)<br>*Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, Sitan Chen* | poster | 直击 monotone MDM 最硬的结构性缺陷：**token 一旦 unmask 就不可撤销**。今年至少四篇在做 self-correction，这篇的「provable」是关键 |

> 第 3、4、5 篇的实际分量取决于理论假设和证明落到真实模型时还剩多少，建议优先核对 theorem setting，而不只是看 headline。

### 其次 —— 真正在动架构的

- [**Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding**](https://arxiv.org/abs/2607.01775)（Marianne Arriola & Volodymyr Kuleshov）
  用 flexible-position、flexible-length token sets 替代固定 block，同时处理 fixed-length generation 和 KV-cache compatibility。
  **Kuleshov 组这条线（MDLM → Block Diffusion → Set Diffusion → d2）连续性极强，是目前最值得跟的单一研究轨迹。**

- [**Scaling Beyond Masked Diffusion Language Models**](https://arxiv.org/abs/2602.15014) + [**Esoteric Language Models**](https://arxiv.org/abs/2506.01928)（均为 Subham Sekhar Sahoo 一作）
  **MDLM 的原作者自己在说「要走出 masked」。这个信号本身比论文内容更重要。** Eso-LM 是 AR + MDM 混合且支持 KV-cache。

- [**WeDLM**](https://arxiv.org/abs/2512.22737)（Oral）
  同方向另一条路径：直接回归标准 causal attention 换基础设施兼容性。

> 这三条讲的是同一件事：**2026 年的主流是「如何把 diffusion 拉回 AR 的基础设施」。**

### 也还不错 —— RL 方向唯三有实质内容的

| 论文 | 要点 |
|---|---|
| **Enhancing Reasoning for dLLMs via Distribution Matching Policy Optimization**（Spotlight） | 用分布匹配 + weighted denoising cross-entropy，减少对 rollout trajectory 的依赖。思路上最激进 |
| **Stabilizing Reinforcement Learning for Diffusion Language Models** | 直接诊断 GRPO 在 dLLM 上严重不稳定的成因。问题定义型工作 |
| **d2: Improved Techniques for Training Reasoning Diffusion Language Models**（Kuleshov 组） | 工程集大成，当 recipe 用 |

其余 dTRPO / LightningRL / Simple Policy Gradients 更像各自的 trick 组合，方法上未互相收敛。

### 有时间可以看

- **Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models**
  潜台词很尖锐：**大量号称"加速"的采样方法可能悄悄改变了目标分布，benchmark 分数掩盖了这一点。** 做实验对比时的校准工具。

- **Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models**（Qualcomm）
  MDLM 长上下文理解的失效模式：denoising objective 理论上给了双向上下文，实际上 mask token 本身在干扰注意力。

- **Tuning the Implicit Regularizer of Masked Diffusion Language Models: Insights from k-Parity**（Jianhao Huang & Baharan Mirzasoleiman）
  用 k-parity 这个可控问题刻画 MDM 的隐式正则化。小而干净。

- **Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner**（Cai Zhou 等）
  连续 + 离散协同演化。如果在看 latent reasoning / world model，值得扫一眼。

- **Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models**（Minjoon Seo 组）
  早期解码决策的 proximity bias 及其对整条轨迹的塑形作用。

- **DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders**
  把 SAE 这套 AR-LLM 的可解释性工具搬到 dLLM。



## 7 总结

我觉得今年 dLLM 正在经历一次很健康的话题收缩：

- arbitrary order 不是自动成立的优势；
- parallel decoding 不是只看每步能生成多少 token；
- inference speed 也不能脱离 cache、kernel 和 serving baseline 来谈。

目前，我觉得更有意思的问题其实是：**我们应该把多少 dependency 交给 parallel prediction，又应该把多少 dependency 显式地变成 generation order？** 这也是我从去年开始思考 editing、remasking 之后，觉得最应该关注的，算是这个方向一个收束点。






