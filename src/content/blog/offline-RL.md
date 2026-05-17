---
title: "Some Basic Thoughts on Offline RL: The Core Problem, Common Solutions, and Generative Policies"
description: "Offline RL is about reliable policy improvement under dataset support constraints."
date: 2025-10-03
lang: zh
translationKey: offline-rl
tags:
  - Diffusion Models
  - Offline RL
---

Offline RL 的核心问题可以用一句话概括：**我们希望从一个固定数据集里学到比数据收集策略更好的 policy，但又不能让 policy 跑到数据集没有覆盖的区域。**

这也是 offline RL 和普通 supervised learning 不一样的地方。Supervised learning 通常假设训练数据和测试数据来自同一个分布，因此模型主要是在训练分布附近进行泛化。而在 offline RL 中，数据是由某个 behavior policy 收集的，但我们最终要学习的是一个可能优于它的新 policy。由于 policy 的 action 会影响未来访问到的 states，一旦 learned policy 和 behavior policy 不同，它诱导出的 state-action distribution $d^π(s,a)$ 就可能偏离数据集中的分布 $d^μ(s,a)$。因此，offline RL 的核心难点不是简单地在固定数据上拟合，而是在实现 policy improvement 的同时控制 distribution shift 和 support mismatch。

## 1 理解offline dataset, behavior policy以及learned policy

假设 offline dataset 是由 behavior policy $\mu$ 收集的：

$$
D \sim d^μ(s,a) = d^μ(s) μ(a|s)
$$

这里 $d^μ(s)$ 是 $μ$ 和环境 dynamics 共同诱导出来的 state visitation distribution，$μ(a|s)$ 是 behavior policy 在 state $s$ 下选择 action 的分布。因此，offline dataset 实际覆盖的是 behavior policy $ μ$ 诱导出来的 state-action distribution。

但训练时我们最终要学的是另一个 policy：

$$
π(a|s)
$$

如果在某个 state $s$ 下，$π$ 选择了 dataset 里很少出现甚至没有出现过的 action，那么这个 $(s,a)$ 就是 OOD state-action pair。这个时候 $Q(s,a)$ 的估计往往是不可靠的，因为 dataset 里没有足够多关于这个 action 的真实 transition 来监督它。

所以 offline RL 里最核心的问题不是“没有数据”，而是：**数据只覆盖了某个 behavior policy 访问过的区域，而 learned policy 可能会想去数据没有覆盖的区域。**

## 2 OOD 和 Distribution Shift

在 offline RL 里，OOD 最常见的表现是 **action OOD**。也就是说，在同一个 state 下，learned policy $π$ 选择了 behavior policy $μ$ 很少选择的 action：

$$
π(a|s) \not\approx μ(a|s)
$$

但更严谨地说，offline RL 里的 distribution shift 是 **state-action distribution shift**：

$$
d^π(s,a)\neq d^D(s,a)
$$

展开来看：

$$
d^π(s,a) = d^π(s)π(a|s)
$$

而 offline dataset 中的分布近似是：

$$
d^D(s,a) \approx d^μ(s)μ(a|s)
$$

因此 distribution shift 可以来自两个方面。

第一是 **action shift**：在同一个 state 下，$π(a|s)$ 和 $μ(a|s)$ 不一样。这是 offline RL 中最直接、最常讨论的问题。比如 dataset 中在 state $s$ 下几乎都是向左走，但 learned policy 因为某个 Q 值估计很高，开始选择向右走。这个向右走的 action 对当前 dataset 来说就是 OOD action。

第二是 **state shift**：如果 policy 选择了和 dataset 不一样的 actions，它可能会进入 dataset 很少访问过的新 states。也就是说，action OOD 可能进一步诱导 state OOD。

所以可以把 offline RL 的问题链条理解成：

$$
\pi(a\mid s) \not\approx \mu(a\mid s)
$$

$$
\Longrightarrow (s,a) \notin \mathrm{supp}\bigl(d^{\mathcal D}(s,a)\bigr)
$$

$$
\Longrightarrow Q(s,a) \text{ is unreliable and may be overestimated}
$$

$$
\Longrightarrow \pi \text{ is optimized toward bad OOD actions}
$$

$$
\Longrightarrow d^{\pi}(s,a) \text{ shifts further away from } d^{\mathcal D}(s,a)
$$

这里有一个容易混淆的点：**不是 Q 估计不准确之后才出现 distribution shift，而是 policy 和 dataset distribution 不匹配本身就是 distribution shift。**  Q 估计不准确是这个 shift 带来的后果；如果 Q 又刚好高估了 OOD actions，那么 policy optimization 会进一步放大这个问题。

## 3 为什么直接 Maximizing Q 会有问题？

在 online RL 里，如果 Q function 对某个 action 估计错了，agent 还有机会通过和环境交互拿到新的反馈，然后修正 Q function。但在 offline RL 里，数据集是固定的，policy 没有机会去环境里验证一个 OOD action 到底好不好。

这会导致一个典型问题：**extrapolation error**。也就是 Q function 在 dataset 没有覆盖的区域只能依靠函数近似进行 extrapolation，而这种外推估计可能是不准确的。

如果某个 OOD action 的 Q 被错误地高估：

$$
Q(s,a_\text{OOD}) > Q(s,a_\text{data})
$$

那么 actor 或 policy improvement step 就可能选择这个 OOD action。问题在于，这个 high Q 不是因为 action 真的好，而是因为 critic 没有足够数据监督，导致估计不可靠。

因此 offline RL 的核心不是简单地做：

$$
\text{maximize}\ Q(s,a)
$$

而是要做：

> maximize Q, but only when the Q estimate is trustworthy enough.

也可以说：

> improve the policy while staying within or near the dataset support.

这里就出现了 offline RL 最核心的 trade-off：

**如果太保守，policy 只是 behavior cloning，很稳定但很难超过 behavior policy；如果太激进，policy 会 exploit OOD Q errors，可能直接崩掉。**

## 4 常见解决思路

Offline RL 的不同方法看起来很多，但大体上都在解决同一个问题：**如何控制 learned policy 和 dataset distribution 之间的 mismatch。**  常见思路可以分成三类。

第一类是 **policy constraint / behavior regularization**。这类方法直接限制 learned policy 不要离 behavior policy 太远：

$$
\max_{\pi} \; \mathbb{E}_{s\sim \mathcal{D},\, a\sim \pi(\cdot\mid s)}\left[Q(s,a)\right]
\quad \text{s.t.} \quad
D\left(\pi(\cdot\mid s),\mu(\cdot\mid s)\right) \leq \epsilon
$$

这里的 $D$ 可以是 KL divergence、MMD、Wasserstein distance，或者 action-space distance。也可以把 constraint 写成 regularization：

$$
\max_{\pi} \; \mathbb{E}_{s\sim \mathcal{D},\, a\sim \pi(\cdot\mid s)}\left[Q(s,a)\right]
- \alpha D\left(\pi(\cdot\mid s),\mu(\cdot\mid s)\right)
$$

直觉很简单：policy 可以 improve，但不要离 dataset 太远。像 BEAR、BRAC、TD3+BC 都可以从这个角度理解。TD3+BC 甚至可以看成是在 actor objective 里加了 behavior cloning regularization，让 actor 不要为了追求 high Q 而输出太 OOD 的 action。

第二类是 **support constraint**。它比 policy distance constraint 更直接：不是说 policy 和 behavior policy 的整体距离要小，而是要求 policy 选择的 action 必须在 dataset support 里面。

$$
a \in \mathrm{supp}\left(\mathcal{D}(\cdot\mid s)\right)
$$

更严格地说，可以写成：

$$
\mathrm{supp}\left(\pi(\cdot\mid s)\right)
\subseteq
\mathrm{supp}\left(\mathcal{D}(\cdot\mid s)\right)
$$

离散 action 下，这很容易理解：如果某个 state 下 dataset 只出现过 $a_1$ 和 $a_2$，那么 policy 就不应该选择 $a_3$。连续 action 下更复杂，因为精确的某个 action 几乎不会重复出现，所以通常需要近似 support。例如学习一个 behavior model 或 generative model，只生成 dataset-like actions，然后在这些 candidate actions 里选 Q 最高的。BCQ 就是这种思想：先从生成模型中采样接近 dataset 的 actions，再允许一个小范围 perturbation，而不是让 actor 在整个 action space 里自由搜索。

第三类是 **conservative value learning / pessimism**。这类方法不一定直接约束 policy，而是让 Q function 对 OOD actions 更保守：

$$
Q(s,a_{\mathrm{OOD}}) \downarrow
$$

典型代表是 CQL。CQL 的核心思想是：压低 dataset 外 actions 的 Q 值，同时相对保留 dataset actions 的 Q 值。这样即使 policy optimization 会寻找 high-Q actions，它也不容易被虚高的 OOD Q-value 吸引。

这类方法的 insight 是：**有时候不需要显式禁止 policy 选择 OOD action，只要让 OOD action 的 Q 值足够低，policy 自然就不会选择它。**

还有一类方法，比如 AWR、AWAC、IQL，可以理解成 **advantage-weighted behavior cloning**，也带有 implicit support constraint 的味道。它们不是在整个 action space 中做

$$
\arg\max_a Q(s,a)
$$

而是在 dataset actions 上做 weighted imitation：

$$
\max_{\pi} \; \mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[w(s,a)\log \pi(a\mid s)\right]
$$

其中 $w(s,a)$ 通常和 advantage 有关。直觉是：dataset 里也有好 action 和坏 action，我们不需要模仿所有 action，而是更多模仿 advantage 高的 action。由于训练信号只来自 dataset actions，这类方法天然减少了对 OOD actions 的依赖，因此通常会更稳定。

## 5 统一理解

Offline RL 的方法虽然形式不同，但可以统一理解成一句话：

> Offline RL is about **reliable** policy improvement under dataset support constraints.

Policy constraint 是从 policy side 控制 distribution shift：让 $π$ 不要离 $μ$ 太远。

Support constraint 是从 action space 控制 distribution shift：只允许 policy 在 dataset 支持的 action 区域里做 improvement。

Conservative value learning 是从 value side 控制 distribution shift：让 OOD actions 的 Q 值变低，避免 policy 被错误的 Q 高估吸引。

Advantage-weighted BC 则是一种更隐式的方式：不去 OOD 区域里 search action，而是在 dataset 内部挑更好的 actions 来模仿。

所以 offline RL 的关键不是“怎么最大化 Q”，而是“什么时候 Q 是可信的”。如果 Q 在 dataset support 内比较可信，那么 policy improvement 是有意义的；如果 Q 来自 OOD 区域的外推估计，那么最大化它反而可能让 policy 变差。

因此，一个更自然的总结是：offline RL 里的 optimization 不是在整个 action space 里无约束地找最大 Q，而是在数据支持的区域内做可靠的 improvement。不同方法只是把“可靠”落实在不同位置：policy、action support、value estimation，或者 dataset action weighting。


## 6 为什么generative policy很适合offline RL？

从上面的角度看，generative policy 的吸引力很自然：它不是让 actor 在整个 action space 里直接搜索 high-Q action，而是先建模 offline dataset 中的 action 或 trajectory distribution。这样采样出来的 candidates 通常更接近 behavior policy 访问过的高密度区域。

这给 policy improvement 加了一层 implicit support constraint：

$$
\text{policy improvement over data-supported actions}
$$

而不是：

$$
\text{unconstrained policy improvement over all actions}
$$

但这并不意味着 generative policy 自动解决 offline RL。单纯学习数据分布，本质上更接近 behavior cloning：它能告诉我们“哪些 actions 像数据”，但不能告诉我们“哪些 data-supported actions 更好”。

所以 generative policy 在 offline RL 里通常还需要 return、value 或 advantage signal。生成模型负责把 search space 限制在 dataset support 附近，value / advantage guidance 负责在这个区域里偏向更优的 actions 或 trajectories。

换句话说，generative policy 的价值不只是表达能力更强，而是它把 offline RL 中最难处理的 unconstrained action search，变成了更可控的 data-supported candidate generation。
