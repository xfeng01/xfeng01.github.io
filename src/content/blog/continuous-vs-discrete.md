---
title: "Some Basic Thoughts on Consistency in Continuous and Discrete Diffusion Language Models"
description: "A note on why continuous diffusion language models may express global consistency more naturally than discrete masked diffusion."
date: 2026-06-02
lang: zh
translationKey: continuous-vs-discrete
tags:
  - Diffusion Language Models
---

最近刚开始做 continuous diffusion language model，一个自然的问题是：continuous DLM 和 discrete DLM 的本质区别是什么？直觉上，continuous DLM 的机制似乎更适合处理一致性问题。比如：

> The capital of X is Y.

如果模型要同时生成 `X` 和 `Y`，它们不能各自独立决定。`X = France` 时，`Y` 应该是 `Paris`；`X = Japan` 时，`Y` 应该是 `Tokyo`。对于 discrete masked diffusion，这通常需要多步完成：先填出 `X`，

> The capital of France is [MASK].

再根据已经确定的 `France` 填 `Paris`。而 continuous diffusion 可以在一次 denoising 过程中，同时把两个位置往一致的方向推：`X` 和 `Y` 在同一个连续空间里共同演化，最后落到一个一致的 pair 上。

这背后的区别不是 backbone。continuous DLM 和 discrete DLM 内部都可以用类似的结构，比如 DiT。它们都可以通过 attention 看到全局 token。所以不是 discrete 看不到全局信息，而 continuous 看得到。真正的区别在于 denoise 的对象。

对于 discrete DLM，当前状态通常是一个离散序列：每个位置要么是已经确定的 token，要么是 `[MASK]`。模型根据已有 context，把 `[MASK]` 恢复成原来的 token。这意味着，一个 `[MASK]` 位置虽然可以通过 attention 看到其他位置也是 `[MASK]`，但它不知道这些 `[MASK]` 最后会被采样成什么。attention 只保证每个位置的 logits 可以依赖全局状态，不保证同一步采样出来的多个 token 之间真的互相依赖。

形式上，如果当前被 mask 的位置是 $\mathcal{M}_t$，未被 mask 的位置是 $\bar{\mathcal{M}}_t$，reverse step 真正需要预测的是 masked subset：

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
$$

而常见的 discrete masked diffusion 会把这个 joint distribution 近似成 per-position 的乘积：

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
\approx
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

如果写成整个 sequence，未被 mask 的位置直接保留：

$$
p_\theta(x_0 \mid x_t)
\approx
\mathbf{1}_{x_{0,\bar{\mathcal{M}}_t}=x_{t,\bar{\mathcal{M}}_t}}
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

这里的重点是：每个因子 $p_\theta(x_{0,i} \mid x_t)$ 都可以依赖完整的 $x_t$，所以模型并不是缺少全局 context；问题是同一步采样时，masked positions 的输出仍然被写成了乘积形式。

回到 capital 这个例子，我们真正想要的不是

$$
p(X,Y \mid c) = p(X \mid c) \cdot p(Y \mid c)
$$

而是 joint distribution：

$$
p(X,Y \mid c)
$$

因为一旦 `X` 被采样成 `France`，`Y` 就应该更倾向于 `Paris`；一旦 `X` 是 `Japan`，`Y` 就应该更倾向于 `Tokyo`。在

> The capital of [MASK] is [MASK].

里面，两个 `[MASK]` 的 logits 都能看到整句话，也都知道另一个位置是 `[MASK]`。但如果它们在同一步被采样，第二个位置并不知道第一个位置最终是 `France`、`Japan`，还是 `China`。这就是一致性问题的来源。因此，discrete masked DLM 的常见做法，是用多步 refinement 把 simultaneous dependency 变成 sequential dependency：先生成一部分，再把已经生成的 token 作为新 context 生成另一部分。context 越来越完整，恢复也越来越容易。但这个过程里的状态仍然是 hard/discrete 的：一个位置要么是 `[MASK]`，要么已经被 commit 成某个 token。


continuous diffusion 的情况不同：它 denoise 的不是离散 token，而是整个 embedding space，或者说整个 sequence latent matrix。假设整句话的 hidden representation 是

$$
X_t \in \mathbb{R}^{L \times d}
$$

reverse process 可以理解成

$$
\frac{dX_t}{dt} = f_\theta(X_t, t)
$$

这里的 $X_t$ 是整句话所有位置的连续表示。每个位置的更新都可以依赖整个 $X_t$，所以它不是把某个 `[MASK]` 单独映射回 token，而是在对整个 sequence latent 做 global denoising。关键是，continuous diffusion 的中间状态不是硬 token，而是连续向量；它可以携带“soft、尚未离散化的假设”。

还是以这句话为例：

> The capital of X is Y.

`X` 和 `Y` 不需要一开始就独立决定自己是什么 token。它们可以在连续空间里共同演化，使整个 latent state 逐渐靠近

$$
(\text{France-like latent}, \text{Paris-like latent})
$$

或者

$$
(\text{Japan-like latent}, \text{Tokyo-like latent})
$$

也就是说，一致性不必等某个 token 被采样出来之后再修正，而可以在最终离散化之前，就在连续空间里形成一个整体的语义假设。

所以我目前的理解是：discrete DLM 的中间状态是hard，一个位置要么 unknown，要么已经 commit 成某个 token；continuous DLM 的中间状态是soft，一个位置可以在 embedding space 里逐渐移动，并和其他位置一起形成 joint semantic state。这也是为什么 continuous DLM 机制上更适合处理全局一致性。

不过这里也要小心：continuous DLM 并不天然保证一致性。更准确地说，它的连续状态允许多个位置在最终离散化之前共同演化，因此 **更容易表达同步一致性**。
它的 vector field 或 score 定义在整个 sequence latent $X_t$ 上，所以更自然地建模 **joint evolution**。但一致性最终能不能做好，仍然取决于 latent representation、decoder、training objective，以及最后的 discretization。如果 latent space 没学好，或者 decoder 仍然是很强的 per-position independent decoding，continuous diffusion 也可能不一致。

所以我么不能说：

> continuous 一定能解决一致性，而 discrete 一定不能。

更准确的说法是：

> discrete masked DLM 的常见形式容易遇到 factorization barrier；continuous DLM 因为在连续空间中进行 global denoising，所以机制上更适合表达 global consistency。

这算是我现在对 continuous DLM 的一些初步理解。continuous DLM 真正有意思的地方，不只是把 token 换成 embedding，也不是简单地把 discrete diffusion 改成 continuous diffusion，而是把 language generation 从“逐个位置恢复 token”变成了“整个 sequence latent 的共同演化”。这可能才是 continuous DLM 和 discrete DLM 最本质的区别。
