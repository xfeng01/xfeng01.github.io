---
title: "Some Basic Thoughts on Consistency in Continuous and Discrete Diffusion Language Models"
description: ""
date: 2026-06-02
tags:
  - Diffusion Language Models
---



因为最近也刚开始做continuous diffusion language model，所以自然的一个问题是：continuous diffusion language model 和 discrete diffusion language model 的本质区别到底是什么？

直觉上，continuous DLM 可以更好地解决一致性问题，而 discrete DLM 不太行。比如一个很简单的例子：

> The capital of X is Y.

如果模型要同时生成 `X` 和 `Y`，那么它不能随便生成。`X` 和 `Y` 之间是有强依赖关系的。如果 `X = France`，那么 `Y` 应该是 `Paris`；如果 `X = Japan`，那么 `Y` 应该是 `Tokyo`。

对于 discrete masked diffusion 来说，这个问题可能需要两步。第一步先把 `X` 填出来：

> The capital of France is [MASK].

然后第二步再根据已经确定的 `France` 去填 `Paris`。

但是 continuous diffusion 似乎可以在一次 denoising 过程中同时把这两个位置往一致的方向推。也就是说，它可以让 `X` 和 `Y` 在同一个连续空间里一起演化，最后一起落到一个一致的 pair 上。

这背后的区别到底是什么？

我们首先看 backbone。

不管是 continuous DLM 还是 discrete DLM，内部结构其实都可以是类似的，比如都可以用 DiT，也就是都可以通过 attention 看到其他 token。所以问题不是说 discrete 看不到全局信息，而 continuous 看得到全局信息。

真正的区别在于：它们 denoise 的对象不一样。

对于 discrete DLM 来说，它本质上是在做 conditional infilling 或者 recovering。当前状态通常是一个离散序列，每个位置要么是一个已经确定的 token，要么是一个 `[MASK]`。模型的目标是根据已有 context，把 `[MASK]` 的位置恢复成原来的 token。

也就是说，对于某一个 `[MASK]` 位置来说，它当然可以通过 attention 看到其他位置也是 `[MASK]`，但是它并不知道其他 `[MASK]` 最后会被采样成什么 token。

这个地方很关键。

attention 只保证当前 token 的 logits 可以依赖全局 context，但是它不保证同一步采样出来的多个 token 之间真的互相依赖。

换句话说，attention 让每个位置的预测都看到了全局状态，但如果最后的采样分布还是写成每个位置独立的 categorical distribution，那么同一步生成的多个 token 之间仍然可能是 factorized 的。



形式上，如果记当前被 mask 的位置为 $\mathcal{M}_t$，未被 mask 的位置为 $\bar{\mathcal{M}}_t$，那么 reverse step 真正需要预测的是 masked subset：

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
$$

而常见的 discrete masked diffusion 会把这个 masked subset 的 joint distribution 近似写成 per-position 的乘积形式：

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
\approx
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

如果要写成整个 sequence 的形式，那么未被 mask 的位置其实是直接保留的，因此可以写成：

$$
p_\theta(x_0 \mid x_t)
\approx
\mathbf{1}_{x_{0,\bar{\mathcal{M}}_t}=x_{t,\bar{\mathcal{M}}_t}}
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

这里的重点是：每一个因子 $p_\theta(x_{0,i} \mid x_t)$ 都可以依赖完整的 $x_t$，所以模型并不是看不到全局 context；但在同一步采样时，masked positions 之间的输出仍然被写成了乘积形式。


但是对于：

> The capital of X is Y.

这种情况，我们真正想要的并不是：

$$
p(X,Y \mid c) = p(X \mid c)p(Y \mid c)
$$

而是一个 joint distribution：

$$
p(X,Y \mid c)
$$

因为 $X$ 和 $Y$ 之间有强依赖关系。一旦 $X$ 被采样成 France，那么 $Y$ 就应该更倾向于 Paris；一旦 $X$ 被采样成 Japan，那么 $Y$ 就应该更倾向于 Tokyo。

所以在：

> The capital of [MASK] is [MASK].

里面，两个 `[MASK]` 的 logits 都可以看到整句话，也都知道另一个位置是 `[MASK]`。但是当它们在同一步被采样时，第二个位置并不知道第一个位置最后到底被采样成了 `France`、`Japan`，还是 `China`。

这就会带来一致性问题。

所以 discrete masked DLM 的常见解决方式，其实就是通过多步 refinement，把这个 simultaneous dependency 变成 sequential dependency。先生成一部分，再用已经生成出来的 token 作为新的 context 去生成另一部分。这样 context 会越来越多，信息会越来越多，每一个 `[MASK]` 也会越来越好恢复。

但是这个过程本质上还是：一个位置要么是 `[MASK]`，要么已经被 commit 成某个 token。它的状态是比较硬的。

continuous DLM 不一样。

continuous diffusion denoise 的不是一个个离散 token，而是整个 embedding space，或者说整个 sequence latent matrix。假设整句话的 hidden representation 是：

$$
X_t \in \mathbb{R}^{L \times d}
$$

那么 reverse process 可以理解成类似：

$$
\frac{dX_t}{dt} = f_\theta(X_t, t)
$$

这里的 $X_t$ 不是某一个 token，而是整句话所有位置的连续表示。每一个位置的更新都可以依赖整个 $X_t$。所以它不是单独把某个 `[MASK]` 映射回 token，而是在对整个句子的 embedding matrix 做 global denoising。

这就带来了一个很重要的区别。

在 continuous diffusion 里面，每个位置从一开始到最后都不是一个硬 token，而是一个连续向量。它可以携带一些“软的、尚未离散化的假设”。

比如在生成：

> The capital of X is Y.

的时候，`X` 位置和 `Y` 位置不需要一开始就分别独立地决定自己是什么 token。它们可以在连续空间里共同演化。整个 latent state 可以逐渐靠近：

$$
(\text{France-like latent}, \text{Paris-like latent})
$$

或者：

$$
(\text{Japan-like latent}, \text{Tokyo-like latent})
$$

也就是说，一致性不是等到某个 token 被采样出来之后再修正，而是在最终离散化之前，就可以在连续空间里形成一个整体的、同步的语义假设。

这也是我觉得 continuous DLM 更自然的地方。

discrete diffusion 的 denoising 更像是：当前某些位置是 `[MASK]`，然后模型尝试把这些 `[MASK]` 恢复成原始 token。整个过程中，context 会随着 token 被填入而变得越来越完整。

而 continuous diffusion 的 denoising 更像是：每个位置的 embedding 都在逐渐去噪。每个位置的信息不是 0 或 1，不是 `[MASK]` 或 token，而是一个连续变化的向量。它可以在中间状态表达不确定性、相关性和全局语义结构。只有到最后一步，模型才需要把这些连续表示离散化成 token。

所以我觉得 continuous 和 discrete 的区别可以这样理解：

discrete DLM 的中间状态是硬的。一个位置要么是 unknown，要么已经 commit 成某个 token。

continuous DLM 的中间状态是软的。一个位置可以在 embedding space 里逐渐移动，并且和其他位置一起形成一个 joint semantic state。

这也是为什么 continuous DLM 更容易处理全局一致性。

不过这里也需要小心一点。不能说 continuous DLM 天然保证一致性。更准确地说，continuous DLM 的连续状态允许多个位置在最终离散化之前共同演化，所以它更容易表达同步一致性。

它的向量场或者 score 是定义在整个 sequence latent $X_t$ 上的，因此它更自然地建模 joint evolution。但是一致性最终能不能真的做好，仍然取决于 latent representation、decoder、training objective，以及最后的 discretization。

如果 latent space 本身没有学好，或者 decoder 最后仍然是很强的 per-position independent decoding，那么 continuous diffusion 也可能出现不一致的问题。

所以我们不能说：

> continuous 一定能解决一致性，而 discrete 一定不能。

我更倾向于：

> discrete masked DLM 的常见形式容易遇到 factorization barrier；continuous DLM 因为在连续空间中进行 global denoising，所以机制上更适合表达 global consistency。

这算是我现在对 continuous DLM 的一些初步理解。

它真正有意思的地方，不只是把 token 换成 embedding，也不是简单地把 discrete diffusion 改成 continuous diffusion，而是它把 language generation 从“逐个位置恢复 token”变成了“整个 sequence latent 的共同演化”。

这可能才是 continuous DLM 和 discrete DLM 最本质的区别。
