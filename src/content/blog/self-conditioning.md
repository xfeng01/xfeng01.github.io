---
title: "Some Basic Thoughts on Self-Conditioning"
description: "A note on self-conditioning."
date: 2026-07-20
lang: zh
translationKey: self-conditioning
tags:
  - Diffusion
---



Self-conditioning 最早由 [Analog Bits](https://arxiv.org/abs/2208.04202) 提出。简单来说，原论文的训练方式是：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
$$

先在相同的 $x_t$ 和 $t$ 下得到一次 prediction：

$$
m=\operatorname{sg}\left[f_\theta(x_t,0,t)\right],
$$

再把它作为 condition 重新预测：

$$
\hat{x}_0=f_\theta(x_t,m,t),
$$

最后计算：

$$
\mathcal{L}=\|\hat{x}_0-x_0\|^2.
$$

推理时则是：

$$
m_k=f_\theta(z_{t_k},m_{k-1},t_k),
$$

也就是把上一个 timestep 的 prediction 作为当前 timestep 的 condition。

所以，self-conditioning 的 training 和 inference 其实没有完全对齐。训练时使用的是当前 timestep 下的第一次 prediction，而推理时使用的是上一个 timestep 留下来的 prediction：

$$
m_{k-1}\neq f_\theta(z_{t_k},0,t_k).
$$

当然，也可以在 inference 的每一步先做一次 unconditional prediction，再用它做 self-conditioning：

$$
a_k=f_\theta(z_{t_k},0,t_k),
$$

$$
m_k=f_\theta(z_{t_k},a_k,t_k).
$$

这样 training 和 inference 就对齐了，但每一步需要两次 forward。所以常见的做法更像是一种 empirical trick：直接复用 previous prediction，减少计算成本。

[Self-conditioned Flow Map Language Models via Fixed-point Flows](https://arxiv.org/abs/2607.00714) 将它解释成一种 fixed-point iteration。

固定 $x_t$ 和 $t$，定义：

$$
T(m)=f_\theta(x_t,m,t).
$$

理想情况下不断迭代：

$$
m^{(0)}=0,
$$

$$
m^{(j+1)}=T(m^{(j)}),
$$

最终收敛到：

$$
m^\star=T(m^\star).
$$

也就是说，当 prediction 已经稳定以后，再把它输入 model，结果基本不会继续变化。

不过，vanilla self-conditioning 的 loss 并不保证这个过程真的会收敛，也不保证第二次 prediction 一定比第一次更好。它只要求：

$$
f_\theta(x_t,m,t)\approx x_0,
$$

但没有要求：

$$
\|f_\theta(x_t,m,t)-x_0\|
<
\|m-x_0\|.
$$

所以 model 可能真的在做 refinement，也可能只是直接复制之前的 prediction。

我目前对 self-conditioning 的理解是：它并不是给 model 提供新的信息。因为这个 condition 本身也是 model 根据 $x_t$ 预测出来的，所以它并没有额外告诉 model 真实的 $x_0$ 是什么。

它也不是一个 critic 或 evaluator，不是在判断前面的 prediction 好不好，然后再决定怎么修改。

它更像是先让 model 有一个当前的 guess。之后再预测时，model 不需要完全从头开始，而是在这个 guess 的基础上继续预测。

所以它主要解决的不是“信息不够”，而是“prediction 不稳定”。

比如当前 $x_t$ 可能同时对应 token A 和 token B。没有 self-conditioning 时，model 每一步都可能重新选择：

$$
A\rightarrow B\rightarrow A\rightarrow B.
$$

加上 self-conditioning 后，如果前一步已经比较偏向 A，后面通常也更容易继续沿着 A 走。

所以它并不是真的降低了 posterior uncertainty，而是减少 model 在不同 mode 之间反复摇摆，让 sampling trajectory 更稳定。

对于 language 这种离散数据，这个作用可能更加明显。

如果只使用 embedding MSE，假设 token A 和 token B 都有可能，那么最优 prediction 可能是：

$$
\frac{e_A+e_B}{2}.
$$

这个 embedding 其实谁都不是，只是两个 token embedding 的平均。

CE loss 则可以直接表示：

$$
p(A)=0.5,\qquad p(B)=0.5.
$$

它不需要把多个 mode 平均成一个中间 embedding，而是可以保留一个离散分布。

再结合 self-conditioning，如果第一步已经有一点偏向：

$$
p(A)=0.6,\qquad p(B)=0.4,
$$

后面可能进一步变成：

$$
p(A)=0.8,\qquad p(B)=0.2.
$$

因此，self-conditioning + CE 可能让 model 逐渐收束到某一个 token mode，而不是停留在多个 token 的平均位置。

不过，CE 本身并不会强制选择一个 mode，self-conditioning 也不保证一定会 sharpen。更准确地说，一旦前面已经产生了一点偏向，self-conditioning 会让这个偏向更容易被维持和放大。

如果前面的方向是对的，prediction 会越来越稳定；如果前面的方向是错的，也可能把错误提前锁死。

所以我目前会把 self-conditioning 理解成：

> 它不提供新信息，而是提供一个当前 hypothesis，减少 mode switching。

对于离散数据，CE 负责避免用一个连续平均值表示多个离散 mode，而 self-conditioning 负责让已经形成的 mode 更容易被维持和继续收束。

**References:**

- [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202)
- [Self-conditioned Flow Map Language Models via Fixed-point Flows](https://arxiv.org/abs/2607.00714)

