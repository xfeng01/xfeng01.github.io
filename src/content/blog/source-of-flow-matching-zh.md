---
title: "Does a Better Source Distribution Really Help Flow Matching?"
description: "Some thinking on the source distribution of flow matching."
date: 2026-11-10
lang: zh
translationKey: source-of-flow-matching
tags:
  - Flow Matching
---


之前思考的一个问题：**source distribution 的选择能在多大程度上改善 flow matching？**

一个很自然的想法是：如果 source distribution 更接近 target distribution，那么 transport 应该更简单，生成时也应该需要更少的 sampling steps。尤其是在 sequential generation 中，相邻状态通常具有很强的相关性。因此，相比每次从 Gaussian noise 开始生成，使用上一轮的预测分布作为 source 似乎更加合理。

但经过和友人H讨论以及仔细分析后，我发现这个想法并不一定成立。至少在 **independent coupling + linear interpolation** 的设定下，source 和 target 的距离并不能直接反映 transport 的难度，也不能决定少步生成的误差。

## 1 Problem Setup

考虑标准的 conditional flow matching：

$$
X_0\sim Q,\quad X_1\sim P,\quad X_0\perp X_1,
$$

其中 $Q$ 是 source distribution，$P$ 是 target distribution。

使用 linear interpolation：

$$
X_t=(1-t)X_0+tX_1.
$$

对应的 optimal velocity field 为：

$$
u_t(x)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

这里先做两个假设：

1. 模型能够学习到精确的 velocity field。
2. 使用 Euler method 进行离散积分。

这样可以先排除 neural network approximation error，只研究 source distribution 对 numerical integration error 的影响。

如果使用精确的 ODE solver，那么无论选择什么 source distribution，最终的分布都应该是 $P$（假设相应的 ODE flow 存在且能够正确输运）。

因此，source selection 的潜在优势主要体现在**有限步积分**以及**实际网络学习**上。

## 2 NFE = 1: Source Distribution Does Not Matter

先考虑最简单的情况，只使用一步 Euler：

$$
Y_1=Y_0+u_0(Y_0),\quad Y_0\sim Q.
$$

由于 source 和 target 独立：

$$
\begin{aligned}
u_0(x)
&=\mathbb{E}[X_1-X_0\mid X_0=x]\\
&=\mathbb{E}[X_1]-x.
\end{aligned}
$$

因此：

$$
Y_1=Y_0+\mathbb{E}[X_1]-Y_0
=\mathbb{E}[X_1].
$$

也就是说：

> 无论 source distribution 是什么，一步 Euler 最终都会得到 target distribution 的均值。

对应的 Wasserstein error 为：

$$
W_2^2(\mathcal{L}(Y_1),P)
=\operatorname{Tr}(\operatorname{Cov}(P)).
$$

这个结果与 $Q$ 完全无关。即使 $Q$ 与 $P$ 非常接近，甚至 $Q=P$，一步 Euler 仍然只会输出 target mean。

所以，这个结论让我开始思考：**让 source distribution 更接近 target，是否真的能让 flow matching 更容易生成？**

当然，一步 Euler 是一个极端情况，我们继续考虑两步。

## 3 NFE = 2: Closer Does Not Mean Better

考虑两个等长的 Euler steps $0\rightarrow \frac12\rightarrow 1$, 第一步：

$$
Y_{1/2}=Y_0+\frac12u_0(Y_0)
=\frac{Y_0+m_P}{2},
$$

其中 $m_P=\mathbb{E}[X_1]$。

定义：

$$
M(w)=\mathbb{E}[X_1\mid X_0+X_1=w].
$$

在 $t=1/2$ 时，有：

$$
u_{1/2}(x)=2M(2x)-2x.
$$

代入第二步 Euler：

$$
\begin{aligned}
Y_1
&=Y_{1/2}+\frac12u_{1/2}(Y_{1/2})\\
&=M(2Y_{1/2})\\
&=M(Y_0+m_P).
\end{aligned}
$$

因此，两步 Euler 的最终输出为：

$$
\boxed{Y_1=M(Y_0+m_P)}
$$

这意味着 source distribution 的选择确实会影响生成结果，但影响来自两个方面：

- Source 与 target 共同决定的 posterior mean function $M$。
- Source samples 在 $M$ 上形成的 query distribution。

而不是简单由 $W_2(Q,P)$ 决定。

### 一个反直觉的例子：Source = Target

假设：

$$
Q=P.
$$

由于 $X_0$ 和 $X_1$ 独立同分布，可以利用 exchangeability 得到：

$$
\mathbb{E}[X_1\mid X_0+X_1=w]=\frac{w}{2}.
$$

因此：

$$
M(w)=\frac{w}{2}.
$$

最终两步 Euler 的输出为：

$$
Y_1=\frac{Y_0+m_P}{2}.
$$

这实际上会将整个分布向均值收缩。如果 $P$ 是一个均值为 0 的 bimodal distribution，那么生成结果只是将原分布缩小了一半，而不是重新得到 $P$。

> 换句话说：**即使 source 已经等于 target，independent coupling 仍然可能构造出一个不必要的、非平凡的 transport process，并且少步 Euler 仍然会产生明显误差。**

实际上，由于 exchangeability：

$$
u_{1/2}(x)=0.
$$

但是这不代表整个 velocity field 都是零。对于 independent coupling，intermediate distributions 通常仍然会发生变化。

例如，当 $Q=P=\mathcal{N}(0,1)$ 时：

$$
X_t\sim\mathcal{N}(0,(1-t)^2+t^2).
$$

这个分布会先收缩再扩张，尽管 source 和 target 完全相同。

对应的 velocity field 是：

$$
u_t(x)=
\frac{2t-1}{(1-t)^2+t^2}x.
$$

因此，一个本来可以不移动任何样本就完成的任务，在 independent coupling 下反而需要先收缩、再扩张。

### 更远的 Source 反而可以更好

还可以构造更强的反例。考虑一个 bimodal target：

$$
P_\epsilon=
\frac12\mathcal{N}(-1,\epsilon^2)
+\frac12\mathcal{N}(1,\epsilon^2).
$$

比较两个 source：

$$
Q_1=P_\epsilon,
$$

以及

$$
Q_2=(D_{1/2})_\#P_\epsilon,
\qquad D_{1/2}(x)=x/2.
$$

其中 $Q_2$ 就是将 $P_\epsilon$ 缩小一半。显然：

$$
W_2(Q_1,P_\epsilon)=0,
$$

而

$$
W_2(Q_2,P_\epsilon)
=\frac12\sqrt{1+\epsilon^2}>0.
$$

但对于两步 Euler：

- 使用 $Q_1$，由于 source 和 target 同分布，最终输出为 $Q_2$，因此生成误差为 $\frac12\sqrt{1+\epsilon^2}$。
- 使用 $Q_2$，当 $\epsilon\rightarrow0$ 时，两步 Euler 的生成误差却趋近于 0。

也就是说：

$$
\boxed{
W_2(Q_1,P)<W_2(Q_2,P)
}
$$

但同时：

$$
\boxed{
D_2(Q_1,P)>D_2(Q_2,P)
}
$$

这里 $D_2$ 表示精确 velocity field 下，两步 Euler 输出与目标分布的 Wasserstein distance。

这个反例说明：**source 和 target 越接近，并不意味着少步 Euler 的生成误差越小。** 对于 $Q_1=P$，exchangeability 使中点 velocity 为零；而对于 $Q_2$，posterior mean 能够利用双峰结构，将 source samples 更准确地映射回 target modes。

所以真正起作用的是 velocity field 的结构，而不是 source-target distance 本身。

## 4 What About Sequential Generation?

我最初考虑这个问题，是因为 sequential generation 中存在一个非常intuitive的想法：如果相邻状态高度相关，那么上一轮预测分布应该与当前 target 很接近。因此，直接复用上一轮预测作为 source，可能比从 Gaussian noise 开始更有效。

例如，一个缓慢变化的物理系统：

$$
S_{n+1}=S_n+\epsilon_n,
$$

其中 $\epsilon_n$ 很小。从真实 dynamics 来看，预测下一状态可能只需要做一个很小的更新。

但是这里容易混淆两个不同的概念：

1. 真实状态之间的 temporal coupling，即 $(S_n,S_{n+1})$ 的联合分布。
2. Flow matching 训练时人为构造的 source-target coupling。

如果训练时仍然采用：

$$
X_0\sim Q_n,\quad X_1\sim P_n,
\quad X_0\perp X_1,
$$

那么模型实际上并没有利用真实状态之间的 sample-level correspondence。

即使 $Q_n$ 和 $P_n$ 很接近，甚至完全相同，independent coupling 仍然会重新随机配对两个分布中的样本。前面的 $Q=P$ 例子已经说明，这种情况下 transport 并不一定简单。

更进一步，还可以构造一个 highly correlated sequential process，使得：

$$
\mathbb{E}\|S_{n+1}-S_n\|^2\rightarrow0
$$

并且历史预测分布满足：

$$
W_2(Q_n^{\mathrm{hist}},P_n)\rightarrow0.
$$

但是在两步 Euler 下，使用历史预测分布仍然可以比使用 standard Gaussian source 产生更大的生成误差。

> 这说明，**真实 dynamics 中很强的 temporal correlation，并不会自动转化为 independent flow matching 下的少步生成优势。**
> 关键在于, 如果这种 correlation 没有进入 coupling，flow matching 只看到了两个 marginal distributions，并没有真正利用状态之间的对应关系。

## 5 Maybe Coupling Matters More Than Source Distance

一个值得思考的例子是，当 $Q=P$ 时，比较两种 coupling。

**Independent coupling:**

$$
X_0\sim P,\quad X_1\sim P,\quad X_0\perp X_1.
$$

在这种情况下，即使 source 和 target 相同，intermediate distributions 仍然可能变化，velocity field 也不一定为零。

**Identity coupling:**

$$
X_1=X_0,\quad X_0\sim P.
$$

此时：

$$
X_t=X_0
$$

并且：

$$
u_t(x)=0.
$$

因此，使用任意步数的 Euler 都不会产生离散误差。这两种设定具有完全相同的 source 和 target，但得到的 transport path 完全不同。当然，identity coupling 在这里主要是一个理论对照。如果我们本来就能直接采样 $P$，就没有必要再训练生成模型。实际生成中，构造这种 coupling 并不一定可行。

不过它说明了一个重要问题：**仅仅知道 source 和 target 的 marginal distributions，可能不足以判断 transport 的难度；coupling 本身同样重要。**

对于 sequential generation，如果我们希望利用历史状态的信息，可能需要考虑如何将真实的 temporal correspondence 引入 transport construction，而不是简单地把历史预测分布作为新的 source。

## 6 What Does This Mean for Learnability?

前面的分析有一个很重要的限制：我们一直假设 velocity field 是精确的。因此，我们分析的是 numerical integration error，而不是实际训练神经网络时的 learnability。现实中，还需要考虑 velocity field 是否容易被有限容量的神经网络拟合。

对于 flow matching 的 regression objective：

$$
\mathcal{L}
=\mathbb{E}\left[
\|v_\theta(t,X_t)-(X_1-X_0)\|^2
\right].
$$

定义：

$$
u_t(x)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

在平方损失下，可以将 objective 分解为：

$$
\begin{aligned}
\mathcal{L}
=&\ \mathbb{E}\left[
\|v_\theta(t,X_t)-u_t(X_t)\|^2
\right]\\
&+\mathbb{E}\left[
\operatorname{Tr}\operatorname{Cov}
(X_1-X_0\mid t,X_t)
\right].
\end{aligned}
$$

第一项对应模型对真实 velocity field 的 approximation error。第二项则是 conditional velocity target 本身的不可约方差。

这也意味着 source 和 coupling 的选择至少可能从两个方面影响训练：

1. **Velocity field complexity.** 不同的 source/coupling 会产生不同的 velocity field，影响其 smoothness、curvature，以及神经网络拟合的难度。
2. **Regression target variance.** 即使 optimal velocity field 很简单，训练样本中的 conditional velocity variance 仍然可能很大，从而影响有限数据和有限计算下的优化过程。

但需要强调，较低的 regression target variance 并不直接等价于更好的 generation quality。同样，较小的 velocity field curvature 也不一定能够单独保证更小的 Euler error。

真正值得分析的应该是这些因素如何共同影响有限容量、有限训练计算和有限 NFE 下的生成误差。

## 7 Some Thoughts

目前的分析让我对单纯的 source selection 有些怀疑。如果一个方法的主要 motivation 是：

> Historical source is closer to target, so flow matching should require fewer steps.

那么这个逻辑至少在 independent coupling 的设定下并不充分。因为：

- NFE = 1 时，source selection 不改变最终输出。
- NFE = 2 时，source selection 会影响误差，但 source-target distance 不能决定误差大小。
- 即使 source = target，independent coupling 仍然可能产生非平凡的 transport。
- 即使真实相邻状态高度相关，independent coupling 也不一定能够利用这种相关性。

这并不是说 source selection 没有意义。Source 确实会改变 probability path 和 velocity field，也可能影响实际模型的 learnability 和少步生成效率。

但我认为，更值得研究的问题应该是：source distributions, couplings 以及probability paths怎么共同影响learnability和flow matching的efficiency？我其实感觉，怎么利用distribution之间的correspondence或者说source和target distribution的特征来构造更加容易学习、更适合少步积分的transport path，可能更加有趣也更加有意义一点。至少对于目前我所关注的continuous DLM的问题来讲，这个也更加重要一点。
