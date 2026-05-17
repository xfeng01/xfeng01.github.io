---
title: "From Policy Improvement to PPO, GRPO, and DPO"
description: "A note on policy improvement for LLM post-training, from policy ratios and advantages to PPO clipping, GRPO group baselines, and DPO preference optimization."
date: 2025-12-22
lang: zh
translationKey: policy-improvement
tags:
  - LLM
  - Policy Optimization
---

## 1. Policy improvement 的目标

在 RL 里，policy improvement 的目标很直接：给定旧 policy $\pi_{\text{old}}$，学习一个更好的新 policy $\pi_\theta$。这里的“更好”指：在旧 policy 访问的 states 上，让新 policy 更倾向于选择高价值 actions。

一个常见的 policy improvement objective 可以写成：

$$
\max_{\pi_\theta}
\mathbb{E}_{s \sim d_{\pi_{\text{old}}},\, a \sim \pi_\theta(\cdot|s)}
\left[
Q_{\pi_{\text{old}}}(s,a)
\right]
$$

这里，$d_{\pi_{\text{old}}}$ 是旧 policy 诱导出的 state distribution。$Q_{\pi_{\text{old}}}(s,a)$ 表示在 state $s$ 下先选 action $a$，之后仍按旧 policy 执行时的 expected return。

之所以用 $Q_{\pi_{\text{old}}}$，是因为 policy improvement 通常是 **local improvement**：先固定旧 policy，用它的 value function 作为 reference，判断某个 action 是否比旧 policy 的平均行为更好。如果直接用 $Q_{\pi_\theta}$，target 会随着新 policy 一起变化，也很难用旧 samples 稳定估计。

因此，这个 objective 的意思是：

> 在旧 policy 遇到的 states 上，让新 policy 把更高概率分给相对旧 policy 更好的 actions。

实际优化时，通常不用 $Q$ 的绝对值，而是用 advantage：

$$
A_{\pi_{\text{old}}}(s,a)
=
Q_{\pi_{\text{old}}}(s,a)-V_{\pi_{\text{old}}}(s)
$$

于是 objective 变成：

$$
\max_{\pi_\theta}
\mathbb{E}_{s \sim d_{\pi_{\text{old}}},\, a \sim \pi_\theta(\cdot|s)}
\left[
A_{\pi_{\text{old}}}(s,a)
\right]
$$

但这里还有一个 mismatch：objective 里的 action 来自新 policy，而训练数据通常来自旧 policy。为了用旧 samples 估计新 policy 的 objective，需要 importance sampling：

$$
\mathbb{E}_{a \sim \pi_\theta}
\left[
A(s,a)
\right]
=
\mathbb{E}_{a \sim \pi_{\text{old}}}
\left[
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
A(s,a)
\right]
$$

由此得到 policy ratio：

$$
r(\theta)
=
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
$$

所以 policy improvement 的 surrogate objective 可以写成：

$$
\max_\theta
\mathbb{E}
\left[
r(\theta)A(s,a)
\right]
$$

## 2. PPO：在 policy ratio 上加 clipping，限制更新幅度

PPO 在上面的 surrogate 上加了一个限制：**新 policy 不能相对旧 policy 变化太大。**

原因是训练数据来自旧 policy。如果新 policy 更新太大，旧 samples 就不再能可靠代表新 policy，importance sampling 的估计也会变得不稳定。

所以 PPO 不直接优化：

$$
r_t(\theta)A_t
$$

而是使用 clipped surrogate objective：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\text{old}}(a_t|s_t)}
$$

clipping 的含义很简单：
- 如果 $A_t>0$，说明这个 action 好，新 policy 应该提高它的概率；但当 $r_t(\theta)>1+\epsilon$，概率已经提高太多，PPO 就不再继续奖励这个方向。
- 如果 $A_t<0$，说明这个 action 不好，新 policy 应该降低它的概率；但当 $r_t(\theta)<1-\epsilon$，概率已经降低太多，PPO 也会限制继续下降。

所以 PPO 的核心是：

> 用 ratio 表示新旧 policy 的概率变化，用 advantage 决定变化方向，再用 clipping 限制变化幅度。

### 2.1 PPO 在 LLM 里：action 是 token

在 LLM fine-tuning 里，语言模型通常被看成一个 **token-level policy**。给定 prompt $x$，模型生成回答：

$$
y=(y_1,y_2,\dots,y_T)
$$

在第 $t$ 步，state 是上下文 $s_t=(x,y_{<t})$，action 是当前 token $a_t=y_t$。因此 LLM policy 可以写成：

$$
\pi_\theta(y_t|x,y_{<t})
$$

整段回答的概率是：

$$
\pi_\theta(y|x)
=
\prod_{t=1}^T
\pi_\theta(y_t|x,y_{<t})
$$

所以 LLM PPO 实际上是在 token level 上做 policy improvement。对应的 ratio 是：

$$
r_t(\theta)
=
\frac{\pi_\theta(y_t|x,y_{<t})}
{\pi_{\text{old}}(y_t|x,y_{<t})}
$$

它表示：新模型相比旧模型，对第 $t$ 个 token 的概率提高或降低了多少。

这也解释了为什么通常不把“整段回答”当成一个不可分解的 action。理论上，整段回答的 ratio 可以写成：

$$
\frac{\pi_\theta(y|x)}
{\pi_{\text{old}}(y|x)}
=
\prod_{t=1}^T
\frac{\pi_\theta(y_t|x,y_{<t})}
{\pi_{\text{old}}(y_t|x,y_{<t})}
$$

但这个 product 对长度非常敏感，容易变得极大或极小，训练不稳定。而且整段回答作为 action 会让 credit assignment 更粗：一个回答错了，不代表每个 token 都错了。因此 LLM PPO 通常在 token level 上计算 ratio 和 advantage。

### 2.2 PPO 为什么需要 value model？

PPO 的更新依赖 token-level advantage：

$$
A_t = Q(s_t,a_t)-V(s_t)
$$

但在 LLM 里，reward 往往是 sequence-level 的。例如数学题通常是整段回答生成完后，根据最终答案是否正确给 reward：

$$
R(x,y)
$$

这时通常不会单独学习一个 $Q(s_t,a_t)$。对于一条 sampled trajectory，可以用 rollout return 近似 $Q$：

$$
Q(s_t,a_t) \approx G_t
$$

如果 reward 只在最后给，可以近似理解为：

$$
G_t \approx R(x,y)
$$

但直接用 final reward 更新所有 token，variance 会很大。所以 PPO 通常会学习一个 value function：

$$
V_\phi(s_t)
\approx
\mathbb{E}[G_t|s_t]
$$

然后构造 advantage：

$$
A_t = G_t - V_\phi(s_t)
$$

这里 $V_\phi(s_t)$ 的作用是 baseline：它估计在当前 prefix $s_t$ 下，模型继续生成平均能拿到多少 return。

所以更准确地说：

> PPO 通常不显式学习 $Q$-function，而是用 sampled return 作为 $Q$ 的估计；真正额外学习的是 $V$-function，用来把 raw return 转成低方差的 advantage。

这也是 PPO 在 LLM 里开销较大的原因之一：除了 policy model，通常还需要 value model / critic，有时还会涉及 reference model 和 reward model。

## 3. GRPO：用 group baseline 替代 value model

GRPO 可以看成是对 PPO 的一个简化：保留 policy ratio 和 PPO-style clipping，但去掉 learned value model。

PPO 中 advantage 的形式是：

$$
A_t = G_t - V(s_t)
$$

其中 $V(s_t)$ 需要额外训练。

GRPO 的想法是：很多 LLM reasoning 任务的 reward 是 sequence-level 的，而且可以对同一个 prompt sample 多个回答。于是，对同一个 prompt $x$，从旧 policy sample 一组 responses：

$$
y_1,y_2,\dots,y_G
$$

然后分别计算 reward：

$$
R_1,R_2,\dots,R_G
$$

接着用组内 reward 均值作为 baseline：

$$
A_i
=
R_i
-
\frac{1}{G}\sum_{j=1}^G R_j
$$

有时也会进一步标准化：

$$
A_i
=
\frac{
R_i-\mathrm{mean}(R_1,\dots,R_G)
}{
\mathrm{std}(R_1,\dots,R_G)
}
$$

这个 advantage 的含义是：

> 对同一个 prompt 来说，这个回答比同组其他回答好多少。

然后 GRPO 仍然可以使用类似 PPO 的 ratio objective。对于回答 $y_i$ 中的 token $y_{i,t}$，ratio 是：

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}|x,y_{i,<t})
}{
\pi_{\text{old}}(y_{i,t}|x,y_{i,<t})
}
$$

所以 GRPO 和 PPO 的主要区别在于 advantage 的估计方式：

- PPO：$A_t = G_t - V(s_t)$，需要额外训练 value model。
- GRPO：$A_i = R_i - \mathrm{mean}(R_1,\dots,R_G)$，不需要 value model。

因此，GRPO 的核心可以理解为：

> 仍然用 policy ratio 做 policy improvement，但不再训练 value function，而是用同一个 prompt 下多个回答的相对好坏来估计 advantage。

这也是为什么 GRPO 很适合 math/code 这类任务：这些任务通常可以对同一个 prompt 采样多个回答，并用 rule-based verifier 快速得到 outcome reward。

## 4. DPO：另一种不显式做 rollout 的 policy improvement

除了 PPO/GRPO，DPO 也是 LLM post-training 里常见的 preference optimization 方法。不过它和 PPO/GRPO 的出发点不同。

PPO/GRPO 通常是：

1. 用当前 policy sample responses；
2. 用 reward model、verifier 或 rule-based reward 打分；
3. 估计 advantage；
4. 用 policy ratio 做 policy improvement。

而 DPO 直接使用 preference data：

$$
(x, y_w, y_l)
$$

其中 $y_w$ 是 preferred response，$y_l$ 是 rejected response。DPO 的目标是让模型相对 reference model 更偏向 preferred response，也就是让下面这个差值更大：

$$
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

它的 loss 通常写成：

$$
\mathcal{L}_{\text{DPO}}
=
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

直觉上，DPO 也在做 policy improvement：提高 preferred response 的相对概率，降低 rejected response 的相对概率，同时用 reference model 控制 policy 不要偏离太远。

但和 PPO/GRPO 不同的是，DPO 不需要 rollout 后估计 advantage，也不需要显式训练 reward model。它更像是一种 offline preference optimization：直接从 preference pairs 中学习“chosen 应该比 rejected 更可能”。

所以可以简单对比为：

- PPO：用 policy samples + reward，需要 rollout 和 value model，核心信号是 token-level advantage。
- GRPO：用 group samples + reward，需要 rollout，但不需要 value model，核心信号是 group-relative reward。
- DPO：用 preference pairs，不需要 rollout，也不需要 value model，核心信号是 chosen vs. rejected preference。
