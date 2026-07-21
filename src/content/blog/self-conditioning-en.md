---
title: "Some Basic Thoughts on Self-Conditioning"
description: "A note on self-conditioning."
date: 2026-07-20
lang: en
translationKey: self-conditioning
tags:
  - Diffusion
---

Self-conditioning was first introduced in [Analog Bits](https://arxiv.org/abs/2208.04202). In brief, the training procedure in the original paper is as follows:

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
$$

First, obtain a prediction using the same $x_t$ and $t$:

$$
m=\operatorname{sg}\left[f_\theta(x_t,0,t)\right],
$$

then use it as a condition to make another prediction:

$$
\hat{x}_0=f_\theta(x_t,m,t),
$$

and finally compute:

$$
\mathcal{L}=\|\hat{x}_0-x_0\|^2.
$$

At inference time, the procedure is:

$$
m_k=f_\theta(z_{t_k},m_{k-1},t_k),
$$

meaning that the prediction from the previous timestep is used as the condition at the current timestep.

Thus, self-conditioning is not perfectly aligned between training and inference. Training uses the first prediction made at the current timestep, whereas inference uses the prediction retained from the previous timestep:

$$
m_{k-1}\neq f_\theta(z_{t_k},0,t_k).
$$

Of course, one could first make an unconditional prediction at every inference step and then use it for self-conditioning:

$$
a_k=f_\theta(z_{t_k},0,t_k),
$$

$$
m_k=f_\theta(z_{t_k},a_k,t_k).
$$

This would align training and inference, but it would require two forward passes per step. The common approach is therefore closer to an empirical trick: directly reuse the previous prediction to reduce computation.

[Self-conditioned Flow Map Language Models via Fixed-point Flows](https://arxiv.org/abs/2607.00714) interprets self-conditioning as a fixed-point iteration.

For fixed $x_t$ and $t$, define:

$$
T(m)=f_\theta(x_t,m,t).
$$

Ideally, we iterate repeatedly:

$$
m^{(0)}=0,
$$

$$
m^{(j+1)}=T(m^{(j)}),
$$

and eventually converge to:

$$
m^\star=T(m^\star).
$$

In other words, once the prediction has stabilized, feeding it back into the model should produce almost no further change.

However, the vanilla self-conditioning loss neither guarantees that this process will converge nor that the second prediction will be better than the first. It only requires:

$$
f_\theta(x_t,m,t)\approx x_0,
$$

but does not require:

$$
\|f_\theta(x_t,m,t)-x_0\|
<
\|m-x_0\|.
$$

The model may genuinely be performing refinement, or it may simply be copying its previous prediction.

My current understanding of self-conditioning is that it does not provide the model with new information. The condition is itself predicted by the model from $x_t$, so it reveals nothing additional about the true $x_0$.

Nor is it a critic or evaluator that judges whether the previous prediction is good before deciding how to revise it.

Instead, it gives the model a current guess. On the next prediction, the model does not have to start completely from scratch; it can continue from that guess.

The main problem it addresses is therefore not “insufficient information,” but “unstable predictions.”

For example, the current $x_t$ may be compatible with both token A and token B. Without self-conditioning, the model might reconsider its choice at every step:

$$
A\rightarrow B\rightarrow A\rightarrow B.
$$

With self-conditioning, if the previous step already leaned toward A, later steps are more likely to continue along A.

This does not truly reduce posterior uncertainty; rather, it reduces oscillation between different modes and makes the sampling trajectory more stable.

This effect may be especially pronounced for discrete data such as language.

If we use only embedding MSE and both token A and token B are plausible, the optimal prediction may be:

$$
\frac{e_A+e_B}{2}.
$$

This embedding is neither token—it is simply the average of their embeddings.

Cross-entropy loss can instead represent the distribution directly:

$$
p(A)=0.5,\qquad p(B)=0.5.
$$

It does not need to average multiple modes into an intermediate embedding and can preserve a discrete distribution instead.

Combined with self-conditioning, if the first step has already developed a slight preference:

$$
p(A)=0.6,\qquad p(B)=0.4,
$$

a later step may move further toward:

$$
p(A)=0.8,\qquad p(B)=0.2.
$$

Self-conditioning with cross-entropy may therefore let the model gradually settle on one token mode rather than remain at an average of several tokens.

However, cross-entropy alone does not force the model to choose one mode, and self-conditioning does not guarantee sharpening. More precisely, once a slight preference has emerged, self-conditioning makes that preference easier to preserve and amplify.

If the initial direction is correct, the prediction becomes increasingly stable; if it is wrong, the error may instead become locked in early.

I therefore currently think of self-conditioning as follows:

> It provides no new information. Instead, it provides a current hypothesis that reduces mode switching.

For discrete data, cross-entropy prevents multiple discrete modes from being represented by a continuous average, while self-conditioning helps an established mode persist and continue to sharpen.

**References:**

- [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202)
- [Self-conditioned Flow Map Language Models via Fixed-point Flows](https://arxiv.org/abs/2607.00714)
