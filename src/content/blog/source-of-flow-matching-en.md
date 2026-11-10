---
title: "Does a Better Source Distribution Really Help Flow Matching?"
description: "Some thinking on the source distribution of flow matching."
date: 2026-11-10
lang: en
translationKey: source-of-flow-matching
tags:
  - Flow Matching
---

A question I had been thinking about: **how much can the choice of source distribution improve flow matching?**

One natural idea is that if the source distribution is closer to the target distribution, transport should be simpler, and generation should require fewer sampling steps. This seems especially plausible in sequential generation, where adjacent states are often strongly correlated. Using the previous round's predictive distribution as the source therefore seems more reasonable than starting from Gaussian noise every time.

But after discussing this with my friend H and analyzing it more carefully, I realized that this intuition does not necessarily hold. At least under **independent coupling + linear interpolation**, the distance between source and target does not directly reflect the difficulty of transport, nor does it determine the error of few-step generation.

## 1 Problem Setup

Consider standard conditional flow matching:

$$
X_0\sim Q,\quad X_1\sim P,\quad X_0\perp X_1,
$$

where $Q$ is the source distribution and $P$ is the target distribution.

We use linear interpolation:

$$
X_t=(1-t)X_0+tX_1.
$$

The corresponding optimal velocity field is:

$$
u_t(x)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

Let us start with two assumptions:

1. The model can learn the exact velocity field.
2. We use the Euler method for numerical integration.

This lets us set aside neural network approximation error and focus on how the source distribution affects numerical integration error.

With an exact ODE solver, the final distribution should be $P$ regardless of the source distribution, assuming the corresponding ODE flow exists and transports the distribution correctly.

The potential advantages of source selection therefore lie mainly in **integration with a finite number of steps** and **learning with actual neural networks**.

## 2 NFE = 1: Source Distribution Does Not Matter

First, consider the simplest case: a single Euler step:

$$
Y_1=Y_0+u_0(Y_0),\quad Y_0\sim Q.
$$

Because source and target are independent:

$$
\begin{aligned}
u_0(x)
&=\mathbb{E}[X_1-X_0\mid X_0=x]\\
&=\mathbb{E}[X_1]-x.
\end{aligned}
$$

Therefore:

$$
Y_1=Y_0+\mathbb{E}[X_1]-Y_0
=\mathbb{E}[X_1].
$$

In other words:

> Regardless of the source distribution, a single Euler step always outputs the mean of the target distribution.

The corresponding Wasserstein error is:

$$
W_2^2(\mathcal{L}(Y_1),P)
=\operatorname{Tr}(\operatorname{Cov}(P)).
$$

This result is completely independent of $Q$. Even if $Q$ is very close to $P$, or even if $Q=P$, a single Euler step still outputs only the target mean.

This made me start wondering: **does bringing the source distribution closer to the target really make generation easier for flow matching?**

Of course, a single Euler step is an extreme case. Let us consider two steps next.

## 3 NFE = 2: Closer Does Not Mean Better

Consider two equally sized Euler steps, $0\rightarrow \frac12\rightarrow 1$. The first step gives:

$$
Y_{1/2}=Y_0+\frac12u_0(Y_0)
=\frac{Y_0+m_P}{2},
$$

where $m_P=\mathbb{E}[X_1]$.

Define:

$$
M(w)=\mathbb{E}[X_1\mid X_0+X_1=w].
$$

At $t=1/2$, we have:

$$
u_{1/2}(x)=2M(2x)-2x.
$$

Substituting this into the second Euler step:

$$
\begin{aligned}
Y_1
&=Y_{1/2}+\frac12u_{1/2}(Y_{1/2})\\
&=M(2Y_{1/2})\\
&=M(Y_0+m_P).
\end{aligned}
$$

Thus, the final output of two-step Euler is:

$$
\boxed{Y_1=M(Y_0+m_P)}
$$

This means that the choice of source distribution does affect the generated result, but through two factors:

- The posterior mean function $M$, which is jointly determined by source and target.
- The query distribution induced by source samples when evaluating $M$.

It is not simply determined by $W_2(Q,P)$.

### A Counterintuitive Example: Source = Target

Suppose:

$$
Q=P.
$$

Since $X_0$ and $X_1$ are independent and identically distributed, exchangeability gives:

$$
\mathbb{E}[X_1\mid X_0+X_1=w]=\frac{w}{2}.
$$

Therefore:

$$
M(w)=\frac{w}{2}.
$$

The final output of two-step Euler is:

$$
Y_1=\frac{Y_0+m_P}{2}.
$$

This actually contracts the entire distribution toward its mean. If $P$ is a bimodal distribution with mean zero, the generated result is simply the original distribution scaled down by half, rather than $P$ itself.

> In other words, **even when the source already equals the target, independent coupling can still construct an unnecessary, nontrivial transport process, and few-step Euler can still introduce substantial error.**

In fact, exchangeability implies:

$$
u_{1/2}(x)=0.
$$

But this does not mean that the velocity field is zero everywhere in time. Under independent coupling, the intermediate distributions generally still change.

For example, when $Q=P=\mathcal{N}(0,1)$:

$$
X_t\sim\mathcal{N}(0,(1-t)^2+t^2).
$$

The distribution first contracts and then expands, even though source and target are identical.

The corresponding velocity field is:

$$
u_t(x)=
\frac{2t-1}{(1-t)^2+t^2}x.
$$

Thus, a task that could have been completed without moving any samples instead requires contraction followed by expansion under independent coupling.

### A More Distant Source Can Actually Be Better

We can construct an even stronger counterexample. Consider a bimodal target:

$$
P_\epsilon=
\frac12\mathcal{N}(-1,\epsilon^2)
+\frac12\mathcal{N}(1,\epsilon^2).
$$

Compare two sources:

$$
Q_1=P_\epsilon,
$$

and

$$
Q_2=(D_{1/2})_\#P_\epsilon,
\qquad D_{1/2}(x)=x/2.
$$

Here, $Q_2$ is simply $P_\epsilon$ scaled down by half. Clearly:

$$
W_2(Q_1,P_\epsilon)=0,
$$

while

$$
W_2(Q_2,P_\epsilon)
=\frac12\sqrt{1+\epsilon^2}>0.
$$

But with two-step Euler:

- With $Q_1$, since source and target have the same distribution, the final output has distribution $Q_2$, giving a generation error of $\frac12\sqrt{1+\epsilon^2}$.
- With $Q_2$, the generation error of two-step Euler instead approaches zero as $\epsilon\rightarrow0$.

In other words:

$$
\boxed{
W_2(Q_1,P)<W_2(Q_2,P)
}
$$

but at the same time:

$$
\boxed{
D_2(Q_1,P)>D_2(Q_2,P)
}
$$

Here, $D_2$ denotes the Wasserstein distance between the two-step Euler output and the target distribution when using the exact velocity field.

This counterexample shows that **a source closer to the target does not necessarily yield a smaller generation error with few-step Euler.** For $Q_1=P$, exchangeability makes the midpoint velocity zero. For $Q_2$, the posterior mean can exploit the bimodal structure to map source samples back to the target modes more accurately.

What really matters here is the structure of the velocity field, rather than the source-target distance itself.

## 4 What About Sequential Generation?

I originally considered this question because of a very intuitive idea in sequential generation: if adjacent states are highly correlated, the previous round's predictive distribution should be close to the current target. Reusing the previous prediction directly as the source might therefore be more efficient than starting from Gaussian noise.

For example, consider a slowly changing physical system:

$$
S_{n+1}=S_n+\epsilon_n,
$$

where $\epsilon_n$ is small. From the perspective of the true dynamics, predicting the next state may require only a small update.

But it is easy to conflate two different concepts here:

1. The temporal coupling between true states, namely the joint distribution of $(S_n,S_{n+1})$.
2. The source-target coupling constructed for flow matching training.

If training still uses:

$$
X_0\sim Q_n,\quad X_1\sim P_n,
\quad X_0\perp X_1,
$$

then the model is not actually using the correspondence between individual samples of the true states.

Even if $Q_n$ and $P_n$ are very close, or even identical, independent coupling still randomly pairs samples from the two distributions. The earlier $Q=P$ example already shows that transport need not be simple in this setting.

Going further, we can construct a highly correlated sequential process such that:

$$
\mathbb{E}\|S_{n+1}-S_n\|^2\rightarrow0
$$

and the historical predictive distribution satisfies:

$$
W_2(Q_n^{\mathrm{hist}},P_n)\rightarrow0.
$$

Yet with two-step Euler, using the historical predictive distribution can still produce a larger generation error than using a standard Gaussian source.

> This shows that **strong temporal correlation in the true dynamics does not automatically translate into an advantage for few-step generation under independent flow matching.**
> The key is that if this correlation is not incorporated into the coupling, flow matching sees only two marginal distributions and does not actually exploit the correspondence between states.

## 5 Maybe Coupling Matters More Than Source Distance

An example worth considering is to compare two couplings when $Q=P$.

**Independent coupling:**

$$
X_0\sim P,\quad X_1\sim P,\quad X_0\perp X_1.
$$

In this case, even though source and target are identical, the intermediate distributions may still change, and the velocity field need not be zero.

**Identity coupling:**

$$
X_1=X_0,\quad X_0\sim P.
$$

Then:

$$
X_t=X_0
$$

and:

$$
u_t(x)=0.
$$

Therefore, Euler integration introduces no discretization error, regardless of the number of steps. These two settings have exactly the same source and target, yet produce completely different transport paths. Of course, identity coupling mainly serves as a theoretical comparison here. If we could already sample directly from $P$, there would be no need to train a generative model. Constructing such a coupling may not be feasible in practical generation.

Still, it illustrates an important point: **knowing only the source and target marginal distributions may not be enough to judge the difficulty of transport; the coupling itself matters as well.**

For sequential generation, if we want to exploit information from historical states, we may need to consider how to incorporate the true temporal correspondence into the transport construction, rather than simply using the historical predictive distribution as a new source.

## 6 What Does This Mean for Learnability?

The analysis so far has an important limitation: we have assumed an exact velocity field throughout. We are therefore analyzing numerical integration error, rather than learnability when training an actual neural network. In practice, we also need to consider whether the velocity field is easy for a neural network with finite capacity to fit.

Consider the flow matching regression objective:

$$
\mathcal{L}
=\mathbb{E}\left[
\|v_\theta(t,X_t)-(X_1-X_0)\|^2
\right].
$$

Define:

$$
u_t(x)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

Under squared loss, the objective can be decomposed as:

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

The first term is the model's approximation error relative to the true velocity field. The second is the irreducible variance of the conditional velocity target itself.

This means that the choice of source and coupling can affect training in at least two ways:

1. **Velocity field complexity.** Different sources and couplings produce different velocity fields, affecting their smoothness, curvature, and how difficult they are for a neural network to fit.
2. **Regression target variance.** Even when the optimal velocity field is simple, the conditional velocity variance in the training samples may still be large, affecting optimization under finite data and compute budgets.

But lower regression target variance does not directly imply better generation quality. Likewise, smaller velocity field curvature alone does not necessarily guarantee smaller Euler error.

What is really worth analyzing is how these factors jointly affect generation error under finite model capacity, finite training compute, and finite NFE.

## 7 Some Thoughts

The current analysis leaves me somewhat skeptical of source selection alone. If a method's main motivation is:

> Historical source is closer to target, so flow matching should require fewer steps.

then this reasoning is insufficient, at least under independent coupling, because:

- At NFE = 1, source selection does not change the final output.
- At NFE = 2, source selection affects the error, but source-target distance does not determine its magnitude.
- Even when source = target, independent coupling can still produce nontrivial transport.
- Even when true adjacent states are highly correlated, independent coupling does not necessarily exploit that correlation.

This does not mean that source selection is pointless. The source does change the probability path and velocity field, and may affect both the learnability of actual models and the efficiency of few-step generation.

But I think a more worthwhile question is: how do source distributions, couplings, and probability paths jointly affect learnability and the efficiency of flow matching? I feel that using the correspondence between distributions, or the characteristics of the source and target distributions, to construct transport paths that are easier to learn and better suited to few-step integration may be more interesting and meaningful. At least for the continuous DLM problems I am currently focused on, this also seems more important.
