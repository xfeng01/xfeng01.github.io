---
title: Efficient Diffusion Policies for Offline Reinforcement Learning
published: 2024-09-12
tags:
  - diffusion-model
  - RL
authors: 
  - Bingyi Kang
  - Xiao Ma
  - Chao Du
  - Tianyu Pang
  - Shuicheng Yan
draft: false
---

Diffusion-QL suffers from two critical limitations
- computationally inefficient to forward and backward through the whole Markov chain during training
- incompatible with maximum likelihood-based RL algorithms (e.g., policy gradient methods) as the <font color="#ff0000">likelihood of diffusion models is intractable</font>.

EDP approximately constructs actions from corrupted ones at training to avoid running the sampling chain.

## Efficient Diffusion Policy

present a novel algorithm termed Reinforcement-Guided Diffusion Policy Learning (RGDPL)

### Diffusion Policy

We use the reverse process of a conditional diffusion model as a parametric policy:
$$
\pi_\theta(\boldsymbol{a}|s)=p_\theta(\boldsymbol{a}^{0:K}|\boldsymbol{s})=p(\boldsymbol{a}^K)\prod_{k=1}^Kp_\theta(\boldsymbol{a}^{k-1}|\boldsymbol{a}^k,\boldsymbol{s}),
$$
where $a^K\sim\mathcal{N}(0,I).$ 

Given a dataset, we can easily and efficiently train a diffusion policy in a <font color="#ff0000">behavior-cloning</font> manner as we only need to forward and backward through the network once each iteration.

### Reinforcement-Guided Diffusion Policy Learning

how we can efficiently use $Q_{\phi}$ to guide diffusion policy training procedure.
- We now show that this can be achieved without sampling actions from diffusion policies.

Using the reparameterization trick, we are able to connect $\boldsymbol{a}^k,\boldsymbol{a}^0$ and $\epsilon$ by:
$$
a^k=\sqrt{\bar{\alpha}^k}a^0+\sqrt{1-\bar{\alpha}^k}\epsilon,\quad\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}).
$$

Recall that our diffusion policy is parameterized to predict $\epsilon$ with $\epsilon_\theta(a^k,k;s).$ By relacing $\epsilon$ with $\boldsymbol{\epsilon}_\theta(\boldsymbol{a}^k,k;\boldsymbol{s})$, we obtain the approximated action:
$$
\hat{a}^0=\frac1{\sqrt{\bar{\alpha}^k}}\boldsymbol{a}^k-\frac{\sqrt{1-\bar{\alpha}^k}}{\sqrt{\bar{\alpha}^k}}\boldsymbol{\epsilon}_\theta(\boldsymbol{a}^k,k;\boldsymbol{s}).
$$
Accordingly, the policy improvement for diffusion policies is modified as follows:
$$
L_\pi(\theta)=-\mathbb{E}_{\boldsymbol{s}\sim\mathcal{D},\hat{\boldsymbol{a}}^0}\left[Q_\phi(s,\hat{\boldsymbol{a}}^0)\right].
$$
To improve the efficiency of policy evaluation, we propose to replace the DDPM sampling with DPM-Solver [20], which is an ODE-based sampler. 

### Generalization to Various RL algorithms

**Direct policy optimization.** It maximizes Q values and directly backpropagate the gradients from Q network to policy network.
$$
\nabla_\theta L_\pi(\theta)=-\frac{\partial Q_\phi(\boldsymbol{s},\boldsymbol{a})}{\partial\boldsymbol{a}}\frac{\partial\boldsymbol{a}}{\partial\theta}.
$$

This is only applicable to cases where $\frac{\partial \boldsymbol{a}}{\partial \theta}$ is tractable, e.g., when a deterministic policy $\boldsymbol{a}=\pi_{\theta}(\boldsymbol{s})$ is used or when the sampling process  can be reparametrized.

**Likelihood-based policy optimization.** It tries to distill the knowledge from the Q network into the policy network indirectly by performing weighted regression or weighted maximum likelihood
$$
\max_{\boldsymbol{\theta}}\quad\mathbb{E}_{(\boldsymbol{s},\boldsymbol{a})\thicksim\mathcal{D}}\left[f(Q_{\phi}(\boldsymbol{s},\boldsymbol{a}))\log\pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\boldsymbol{s})\right],
$$
where $f( Q_\phi ( \boldsymbol{s}, \boldsymbol{a}) )$ is a monotonically increasing function that assigns a weight to each state- action where $f( Q_\phi ( \boldsymbol{s}, \boldsymbol{a}) )$ is a monotonically increasing function that assigns a weight to each state- action ser$f( Q_\phi ( \boldsymbol{s}, \boldsymbol{a}) )$ is pair in the dataset. This objective requires the log-likelihood of the policy to be tractable and differentiable.

**In this paper**

First,  instead of computing the likelihood,  we turn to a lower bound for log $\pi _{\theta }( \boldsymbol{a}| \boldsymbol{s})$ introduced in DDPM.  By discarding the constant term that does not depend on $\theta$,we can have the objective:
$$
\mathbb{E}_{k,\boldsymbol{\epsilon},(\boldsymbol{a},\boldsymbol{s})}\left[\frac{\beta^k\cdot f(Q_\phi(\boldsymbol{s},\boldsymbol{a}))}{2\alpha^k(1-\bar{\alpha}^{k-1})}\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\boldsymbol{a}^k,k;\boldsymbol{s}\right)\right\|^2\right].
$$
Second, instead of directly optimizing $\log\pi_\theta(\boldsymbol{a}|s)$, we propose to replace it with an approximated policy $\hat{\pi}_\theta(a|s)\triangleq\mathcal{N}(\hat{a}^0,\boldsymbol{I})$. Then, we get the following objective:
$$
\mathbb{E}_{k,\boldsymbol{\epsilon},(\boldsymbol{a},\boldsymbol{s})}\left[f(Q_\phi(s,\boldsymbol{a}))\left\|\boldsymbol{a}-\hat{\boldsymbol{a}}^0\right\|^2\right].
$$
Empirically, we find these two choices perform similarly, but the latter is easier to implement. So we will report results mainly based on the second realization. In our experiments, we consider two offline RL algorithms under this category, *i.e*., CRR, and IQL. They use two weighting schemes: $f_{\mathbf{CRR}}=\exp\left[\left(Q_\phi(s,a)-\mathbb{E}_{a^{\prime}\sim\hat{\pi}(a|s)}Q(s,a^{\prime})\right)/\tau_{\mathrm{CRR}}\right]$ and $f_{\mathrm{IQL}}=\exp\left[\left(Q_\phi(s,a)-V_\psi(s)\right)/\tau_{\mathrm{IQL}}\right]$, where $\tau$ refers to the temperature parameter and $V_\psi(s)$ is an additional value network parameterized by $\psi.$ 

