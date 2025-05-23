---
title: Paper Notes - Energy-Guided Diffusion Sampling for Offline-to-Online Reinforcement Learning
published: 2024-09-08
description: Notes for this paper.
tags:
  - paper-notes
  - diffusion-model
  - RL
authors:
  - Xu-Hui Liu
  - Tian-Shuo Liu
  - Shengyi Jiang
  - Ruifeng Chen
  - Zhilong Zhang
  - Xinwei Chen
  - Yang Yu
draft: false
---
Replay offline data directly in the online phase
- data distribution shift
- inefficiency in online fine-tuning

Introduce Energy-guided Diffusion Sampling (EDIS)
- use a **diffusion model** to <font color="#ff0000">extract prior knowledge</font> from the offline dataset
- employs **energy functions** to <font color="#ff0000">distill this knowledge</font> for enhanced data degeneration in the online phase

## 1 Introduction

However, utilizing a diffusion model trained on an offline dataset introduces a challenge—it can only <font color="#ff0000">generate samples adhering to the dataset distribution</font>, thus still being susceptible to distribution shift issues.

The desired distribution for RL has three crucial characteristics：
1. the state distribution should align with that in the online training phase
2. actions should be consistent with the current policy
3. the next states should conform to the transition function

To achieve this, we formulate three distinct energy functions to guide the diffusion sampling process, ensuring alignment with the aforementioned features.


## 2 EDIS: Energy-Guided Diffusion Sampling

To <font color="#ff0000">extract prior knowledge</font> from the offline dataset and <font color="#ff0000">generate samples to conform to the online data distribution</font>, we introduce our innovative approach, named Energy-guided Diffusion Sampling (EDIS).

At the heart of our method is to accurately generate a desired online data distribution, denoted as $q_\pi(s,a,s^{\prime})$, from pre-gathered data. The distribution does not include reward $r$ because we assume that the reward function $r(s,a)$ is accessible, either directly or through learning from the dataset. 

To achieve this, we have integrated a diffusion model into our framework, capitalizing on its exceptional capability for modeling complex distributions.

### 2.1 Distribution Adjustment via Energy Guidance

One challenge in this process is the inherent limitation of directly training a diffusion model on an offline dataset. Such a model typically yields an offline data distribution $p_\mathcal{D}(s,a,s^{\prime})$,which does not <font color="#ff0000">align perfectly with</font> online data and causes distribution shift issues. 

To address this, our method needs to guide the diffusion sampling process towards the online distribution. This is achieved by decomposing the online data distribution into the following form:
$$
q_\pi(s,a,s')\propto p_\theta(s,a,s')e^{-\mathcal{E}(s,a,s')},
$$
where $p_\theta(s,a,s^{\prime})$ is the distribution generated by the denoiser network, parameterized by $\theta$. $\mathcal{E}(s,a,s^\prime)$ is the energy function, which serves as the <font color="#ff0000">guidance to bridge the gap between generated distribution and online data distribution</font>. The following theorem shows such an energy function exists.

---
**Theorem 3.1.** Let $p_\theta(s)$ be the marginal distribution of $p_\theta ( s, a, s^{\prime })$, $p_{\theta}(a|s)$ and $p_{\theta}(s'|s,a)$ be the conditional distribution of $p_\theta ( s, a, s^{\prime })$ given $s$ and $( s, a)$. The former equation is valid if the energy function $\mathcal{E} ( s, a, s^\prime )$  is structured as follows: 
$$
\mathcal{E}(s,a,s')=\mathcal{E}_1(s)+\mathcal{E}_2(a|s)+\mathcal{E}_3(s'|s,a),
$$
such that $e^{\mathcal{E} _1( s) }\propto \frac {p_\theta ( s) }{d^\pi ( s) }$, $e^{\mathcal{E} _2( a| s) }\propto \frac {p_\theta ( a| s) }{\pi ( a| s) }$, $e^{\mathcal{E} _3( s^{\prime }| s, a) }\propto \frac{p_\theta(s^{\prime}|s,a)}{T(s^{\prime}|s,a)}$.

Each part is responsible for aligning the generated distribution with different aspects of the online data: 
- the online state distribution
- the current policy action distribution
- the environmental dynamics.

### 2.2 Learning Energy Guidance by Contrastive Energy Prediction

- the energy is estimated using a neural network denoted as $\mathcal{E}_{\phi_1}(s).$ 
- Let $K$ and $K_\mathrm{neg}$ be two positive numbers. Given $s_1,s_2,\ldots,s_K,K$ i.i.d. samples drawn from the distribution $p_\theta(s)$, and $s_i^1,s_i^2,\ldots,s_i^{K_{\mathrm{neg}}},K_{\mathrm{Neg}}$ negative samples for $s_i.$ We employ the Information Noise Contrastive Estimation (InfoNCE):
$$
\mathcal{L}(\phi_1)=-\sum_{i=1}^K\log\frac{e^{-\mathcal{E}_{\phi_1}(s_i)}}{e^{-\mathcal{E}_{\phi_1}(s_i)}+\sum_{j=1}^{K_{\mathrm{neg}}}e^{-\mathcal{E}_{\phi_1}(s_i^j)}},
$$
- Then, we devise positive and negative samples to achieve the target energy function established by Thm. 3.1. 
- Suppose the distribution of positive samples is $\mu(s)$, the distribution of negative samples is $\nu(s)$, the final optimized results is $e^{\mathcal{E}_{\phi_1}(s)}\propto\frac{\nu(s)}{\mu(s)}.$ Compared to the function indicated bv Thm. 3.1. the result can be achieved by selecting $\mu(s)=d^\pi(s),\nu(s)=p_\theta(s).$ 
- Following the approach of Sinha et al. (2022);Liu et al. (2021),we construct a positive buffer, containing only a small set of trajectories from very recent policies. The data distribution in this buffer can be viewed as an approximation of the on-policy distribution $d^\pi(s).$ While $p_\theta(s)$ is the distribution of the data generated during the denoising steps. 
- Therefore, the positive samples is sampled from the positive buffer and the negative samples is sampled from the denoiser.

### 2.3 Sampling under Energy Guidance

Score function in sampling process:
$$
\begin{aligned}&\nabla_{(s,a,s^{\prime})}\log q_\pi(s,a,s^{\prime})\\=&\nabla_{(s,a,s^{\prime})}\log p_\theta(s,a,s^{\prime})-\nabla_{(s,a,s^{\prime})}\mathcal{E}(s,a,s^{\prime})\end{aligned}
$$
In the denoising process, we need to obtain the score function at each timestep. Denote the forward distribution at time $t$ starting from $p_0(s,a,s^{\prime})$ as $p_t(s,a,s^{\prime}).$ Remember that the denoiser model $D_\theta(s,a,s^{\prime};\sigma)$ is designed to match the score with the expression:
$$
\nabla\log p_\theta(s,a,s')=(D_\theta(s,a,s';\sigma)-(s,a,s'))/\sigma^2.
$$
Thus, we can obtain the gradient through the denoiser model.