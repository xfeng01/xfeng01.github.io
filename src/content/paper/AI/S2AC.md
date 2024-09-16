---
title: "S2AC: energy-based reinforcement learning with stein soft actor critic"
published: 2024-06-16
tags:
  - Max-Ent-RL
authors: 
  - Safa Messaoud
  - Billel Mokeddem
  - Zhenghai Xue
  - Linsey Pang
  - Bo An
  - Haipeng Chen
  - Sanjay Chawla
draft: false
---

## 1 Introduction
MaxEnt RL learns a stochastic policy that captures the intricacies of the action space.
- better exploration
- better robustness to environmental perturbations
- learning policies that maximize the sum of *expected future reward* and *expected future entropy*.
 - **estimating the entropy** is a problem.

### 1.1 Related Work
> go around the entropy computation or make limiting assumptions on the policy
- poor scalability
- convergence to suboptimal solutions.

SQL (Haarnoja et al., 2017)
- implicitly incorporates entropy in the Q-function computation
- using importance sampling
- high variability and hence poor training stability and limited scalability to high dimensional action spaces.

SAC et al., 2018a)
- fitting a Gaussian distribution to the EBM policy --- closed form evaluation of entropy
- suboptimal solution (in multimodal action distributions)

IAPO (Marino et al., 2021)
- models the policy as a uni-modal Gaussian
- achieves multimodal policies by learning a collection of parameter estimates (mean, variance) through different initializations for different policies.

### 1.2 Proposed Method
To achieve expressivity, S2AC **models the policy** as a Stein Variational Gradient Descent (SVGD) (Liu, 2017) sampler from an EBM over Q-values (target distribution).

SVGD proceeds by first sampling a set of particles from an initial distribution, and then iteratively transforming these particles via a sequence of updates to fit the target distribution.

To compute a closed-form estimate of the **entropy** of such policies, we use the change-of-variable formula for pdfs (Devore et al., 2012).

To improve *scalability*, we model the initial distribution of the SVGD sampler as an isotropic Gaussian and learn its parameters
- faster convergence to the target distribution

Beyond RL, the backbone of S2AC is a **new variational inference algorithm** with a more expressive and scalable distribution characterized by a closed-form entropy estimate.

## 2 Preliminaries
### 2.1 Samplers for energy-based models
- SVGD is a particle-based Bayesian inference algorithm.
- SVGD samples a set of m particles $\{a_{j}\}^{m}_{j=1}$ from an initial distribution $q_{0}$ which it then transforms through a sequence of updates to fit the target distribution.
- SVGD applies a form of functional gradient descent $∆f$ that minimizes the KL-divergence between the target distribution $p$ and the proposal distribution $q^{l}$ induced by the particles. $a_i^{l+1}=a_i^l+\epsilon\Delta f(a_i^l)$. $$
\Delta f(a_i^l)=\mathbb{E}_{a_j^l\thicksim q^l}\big[k(a_i^l,a_j^l)\nabla_{a_j^l}\log p(a_j^l)+\nabla_{a_j^l}k(a_i^l,a_j^l)\big]
$$

### 2.2 Maximum-entropy RL
MaxEnt RL learns a policy $π^{∗}(a_{t}|s_{t})$, that instead of maximizing the expected future reward, maximizes the sum of the expected future reward and entropy:
$$
\pi^*=\arg\max_\pi\sum_t\gamma^t\mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot|s_t))\big],
$$
equivalent to approximating the policy, modeled as an EBM over Q-values, by a variational distribution $π(a_{t}|s_t)$.
$$
\pi^*=\arg\min_\pi\sum_t\mathbb{E}_{s_t\sim\rho_\pi}\big[D_{KL}\big(\pi(\cdot|s_t)\|\exp(Q(s_t,\cdot)/\alpha)/Z\big)\big],
$$
#### SAC: actor-critic algorithm
policy evaluation:
$$
Q_\phi(s_t,a_t)\leftarrow r(s_t,a_t)+\gamma\mathbb{E}_{s_{t+1},a_{t+1}\thicksim\rho_{\pi_\theta}}\left[Q_\phi(s_{t+1},a_{t+1})+\alpha\mathcal{H}(\pi_\theta(\cdot|s_{t+1}))\right]
$$
> Don't include the entropy of current state

policy improvement:
$$
\pi_\theta=\arg\max_\theta\sum_t\mathbb{E}_{s_t,a_t\thicksim\rho_{\pi_\theta}}\left[Q_\phi(a_t,s_t)+\alpha\mathcal{H}(\pi_\theta(\cdot|s_t))\right].
$$
SAC models $π_{θ}$ as an isotropic Gaussian, i.e., $π_{\theta}(·|s) = \mathcal{N} (\mu_{\theta}, \sigma_{\theta}I)$.
> If we use diffusion model to learn the policy, **how to compute its entropy**. 

**Weakness**
- over-simplification of the true action distribution
- cannot represent complex distributions, e.g., multimodal distributions.

#### SQL
goes around the entropy computation, by defining a soft version of the value function 
$$
V_{\phi} = \alpha\log\big(\int_{\mathcal{A}}\exp\big(\frac{1}{\alpha}Q_{\phi}(s_{t},a')\big)da'\big)
$$
This lead to the expression of Q-value
$$
Q_{\phi}(s_{t},a_{t})=r(s_{t},a_{t})+\gamma\mathbb{E}_{s_{t+1}\thicksim p}[V_{\phi}(s_{t+1})]
$$
SQL follows a soft value iteration which alternates between the updates of the “soft” versions of $Q$ and value functions
$$
\begin{align}
Q_{\phi}(s_{t},a_{t})\leftarrow r(s_{t},a_{t})+\gamma\mathbb{E}_{s_{t+1}\thicksim p}[V_{\phi}(s_{t+1})],\forall(s_{t},a_{t})\\
V_{\phi}(s_{t})\leftarrow\alpha\log\big(\int_{\mathcal{A}}\exp\big(\frac{1}{\alpha}Q_{\phi}(s_{t},a')\big)da'\big),\forall s_{t}.
\end{align}
$$
- let $Q_{\phi}$ and $V_{\phi}$ converge first
- uses amortized **SVGD** to learn a **stochastic sample network** $f_{\theta}(\xi,s_{t})$ that maps noise samples $\xi$ into the action samples from the EBM policy distribution $\pi^{*}(a_{t}|s_{t})=\exp\left(\frac{1}{\alpha}(Q^{*}(s_{t},a_{t})-V^{*}(s_{t}))\right)$
- $\theta$ obtained by minimizing the loss $J_{\theta}(s_{t})=D_{KL}\big(\pi_{\theta}(\cdot|s_{t})||\exp\big(\frac{1}{\alpha}(Q_{\phi}^{*}(s_{t},\cdot)-V_{\phi}^{*}(s_{t}))\big)$
- the integral is approximated via **importance sampling** ---  high variance estimates and hence poor scalability to high dimensional action spaces
- amortized generation is usually unstable and prone to mode collapse

## 3 Approach
S<sup>2</sup>AC: a new actor-critic MaxEnt RL algorithm
- uses SVGD as the underlying actor to generate action samples from policies represented using EBMs. （expressivity）
- derive a closed-form entropy **estimate** of the SVGD-induced distribution
- propose a parameterized version of SVGD to enable **scalability** to high-dimensional action spaces and non-smooth Q-function landscapes.

### 3.1 Stein Soft Actor Critic
- model the **actor** as a parameterized sampler from an EBM.
#### Critic
$$
\phi^*=\arg\min_\phi\mathbb{E}_{(s_t,a_t)\sim\rho_{\pi_\theta}}\left[(Q_\phi(s_t,a_t)-\hat{y})^2\right]
$$
where target $\hat{y}=r_t(s_t,a_t)+\gamma\mathbb{E}_{(s_{t+1},a_{t+1})\thicksim\rho_\pi}\left[Q_{\bar{\phi}}(s_{t+1},a_{t+1})+\alpha\mathcal{H}(\pi(\cdot|s_{t+1}))\right]$.
#### Actor as an EBM sampler
- samples a set of particles $\{a^{0}\}$ from an initial distribution $q^{0}$ (e.g., Gaussian).
- These particles are then updated over several iterations $l ∈ [1, L]$.

If  $q^{0}$ is tractable and $h$ is invertible, it’s possible to compute a **closed-form expression of the distribution** of the particles at the $l$<sup>th</sup> iteration via the change of variable formula.

- The **policy** is represented using the particle distribution at the final step $L$ of the sampler dynamics, i.e., $π(a|s) = q^{L}(a^{L}|s)$ 
- The **entropy** can be *estimated* by averaging $\log q^{L}(a^{L}|s)$ over a set of particles.
#### Parameterized initialization
- To speed up convergence, modeling the initial distribution as a parameterized isotropic Gaussian, i.e., $a^{0}\sim \mathcal{N}(\mu_{\theta}(s),\sigma_{\theta}(s))$.
- To deal with the non-smooth nature of deep Q-function landscapes, bound the particle updates $-t\sigma_{\theta}\leq a^{l}_{\theta}\leq t\sigma_{\theta}$, $\forall l \in [1,L]$.

$$
\begin{aligned}&\theta^*=\arg\max_\theta\mathbb{E}_{s_t\sim\mathcal{D},a_\theta^L\sim\pi_\theta}\left[Q_\phi(s_t,a_\theta^L)\right]+\alpha\mathbb{E}_{s_t\sim\mathcal{D}}\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right]\\&\mathrm{s.t.~}-t\sigma_\theta\leq a_\theta^l\leq t\sigma_\theta,\quad\forall l\in[1,L].\end{aligned}
$$
$\mathcal{D}$ is the replay buffer.

### 3.2 A Closed-form Expression of the Policy’s entropy
> A critical challenge in MaxEnt RL is *how to efficiently compute the entropy* term $H(π(·|st+1))$.
$$
\mathcal{H}(\pi_\theta(\cdot|s))=-\mathbb{E}_{a_\theta^0\sim q_\theta^0}\Big[\log q_\theta^L(a_\theta^L|s)\Big]\approx-\mathbb{E}_{a_\theta^0\sim q_\theta^0}\Big[\log q_\theta^0(a^0|s)-\epsilon\sum_{l=0}^{L-1}\mathrm{Tr}\Big(\nabla_{a_\theta^l}h(a_\theta^l,s)\Big)\Big]
$$
### 3.3 Invertible Policies
## 4 Results
### Entropy Evaluation
- compare the estimated entropy for distributions (with known ground truth entropy or log-likelihoods) using different samplers.
- study the **sensitivity** of the formula to different samplers’ parameters.

### Multi-goal Experiments
- To check if S<sup>2</sup>AC learns a better solution to the max-entropy objective, we design a new multi-goal environment.

### Mujoco Experiments
- Performance and sample efficiency
- run-time

