---
title: Diffusion Policies creating a Trust Region for  Offline Reinforcement Learning
published: 2024-09-16
tags:
  - diffusion-model
  - offline-RL
authors:
  - Tianyu Chen
  - Zhendong Wang
  - Mingyuan Zhou
draft: false
---
- We introduce a dual policy approach, Diffusion Trusted Q-Learning (DTQL), which comprises a diffusion policy for pure behavior cloning and a practical one-step policy.
- We bridge the two polices by a newly introduced <font color="#ff0000">diffusion trust region loss</font>.
- The diffusion policy maintains expressiveness, while the trust region loss directs the one-step policy to explore freely and seek modes within the region defined by the diffusion policy.

## 1 Introduction

- Behavior-regularized policy optimization techniques are employed to constrain the divergence between the learned and in-sample policies during training
- Distilling the iterative denoising process of diffusion models into a one-step generator.

This paper
- introduce a <font color="#ff0000">diffusion trust region loss</font> that moves away from focusing on distribution matching; instead, it emphasizes establishing a safe, **in-sample behavior region**.
- simultaneously <font color="#ff0000">train two cooperative policies</font>: a diffusion policy for pure behavior cloning and a one-step policy for actual deployment.
- The one-step policy is optimized based on two objectives: the diffusion trust region loss, which ensures safe policy exploration, and the maximization of the Q-value function, guiding the policy to generate actions in high-reward regions.

<font color="#00b050">What's "in-sample"</font>

## 2 Diffusion Trusted Q-Learning

### Diffusion Policy

This paper only trains a diffusion model and <font color="#ff0000">avoids using it for inference</font>, thus significantly reducing both training and inference times.

The objective function of the diffusion model aims to train a predictor for denoising noisy samples back to clean samples, represented by the optimization problem:
$$
\min_\phi\mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\boldsymbol{I})}[w(t)\|\mu_\phi(\boldsymbol{x}_t,t)-\boldsymbol{x}_0\|_2^2]
$$
where $w(t)$ is a weighted function dependent only on $t.$ 

In offline RL, since the training data is state-action pairs, we train a diffusion policy using a conditional diffusion model as follows:
$$
\mathcal{L}(\phi)=\mathbb{E}_{t,\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\boldsymbol{I}),(\boldsymbol{a}_0,\boldsymbol{s})\sim\mathcal{D}}[w(t)\|\mu_\phi(\boldsymbol{a}_t,t|\boldsymbol{s})-\boldsymbol{a}_0\|_2^2]
$$
where $a_0,s$ are the action and state samples from offline datasets $\mathcal{D}$, and $\boldsymbol a_t=\alpha_t\boldsymbol{a}_0+\sigma_t\boldsymbol{\varepsilon}.$ 

**The ELBO Objective** 
The ELBO for continuous-time diffusion models can be simplified to the following expression (adopted in our setting):
$$
\log p(\boldsymbol{a}_0|s)\geq\mathrm{ELBO}(\boldsymbol{a}_0|s)=-\frac12\mathbb{E}_{t\thicksim\mathcal{U}(0,1),\boldsymbol{\varepsilon}\thicksim\mathcal{N}(0,\boldsymbol{I})}\left[w(t)\|\mu_\phi(\boldsymbol{a}_t,t|\boldsymbol{s})-\boldsymbol{a}_0\|_2^2\right]+c,
$$
where $\boldsymbol{a}_t= \alpha _t\boldsymbol{a}_0+ \sigma _t\boldsymbol{\varepsilon }, w( t) = - \frac {\mathrm{dSNR}( t) }{\mathrm{d} t}$, and the signal-to-noise ratio SNR$( t) = \frac {\alpha _t^2}{\sigma _t^2}, c$ is a constant not relevant to $\phi.$ 
- Since we always assume that the SNR$(t)$ is strictly monotonically decreasing in $t$, thus $w(t)>0$. The validity of the ELBO is maintained regardless of the schedule of $\alpha_t$ and $\sigma_t.$
- Kingma and Gao [2024] generalized this theorem stating that if the weighting function $w(t)=-v(t)\frac{\mathrm{dSNR}(t)}{\mathrm{d}t}$, where $v(t)$ is monotonic increasing function of $t$, then this weighted diffusion denoising loss is equivalent to the ELBO as defined in Equation 3. 


### Diffusion Trust Region Loss

For any given $s$ and a fixed diffusion model $\mu_\phi$, the loss is to find the optimal generation function $\pi_{\boldsymbol{\theta}}(\cdot|\boldsymbol{s})$ that can minimize the diffusion-based trust region (TR) loss:
$$
\mathcal{L}_{\mathrm{TR}}(\theta)=\mathbb{E}_{t,\boldsymbol{\varepsilon}\thicksim\mathcal{N}(0,\boldsymbol{I}),\boldsymbol{s}\thicksim\mathcal{D},\boldsymbol{a}_\theta\thicksim\pi_\theta(\cdot|\boldsymbol{s})}[w(t)\|\mu_\phi(\alpha_t\boldsymbol{a}_\theta+\sigma_t\boldsymbol{\varepsilon},t|\boldsymbol{s})-\boldsymbol{a}_\theta\|_2^2],
$$
where $\pi_{\theta}(\boldsymbol{a}|\boldsymbol{s})$ is a <font color="#ff0000">one-step generation policy</font>, such as a Gaussian policy.

---
**Theorem 1.**  policy $\mu$ satisfies the ELBO condition of Equation 3, then the Diffusion Trust Region Loss aims to maximize the lower bound of the distribution mode $\max _{\boldsymbol{a}_0}\log p( \boldsymbol{a}_0| \boldsymbol{s})$ for any given $s$.

$Proof.$ For any given state $\boldsymbol{s}$
$$
\begin{aligned}\max_{\boldsymbol{a}_0}\log p(\boldsymbol{a}_0|\boldsymbol{s})&\geq\max_\theta\mathbb{E}_{\boldsymbol{a}_\theta\sim\pi_\theta(\cdot|\boldsymbol{s})}\left[\log p(\boldsymbol{a}_\theta|\boldsymbol{s})\right]\geq\max_\theta\mathbb{E}_{\boldsymbol{a}_\theta\sim\pi_\theta(\cdot|\boldsymbol{s})}\left[\mathrm{ELBO}(\boldsymbol{a}_\theta|\boldsymbol{s})\right]\\&=\min_\theta\frac12\mathbb{E}_{t\thicksim\mathcal{U}(0,1),\boldsymbol{\varepsilon}\thicksim\mathcal{N}(0,\boldsymbol{I}),\boldsymbol{a}_\theta\thicksim\pi_\theta(\cdot|\boldsymbol{s})}\left[w(t)\|\mu_\phi(\boldsymbol{a}_\theta+\sigma_t\boldsymbol{\varepsilon},t|\boldsymbol{s})-\boldsymbol{a}_\theta\|_2^2\right]\end{aligned}
$$
Then, during training, we consider all states $s$ in $\mathcal{D}.$ Thus, by taking the expectation over $s\sim\mathcal{D}$ on both sides and setting $t\sim\mathcal{U}(0,1)$, we derive the loss described in Equation 4.

<font color="#00b050">Not fully understand</font>

---

Unlike other diffusion models that generate various modalities by optimizing $\phi$ to learn the data distribution, our method specifically aims to generate actions (data) that <font color="#ff0000">reside in the high-density region</font> of the data manifold specified by $\mu_\phi$ through optimizing $\theta.$ 

> The idea is similar to Langevin Dynamics.

Thus, the loss effectively creates a trust region defined by the diffusion-based behavior-cloning policy, within which the one-step policy $\pi_{\theta}$ can move freely. If the generated action deviates significantly from this trust region, it will be heavily penalized.

---
**Remark 2.** This loss is also closely connected with Diffusion- GAN and and EB-GAN, where the discriminator loss is considered as:
$$
D(\boldsymbol{a}_\theta|s)=\|Dec(Enc(\boldsymbol{a}_\theta)|s)-\boldsymbol{a}_\theta\|_2^2
$$
In our model, the process of adding noise, $\alpha _t\boldsymbol{a}_\theta + \sigma _t\boldsymbol{\epsilon }$, functions as an encoder, and $\mu_{\phi}(\cdot|s)$ acts as a decoder. Thus, this loss can also be considered as a discriminator loss, which determines whether the generated action $a_{\theta}$ resembles the training dataset.

---
This approach makes the generated action $a_\theta$ appear similar to in-sample actions and penalizes those that differ, thereby effectuating behavior regularization. 

### Diffusion Trusted Q-Learning

- Introduce a dual-policy approach Diffusion Trusted Q-Learning (DTQL): a diffusion policy for pure **behavior cloning** and a **one-step policy** for actual deployment.
- bridge the two policies through our newly introduced diffusion trust region loss
- The trust region loss is optimized efficiently through each diffusion timestep without requiring the inference of the diffusion policy.
- DTQL not only maintains an <font color="#00b050">expressive exploration region</font> but also facilitates efficient optimization.

**Policy Learning** 
- we utilize an <font color="#ff0000">unlimited number of timesteps</font> and construct the diffusion policy $\mu_\phi$ in a continuous time setting, based on the schedule outlined in EDM. 
- we can instantiate one typical one-step policy $\pi_\theta(a|s)$ in two cases, Gaussian $\pi_{\theta}(\boldsymbol{a}|\boldsymbol{s})=\mathcal{N}(\mu_{\theta}(\boldsymbol{s}),\sigma_{\theta}(\boldsymbol{s}))$ or Implicit $\boldsymbol a_\theta=\pi_{\theta}(\boldsymbol{s},\boldsymbol{\varepsilon}),\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\boldsymbol{I}).$ 
- Then, we optimize $\pi_{\theta}$ by minimizing the introduced diffusion trust region loss and typical Q-value function maximization, as follows.
$$
\mathcal{L}_\pi(\theta)=\alpha\cdot\mathcal{L}_{\mathrm{TR}}(\theta)-\mathbb{E}_{\boldsymbol{s}\sim\mathcal{D},\boldsymbol{a}_\theta\sim\pi_\theta(\boldsymbol{a}|\boldsymbol{s})}[Q_\eta(s,\boldsymbol{a}_\theta)],
$$
where $\mathcal{L}_\mathrm{TR}(\theta)$ serves primarily as a behavior-regularization term, and maximizing the Q-value function enables the model to preferentially sample actions associated with higher values. 
- Here we use the <font color="#ff0000">double Q-learning trick</font> where $Q_\eta(\boldsymbol{s},\boldsymbol{a}_{\boldsymbol{\theta}})=\min(Q_{\eta_1}(\boldsymbol{s},\boldsymbol{a}_{\boldsymbol{\theta}}),Q_{\eta_2}(\boldsymbol{s},\boldsymbol{a}_{\boldsymbol{\theta}})).$ 
- If Gaussian policy is employed, it necessitates the introduction of an <font color="#ff0000">entropy term</font> $-\mathbb{E}_{\boldsymbol{s},\boldsymbol{a}\sim\mathcal{D}}[\log\pi_{\boldsymbol{\theta}}(\boldsymbol{a}|s)]$ to maintain an exploratory nature during training. 

**Q-Learning**
We utilize Implicit Q-Learning (IQL) to train a Q function by maintaining two Q-functions $(Q_{\eta_1},Q_{\eta_2})$ and one value function $V_\psi$, following the methodology outlined in IQL. The loss function for the value function $V_\psi$ is defined as:
$$
\mathcal{L}_V(\psi)=\mathbb{E}_{(\boldsymbol{s},\boldsymbol{a}\thicksim\mathcal{D})}\left[L_2^\tau\left(\min(Q_{\eta_1^{\prime}}(s,\boldsymbol{a}),Q_{\eta_2^{\prime}}(\boldsymbol{s},\boldsymbol{a}))-V_\psi(\boldsymbol{s})\right)\right],
$$

where $\tau$ is a quantile in [0,1], and $L_2^\tau(u)=|\tau-\mathbf{1}(u<0)|u^2.$ When $\tau=0.5,L_2^\tau$ simplifies to the $L_2$ loss. When $\tau>0.5,L_\psi$ encourages the learning of the $\tau$ quantile values of $Q.$ The loss function for updating the Q-functions, $Q_{\eta_i}$, is given by:
$$
\mathcal{L}_Q(\eta_i)=\mathbb{E}_{(\boldsymbol{s},\boldsymbol{a},\boldsymbol{s}^{\prime}\boldsymbol{\sim}\mathcal{D})}\left[||r(\boldsymbol{s},\boldsymbol{a})+\gamma*V_\psi(\boldsymbol{s}^{\prime})-Q_{\eta_i}(\boldsymbol{s},\boldsymbol{a})||^2\right],
$$where $\gamma$ denotes the discount factor. This setup aims to minimize the error between the predicted Q-values and the target values derived from the value function $V_\psi$ and the rewards. The algorithm is as follows:

---
**Algorithm 1** Diffusion Trusted Q-Learning
Initialize policy network $\pi_\theta,\mu_\phi$, critic networks $Q_{\eta_1}$ and $Q_{\eta_2}$, and target networks $Q_{\eta_1^{\prime}}$ and $Q_{\eta_2^{\prime}}$, value function $V_\psi$
**for** each iteration **do**
   Sample transition mini-batch $\mathcal{B}=\left\{(\boldsymbol{s}_t,\boldsymbol{a}_t,r_t,\boldsymbol{s}_{t+1})\right\}\sim\mathcal{D}.$
1. Q-value function learning: Update $Q_{\eta_1},Q_{\eta_2}$ and $V_\psi$ by $\mathcal{L}_Q$ and $\mathcal{L}_V$ (Eqs. 6 and 7).
2. Diffusion Policy learning: Update $\mu_\phi$ by $\mathcal{L}(\phi)$ (Eq. 2).
3. Diffusion Trust Region Policy learning:  $\boldsymbol{a}_\theta \sim \pi _\theta ( \boldsymbol{a}| \boldsymbol{s})$, Update $\pi_\theta$ by $\mathcal{L}_\pi(\theta)$ (Eq. 5).
4. Update target networks: $\eta _i^\prime = \rho \eta _i^{\prime }+ ( 1- \rho ) \eta _i$ for $i=\{1,2\}.$
$\textbf{end for}$
---

## 3 Mode seeking behavior regularization comparison

Another approach to accelerate training and inference in diffusion-based policy learning involves utilizing <font color="#ff0000">distillation</font> techniques.
- using a trained diffusion model alongside another diffusion network to <font color="#ff0000">minimize the KL divergence between the two models</font>.

In our experimental setup, this strategy is employed for behavior regularization by
$$
\mathcal{L}_{\mathrm{KL}}(\theta)=D_{\mathrm{KL}}[\pi_\theta(\cdot|s)||\mu_\phi(\cdot|s)]=\mathbb{E}_{\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\boldsymbol{I}),\boldsymbol{s}\sim\mathcal{D},\pi_\theta(\boldsymbol{s},\boldsymbol{\varepsilon})}\left[\log\frac{p_\mathrm{fake}(\boldsymbol{a}_\theta|\boldsymbol{s})}{p_\mathrm{real}(\boldsymbol{a}_\theta|\boldsymbol{s})}\right]\
$$
where $\pi_\theta(\boldsymbol{s},\boldsymbol{\varepsilon})$ is instantiates as an one-step Implicit policy.

As we do not have access to the log densities of the fake and true conditional distributions of actions, the loss itself cannot be calculated directly. However, we are able to **compute the gradients**. The gradient of $\log p_{\mathrm{real}}(\boldsymbol{a}_{\boldsymbol{\theta}}|\boldsymbol{s})$ can be estimated by the diffusion model $\mu_\phi(\cdot|\boldsymbol{s})$, and the gradient of $\log p_{\mathrm{fake}}(\boldsymbol{a}_{\boldsymbol{\theta}}|\boldsymbol{s})$ can also be estimated by a diffusion model trained from fake action data $\boldsymbol{a}_\theta.$ 