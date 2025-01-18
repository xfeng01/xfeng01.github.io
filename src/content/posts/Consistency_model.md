---
title: Paper Notes -- Consistency Models
published: 2025-01-18
description: Notes for this paper
tags:
  - paper-notes
  - generative-AI
category: Paper
draft: false
---
## Consistency Training

In typical setups, we rely on a pre-trained score model, $\boldsymbol{s}_\phi(\mathbf{x}, t)$, to approximate the true score function $\nabla \log p_t(\mathbf{x})$. However, we can bypass this pre-trained model by using the following unbiased estimator:
$$
\nabla \log p_t\left(\mathbf{x}_t\right)=-\mathbb{E}\left[\left.\frac{\mathbf{x}_t-\mathbf{x}}{t^2} \right\rvert\, \mathbf{x}_t\right]
$$
where $\mathbf{x} \sim p_{\text{data}}$ and $\mathbf{x}_t \sim \mathcal{N}(\mathbf{x}; t^2 \boldsymbol{I})$. This implies that, given $\mathbf{x}$ and $\mathbf{x}_t$, we can estimate $\nabla \log p_t(\mathbf{x}_t)$ as $-\left(\mathbf{x}_t - \mathbf{x}\right) / t^2$.

This unbiased estimate serves as a sufficient replacement for the pre-trained diffusion model in consistency distillation, particularly when using the Euler method as the ODE solver in the limit of $N \rightarrow \infty$.

---
A **key trick** commonly used is:
$$
\nabla \log f = \frac{{\nabla f}}{f}
$$
or equivalently,
$$
\nabla f = f \cdot \nabla \log f
$$
