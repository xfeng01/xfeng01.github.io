---
title: Paper Notes -- Consistency Models
published: 2025-01-18
description: Notes for this paper
tags:
  - generative-AI
category: Paper
draft: false
---
## Consistency Training

Usually, we rely on a pre-trained score model $\boldsymbol{s}_\phi(\mathbf{x}, t)$ to approximate the ground truth score function $\nabla \log p_t(\mathbf{x})$. It turns out that we can avoid this pre-trained score model altogether by leveraging the following unbiased estimator:
$$
\nabla \log p_t\left(\mathbf{x}_t\right)=-\mathbb{E}\left[\left.\frac{\mathbf{x}_t-\mathbf{x}}{t^2} \right\rvert\, \mathbf{x}_t\right]
$$
where $\mathbf{x} \sim p_{\text {data }}$ and $\mathbf{x}_t \sim \mathcal{N}\left(\mathbf{x} ; t^2 \boldsymbol{I}\right)$. That is, given $\mathbf{x}$ and $\mathbf{x}_t$, we can estimate $\nabla \log p_t\left(\mathbf{x}_t\right)$ with $-\left(\mathbf{x}_t-\mathbf{x}\right) / t^2$.

This unbiased estimate suffices to replace the pre-trained diffusion model in consistency distillation when using the Euler method as the ODE solver in the limit of $N \rightarrow \infty$.

One **important trick** will always be used is 
$$
\nabla \log f = \frac{{\nabla f}}{f}
$$
or 
$$
\nabla f = f \cdot \nabla \log f
$$
