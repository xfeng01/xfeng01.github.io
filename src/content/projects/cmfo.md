---
title: "Solving Constrained Optimization Problems as ODE-based Models Using Reinforcement Learning"
authors: [Han Meng, Xinsong Feng, Yang Li, Chenan Wang, Kishansingh Rajput, Malachi Schram, Haipeng Chen]
pub: "Submitted to AISTATS 2026"
image: "/cmfo.png"
date: 2025-10-02
description: "We propose CMFO (Constrained Markov Flow Optimizer), which unifies flow-matching generative models and reinforcement learning to solve constrained optimization problems with improved efficiency and feasibility."
# link: "https://arxiv.org/abs/2502.05537"
---

Previous learning-to-optimize (L2O) methods on constrained optimization problems often treat neural networks as initializers that generate approximate solutions requiring substantial post-hoc refinements. 
This approach overlooks a key insight: Solving complex optimization problems often requires iterative refinement of candidate solutions, a process naturally aligned with the Markov Decision Process (MDP) and reinforcement learning (RL) framework. 
We show that within the MDP framework, RL and Ordinary Differential Equation (ODE)-based generative models (e.g., diffusion, flow matching) are formally equivalent, unifying them as trainable optimizers. 
Building on our unified perspective, we propose to train a flow-matching model within an RL paradigm as a learnable refinement mechanism, thereby incorporating constraint satisfaction directly into the optimization process.
To further enhance feasibility, we introduce a minimal correction step that adjusts solutions to ensure constraint compliance.
Empirical results demonstrate that our approach achieves state-of-the-art performance across a range of constrained optimization tasks, yielding improvements in efficiency, solution quality, and feasibility over prior baselines.
