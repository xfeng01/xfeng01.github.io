---
title: "Offline Reinforcement Learning with Generative Trajectory Policies"
authors: ["Xinsong Feng", "Leshu Tang", "Chenan Wang", "Haipeng Chen"]
pub: "Submitted to ICLR 2026"
image: "/gtp.svg"
date: 2025-09-24
description: "We propose Generative Trajectory Policies (GTPs), an ODE-based framework that unifies generative policies in offline RL, overcoming the performance–efficiency trade-off and achieving state-of-the-art results on D4RL benchmarks."
# paper: "https://arxiv.org/abs/2502.05537"
---

Generative models have emerged as a powerful class of policies for offline reinforcement learning (RL) due to their ability to capture complex, multi-modal behaviors. 
However, existing methods face a stark trade-off: slow, iterative models like diffusion policies are computationally expensive, while fast, single-step models like consistency policies often suffer from degraded performance. 
In this paper, we demonstrate that it is possible to bridge this gap.
The key to moving beyond the limitations of individual methods, we argue, lies in a unifying perspective that views modern generative models—including diffusion, flow matching, and consistency models—as specific instances of learning a continuous-time generative trajectory governed by an Ordinary Differential Equation (ODE).
This principled foundation provides a clearer design space for generative policies in RL and allows us to propose *Generative Trajectory Policies* (GTPs), a new and more general policy paradigm that learns the entire solution map of the underlying ODE.
To make this paradigm practical for offline RL, we further introduce two key theoretically principled adaptations. 
Empirical results demonstrate that GTP achieves state-of-the-art performance on D4RL benchmarks -- it significantly outperforms prior generative policies, achieving perfect scores on several notoriously hard AntMaze tasks.
