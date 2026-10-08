---
title: "Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization"
authors: ["Xinsong Feng", "Peng Du", "Zhizhuo Yang", "Daniel M. Bikel", "Jiayun Wang", "Haipeng Chen"]
pub: "Submitted to ICLR 2027"
image: "/bld.svg"
date: 2026-10-07
description: "We introduce Branching Latent Diffusion (BLD), which compresses 1024 tokens into 64 block latents and reconstructs text through parallel autoregressive branches, achieving over 6× higher throughput than both ELF-L and the AR baseline in same-GPU evaluations."
paper: "https://arxiv.org/abs/2610.09311"
---

Diffusion language models (DLMs) generate text through iterative parallel refinement, offering the potential for higher throughput than autoregressive (AR) decoding.
However, most DLMs still maintain one generative state per token, so every denoising step processes a state sequence as long as the output sequence, limiting the throughput gains from parallel generation.
Continuous DLMs provide an additional degree of freedom: a single continuous state can represent multiple tokens, allowing diffusion to operate on a much shorter latent sequence.
We introduce *Branching Latent Diffusion (BLD)*, which exploits this flexibility by compressing a 1024-token sequence into only 64 block latents, a $16\times$ reduction.
BLD combines latent compression with *branching token realization*, where each latent is decoded by a local AR branch and all branches run in parallel.
Because strong compression makes joint latent generation difficult, BLD generates the latents in groups, conditioning each group on previously generated latents.
In end-to-end evaluation on the same GPU, BLD reduces generation FLOPs by more than $80\times$ and increases throughput by more than $6\times$ relative to the similarly sized ELF-L baseline.
Compared with the AR baseline, BLD achieves more than $6\times$ higher throughput and more than $4\times$ lower latency.
Despite the compression, BLD maintains competitive local fluency and diversity, although long-range coherence remains challenging.
Overall, BLD shows that moving diffusion from token-level states to compressed latent sequences can substantially improve the efficiency of long-sequence generation.
