---
title: "Some Basic Thoughts on Consistency in Continuous and Discrete Diffusion Language Models"
description: "A note on why continuous diffusion language models may express global consistency more naturally than discrete masked diffusion."
date: 2026-06-02
lang: en
translationKey: continuous-vs-discrete
tags:
  - Diffusion Language Models
---

Recently I have started working on continuous diffusion language models, and a natural question is: what is the essential difference between continuous DLMs and discrete DLMs? My intuition is that continuous DLMs are mechanically better suited for handling consistency. For example:

> The capital of X is Y.

If the model needs to generate both `X` and `Y`, the two positions cannot decide independently. If `X = France`, then `Y` should be `Paris`; if `X = Japan`, then `Y` should be `Tokyo`. For discrete masked diffusion, this usually requires multiple steps: first fill in `X`,

> The capital of France is [MASK].

and then fill in `Paris` based on the already fixed `France`. Continuous diffusion, by contrast, can push both positions toward a consistent direction during the same denoising process: `X` and `Y` can co-evolve in the same continuous space and eventually land on a consistent pair.

The difference here is not the backbone. Continuous DLMs and discrete DLMs can use similar internal architectures, such as DiT. Both can see the global sequence through attention. So the issue is not that discrete models cannot see global information while continuous models can. The real difference is what they denoise.

For a discrete DLM, the current state is usually a discrete sequence: each position is either an already fixed token or `[MASK]`. The model uses the existing context to recover the masked tokens. This means that although one `[MASK]` position can attend to other positions and see that they are also `[MASK]`, it does not know what those other masked positions will eventually be sampled as. Attention only ensures that each position's logits can depend on the global state. It does not ensure that multiple tokens sampled in the same step are actually mutually dependent.

Formally, if the currently masked positions are $\mathcal{M}_t$ and the unmasked positions are $\bar{\mathcal{M}}_t$, then the reverse step really needs to predict the masked subset:

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
$$

But common discrete masked diffusion approximates this joint distribution as a product over positions:

$$
p_\theta(x_{0,\mathcal{M}_t} \mid x_t)
\approx
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

If we write this for the whole sequence, the unmasked positions are simply preserved:

$$
p_\theta(x_0 \mid x_t)
\approx
\mathbf{1}_{x_{0,\bar{\mathcal{M}}_t}=x_{t,\bar{\mathcal{M}}_t}}
\prod_{i \in \mathcal{M}_t}
p_\theta(x_{0,i} \mid x_t)
$$

The key point is that each factor $p_\theta(x_{0,i} \mid x_t)$ can depend on the full $x_t$, so the model is not missing global context. The problem is that, during same-step sampling, the outputs at masked positions are still written in a factorized form.

Returning to the capital example, what we really want is not

$$
p(X,Y \mid c) = p(X \mid c) \cdot p(Y \mid c)
$$

but a joint distribution:

$$
p(X,Y \mid c)
$$

because once `X` is sampled as `France`, `Y` should become more likely to be `Paris`; once `X` is `Japan`, `Y` should become more likely to be `Tokyo`. In

> The capital of [MASK] is [MASK].

the logits at both `[MASK]` positions can see the whole sentence, and both know that the other position is `[MASK]`. But if they are sampled in the same step, the second position does not know whether the first one finally became `France`, `Japan`, or `China`. This is the source of the consistency issue. Therefore, the common solution in discrete masked DLMs is to use multi-step refinement to turn simultaneous dependency into sequential dependency: generate part of the sequence first, then use the generated tokens as new context to generate another part. The context becomes more complete, and recovery becomes easier. But the state in this process is still hard/discrete: a position is either `[MASK]` or already committed to a token.

Continuous diffusion is different: it denoises not discrete tokens, but the entire embedding space, or the whole sequence latent matrix. Suppose the hidden representation of a sentence is

$$
X_t \in \mathbb{R}^{L \times d}
$$

The reverse process can be understood as

$$
\frac{dX_t}{dt} = f_\theta(X_t, t)
$$

Here, $X_t$ is the continuous representation of all positions in the sentence. The update at each position can depend on the entire $X_t$, so the model is not separately mapping a particular `[MASK]` back to a token. It is performing global denoising over the whole sequence latent. The key point is that the intermediate state of continuous diffusion is not a hard token, but a continuous vector. It can carry soft hypotheses that have not yet been discretized.

Using the same sentence again:

> The capital of X is Y.

`X` and `Y` do not need to independently decide what tokens they are at the beginning. They can co-evolve in continuous space, so that the whole latent state gradually moves toward

$$
(\text{France-like latent}, \text{Paris-like latent})
$$

or

$$
(\text{Japan-like latent}, \text{Tokyo-like latent})
$$

In other words, consistency does not have to wait until one token is sampled and then be repaired afterwards. It can form as a coherent semantic hypothesis in continuous space before final discretization.

So my current understanding is: the intermediate state of a discrete DLM is hard. A position is either unknown or already committed to a token. The intermediate state of a continuous DLM is soft. A position can gradually move in embedding space and form a joint semantic state together with other positions. This is why continuous DLMs are mechanically better suited for global consistency.

But this also needs care: continuous DLMs do not automatically guarantee consistency. More precisely, their continuous states allow multiple positions to co-evolve before final discretization, so they can express synchronous consistency more easily. Their vector field or score is defined over the whole sequence latent $X_t$, so they model joint evolution more naturally. Whether consistency actually works still depends on the latent representation, decoder, training objective, and final discretization. If the latent space is not well learned, or if the decoder is still strongly per-position independent, continuous diffusion can also be inconsistent.

So we should not say:

> continuous always solves consistency, and discrete never can.

A more accurate statement is:

> common forms of discrete masked DLMs tend to face a factorization barrier; continuous DLMs, because they perform global denoising in continuous space, are mechanically better suited for expressing global consistency.

This is my current preliminary understanding of continuous DLMs. What is interesting about continuous DLMs is not just replacing tokens with embeddings, nor simply turning discrete diffusion into continuous diffusion. It is that language generation changes from "recovering tokens position by position" into "the joint evolution of the whole sequence latent." This may be the most essential difference between continuous DLMs and discrete DLMs.
