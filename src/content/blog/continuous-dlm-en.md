---
title: "Diffusion for Language: Should the Basic Unit Really Be a Token?"
description: "A note on diffusion language models as a signal-design problem: if diffusion is a refinement process, what representation should it refine?"
date: 2026-05-29
lang: en
translationKey: continuous-dlm
tags:
  - Diffusion Language Models
---

Recently I have been thinking about a question: for diffusion language models, what should the basic unit of diffusion really be?

Many current diffusion language models, whether discrete mask diffusion or continuous embedding diffusion, still use the token position as the basic modeling unit. Even when the model denoises a full token embedding matrix, and even when the denoiser can see the global context through self-attention, its state space is still token-aligned. In other words, each diffusion state still corresponds to a token position.

This does not mean that token-level diffusion has no global semantics. On the contrary, one strength of continuous diffusion is exactly that it can perform global refinement over the whole $L \times d$ continuous representation field. It does not repair each token independently. It can jointly revise the consistency of the entire sequence. This ability is important, and it is part of why continuous diffusion has potential for language modeling.

But this also leads to a more fundamental question: if the advantage of continuous diffusion is global refinement over a continuous field, does the coordinate system of that field have to be tied to token positions?

I think the real issue here is signal design:

> token is a natural unit for surface realization, but it may not be the most natural diffusion state.

## 1 The Problem with Tokens Is Not Globality

One easy misunderstanding is to say that token-level diffusion is not global enough, so we need a higher-level representation. I do not think this is accurate.

If the denoiser applies self-attention over the entire sequence, then token-level embedding diffusion is already global. The model can certainly use the full context to revise local representations. So the problem with token-level diffusion is not that it cannot see the global context.

The more accurate issue is:

> token-level diffusion can be globally conditioned, but it is still token-aligned in its state space.

That is, the denoising objects are still continuous vectors at token positions. High-level semantics, reasoning structure, discourse relations, local wording, and syntactic details are all mixed together inside the same token-aligned space.

This means planning and realization are not clearly separated. The model has to decide both what the passage should express and what each token should be. For an autoregressive language model, this mixture is natural, because next-token prediction unfolds text step by step. But for a diffusion model, this may not be the most natural state space.

## 2 What Should the New Unit Be?

Of course, even if we think tokens are not necessarily the most natural diffusion state, we cannot immediately conclude that we should replace them with another unit, such as a semantic unit. That conclusion also needs care.

The issue is that tokens, despite their limitations, are at least explicit, supervised, and recoverable. Text is naturally a token sequence, so token-level modeling has a clear training target and a clear generation interface. By contrast, a higher-level diffusion unit is much harder to define. Should it correspond to a phrase, a clause, a sentence, or some learned latent block? Different tasks may also require different granularities.

This is why I do not want to state the claim as:

> the basic unit of diffusion definitely should not be a token.

A more careful statement is:

> token should not be assumed to be the only or the most natural diffusion unit.

In other words, this is not a simple choice between token and semantic unit. It is a representation question: what signal should continuous diffusion operate on?

If we really want to introduce a diffusion unit above tokens, it should satisfy several conditions. It should be compositional, so that it can represent the compositional structure of language. It should be recoverable, so that it can be converted back into text. It should also be jointly modelable, because what we care about is not a single unit, but the joint structure of the whole sequence.

Only under these conditions would a semantic unit be more than a vague intuition. It could become a new generative state space.

## 3 A Possible Standard: Compositional, Recoverable, Jointly Modelable

If we do not use tokens, then the new diffusion unit should satisfy at least a few conditions.

First, it should be compositional. The meaning of language is not a simple stack of tokens. It is built from phrases, sentences, reasoning steps, discourse structure, and other levels of composition. A good semantic unit should be able to express this higher-level compositional structure.

Second, it should be recoverable. A latent representation cannot just be an unrecoverable compressed vector. It must be recoverable into text. Otherwise, it is not really a generative language state.

Third, it should be jointly modelable. Semantic units should not be generated independently. What really matters is modeling the joint structure of the whole semantic sequence:

$$
p_\theta(z_{1:K}).
$$

Only then can the model capture discourse order, long-range consistency, and reasoning structure.

From this perspective, the goal of semantic-block diffusion is not simply to replace tokens with blocks. It is to define a new continuous signal: a shorter, more abstract, recoverable semantic latent sequence that can be jointly modeled.

## 4 NCP Provides a Related Signal

Next Concept Prediction gives an interesting reference point. It is not a diffusion model, but an autoregressive model. Its concepts are also not continuous semantic blocks, but discrete latent concepts obtained through VQ. Still, it shows one thing: prediction targets above the token level may indeed be valuable.

The basic idea of NCP is to aggregate the hidden states of multiple tokens into a concept-level representation, use VQ to build a concept vocabulary, predict the next concept, and then use the predicted concept to guide subsequent token generation.

This is close to the intuition behind semantic-block diffusion: instead of only predicting the next token, first predict a higher-level "concept" or "plan", and then unfold it into tokens.

But there are also key differences. NCP is still autoregressive:

$$
p(c_{t+1} \mid c_{\leq t}),
$$

while semantic-block diffusion aims to model the joint distribution of the entire semantic plan:

$$
p_\theta(z_{1:K}).
$$

In addition, NCP concepts are more like fixed-size token chunks, whereas semantic-block diffusion would ideally explore variable-length and compositional semantic units. The decoder in NCP broadcasts a predicted concept back to a fixed number of token positions, while a more natural form for semantic-block diffusion is a branching decoder:

$$
z_i \rightarrow x_{i,1:m_i}.
$$

That is, one semantic latent can unfold into multiple tokens, instead of being forcibly aligned with fixed token positions.

## 5 What This Means for Long Context

I think concept-level or semantic-block-level modeling can help with long context, but the benefit is not that it directly enlarges the context window.

More precisely, it reduces the effective sequence length for high-level modeling.

If the original sequence has $L$ tokens and is compressed into $K$ semantic blocks, where $K \ll L$, then the model faces a shorter semantic trajectory when doing high-level planning, rather than the full token sequence.

This is important for long-form generation. The difficulty in long text is often not a local token. It is whether the topic remains consistent, whether the argument is coherent, whether the reasoning steps are correct, and whether the earlier and later structures match. A semantic-block representation can let the model handle these high-level dependencies over a shorter sequence.

But this should not be overstated. Semantic blocks are more suitable for long-range semantic planning, not necessarily for all long-context retrieval problems. If the task requires precisely finding a name, number, code variable, or citation from earlier context, a compressed representation may be less stable than a token-level representation.

So the more accurate statement is:

> semantic-level modeling may improve long-range planning, but it does not automatically solve all long-context problems.

## 6 Back to Diffusion: What Should Be Denoised?

In the end, I think the core question in this direction is not whether tokens are good or bad. It is:

> What is the right continuous signal for language diffusion?

The advantage of continuous diffusion is that it can perform iterative refinement over a continuous structure. Existing token-level continuous diffusion has already shown that this kind of refinement has potential in language. But if diffusion's strength is global refinement, then we should seriously ask whether the object being refined should still be token-aligned embeddings.

Maybe tokens are better understood as realization units, while semantic blocks are better understood as planning units.

One possible architecture is:

$$
x_{1:L} \rightarrow z_{1:K}, \quad K \ll L.
$$

Here, $z_{1:K}$ is a sequence of semantic-block latents. The diffusion model learns the joint distribution over this latent sequence:

$$
p_\theta(z_{1:K}).
$$

Then a branching decoder unfolds each semantic latent into a variable-length token span:

$$
z_i \rightarrow x_{i,1:m_i}.
$$

In this setup, diffusion is responsible for generating and refining a global semantic plan, while the decoder is responsible for realizing that plan into concrete text.
