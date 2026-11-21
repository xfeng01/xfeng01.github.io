---
title: "From Recursive Generalization to Generation Order: Some Thoughts on Neural Network Learning and Generation"
description: "A note on 'neural networks are already clean'."
date: 2026-11-20
lang: en
translationKey: recursive-generalization
tags:
  - Representation Learning
  - Generative Models
  - Mechanistic Interpretability
---

I recently read Goodfire's [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) and found it very interesting.

The article mainly discusses mechanistic interpretability. What interested me most was the authors' discussion of **the nature of neural networks**, especially how neural networks learn and organize their internal representations.

## 1 Recursive Generalization: What Are Neural Networks Actually Learning?

A longstanding question in mechanistic interpretability is polysemanticity: why does the same neuron respond to multiple seemingly unrelated concepts? One influential explanation has been superposition. Because a network has limited representational capacity, it may have to encode different features in a shared representation space, even when those features have no obvious semantic connection. Goodfire proposes another explanation: **Recursive Generalization**.

The authors found that many seemingly unrelated neuronal responses can actually be traced back to a series of local, reasonable generalizations. For example, a neuron may initially respond to `"iy"`, then extend to `"diy"`, and eventually respond to `"do it yourself"`, even though the latter no longer contains the original character pattern. This is more like a tree continually growing new branches: an existing feature extends in related directions, and those new branches can generalize further. The authors summarize this as: **Generalization is fractal.**

This suggests that the seemingly complex and messy representations inside neural networks may be organized structures formed through recursive generalization, rather than arbitrary combinations of unrelated concepts. The article also makes another interesting observation: when model capacity is small, neurons tend to represent broader concepts; as capacity increases, those concepts may split into more specific features.

<figure>
  <img src="/blog/learning-of-neural-networks/recursive-generalization.svg" alt="Left: a tree branching to the right; a neuron's response grows from iy to diy to do it yourself, while gray side branches keep splitting. Right: with small capacity there is one broad feature; with larger capacity it splits into three more specific features." width="680" height="252" loading="lazy" />
  <figcaption>Schematic. Left: a neuron first responds to <code>"iy"</code>, then extends to <code>"diy"</code>, and eventually responds to <code>"do it yourself"</code>; each existing feature may keep branching in related directions (gray branches indicate other possible generalizations). Right: when model capacity is small, neurons tend to represent broader concepts; as capacity increases, those concepts may split into more specific features.</figcaption>
</figure>

This makes me think that coarse-to-fine organization may be more than a hierarchy we design by hand. It may also be related to how neural networks naturally form representations.

## 2 Circuit Ecology: Why Does Recursive Generalization Emerge?

Beyond Recursive Generalization itself, I was also interested in the authors' explanation of how it arises: **Circuit Ecology**.

The authors hypothesize that different neurons and circuits within a neural network compete with one another. When a model needs to learn a new pattern, an existing circuit that can already partially process the relevant information may capture the training error signal faster than a new circuit built from scratch. In other words, **existing computational structures have a first-mover advantage in optimization.** New functions do not necessarily require independent neurons; they may grow directly out of existing computational structures. As training continues, these extensions become the foundation for further learning.

Recursive Generalization may therefore be a result of ordinary gradient-based optimization and feature reuse, rather than a special encoding trick. I think this can be viewed as a theoretical hypothesis about learning dynamics, although it is not yet a rigorously established learning theory.

It made me start thinking: why can neural networks spontaneously develop reusable, organized representations at different levels of abstraction, starting from random initialization? Why would optimization tend to reuse related features instead of building a completely independent representation for every new concept? The authors do not fully answer these questions, but Circuit Ecology offers an interesting direction: **perhaps representational structure depends not only on the data itself, but also on competition and reuse among computational structures during optimization.**

## 3 What Does This Have to Do with Coarse-to-Fine Generation?

Recently, I have been thinking about coarse-to-fine diffusion and what it means for the generation process itself. Goodfire's article led me to reconsider a question: **could coarse-to-fine be more than an effective generation strategy, and also reflect a more general principle of learning?**

Of course, Recursive Generalization describes how neurons develop during learning, while coarse-to-fine generation describes how information is progressively determined during generation. They are not the same thing. But there is a structural connection worth studying: **first establish a reusable shared structure, then handle more specific information on top of it.** For example, in video generation, different frames share many dependencies: scene layout, object identity, motion state, and long-term temporal relationships. Suppose there is a coarse representation that captures this globally shared information. Given that representation, the remaining dependencies between different frames and spatial regions may be substantially weaker. This means we could generate the global state first, then generate local details conditioned on it. By comparison, if all variables are generated jointly from the beginning, the model may need to continually coordinate these global dependencies throughout sampling. This may help explain why coarse-to-fine generation is particularly attractive for tasks such as video and long-form text, which require long-term consistency.

<figure>
  <img src="/blog/learning-of-neural-networks/shared-dependencies.svg" alt="Left: five frames generated jointly, with a dependency link between every pair of frames, including long-range pairs. Right: a coarse representation z is generated first and points to each frame; only faint dashed links remain between neighboring frames." width="680" height="240" loading="lazy" />
  <figcaption>Schematic. Left: when all frames are generated jointly, every pairwise dependency, including long-range ones, has to be coordinated during sampling. Right: first determine a coarse representation <em>z</em> that captures globally shared information, then generate the frames conditioned on it; the remaining dependencies between frames (dashed) may be substantially weaker.</figcaption>
</figure>

But I think the key question here goes beyond coarse-to-fine in the sense of spatial scale: **Can we resolve shared dependencies first, so that the remaining generation becomes easier?** If a coarse representation can account for dependencies among many variables, determining it first may substantially simplify subsequent conditional generation. Whether this actually improves the learnability of generative models, however, still requires independent validation.

## 4 Coarse-to-Fine Is Fundamentally Part of a Broader Generation Order Problem

Thinking further, I feel that coarse-to-fine may simply be a special case of generation order. Generation order has long been an important research question in discrete diffusion language models. For example, masked diffusion can use different unmasking schedules to determine which tokens are generated first and which come later. But these methods are usually still constrained by a fixed token representation. They mainly select a better generation order for a given set of tokens.

What interests me more now is a broader question: **In what order should a generative model resolve different kinds of uncertainty?** The objects being generated need not be tokens. They could also be:

- Global semantics or abstract concepts
- Local content and details
- Spatial frequencies or structures at different scales
- Dynamic states shared across time
- Learned latent variables or certain continuous representation directions

Even the order itself need not be a fixed, discrete permutation. In continuous diffusion, different kinds of information can evolve simultaneously, while their uncertainty may be resolved at different rates. Generation order can therefore also be a continuous, dynamic process of information resolution. From this perspective, autoregressive generation, discrete diffusion, continuous diffusion, coarse-to-fine generation, and hierarchical generation can all be viewed as different ways of organizing information generation.

<figure>
  <img src="/blog/learning-of-neural-networks/resolution-order.svg" alt="Three rows for global, mid-scale and local-detail information; shading is the remaining uncertainty over generation progress. Left: each row drops to zero in a single step, one after another. Right: all rows decay smoothly and overlap in time, with global information resolved earliest and local detail latest." width="680" height="250" loading="lazy" />
  <figcaption>Schematic. Shading shows the remaining uncertainty of each kind of information. Left: a fixed discrete order, where each kind of information is determined at a single step. Right: a continuous process of information resolution, where different kinds of information evolve simultaneously but their uncertainty is resolved at different rates.</figcaption>
</figure>

A natural question then is: **what makes a good generation order?** I currently lean toward the following hypothesis:

> **The learnability of a generative model depends not only on its architecture and objective, but also on the order in which different dependencies are resolved.**

In other words, the difficulty of learning a generative model may depend not only on its network architecture and training objective, but also on the order in which it handles different dependencies in the data. The key is not simply to generate the variables with the lowest conditional entropy, or those that are easiest to predict, first. Some variables may themselves be difficult to generate, but once determined, they can greatly reduce the conditional complexity of many other variables. Such variables may be worth generating first. This also reminds me of an interesting finding in *The Flexibility Trap*: some high-entropy logical connectives are key branching points that determine the reasoning trajectory. Yet a discrete DLM may bypass these tokens because they are difficult to predict, generating other low-entropy tokens first and prematurely constraining subsequent reasoning paths. This also seems to suggest that **generation order should depend not only on how easy a variable itself is to generate, but also on its effect on the rest of the generation process.**

Therefore, **the best generation order is not necessarily coarse-to-fine, nor a fixed token order. It should be determined jointly by the dependency structure of the data and the model's learning capabilities.**

## 5 Can We Derive an Optimal Generation Process?

Based on the ideas above, I think one research question is:

**Given a data distribution and its dependency structure, can we derive a generation process that achieves the best learnability and efficiency?**

In other words, if we know how the variables in the data depend on one another, can we systematically determine which variables should be generated first, which dependencies should be resolved first, which variables should be generated jointly, and which variables should be generated only after certain shared states have been determined? Going further, can we theoretically explain why one generation order is easier to learn than another?

Learnability needs a rigorous definition here. It could concern generation error under finite data, model capacity, and training compute. It could also involve learning complexity, the complexity of conditional distributions, and the final generation quality under a finite sampling budget. I may think about this more carefully later.

It is worth noting that, from the perspective of probabilistic factorization, different generation orders do not necessarily change the amount of information contained in the data distribution itself. The advantage of a generation order may therefore come from **reorganizing a complex joint generation problem into a set of subproblems that are easier for finite models to learn, approximate, and sample from**, rather than reducing the total amount of information that needs to be generated. Of course, some data distributions may naturally be better suited to joint generation than to an explicit coarse-to-fine process. So coarse-to-fine is not necessarily a particularly good way to generate data either.

What interests me is: **can we start from the dependency structure of the data and find more general principles that determine the learnability of a generation process?** Goodfire's Recursive Generalization offers an inspiring perspective: the internal representations of neural networks may be organized into reusable structures that emerge naturally during learning, rather than being arranged arbitrarily. If there is a deeper connection between these structures and the learnability of generation processes, we may be able to better understand what should be generated first, and why.
