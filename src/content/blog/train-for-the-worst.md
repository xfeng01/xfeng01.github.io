---
title: "Train for the Worst, Plan for the Best: What Really Matters in Masked Diffusion"
description: "A note on why masked diffusion models pay a harder training objective in exchange for inference-time flexibility, and why token ordering matters most when the best decoding order is instance-dependent."
date: 2026-03-27
tags:
  - Diffusion Language Models
  - Inference
  - Reasoning
---

One paper I enjoyed recently is **Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions**.

What I like about this paper is that the main idea is actually very simple. It is not primarily a paper about building a much more complicated model. Instead, it clarifies a fundamental trade-off in masked diffusion models (MDMs):

> **MDMs trade off complexity at training time with flexibility at inference time.**

I think this is exactly the right lens for understanding the difference between autoregressive models (ARMs) and MDMs.

For generative modeling, I often think in terms of

$$
\text{generation} = \text{model} + \text{process}.
$$

AR models fix the process almost completely: generate token 1, then token 2, then token 3, always in a predetermined order. This makes training easier, because the model only needs to learn one factorization. But it also limits what inference can do.

MDMs sit on the opposite side. They do **not** commit to a single generation order during training. That makes the learning objective much harder, but in return inference becomes much more flexible.

So the paper asks a very clean question:

> **Are the benefits of inference flexibility for MDMs enough to outweigh the drawbacks of training complexity?**

My short answer is: **yes, but only when the decoding strategy is good enough.**



## 1 Order-agnostic training: MDM learns many more conditional problems

Let us first compare the training objectives.

For a standard left-to-right autoregressive model, the objective is

$$
\log p_\theta(x)
=
\sum_{i=0}^{L-1} \log p_\theta(x_i \mid x_{<i}).
$$

This is a very special objective: it only learns **one fixed order**, usually left-to-right.

The key theoretical insight of this paper is that the MDM objective can be rewritten as a loss over **all possible infilling masks**:

$$
\mathcal{L}_\theta
=
-
\sum_{M \subseteq [L],\, i \in M}
\frac{1}{|M|}
\frac{1}{\binom{L}{|M|}}
\mathbb{E}_{x_0 \sim p_{\text{data}}}
\left[
\log p_\theta \bigl(x_0^i \mid x_0[M]\bigr)
\right].
$$

This means that MDM is not just learning one conditional prediction problem. It is learning a huge family of masked prediction problems.

The paper then gives an even more revealing reformulation:

$$
\mathcal{L}_\theta
=
-
\mathbb{E}_{x_0 \sim p_{\mathrm{data}},\, \pi \sim \mathrm{Unif}(S_L)}
\left[
\sum_{i=0}^{L-1}
\log p_\theta \bigl(x_0^{\pi(i)} \mid x_0[\pi\{i,\dots,L-1\}]\bigr)
\right].
$$

This is the part I found especially important.

Conceptually, this says that MDM can be viewed as learning an **any-order autoregressive model**, averaged over all possible permutations. If we only keep one fixed permutation, this reduces to the autoregressive case. In that sense, standard AR training is just a special case of this broader objective.

So the difference becomes very clear:

- **AR training** learns one factorization under one fixed process.
- **MDM training** learns an expectation over all token orderings.

This is why MDM training is fundamentally harder. It has to solve exponentially many conditional subproblems rather than just one sequence of next-token predictions.

For me, this is the cleanest way to summarize the trade-off:

- **Training:** learn to solve an exponentially large number of infilling problems.
- **Inference:** decode tokens in essentially arbitrary order.



## 2 Training for the worst: not all masking subproblems are equally easy

Once we view MDM training in this way, the next point becomes almost inevitable:

> **Not all infilling subproblems are equally easy.**

Some masking problems are easy, while others are much harder. If the data has a natural token generation order, then order-aware training is usually much more tractable than order-agnostic training.

This is exactly the story developed in the paper. The authors show both theoretically and empirically that MDMs can end up training on computationally hard subproblems. In contrast, AR models only need to learn the much simpler subproblems induced by a fixed natural order.

The practical consequence is very important:

> **Some of the subproblems are poorly trained, and vanilla MDM inference that unmasks tokens in random order ends up evaluating exactly those poorly trained marginals.**

This sentence is, in my opinion, the bridge between the theory and the final algorithmic insight.

The issue is not merely that MDM training is harder in the abstract. The real issue is that the difficulty is highly imbalanced across subproblems. As a result, the model learns some conditional marginals much better than others.

This also helps explain **when** MDMs are actually useful.

The paper argues that the real advantage of MDMs appears in tasks that **do not share the same natural token generation order across all sequences**, such as logic puzzles, coding, and math reasoning.

This is a very important point.

MDMs are **not** strongest on tasks where every example naturally follows the same good order. They are strongest when the best order is **instance-dependent**.

In other words:

> **The real advantage of MDMs comes from being able to choose which token to generate next at inference time.**

When there is no universal natural order shared by all sequences, this flexibility becomes genuinely valuable.



## 3 Planning for the best: the whole paper is really about inference strategy

This is why, in the end, I see this paper as being primarily about **decoder strategy**.

Vanilla MDM inference works by repeatedly:

1. selecting a set of currently masked positions,
2. predicting token values for those positions.

The paper points out that the first step should **not** be random.

Instead of randomly selecting which masked tokens to unmask next, we should choose them strategically:

$$
S = \mathcal{F}(\theta, x_t) \subseteq \{ i \mid x_t^i = 0 \}.
$$

Then for each selected position $i \in S$, we sample from

$$
x_s^i \sim p_\theta(x^i \mid x_t).
$$

This is the core algorithmic move in the paper:

> **replace random decoding order with adaptive decoding order.**

The key insight is very simple:

> **Adaptive inference can sidestep the hard subproblems created during training.**

Instead of forcing the model to solve an arbitrary masking problem next, we let it pick the easiest or most confident positions first.

The paper proposes simple confidence-based strategies, such as:

- choosing positions with the largest top probability,
- or choosing positions with the largest margin between the top two probabilities.

I especially like the margin-based version. It captures uncertainty more faithfully: a high top probability is not always enough if the second-best option is almost equally likely.

What is elegant here is that the paper does **not** change the training objective. It argues that the logits of a pretrained MDM already contain enough information to identify a better decoding order. So the gain comes purely from a better inference process.

This is why I like the style of the paper. The main contribution is insight, and the final method is correspondingly simple.



## 4 My takeaway

My main takeaway from this paper is:

> **MDMs pay for inference flexibility with a much harder training objective.**

AR models fix the process and make training easier. MDMs leave the process open, so training becomes much harder, but inference becomes much more flexible.

This flexibility is not always useful. If the task has a single natural order shared by all examples, AR-style training still has a strong advantage. But if the best generation order depends on the instance, then MDMs can become very attractive, especially when the decoding process is adaptive.

So for me, the most important message of the paper is not merely that MDMs are flexible. It is that

> **the real power of MDMs lies in inference-time control over token ordering.**

That is why I see this as a paper about **process design** as much as model design.



## 5 An open question I care about: remasking

One limitation of the current analysis is that it focuses on **monotone unmasking**: once a token is revealed, it stays revealed.

But MDMs can in principle do more than that. One could imagine allowing already revealed tokens to be masked again and revised later.

This leads to a more general question:

> **Can remasking improve not only error correction, but also the induced joint distribution?**

To me, this is the most interesting next step beyond the current paper.

Once remasking is allowed, inference is no longer just about choosing a decoding order. It becomes a more general iterative editing or refinement process. That feels like a much richer view of diffusion-based generation, and potentially a much more powerful one as well.