---
title: "What Should a Continuous Language Latent Represent?"
description: "Lessons from trying to diffuse compressed language representations."
date: 2026-08-17
lang: en
translationKey: latent-represent
tags:
  - Diffusion Language Models
  - Continuous Latents
  - Representation Learning
---

In my previous post, I asked whether tokens are really the right basic unit for language diffusion.

A natural alternative is to compress multiple tokens into a shorter sequence of continuous latents, run diffusion over those latents, and then decode them back into text. Conceptually, this sounds attractive: language has structure above the token level, so perhaps diffusion should operate on a coarser unit.

That was the idea I started with. For example, take a sequence of $N=1024$ tokens and divide it into blocks of 16 tokens. An encoder compresses each block into one latent, giving $K=64$ continuous vectors:

$$
x_{1:1024}
\longrightarrow
z_{1:64}.
$$

A diffusion model learns the latent distribution,

$$
p_\theta(z_{1:64}),
$$

and a decoder expands each latent block back into tokens, for example,

$$
p_D(x_{k,i}\mid x_{k,<i},z_{1:K}).
$$

At first, the problem seemed straightforward:

> If the autoencoder can reconstruct the original text well, then we have learned a good latent space. The remaining task is just to train diffusion on it.

After spending a lot of time testing this idea, I now think this intuition is badly incomplete.

The central lesson is:

$$
\boxed{\text{Reconstruction is easy. Diffusion is not.}}
$$

And the harder question is no longer simply how much we should compress language.
It is:

$$
\boxed{\text{What should a continuous language latent actually represent?}}
$$





## 1 Reconstruction is surprisingly easy

Suppose every 16-token block is mapped into a 512-dimensional or even 192-dimensional continuous vector. This may look like a strong compression ratio in sequence length:

$$
1024\text{ tokens}\rightarrow64\text{ latents},
$$

but it is not necessarily a strong information bottleneck.
A continuous vector has a lot of representational capacity. If the encoder and decoder are trained jointly with token-level reconstruction loss, they can learn a very efficient codec.

The usual objective can be written as

$$
\mathcal L_{\mathrm{AE}}
=
-\log p_D(x\mid E(x)),
$$

which, for a token decoder, is simply the usual token-level cross entropy:

$$
\mathcal L_{\mathrm{AE}}
=
-\sum_i
\log p_D(x_i\mid x_{<i},E(x)).
$$

This objective does not care *why* some information helps reconstruction.
- If exact punctuation helps, the latent can store punctuation.
- If the exact word choice helps, the latent can store lexical identity.
- If document-specific details help, the latent can store those too.

As long as the information reduces token cross entropy, the reconstruction objective rewards putting it into the latent. And in practice, this works extremely well.

A learned 192-dimensional bottleneck can still reconstruct almost perfectly. In one of my experiments, a learned 192-dimensional representation achieved nearly the same oracle language-model perplexity as the original high-dimensional latent and recovered more than 99.9% of tokens exactly.

> So the autoencoding problem was not hard. The diffusion problem was.


## 2 Better reconstruction can make generation much worse

The most important counterexample came from comparing two different 192-dimensional representations.

A simple PCA projection of the original latent degraded reconstruction:

$$
\text{Oracle PPL}\approx55.7,
$$

but diffusion generation improved substantially:

$$
\text{Generated PPL}\approx394.
$$

Then I trained a learned 192-dimensional bottleneck.
Its reconstruction improved dramatically:

$$
55.7\rightarrow30.8.
$$

It was almost lossless.
But its generated text became dramatically worse:

$$
394\rightarrow3000+.
$$

The samples were not collapsing into repetitive loops. In fact, token diversity was often higher and repetition lower.
They were simply word salad.

This was the first very strong sign that
> reconstruction quality is not a reliable proxy for generative quality.

The learned representation had become a better codec and a much worse diffusion state.


## 3 Denoising MSE is not enough either

One might still think that the learned latent space was simply harder for the diffusion model to regress. But that explanation also failed.

Across several experiments, I repeatedly observed cases where pointwise denoising improved while generation became worse.

For example, some learned representations had substantially lower $x_0$ prediction MSE than PCA-192 at intermediate and late noise levels, yet their unconditional generation was several times worse.

Similarly, increasing high-$t$ training coverage improved the model on exact noised data:

$$
\mathrm{MSE}\downarrow,
\qquad
\text{decoder NLL on true }q_t\downarrow,
$$

but unconditional generation got worse. This suggests a broader lesson:

> Euclidean closeness in latent space is not the same as generative compatibility.

A diffusion model can be closer to the target latent in MSE while still producing a latent configuration that the decoder interprets badly.



## 4 Throwing information away can help

The PCA experiments made this even clearer. 
- Reducing the effective latent dimension from 512 to 192 made reconstruction worse but generation better. 
- At 128 dimensions, the prior became extremely easy to learn, but the representation itself had become too lossy.

Here, I use "rate" loosely to refer to how much information the latent can retain, with the number of retained PCA dimensions serving as a controllable proxy.

This produced a clear three-way trade-off:

$$
\boxed{
\text{Rate}
\;\longleftrightarrow\;
\text{Reconstruction Distortion}
\;\longleftrightarrow\;
\text{Transport Difficulty}.
}
$$

- Too much information makes the prior difficult.
- Too little information makes the decoder under-conditioned.
- Somewhere in the middle is a better generative operating point.

But even this is not the whole story.


A learned 192-dimensional bottleneck could recover almost all of the lost information and still fit into the same nominal dimension. The encoder simply reorganized the information into a more efficient continuous code. So:

> dimension is not information rate.

And even effective rate is not enough to characterize the problem.

> How the information is *organized* matters.




## 5 Local robustness does not solve global transport

Another natural hypothesis was that the **decoder was only reliable in a very small neighborhood around clean latents**.
This also turned out not to be the main issue.

The decoder was surprisingly robust to isotropic noise. Adding substantial random perturbations to normalized clean latents often preserved reconstruction almost perfectly.
I also trained stochastic autoencoders where the decoder explicitly learned to reconstruct from noisy latent inputs. 
This substantially enlarged the region around each clean latent in which the decoder could still reconstruct the text correctly.
But unconditional generation still became much worse.


The latent space behaved almost like an error-correcting code: 
- each clean latent had a wide decoding neighborhood; 
- small random perturbations were tolerated; 
- yet the set of **globally valid** latent configurations remained highly structured.


Diffusion could match many marginal statistics of the latent space while still failing to reproduce the correct joint distribution across latent features and blocks.

So another lesson emerged:
> local robustness does not imply global transportability.



## 6 The decoder is part of the problem

Initially, I thought most of the difficulty belonged to the latent prior.
That was also incomplete.

If I continued training the decoder only on clean latents, oracle reconstruction improved. But decoding from diffusion-generated latents became dramatically worse.
In one experiment, the exact same generated latent endpoint had roughly $\mathrm{PPL}\approx500$ under an older decoder, but around $\mathrm{PPL}\approx2000$ under a decoder that had been trained longer on clean latents.

The generated latent was exactly the same. Only the decoder had changed.


The clean-only decoder had become better at exploiting fine-grained details of the clean latent representation that diffusion-generated latents could not reliably reproduce.

So:
$$
\boxed{
\text{A decoder optimized for clean latents}
\neq
\text{a decoder robust to generated latents}.
}
$$



This creates a **train–inference distribution mismatch**.


The decoder is trained on clean latents $z_{\mathrm{clean}}$ but evaluated on generated ones $z_{\mathrm{generated}}$. Better clean reconstruction can therefore increase sensitivity to the structured errors produced by the diffusion model.



## 7 Useful latent features can be difficult to generate jointly

The problem was not simply that some latent dimensions were individually difficult to model.

I found cases where two latent features were each useful and relatively easy to generate on their own, yet their combination caused a much larger generation error.

This suggests that matching individual latent features is not enough. Diffusion also has to reproduce the correct **joint relationships** among them.

Importantly, the marginal distributions of individual features — and even simple pairwise correlations — could look almost correct, while the generated latent still followed the wrong context-dependent relationships.

In one case, several latent features encoded nonlinear lexical/content relationships in later text blocks. Diffusion did not simply lose these features; it combined them in a way that was internally plausible but inconsistent with the structure learned by the decoder.

So the issue was not that the information disappeared.

**The individual features were present, but they were combined in the wrong way.**



## 8 Static latent metrics often failed

This became one of the most useful practical lessons. I tried many plausible proxies for harmful latent directions:

- variance mismatch;
- decoder gradients;
- denoising error energy;
- gradient-error alignment;
- local curvature;
- covariance;
- marginal distance.

They often ranked the truly harmful directions incorrectly.

One direction could have very large decoder curvature, large prior error, and large coupling mismatch—and still have essentially zero causal effect on final text quality.

The metric that consistently worked was much simpler:

> perturb the generated latent a little and see whether the actual serving text gets better.

A small causal intervention,

$$
z_i'
=
z_i-\epsilon c_i(z),
\qquad
\epsilon\ll1,
$$

that slightly weakened a candidate latent relationship could consistently identify which relationships actually hurt generation, across different sampling seeds and independently trained diffusion priors.

> The useful signal was not static sensitivity, but whether a small intervention actually improved generation at inference time.



## 9 From causal diagnosis to latent calibration

The previous experiments suggested that some generated latent features were individually reasonable, but were combined in ways that hurt decoding.

This raised a simple question:

> If a latent relationship is actually harmful, can we identify it by slightly weakening that relationship and checking whether the generated text improves?

This led to a small method I call **Causal Latent Interface Calibration (CLIC)**.

The basic idea is simple:
1. find a suspicious latent relationship
2. perturb it slightly
3. measure whether generation improves

Rather than relying on static statistics such as variance, MSE, or decoder sensitivity, CLIC directly tests the effect of a small latent intervention at inference time.

For example, suppose a particular latent feature depends too strongly on the rest of the generated latent context. We can slightly weaken that dependence,

$$
z' = C(z),
$$

decode the modified latent, and measure whether the resulting text improves.

By repeating this test across different latent directions and block positions, we can automatically identify a small number of relationships that actually hurt generation.

Importantly, the correction does not need to be large. In fact, partially weakening a harmful relationship often worked better than removing it completely. This again suggests that the feature itself was useful; what was wrong was how strongly it was coupled to the rest of the generated latent.

### Learning the correction automatically

The intervention above gives us a useful correction rule \(C\), but applying a hand-designed rule is not very satisfying.

So the next step is to train a small adapter

$$
A_\phi(z)
$$

between the diffusion model and the decoder:

$$
z_{\mathrm{generated}}
\rightarrow
A_\phi(z_{\mathrm{generated}})
\rightarrow
D(A_\phi(z_{\mathrm{generated}})).
$$

Interestingly, the adapter does not need to reproduce the corrected latent itself.

Instead of requiring

$$
A_\phi(z) \approx C(z),
$$

we only require the decoder to behave similarly:

$$
D(A_\phi(z))
\approx
D(C(z)).
$$

This is important because different latent vectors can lead the decoder to nearly the same output. The goal is therefore not to recover one particular "correct" latent coordinate, but to reproduce the useful decoding behavior produced by the causal intervention.

At the same time, we keep a reconstruction loss on clean latents so that the adapter does not destroy the original representation.

The resulting objective is roughly

$$
\mathcal L_A
=
\rho
\underbrace{
\left[
-\log p_D
\left(
x \mid A_\phi(z_{\mathrm{clean}})
\right)
\right]
}_{\mathcal L_{\mathrm{clean}}}
+
\operatorname{KL}
\left[
D(C(z_{\mathrm{generated}}))
\Vert
D(A_\phi(z_{\mathrm{generated}}))
\right].
$$

The first term keeps the adapter grounded by optimizing reconstruction on clean latents, while the second term distills the improved decoder behavior discovered through causal intervention on generated latents.

So the overall idea is:

> **use causal interventions to discover what should change in latent space, then train a small adapter to reproduce the resulting decoder behavior.**

This worked surprisingly well for the particular latent mismatch we had identified. The adapter was small, the diffusion model and decoder remained frozen, and the improvement transferred across independently trained diffusion priors.

But this result also revealed an important limitation.

CLIC could repair a specific mismatch between generated latents and the decoder, but it did **not** solve the broader language-generation problem. When I applied the same procedure to stronger and more transportable latent representations, the improvement disappeared, and the generated text was still largely word salad.

So CLIC fixed a real problem, but not the fundamental one.

That pushed me toward a more basic question:

> Maybe the main problem is not how to correct reconstruction latents after diffusion. Maybe we should reconsider what information those latents are asked to represent in the first place.



## 10 So what is actually wrong with reconstruction latents?

At this point, I think the most useful conceptual decomposition is:

$$
z
=
\underbrace{
z_{\mathrm{semantic/predictive}}
}_{\text{information genuinely needed for generation}}
+
\underbrace{
z_{\mathrm{exact\ lexical}}
}_{\text{information mainly needed to reproduce this exact text}}.
$$

This is a conceptual decomposition, not necessarily a literal orthogonal subspace.

The important point is that ordinary reconstruction does not distinguish the two.
If a piece of lexical information lowers cross entropy, the encoder has an incentive to preserve it.
So a reconstruction latent answers something like:

> What exactly was this text?

But a generative latent may need to answer a different question:

> What needs to be decided before a reasonable continuation can be produced?

These are not the same objective.


## 11 Maybe exact reconstruction is the wrong goal

Suppose the semantic state says:

> report that the method improves performance over the baseline.

A decoder could reasonably produce:

> Our method substantially outperforms the baseline.

or:

> The model achieves significantly better performance.

or:

> Evaluation shows clear gains over the baseline.

If all of these are reasonable realizations of the same state, why should the latent distinguish them?

For a generative representation, it may actually be desirable that

$$
H(X\mid Z)>0.
$$

The decoder should absorb some lexical uncertainty. The latent should not be required to uniquely identify the training sentence. This suggests three design principles:

1. Let the latent decide content, not exact wording.
2. Let the decoder model lexical realization uncertainty.
3. Structurally prevent the encoder from turning the latent back into a lossless codec.


The third point is important.
Simply lowering the latent dimension was not enough.
Adding noise was not enough.
A powerful encoder and decoder can still construct a very efficient error-correcting code.

The information constraint must come from what the latent is *allowed and encouraged to represent*, not only how many floating-point coordinates it has.



## 12 From compressed answers to predictive states?

This leads to the question I am currently most interested in.

Perhaps a continuous language latent should not be a compressed representation of the block it is supposed to decode. Maybe it should be a **predictive language state**.

Instead of

$$
x_k
\rightarrow
z_k
\rightarrow
x_k,
$$

consider

$$
x_{<k}
\rightarrow
s_k
\rightarrow
x_k.
$$

Here $s_k$ never sees the exact future block.

It cannot encode the precise realization of $x_k$.
It can only summarize information from the context that is useful for predicting what should come next. Conceptually:

$$
\boxed{
z_k
\approx
\text{“what this block is”}
}
$$

versus

$$
\boxed{
s_k
\approx
\text{“what the next block should be like”}.
}
$$

A predictive state might contain things like:

- current topic;
- discourse role;
- relevant entities;
- expected semantic direction;
- local continuation constraints.

But it should not know whether the next sentence happens to use *significantly* or *substantially*. That choice belongs to the decoder.

There are still many unresolved questions. A cumulative predictive state,

$$
s_k=E(x_{<k}),
$$

is highly non-stationary: $s_1$ sees almost nothing while $s_{64}$ summarizes almost the whole document.

A fixed-window predictive state may be easier to model but may lose long-range information. A global-plan-plus-local-predictive-state decomposition may be necessary.

And fully parallel generation introduces a state-text consistency problem: a jointly generated predictive-state trajectory may not exactly match the text that the decoder realizes.

So I do not yet know whether predictive states are the answer.

But after all these experiments, I think this is a much better question than the one I started with.



## 13 The question has changed

I started with:

> How can we compress 1024 tokens into 64 latents and diffuse them?

The more useful question now seems to be:

> What should a continuous language latent represent if its purpose is generation rather than reconstruction?

- A latent that reconstructs perfectly may simply be a very good compressed answer.
- A generative state should instead contain the information that must be decided before language is realized.

Those are different objects.

And I increasingly suspect that designing the latter—not building an ever-better autoencoder—is the real representation-learning problem behind continuous diffusion language models.
