---
title: "From Policy Improvement to PPO, GRPO, and DPO"
description: "A note on policy improvement for LLM post-training, from policy ratios and advantages to PPO clipping, GRPO group baselines, and DPO preference optimization."
date: 2025-12-22
lang: en
translationKey: policy-improvement
tags:
  - LLM
  - Policy Optimization
---

## 1. The Goal of Policy Improvement

In RL, policy improvement starts from an old policy $\pi_{\text{old}}$ and learns a better new policy $\pi_\theta$. Here "better" means: on states that the old policy often visits, the new policy should put more probability on high-value actions.

A common policy improvement objective can be written as:

$$
\max_{\pi_\theta}
\mathbb{E}_{s \sim d_{\pi_{\text{old}}},\, a \sim \pi_\theta(\cdot|s)}
\left[
Q_{\pi_{\text{old}}}(s,a)
\right]
$$

Here, $d_{\pi_{\text{old}}}$ is the state distribution induced by the old policy. $Q_{\pi_{\text{old}}}(s,a)$ is the expected return after taking action $a$ at state $s$, and then following the old policy afterwards.

The reason we use $Q_{\pi_{\text{old}}}$ is that policy improvement is usually a **local improvement** step. We fix the old policy, use its value function as a reference, and ask whether an action is better than the old policy's average behavior. If we directly use $Q_{\pi_\theta}$, the target changes with the new policy and becomes much harder to estimate from old samples.

So the objective means:

> On states visited by the old policy, assign higher probability to actions that are better relative to the old policy.

In practice, we usually use advantage instead of the absolute $Q$ value:

$$
A_{\pi_{\text{old}}}(s,a)
=
Q_{\pi_{\text{old}}}(s,a)-V_{\pi_{\text{old}}}(s)
$$

Then the objective becomes:

$$
\max_{\pi_\theta}
\mathbb{E}_{s \sim d_{\pi_{\text{old}}},\, a \sim \pi_\theta(\cdot|s)}
\left[
A_{\pi_{\text{old}}}(s,a)
\right]
$$

There is still a mismatch: the objective samples actions from the new policy, but the training data usually comes from the old policy. To estimate the new-policy objective with old-policy samples, we use importance sampling:

$$
\mathbb{E}_{a \sim \pi_\theta}
\left[
A(s,a)
\right]
=
\mathbb{E}_{a \sim \pi_{\text{old}}}
\left[
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
A(s,a)
\right]
$$

This gives the policy ratio:

$$
r(\theta)
=
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
$$

So the policy improvement surrogate can be written as:

$$
\max_\theta
\mathbb{E}
\left[
r(\theta)A(s,a)
\right]
$$

## 2. PPO: Clipping the Policy Ratio

PPO adds one constraint to the surrogate above: **the new policy should not move too far from the old policy.**

The reason is simple. The data is sampled from the old policy. If the new policy changes too much, old samples no longer represent the new policy well, and importance sampling becomes unstable.

So PPO does not directly optimize:

$$
r_t(\theta)A_t
$$

Instead, it uses the clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

where:

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\text{old}}(a_t|s_t)}
$$

The intuition behind clipping is straightforward.

If $A_t>0$, the action is good, so the new policy should increase its probability. But when $r_t(\theta)>1+\epsilon$, the probability has already increased too much, so PPO stops rewarding this direction.

If $A_t<0$, the action is bad, so the new policy should decrease its probability. But when $r_t(\theta)<1-\epsilon$, the probability has already decreased too much, so PPO also limits further movement.

So the core of PPO is:

> Use the ratio to measure the probability change, use advantage to decide the direction, and use clipping to control the update size.

### 2.1 PPO for LLMs: Actions Are Tokens

In LLM fine-tuning, a language model is usually treated as a **token-level policy**. Given a prompt $x$, the model generates a response:

$$
y=(y_1,y_2,\dots,y_T)
$$

At step $t$, the state is the context $s_t=(x,y_{<t})$, and the action is the current token $a_t=y_t$. So the LLM policy can be written as:

$$
\pi_\theta(y_t|x,y_{<t})
$$

The probability of the full response is:

$$
\pi_\theta(y|x)
=
\prod_{t=1}^T
\pi_\theta(y_t|x,y_{<t})
$$

Therefore, LLM PPO performs policy improvement at the token level. The corresponding ratio is:

$$
r_t(\theta)
=
\frac{\pi_\theta(y_t|x,y_{<t})}
{\pi_{\text{old}}(y_t|x,y_{<t})}
$$

It measures how much the new model increases or decreases the probability of token $y_t$ compared with the old model.

This also explains why we usually do not treat the whole response as one indivisible action. In principle, the response-level ratio is:

$$
\frac{\pi_\theta(y|x)}
{\pi_{\text{old}}(y|x)}
=
\prod_{t=1}^T
\frac{\pi_\theta(y_t|x,y_{<t})}
{\pi_{\text{old}}(y_t|x,y_{<t})}
$$

But this product is very sensitive to length and can easily become extremely large or small. It also makes credit assignment too coarse: if a response is bad, that does not mean every token is bad. This is why LLM PPO usually computes ratios and advantages at the token level.

### 2.2 Why Does PPO Need a Value Model?

PPO updates depend on token-level advantage:

$$
A_t = Q(s_t,a_t)-V(s_t)
$$

But in LLM tasks, reward is often sequence-level. For example, a math problem may only receive reward after the whole answer is generated:

$$
R(x,y)
$$

In this setting, we usually do not learn a separate $Q(s_t,a_t)$. For a sampled trajectory, the rollout return can be used as an estimate of $Q$:

$$
Q(s_t,a_t) \approx G_t
$$

If reward is only given at the end, we can roughly view it as:

$$
G_t \approx R(x,y)
$$

But directly using the final reward to update every token has high variance. The final reward only tells us whether the whole response is good. It does not tell us how good a token is relative to the average continuation from the current prefix.

So PPO usually learns a value function:

$$
V_\phi(s_t)
\approx
\mathbb{E}[G_t|s_t]
$$

Then advantage is computed as:

$$
A_t = G_t - V_\phi(s_t)
$$

Here, $V_\phi(s_t)$ is a baseline. It estimates the average return the model can get if it continues from the current prefix $s_t$.

More precisely:

> PPO usually does not explicitly learn a $Q$-function. It uses sampled return as an estimate of $Q$, and learns a $V$-function to turn raw return into lower-variance advantage.

This is one reason PPO is expensive for LLM post-training: besides the policy model, it often needs a value model / critic, and may also involve a reference model and a reward model.

## 3. GRPO: Replacing the Value Model with a Group Baseline

GRPO can be viewed as a simplification of PPO. It keeps the policy ratio and PPO-style clipping, but removes the learned value model.

In PPO, advantage has the form:

$$
A_t = G_t - V(s_t)
$$

where $V(s_t)$ must be trained separately.

GRPO uses a different baseline. For many LLM reasoning tasks, reward is sequence-level, and we can sample multiple responses for the same prompt. For a prompt $x$, sample a group of responses from the old policy:

$$
y_1,y_2,\dots,y_G
$$

Then compute their rewards:

$$
R_1,R_2,\dots,R_G
$$

Use the group mean as the baseline:

$$
A_i
=
R_i
-
\frac{1}{G}\sum_{j=1}^G R_j
$$

Sometimes this is further normalized:

$$
A_i
=
\frac{
R_i-\mathrm{mean}(R_1,\dots,R_G)
}{
\mathrm{std}(R_1,\dots,R_G)
}
$$

This advantage means:

> For the same prompt, how much better is this response than the other responses in the group?

GRPO can still use a PPO-like ratio objective. For token $y_{i,t}$ in response $y_i$, the ratio is:

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}|x,y_{i,<t})
}{
\pi_{\text{old}}(y_{i,t}|x,y_{i,<t})
}
$$

So the main difference between PPO and GRPO is how advantage is estimated:

- PPO: $A_t = G_t - V(s_t)$, so it needs a value model.
- GRPO: $A_i = R_i - \mathrm{mean}(R_1,\dots,R_G)$, so it does not need a value model.

The core idea of GRPO is:

> Still use policy ratios for policy improvement, but estimate advantage from the relative quality of multiple responses to the same prompt.

This is why GRPO fits math and code tasks well. These tasks often allow multiple samples per prompt, and a rule-based verifier can provide outcome rewards cheaply.

## 4. DPO: Policy Improvement Without Explicit Rollout

Besides PPO and GRPO, DPO is another common preference optimization method for LLM post-training. Its starting point is different.

PPO and GRPO usually follow this pipeline:

1. Sample responses from the current policy.
2. Score them with a reward model, verifier, or rule-based reward.
3. Estimate advantage.
4. Improve the policy through a policy ratio objective.

DPO directly uses preference data:

$$
(x, y_w, y_l)
$$

where $y_w$ is the preferred response, and $y_l$ is the rejected response. DPO wants the model to prefer $y_w$ over $y_l$ relative to a reference model. Equivalently, it wants the following difference to be larger:

$$
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

The loss is usually written as:

$$
\mathcal{L}_{\text{DPO}}
=
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

Intuitively, DPO also performs policy improvement: it increases the relative probability of the preferred response and decreases the relative probability of the rejected response, while the reference model controls how far the policy moves.

Unlike PPO and GRPO, DPO does not need rollout-based advantage estimation, and it does not explicitly train a reward model. It is closer to offline preference optimization: learn directly from preference pairs that chosen should be more likely than rejected.

A simple comparison is:

- PPO: uses policy samples + reward, needs rollout and a value model, and uses token-level advantage as the main signal.
- GRPO: uses group samples + reward, needs rollout but not a value model, and uses group-relative reward as the main signal.
- DPO: uses preference pairs, needs neither rollout nor a value model, and uses chosen vs. rejected preference as the main signal.
