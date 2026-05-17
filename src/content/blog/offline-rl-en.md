---
title: "Some Basic Thoughts on Offline RL: The Core Problem, Common Solutions, and Generative Policies"
description: "Offline RL is about reliable policy improvement under dataset support constraints."
date: 2025-10-03
lang: en
translationKey: offline-rl
tags:
  - Diffusion Models
  - Offline RL
---

The core problem in offline RL can be summarized in one sentence: **we want to learn a policy that is better than the behavior policy from a fixed dataset, but we cannot let the learned policy move into regions that the dataset does not cover.**

This is also what makes offline RL different from ordinary supervised learning. In supervised learning, we usually assume that training and test examples are drawn from similar distributions, so the model mostly needs to generalize near the training distribution. In offline RL, the data is collected by some behavior policy, but the policy we eventually deploy may be different from it. Since a policy's actions affect the future states it visits, even a small policy change can induce a new state-action distribution. The central challenge is therefore not just fitting a fixed dataset, but improving the policy while controlling distribution shift and support mismatch.

## 1 Offline Dataset, Behavior Policy, and Learned Policy

Suppose the offline dataset is collected by a behavior policy $\mu$:

$$
D \sim d^\mu(s,a) = d^\mu(s)\mu(a|s)
$$

Here, $d^\mu(s)$ is the state visitation distribution induced by $\mu$ and the environment dynamics, and $\mu(a|s)$ is the behavior policy's action distribution at state $s$. Therefore, the offline dataset only covers the state-action distribution induced by the behavior policy.

But during training, we want to learn a new policy:

$$
\pi(a|s)
$$

If $\pi$ chooses an action that rarely appears, or never appears, in the dataset at some state $s$, then this $(s,a)$ pair is out of distribution. The estimate $Q(s,a)$ is usually unreliable in this region, because the dataset does not provide enough real transitions to supervise it.

So the core issue in offline RL is not simply that we have limited data. The issue is that **the data only covers the regions visited by the behavior policy, while the learned policy may want to move outside those regions.**

## 2 OOD Actions and Distribution Shift

In offline RL, the most common form of OOD behavior is **action OOD**. At the same state, the learned policy $\pi$ may choose actions that the behavior policy $\mu$ rarely selected:

$$
\pi(a|s) \not\approx \mu(a|s)
$$

More precisely, the distribution shift in offline RL is a **state-action distribution shift**:

$$
d^\pi(s,a) \neq d^\mathcal{D}(s,a)
$$

Expanding the learned policy distribution:

$$
d^\pi(s,a) = d^\pi(s)\pi(a|s)
$$

The dataset distribution can be approximated as:

$$
d^\mathcal{D}(s,a) \approx d^\mu(s)\mu(a|s)
$$

This shift can come from two sources.

The first is **action shift**. At the same state, $\pi(a|s)$ differs from $\mu(a|s)$. This is the most direct and most commonly discussed problem in offline RL. For example, if the dataset mostly chooses "left" at state $s$, but the learned policy starts choosing "right" because the critic assigns it a high Q-value, then "right" is an OOD action for this dataset.

The second is **state shift**. If the policy chooses actions that differ from the dataset, it may enter states that the dataset rarely visited. In other words, action OOD can further induce state OOD.

A useful chain of failure is:

$$
\pi(a\mid s) \not\approx \mu(a\mid s)
$$

$$
\Longrightarrow (s,a) \notin \mathrm{supp}\bigl(d^{\mathcal D}(s,a)\bigr)
$$

$$
\Longrightarrow Q(s,a) \text{ is unreliable and may be overestimated}
$$

$$
\Longrightarrow \pi \text{ is optimized toward bad OOD actions}
$$

$$
\Longrightarrow d^\pi(s,a) \text{ shifts further away from } d^\mathcal{D}(s,a)
$$

One subtle point is that distribution shift does not appear only after Q estimation becomes inaccurate. The mismatch between the learned policy and the dataset distribution is already distribution shift. Inaccurate Q estimates are a consequence of that shift, and overestimated OOD actions can further amplify the problem.

## 3 Why Directly Maximizing Q Can Fail

In online RL, if the Q function is wrong about some action, the agent can still interact with the environment, collect new feedback, and correct the estimate. In offline RL, the dataset is fixed. The policy cannot test an OOD action in the environment.

This leads to a typical failure mode: **extrapolation error**. In regions not covered by the dataset, the Q function can only extrapolate through function approximation, and that extrapolation can be wrong.

If an OOD action is mistakenly overestimated:

$$
Q(s,a_\text{OOD}) > Q(s,a_\text{data})
$$

then the actor or policy improvement step may choose that OOD action. The problem is that the high Q-value does not necessarily mean the action is good. It may simply mean that the critic has no reliable data there.

Therefore, the goal in offline RL is not simply:

$$
\text{maximize}\ Q(s,a)
$$

It is closer to:

> maximize Q, but only when the Q estimate is trustworthy enough.

Or:

> improve the policy while staying within or near the dataset support.

This creates the central trade-off in offline RL:

**If the method is too conservative, it becomes behavior cloning: stable, but hard to improve beyond the behavior policy. If it is too aggressive, it may exploit OOD Q errors and collapse.**

## 4 Common Solution Directions

Offline RL methods look quite different on the surface, but many of them are trying to solve the same problem: **how to control the mismatch between the learned policy and the dataset distribution.** A useful way to organize them is into three broad directions.

The first direction is **policy constraint / behavior regularization**. These methods directly restrict the learned policy from moving too far away from the behavior policy:

$$
\max_{\pi} \; \mathbb{E}_{s\sim \mathcal{D},\, a\sim \pi(\cdot\mid s)}\left[Q(s,a)\right]
\quad \text{s.t.} \quad
D\left(\pi(\cdot\mid s),\mu(\cdot\mid s)\right) \leq \epsilon
$$

Here, $D$ can be KL divergence, MMD, Wasserstein distance, or an action-space distance. The same idea can also be written as regularization:

$$
\max_{\pi} \; \mathbb{E}_{s\sim \mathcal{D},\, a\sim \pi(\cdot\mid s)}\left[Q(s,a)\right]
- \alpha D\left(\pi(\cdot\mid s),\mu(\cdot\mid s)\right)
$$

The intuition is simple: the policy is allowed to improve, but it should not move too far from the dataset. BEAR, BRAC, and TD3+BC can all be understood from this perspective. TD3+BC, for example, adds a behavior cloning regularizer to the actor objective so that the actor does not output highly OOD actions just because the critic assigns them high values.

The second direction is **support constraint**. This is more direct than constraining policy distance: instead of only asking the learned policy to be close to the behavior policy overall, it asks the policy to choose actions inside the dataset support.

$$
a \in \mathrm{supp}\left(\mathcal{D}(\cdot\mid s)\right)
$$

More strictly:

$$
\mathrm{supp}\left(\pi(\cdot\mid s)\right)
\subseteq
\mathrm{supp}\left(\mathcal{D}(\cdot\mid s)\right)
$$

For discrete actions, this is easy to interpret. If the dataset only contains $a_1$ and $a_2$ at a state, then the learned policy should not choose $a_3$. For continuous actions, exact support is harder because the exact same action may rarely repeat, so support has to be approximated. One common strategy is to learn a behavior model or generative model, sample dataset-like candidate actions, and then select the candidate with the highest Q-value. BCQ follows this idea: generate actions close to the dataset, allow a small perturbation, and avoid searching freely over the entire action space.

The third direction is **conservative value learning / pessimism**. These methods do not necessarily constrain the policy directly. Instead, they make the Q function conservative on OOD actions:

$$
Q(s,a_{\mathrm{OOD}}) \downarrow
$$

CQL is a representative example. Its core idea is to push down Q-values for actions outside the dataset while preserving the values of dataset actions. Then, even if policy optimization searches for high-Q actions, it is less likely to be attracted by spuriously high OOD values.

The insight here is that we do not always need to explicitly forbid OOD actions. If OOD actions have sufficiently low Q-values, the policy will naturally avoid them.

Another family of methods, such as AWR, AWAC, and IQL, can be understood as **advantage-weighted behavior cloning**. They do not perform:

$$
\arg\max_a Q(s,a)
$$

over the entire action space. Instead, they do weighted imitation on dataset actions:

$$
\max_{\pi} \; \mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[w(s,a)\log \pi(a\mid s)\right]
$$

where $w(s,a)$ is usually related to advantage. The intuition is that the dataset contains both good and bad actions. We do not need to imitate all of them equally; we want to imitate high-advantage actions more. Because the training signal only comes from dataset actions, these methods naturally reduce dependence on OOD actions and are often more stable.

## 5 A Unified View

Although offline RL methods take different forms, they can be unified by one sentence:

> Offline RL is about **reliable** policy improvement under dataset support constraints.

Policy constraints control distribution shift from the policy side: make $\pi$ stay close to $\mu$.

Support constraints control distribution shift from the action space: only allow the policy to improve over actions supported by the dataset.

Conservative value learning controls distribution shift from the value side: make OOD actions have lower Q-values so that policy optimization is not attracted to extrapolation errors.

Advantage-weighted BC is a more implicit version: instead of searching in OOD regions, it selects better actions inside the dataset and imitates them more strongly.

So the key question in offline RL is not "how do we maximize Q?" but "when is Q trustworthy?" If Q is estimated inside the dataset support, policy improvement can be meaningful. If Q comes from extrapolation in an OOD region, maximizing it can make the policy worse.

A natural summary is that offline RL optimization is not unconstrained high-Q search over the whole action space. It is reliable improvement inside regions supported by the data. Different methods instantiate "reliable" at different places: the policy, the action support, the value estimate, or the weighting of dataset actions.

## 6 Why Generative Policies Fit Offline RL

From this perspective, the appeal of generative policies is natural. A generative policy does not ask the actor to directly search for high-Q actions over the entire action space. It first models the action or trajectory distribution in the offline dataset. The generated candidates are therefore more likely to stay near high-density regions visited by the behavior policy.

This adds an implicit support constraint to policy improvement:

$$
\text{policy improvement over data-supported actions}
$$

instead of:

$$
\text{unconstrained policy improvement over all actions}
$$

However, this does not mean that a generative policy automatically solves offline RL. Pure generative modeling is closer to behavior cloning: it can tell us which actions look like the data, but not which data-supported actions are better.

This is why generative policies in offline RL usually need return, value, or advantage guidance. The generative model keeps the search space near dataset support; the value or advantage signal biases generation toward better actions or trajectories inside that support.

In other words, the value of a generative policy is not only that it has stronger expressiveness. It also turns one of the hardest parts of offline RL, unconstrained action search, into a more controlled data-supported candidate generation problem.
