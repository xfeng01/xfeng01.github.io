---
title: Hoeffding Bound
published: 2025-04-24
description: A useful equation in reinforcement learning.
tags:
  - generative-AI
draft: false 
---

Let $X_1,\dots,X_m$ be i.i.d. random variables satisfying $|X_i|\le G$ for some constant $G>0$ for all $i$. Then, for any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\begin{aligned}
\left|\frac{1}{m}\sum_{i=1}^{m} X_i - \mathbb{E}[X_1]\right|
&\le
\sqrt{c \cdot \log(\frac{1}{\sigma}) \cdot  \frac{G^2}{m}} \\
&:=
\tilde{\mathcal{O}}\!\left(\frac{G}{\sqrt{m}}\right).

\end{aligned}
$$

for some absolute constant $c>0$.

## Lemma

For iteration $i$ of the algorithm, let $L>0$. Then, with probability at least $1-\frac{\delta}{L}$, for all $(s,a)\in S\times A$,

$$
\begin{aligned}
\left|\frac{1}{m}\sum_{j=1}^m V^{(i-1)}\bigl(Z^{(j)}_{s,a,i}\bigr)
- P(\cdot\mid s,a)^\top\,V^{(i-1)}\right|
&\le
\sqrt{\frac{c\,\log\!\bigl(|S||A|  \frac{L}{\delta}\bigr)}{m}}
\cdot\frac{R_{\max}}{1-\gamma}  \\
&=:\tilde{\mathcal{O}}\!\Bigl(\frac{R_{\max}}{\sqrt{m}\,(1-\gamma)}\Bigr);


\end{aligned}
$$

Define

$$
V^{(i)}(s)\;:=\;\max_a \hat Q^{(i)}(s,a).
$$

‍
