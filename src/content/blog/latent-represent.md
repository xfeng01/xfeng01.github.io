---
title: "What Should a Continuous Language Latent Represent?"
description: "Lessons from trying to diffuse compressed language representations."
date: 2026-08-17
lang: zh
translationKey: latent-represent
tags:
  - Diffusion Language Models
  - Continuous Latents
  - Representation Learning
---

在之前的一篇文章里，我有一个问题：对于 language diffusion 来说，token 真的应该是最基本的建模单位吗？

一个很自然的想法是：先把多个 tokens 压缩成更短的 continuous latent sequence，在这些 latents 上做 diffusion，再把它们 decode 回文本。看起来，这个想法似乎很有吸引力。语言本身就有高于 token 的结构，所以 diffusion 也许应该在一个更粗的粒度上进行。

我一开始是这样想的。比如，把一个 $N=1024$ 的 token sequence 分成每块 16 个 tokens 的 blocks。encoder 把每个 block 压缩成一个 latent，最后得到 $K=64$ 个连续向量：

$$
x_{1:1024}
\longrightarrow
z_{1:64}.
$$

diffusion model 学习这个 latent sequence 的分布：

$$
p_\theta(z_{1:64}),
$$

然后 decoder 再把每个 latent block 展开成 tokens，比如：

$$
p_D(x_{k,i}\mid x_{k,<i},z_{1:K}).
$$

最开始，我觉得这个问题应该很直接：

> 如果 autoencoder 能很好地重建原始文本，就说明我们学到了一个好的 latent space。剩下的事情，只是在这个空间上训练 diffusion。

但花了很多时间做实验以后，我现在觉得这个观点漏掉了很大一部分问题。

最核心的一个体会是：

$$
\boxed{\text{Reconstruction is easy. Diffusion is not.}}
$$

也就是说，重建文本很容易，但在这个表示上做好 diffusion 并不容易。

更难的问题已经不只是“语言应该被压缩多少”，而是：

$$
\boxed{\text{What should a continuous language latent actually represent?}}
$$

一个 continuous language latent，到底应该表示什么？

## 1 Reconstruction 比想象中容易

假设每个 16-token block 被映射成一个 512 维，甚至 192 维的连续向量。从序列长度上看，这似乎已经是很强的压缩：

$$
1024\text{ tokens}\rightarrow64\text{ latents},
$$

但这不一定意味着一个很强的 information bottleneck。

连续向量的表示容量很大。如果 encoder 和 decoder 用 token-level reconstruction loss 联合训练，它们完全可以学出一个很高效的 codec。

通常的训练目标可以写成：

$$
\mathcal L_{\mathrm{AE}}
=
-\log p_D(x\mid E(x)),
$$

对于 token decoder，这就是普通的 token-level cross entropy：

$$
\mathcal L_{\mathrm{AE}}
=
-\sum_i
\log p_D(x_i\mid x_{<i},E(x)).
$$

这个 objective 不关心一条信息*为什么*对 reconstruction 有帮助。

- 如果精确的标点有帮助，latent 就可以存标点。
- 如果具体用了哪个词有帮助，latent 就可以存 lexical identity。
- 如果某篇文档特有的细节有帮助，latent 也可以把这些细节存进去。

只要一条信息能降低 token cross entropy，reconstruction objective 就会鼓励 encoder 把它放进 latent。实际中，这件事可以做得非常好。

一个 learned 192-dimensional bottleneck，仍然可以做到几乎完美的 reconstruction。在我的一个实验里，learned 192-dimensional representation 的 oracle language-model perplexity 几乎和原来的高维 latent 一样，而且可以精确恢复超过 99.9% 的 tokens。

> 所以，难的不是 autoencoding，而是 diffusion。

## 2 Reconstruction 更好，generation 反而可能差很多

最重要的一个反例，来自两个不同的 192-dimensional representations。

先对原始 latent 做一个简单的 PCA projection。reconstruction 变差了：

$$
\text{Oracle PPL}\approx55.7,
$$

但 diffusion generation 明显变好了：

$$
\text{Generated PPL}\approx394.
$$

之后，我训练了一个 learned 192-dimensional bottleneck。它的 reconstruction 大幅改善：

$$
55.7\rightarrow30.8.
$$

这个表示几乎是无损的。但生成文本的质量却差了很多：

$$
394\rightarrow3000+.
$$

这些 samples 并不是陷入了重复循环。实际上，它们的 token diversity 往往更高，repetition 反而更低。只是生成出来的文本成了 word salad：词都在，但拼在一起没有连贯的意思。

这是第一个让我很明确地意识到下面这件事的实验：

> reconstruction quality 不是 generative quality 的可靠 proxy。

learned representation 成了一个更好的 codec，却成了一个差得多的 diffusion state。

## 3 Denoising MSE 也不够

一个可能的解释是：learned latent space 对 diffusion model 来说，只是更难做 regression。但这个解释也没有成立。

在多个实验里，我反复看到这样的情况：pointwise denoising 变好了，generation 却变差了。

比如，有些 learned representations 在中间和后期的噪声水平上，$x_0$ prediction MSE 明显低于 PCA-192，但它们的 unconditional generation 却差了好几倍。

类似地，增加 high-$t$ 阶段的训练覆盖后，模型在按真实前向加噪分布得到的数据上表现更好了：

$$
\mathrm{MSE}\downarrow,
\qquad
\text{decoder NLL on true }q_t\downarrow,
$$

但 unconditional generation 反而变差。这说明一个更一般的问题：

> latent space 里的 Euclidean closeness，不等于这个表示适合 generation。

diffusion model 在 MSE 上可以更接近 target latent，但它生成出来的 latent configuration，仍然可能被 decoder 解读成很差的文本。

## 4 丢掉一些信息，反而可能有帮助

PCA 的实验让这件事更清楚了。

- 把有效 latent dimension 从 512 降到 192，reconstruction 变差，但 generation 变好。
- 降到 128 维时，prior 已经非常容易学习，但 representation 本身又丢掉了太多信息。

这里我比较宽泛地用 “rate” 表示 latent 能保留多少信息，而保留下来的 PCA dimensions 数量，可以作为一个可控的 proxy。

于是，我们可以看到一个很明确的三方 trade-off：

$$
\boxed{
\text{Rate}
\;\longleftrightarrow\;
\text{Reconstruction Distortion}
\;\longleftrightarrow\;
\text{Transport Difficulty}.
}
$$

- 信息太多，prior 很难学。
- 信息太少，decoder 得到的 condition 又不够。
- 中间某个位置，可能更适合 generation。

但这仍然没有解释全部问题。

一个 learned 192-dimensional bottleneck，可以在同样的 nominal dimension 里，几乎恢复所有被 PCA 丢掉的信息。encoder 只是把信息重新组织成了一个更高效的连续编码。所以：

> dimension 不等于 information rate。

而且，即使知道 effective rate，也还不足以描述这个问题。

> 信息是*怎样被组织起来的*，同样很重要。

## 5 Local robustness 不等于 global transport

另一个自然的猜测是：**decoder 只在 clean latents 周围很小的一块区域里可靠**。

但这也不是主要问题。

decoder 对 isotropic noise 的鲁棒性比我预想的强。在 normalized clean latents 上加相当大的随机扰动，很多时候仍然可以保持几乎完美的 reconstruction。

我也训练过 stochastic autoencoders，让 decoder 显式学习从带噪的 latent inputs 重建文本。这确实明显扩大了每个 clean latent 周围能够正确 decode 的区域。但 unconditional generation 仍然变差了很多。

这个 latent space 几乎像一个 error-correcting code：

- 每个 clean latent 都有一个很宽的 decoding neighborhood。
- 小的随机扰动可以被容忍。
- 但**全局有效**的 latent configurations，仍然需要满足很强的结构约束。

diffusion 可以匹配 latent space 的很多 marginal statistics，却仍然无法恢复 latent features 和 blocks 之间正确的 joint distribution。

所以，这里又有一个很重要的区别：

> local robustness 不意味着 global transportability。

## 6 Decoder 也是问题的一部分

最开始，我觉得大部分困难都在 latent prior。但这个理解也不完整。

如果只在 clean latents 上继续训练 decoder，oracle reconstruction 会变好。但用 diffusion-generated latents 做 decoding，结果却会差很多。

在一个实验里，同一个 generated latent endpoint，用较早的 decoder 解码，文本的 $\mathrm{PPL}\approx500$；换成在 clean latents 上训练更久的 decoder，$\mathrm{PPL}$ 却到了大约 $2000$。

generated latent 完全一样。变化的只有 decoder。

只用 clean latents 训练的 decoder，越来越善于利用 clean representation 中的细节，但 diffusion-generated latents 并不能可靠地恢复这些细节。

所以：

$$
\boxed{
\text{A decoder optimized for clean latents}
\neq
\text{a decoder robust to generated latents}.
}
$$

一个在 clean latents 上优化得很好的 decoder，不一定能稳定地处理 generated latents。

这里存在一个 **train–inference distribution mismatch**。

decoder 训练时看到的是 $z_{\mathrm{clean}}$，推理时接收的却是 $z_{\mathrm{generated}}$。因此，更好的 clean reconstruction，反而可能让 decoder 对 diffusion model 产生的结构性误差更加敏感。

## 7 有用的 latent features，不一定容易被联合生成

问题也不只是某几个 latent dimensions 单独很难建模。

我发现过这样的情况：两个 latent features 各自都很有用，单独生成也相对容易，但把它们放在一起，generation error 却会大很多。

这说明，只匹配单个 latent feature 是不够的。diffusion 还需要恢复它们之间正确的 **joint relationships**。

单个 feature 的 marginal distribution，甚至简单的 pairwise correlations，都可能看起来已经基本正确。但 generated latent 里那些依赖 context 的关系，仍然可能是错的。

在一个例子里，几个 latent features 编码了后面 text blocks 中非线性的 lexical/content relationships。diffusion 并不是简单地丢掉了这些 features，而是把它们组合成了一种自身看起来合理、却不符合 decoder 所学结构的状态。

所以，信息并没有消失。

**单个 features 都在，但它们被组合到一起的方式错了。**

## 8 静态 latent metrics 经常不可靠

这是这些实验里，对我最有实际帮助的一个体会。我试过很多看起来合理的指标，用来判断哪些 latent directions 对 generation 有害：

- variance mismatch；
- decoder gradients；
- denoising error energy；
- gradient-error alignment；
- local curvature；
- covariance；
- marginal distance。

但它们经常无法正确排序那些真正有害的方向。

一个方向可以同时有很大的 decoder curvature、很大的 prior error，以及很大的 coupling mismatch，却对最终文本质量几乎没有 causal effect。

真正一直有效的判断方式，反而更简单：

> 对 generated latent 做一点小扰动，看看实际推理时生成的文本有没有变好。

比如，做一个小的 causal intervention：

$$
z_i'
=
z_i-\epsilon c_i(z),
\qquad
\epsilon\ll1,
$$

稍微减弱一个待检验的 latent relationship，就可以比较稳定地识别出哪些关系真的在损害 generation。这个判断在不同 sampling seeds，以及独立训练的 diffusion priors 上，都可以成立。

> 有用的信号，不是静态的 sensitivity，而是一个小的 intervention 能不能真的改善 inference 时的 generation。

## 9 从 causal diagnosis 到 latent calibration

前面的实验说明，有些 generated latent features 单独看是合理的，但组合起来以后，却会损害 decoding。

这就引出了一个很直接的问题：

> 如果一个 latent relationship 真的有害，能不能稍微减弱它，再看生成文本有没有变好，用这种方式把它找出来？

沿着这个想法，我尝试了一个小方法，叫作 **Causal Latent Interface Calibration (CLIC)**。

基本思路很简单：

1. 找到一个可疑的 latent relationship。
2. 对它做一点小扰动。
3. 测量 generation 有没有改善。

CLIC 直接测试 inference 时一个小的 latent intervention 会产生什么效果，而不是依赖 variance、MSE 或 decoder sensitivity 这些静态指标。

比如，假设某个 latent feature 对其余 generated latent context 的依赖太强。我们可以稍微减弱这种依赖：

$$
z' = C(z),
$$

然后 decode 修改后的 latent，测量生成文本有没有变好。

在不同 latent directions 和 block positions 上重复这个测试，就可以自动找出少量真正损害 generation 的关系。

这里的 correction 不需要很大。实际上，部分减弱一个有害关系，往往比彻底移除它更有效。这也再次说明，feature 本身是有用的；有问题的是它和其余 generated latent 耦合得有多强。

### 自动学习这个 correction

上面的 intervention 给出了一个有用的 correction rule $C$，但一直使用一个手工设计的规则，也不太令人满意。

所以，下一步是在 diffusion model 和 decoder 之间，训练一个小的 adapter：

$$
A_\phi(z)
$$

整个接口变成：

$$
z_{\mathrm{generated}}
\rightarrow
A_\phi(z_{\mathrm{generated}})
\rightarrow
D(A_\phi(z_{\mathrm{generated}})).
$$

有意思的是，这个 adapter 不需要复现 corrected latent 本身。

我们不一定需要：

$$
A_\phi(z) \approx C(z),
$$

只需要 decoder 的行为相近：

$$
D(A_\phi(z))
\approx
D(C(z)).
$$

这是因为不同的 latent vectors，可以让 decoder 产生几乎相同的输出。所以，目标不是恢复某个唯一“正确”的 latent coordinate，而是复现 causal intervention 带来的、有用的 decoding behavior。

同时，我们还在 clean latents 上保留一个 reconstruction loss，避免 adapter 破坏原来的 representation。

最后的 objective 大致可以写成：

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

第一项通过 clean latents 上的 reconstruction，让 adapter 保留原来的重建能力。第二项则在 generated latents 上，distill 通过 causal intervention 找到的、更好的 decoder behavior。

所以，整个思路是：

> **先用 causal interventions 找出 latent space 里应该改什么，再训练一个小 adapter，复现这些修改带来的 decoder behavior。**

对于已经识别出的那一类 latent mismatch，这个方法的效果比我预想的好。adapter 很小，diffusion model 和 decoder 都保持 frozen，而且 improvement 可以迁移到独立训练的 diffusion priors 上。

但这个结果也暴露了一个重要的限制。

CLIC 可以修复 generated latents 和 decoder 之间某个具体的 mismatch，但它**没有**解决更一般的 language generation 问题。当我把同样的流程用到更强、也更容易做 transport 的 latent representations 上时，improvement 就消失了，生成文本仍然大多是 word salad。

所以，CLIC 修复了一个真实存在的问题，但没有解决最根本的问题。

这也让我开始想一个更基础的问题：

> 也许我们不应该只想怎么修正经过 diffusion 的 reconstruction latents，而应该重新考虑：一开始，我们到底要求这些 latents 表示什么信息？

## 10 Reconstruction latents 到底有什么问题？

到这里，我觉得最有帮助的一个概念性分解是：

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

一部分信息是 generation 真正需要的 semantic/predictive information；另一部分信息，主要是为了精确复现这段特定文本而保留的 lexical information。

这里是一个概念性的分解，不一定意味着 latent space 里真的存在两个正交子空间。

关键在于，普通的 reconstruction objective 不区分这两类信息。只要一条 lexical information 可以降低 cross entropy，encoder 就有动力把它保留下来。

所以，一个 reconstruction latent 回答的更像是：

> 这段文本具体是什么？

但一个 generative latent，可能需要回答另一个问题：

> 在生成一个合理的 continuation 之前，有哪些事情需要先被决定？

这是两个不同的目标。

## 11 也许 exact reconstruction 就不是正确的目标

假设一个 semantic state 要表达的是：

> 说明这个方法相对于 baseline 有性能提升。

decoder 可以合理地生成：

> Our method substantially outperforms the baseline.

也可以是：

> The model achieves significantly better performance.

或者：

> Evaluation shows clear gains over the baseline.

如果它们都是同一个 state 的合理表达，为什么 latent 一定要区分它们？

对于一个 generative representation 来说，下面这件事可能反而是我们想要的：

$$
H(X\mid Z)>0.
$$

decoder 应该承担一部分 lexical uncertainty。我们不应该要求 latent 唯一确定训练数据里的那一句话。

这给出了三个可能的设计原则：

1. 让 latent 决定内容，而不是精确的措辞。
2. 让 decoder 建模内容如何被表达成文本的不确定性。
3. 从结构上限制 encoder，避免它把 latent 重新变成一个无损 codec。

第三点很重要。

单纯降低 latent dimension 不够。加噪声也不够。一个强的 encoder 和 decoder，仍然可以构造出一个很高效的 error-correcting code。

所以，information constraint 应该来自 latent *被允许、被鼓励表示什么*，而不只是它有多少个浮点坐标。

## 12 从压缩答案到 predictive states？

这就引出了我现在最感兴趣的问题。

也许一个 continuous language latent，不应该是它要 decode 的那个 block 的压缩表示。它也许应该是一个 **predictive language state**。

原来的形式是：

$$
x_k
\rightarrow
z_k
\rightarrow
x_k,
$$

我们可以考虑另一种形式：

$$
x_{<k}
\rightarrow
s_k
\rightarrow
x_k.
$$

这里的 $s_k$ 从来没有看到过那个具体的 future block。

因此，它不能编码 $x_k$ 最后具体会用什么措辞。它只能总结 context 里对预测接下来内容有用的信息。

从概念上看，$z_k$ 更像是：

$$
\boxed{
z_k
\approx
\text{“what this block is”}
}
$$

而 $s_k$ 更像是：

$$
\boxed{
s_k
\approx
\text{“what the next block should be like”}.
}
$$

也就是从“这个 block 是什么”，变成“下一个 block 应该是什么样”。

一个 predictive state 可能包含：

- 当前的 topic；
- 这部分内容在篇章中的作用；
- 相关的 entities；
- 预期的语义方向；
- 局部 continuation 需要满足的约束。

但它不应该知道，下一句话最后到底会用 *significantly* 还是 *substantially*。这个选择应该交给 decoder。

当然，这里还有很多没有解决的问题。比如，一个累积的 predictive state：

$$
s_k=E(x_{<k}),
$$

会非常 non-stationary：$s_1$ 几乎什么都没看到，而 $s_{64}$ 已经在总结几乎整篇文档。

fixed-window predictive state 可能更容易建模，但又可能丢掉 long-range information。也许需要把 global plan 和 local predictive state 分开。

而 fully parallel generation 还会带来 state-text consistency 的问题：联合生成的 predictive-state trajectory，未必和 decoder 最后实现出来的文本完全匹配。

所以，我现在还不知道 predictive states 是不是答案。

但做完这些实验以后，我觉得这个问题比最开始的问题更值得问。

## 13 问题已经变了

我最开始问的是：

> 怎么把 1024 个 tokens 压缩成 64 个 latents，再在上面做 diffusion？

但现在看来，更有用的问题是：

> 如果一个 continuous language latent 的目的是 generation，而不是 reconstruction，它到底应该表示什么？

- 一个可以完美 reconstruction 的 latent，可能只是一个很好的“压缩答案”。
- 一个 generative state，则应该包含在语言被具体表达出来之前，必须先决定的信息。

这是两种不同的对象。

我也越来越怀疑，设计后面这种 state，才是 continuous diffusion language models 背后真正的 representation-learning 问题，而不只是不断训练一个更好的 autoencoder。
