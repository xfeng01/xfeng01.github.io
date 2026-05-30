---
title: "Diffusion for Language: Should the Basic Unit Really Be a Token?"
description: "."
date: 2026-05-29
tags:
  - Diffusion Language Models
---




最近我在思考一个问题：对于 diffusion language model 来说，真正应该被 diffusion 的基本单位是什么？

现在很多 diffusion language model，无论是 discrete mask diffusion，还是 continuous embedding diffusion，本质上仍然把 token (position) 当作基本建模单位。即使模型是在完整的 token embedding matrix 上做 denoising，且 denoiser 可以通过 self-attention 看到全局上下文，它的状态空间依然是 token-aligned 的。换句话说，每一个 diffusion state 仍然对应某个 token position。

这并不是说 token-level diffusion 没有全局语义。相反，continuous diffusion 很强的一点正在于：它可以在整个 $L \times d$ 的连续表示场上做 global refinement。它不是逐个 token 独立修复，而是可以联合修正整段 sequence 的一致性。这个能力非常重要，也解释了为什么 continuous diffusion 在语言建模中有潜力。

但这也引出一个更根本的问题：如果 continuous diffusion 的优势是对一个 continuous field 进行全局 refinement，那么这个 field 的坐标系是否一定要绑定在 token position 上？

我觉得这里真正的问题是 signal design：

> token 是很自然的 surface realization unit，但它不一定是最自然的 diffusion state。

## 1 Token 的问题不在 global

一个容易误解的说法是：token-level diffusion 不够 global，所以需要更 high-level 的表征。这个说法并不准确。

如果 denoiser 对整个 sequence 做 self-attention，那么 token-level embedding diffusion 本身就是 global 的。模型当然可以利用整段上下文来修正局部表示。因此，token-level diffusion 的问题不是“看不到全局”。

更准确的问题是：

> token-level diffusion can be globally conditioned, but it is still token-aligned in its state space.

也就是说，它的 denoising 对象仍然是 token position 上的连续向量。高层语义、推理结构、篇章关系、局部措辞和语法细节，都被混在同一个 token-aligned space 中处理。

这会导致 planning 和 realization 没有被清楚地区分。模型既要决定这段话要表达什么，又要决定每个 token 应该是什么。对于 autoregressive language model 来说，这种混合是自然的，因为 next-token prediction 本来就是逐步展开文本。但对于 diffusion model 来说，这未必是最自然的状态空间。


## 2 新的 unit 应该是什么？

当然，即使我们认为 token 不一定是最自然的 diffusion state，也不能马上得出结论说：那就应该换成另一种 unit，例如 semantic unit。这个判断同样需要谨慎。

问题在于，token 虽然有局限，但它至少是明确、可监督、可还原的。文本天然就是 token sequence，所以 token-level modeling 有非常清楚的训练目标和生成接口。相比之下，一个更高层的 diffusion unit 要困难得多：它应该对应 phrase、clause、sentence，还是某种 learned latent block？不同任务可能需要的粒度也不一样。

这也是为什么我不想把命题说成：

> diffusion 的基本单位一定不应该是 token。

更稳妥的说法是：

> token should not be assumed to be the only or the most natural diffusion unit.

换句话说，这不是一个简单的“token vs. semantic unit”的选择题，而是一个 representation question：continuous diffusion 到底应该发生在什么信号上？

如果我们真的想引入一种高于 token 的 diffusion unit，它至少需要满足几个条件。它应该是 compositional 的，能够表达语言中的组合结构；它应该是 recoverable 的，能够被还原成文本；它还应该是 jointly modelable 的，因为我们真正关心的不是单个 unit，而是整个 sequence 的 joint structure。

只有满足这些条件，所谓 semantic unit 才不只是一个模糊的直觉，而可能成为一种新的 generative state space。












## 3 一个可能的标准：compositional, recoverable, jointly modelable

如果不用 token，那么新的 diffusion unit 至少需要满足几个条件。

第一，它应该是 compositional 的。语言的意义不是 token 的简单堆叠，而是由短语、句子、推理步骤和篇章结构等层次组合出来的。一个好的 semantic unit 应该能够表达这种高层组合结构。

第二，它应该是 recoverable 的。latent representation 不能只是一个不可还原的压缩向量。它必须能够被还原成文本，否则它就不是一个真正可生成的语言状态。

第三，它应该是 jointly modelable 的。semantic units 不能被独立生成。真正重要的是建模整个 semantic sequence 的 joint structure：

$$  
p_\theta(z_{1:K}).  
$$

只有这样，模型才能捕捉 discourse order、long-range consistency 和 reasoning structure。

从这个角度看，semantic-block diffusion 的目标不是简单地把 token 换成 block，而是定义一种新的 continuous signal：一个更短、更抽象、可还原，并且可以联合建模的 semantic latent sequence。

## 4 NCP 提供了一个相关信号

Next Concept Prediction 给了一个很有意思的参考。它不是 diffusion model，而是 autoregressive model；它的 concept 也不是 continuous semantic block，而是通过 VQ 得到的 discrete latent concept。但它说明了一件事：高于 token 的预测目标可能确实有价值。

NCP 的基本思想是把多个 token 的 hidden states 聚合成 concept-level representation，再用 VQ 构建 concept vocabulary，然后预测 next concept，并用预测出的 concept 指导后续 token generation。

这和 semantic-block diffusion 的直觉很接近：不要只预测下一个 token，而是先预测一个更高层的“概念”或“计划”，再把它展开成 tokens。

但两者也有关键区别。NCP 仍然是 autoregressive 的：

$$  
p(c_{t+1} \mid c_{\leq t}),  
$$

而 semantic-block diffusion 想建模的是整个 semantic plan 的 joint distribution：

$$  
p_\theta(z_{1:K}).  
$$

此外，NCP 的 concept 更像 fixed-size token chunk，而 semantic-block diffusion 更希望探索 variable-length、可组合的 semantic unit。NCP 的 decoder 是把 predicted concept broadcast 回固定数量的 token positions，而 semantic-block diffusion 更自然的形式是 branching decoder：

$$  
z_i \rightarrow x_{i,1:m_i}.  
$$

也就是说，一个 semantic latent 可以展开成多个 token，而不是和固定 token positions 强行对齐。

## 5 对 long context 的意义

我觉得 concept-level 或 semantic-block-level modeling 对 long context 是有帮助的，但这种帮助不是“直接扩大 context window”。

更准确地说，它降低了高层建模的有效序列长度。

如果原来有 $L$ 个 tokens，现在被压缩成 $K$ 个 semantic blocks，其中 $K \ll L$，那么模型做 high-level planning 时面对的是更短的 semantic trajectory，而不是完整的 token sequence。

这对 long-form generation 很重要。长文本中的困难往往不是某个局部 token，而是主题是否一致、论证是否连贯、推理步骤是否正确、前后结构是否匹配。semantic-block representation 可以让模型在更短的序列上处理这些高层依赖。

但这也不应该被夸大。semantic blocks 更适合 long-range semantic planning，不一定适合所有 long-context retrieval 问题。如果任务要求精确找到前文某个名字、数字、代码变量或引用，压缩表示可能反而不如 token-level representation 稳定。

所以更准确的说法是：

> semantic-level modeling may improve long-range planning, but it does not automatically solve all long-context problems.

## 6 回到 diffusion：真正的问题是 denoise 什么

最终，我觉得这个方向最核心的问题不是“token 好不好”，而是：

> What is the right continuous signal for language diffusion?

continuous diffusion 的优势在于它可以对一个连续结构进行 iterative refinement。现有 token-level continuous diffusion 已经说明，这种 refinement 在语言中是有潜力的。但如果 diffusion 的强项是 global refinement，那么我们就应该认真思考：被 refinement 的对象是否应该仍然是 token-aligned embeddings？

也许 token 更适合作为 realization unit，而 semantic block 更适合作为 planning unit。

一个可能的架构是：

$$  
x_{1:L} \rightarrow z_{1:K}, \quad K \ll L.  
$$

其中 $z_{1:K}$ 是一组 semantic-block latents。diffusion model 在这个 latent sequence 上建模 joint distribution：

$$  
p_\theta(z_{1:K}).  
$$

然后 branching decoder 把每个 semantic latent 展开成 variable-length token span：

$$  
z_i \rightarrow x_{i,1:m_i}.  
$$

这样，diffusion 负责生成和 refinement 一个 global semantic plan，而 decoder 负责把这个 plan realize 成具体文本。

