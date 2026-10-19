---
title: "Why $O(TL^2)$ Diffusion Can Beat $O(L)$ AR in Practice"
description: "A hardware-efficiency perspective on why diffusion models can sometimes generate faster than autoregressive models despite worse theoretical complexity."
date: 2026-03-06
lang: zh
translationKey: diffusion-vs-ar-speed
tags:
  - Diffusion
  - LLM
  - Systems
---

## 1 一个看起来反直觉的问题

在比较 autoregressive model 和 diffusion language model 时，一个很自然的问题是：diffusion 明明做了更多计算，为什么实际生成速度有时反而更快？

对于带 KV cache 的 AR model，生成一个新 token 时，计算它的 $Q$、$K$、$V$ 不依赖于已有 sequence 的长度，可以看成 $O(1)$。但新的 query 仍然需要和历史的 keys、values 做 attention，所以当 context 长度是 $L$ 时，单步 attention 的计算量是 $O(L)$。

而 diffusion language model（DLM）在每一个 denoising step，都要对整个 sequence 做 full self-attention。单步 attention 是 $O(L^2)$，如果需要 $T$ 步，总共就是 $O(TL^2)$。这里先区分一下：AR 的 $O(L)$ 指的是单个 decoding step，不是生成整段文本的总计算量。

从这个角度看，diffusion 似乎明显更贵。那为什么在真实 GPU inference 中，它仍然可能和 AR 竞争，甚至在某些设置下更快？

我觉得这里最关键的一点是：**理论计算量和实际运行时间，并不是同一个问题。** Big-O 告诉我们做了多少计算，但没有告诉我们这些计算能不能把 GPU 用起来。

下面主要从 hardware efficiency 的角度，比较一下 AR、full-sequence DLM 和 Block Diffusion（BDLM）的执行方式。

## 2 AR 为什么容易慢：计算很少，但一直在搬数据

先看两个决定 GPU 运行速度的基本量。为了直观一点，可以把 GPU 想象成一个很大的数据加工工厂：

- **Global memory（HBM/显存）**是工厂里的大仓库。容量很大，比如 80GB，model weights 和完整的 KV cache 都放在这里。
- **Compute units（Tensor Cores）**是负责计算的工人，真正执行矩阵运算。
- **On-chip cache（SRAM）**是工人旁边的工作台。空间很小，但拿取数据非常快。
- **Peak compute throughput $\pi$** 表示工人每秒最多能做多少计算，也就是 GPU 的峰值 FLOPs/s。
- **Memory bandwidth $\beta$** 表示每秒能从仓库搬多少数据，相当于给工作台送料的传送带速度。

现代 GPU 的一个特点是，计算能力相对于 memory bandwidth 非常强。用这个比喻来说，就是工人处理得很快，但传送带不一定来得及把数据送过来。

这也是 AR decoding 的一个核心问题。

生成第 $i$ 个 token 时，如果已经有 KV cache，单步 FLOPs 大约是

$$
O(d^2 + i \cdot d),
$$

其中 $d$ 是 hidden dimension。$d^2$ 对应线性投影等计算，$i \cdot d$ 对应当前 query 和历史 KV 的 attention。

单步计算量并没有很大，但 GPU 每一步仍然需要读取 model weights 和历史 KV cache。问题在于，这一次搬来的 weights 只服务于一个新 token，不能在很多 token 之间充分复用。工人每次只处理一小份任务，然后又要等下一批数据。

从 roofline model 的角度看，这意味着 AR decoding 的 **arithmetic intensity（算术强度）**通常比较低：

$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}} \approx O(1).
$$

这里的 arithmetic intensity，可以简单理解成：每搬一份数据，能做多少计算。这个值越低，就越容易出现“数据搬得很辛苦，但真正计算没做多少”的情况。

还需要引入一个概念：**overlap**。

直觉上，一步的运行时间似乎应该是

$$
\text{data movement time} + \text{compute time}.
$$

但经过优化的 GPU pipeline 可以一边计算当前任务，一边搬运后续需要的数据。因为两者可以重叠，运行时间更适合近似成较慢的那一项：

$$
T_{\text{real}} \approx \max\left(\text{compute time}, \text{memory time}\right)
= \max\left(\frac{\text{FLOPs}}{\pi}, \frac{\text{Memory I/O}}{\beta}\right).
$$

所以，$I \approx O(1)$ 意味着什么？

它意味着计算量和数据搬运量基本按同一个量级增长。而在计算能力很强的 GPU 上，这样的计算密度通常不足以让 Tensor Cores 持续忙起来。工人很快算完，然后继续等数据。这个时候，系统就容易进入 **memory-bound** 状态：决定速度的主要是 $\beta$，而大量的计算能力 $\pi$ 没有被充分利用。

AR 的问题还不止单步效率低。它需要一个 token 接一个 token 地生成，所以生成 $L$ 个 tokens，就要把这个过程串行重复 $L$ 次。

在 memory-bound 的简化假设下，忽略固定的数据类型、层数等系数，总时间可以粗略写成

$$
T_{\text{AR}} \approx \sum_{i=1}^{L} \frac{O(d^2 + i \cdot d)}{\beta}
\approx O\left(\frac{L d^2 + L^2 d}{\beta}\right).
$$

也就是说，KV cache 虽然避免了重复计算历史 token 的 $K$、$V$，但历史 KV 仍然需要被读取；而 token 之间的串行依赖也仍然存在。

## 3 Diffusion 的优势：一次搬来的 weights，可以服务更多 tokens

Diffusion language model 的执行方式不同。每一个 denoising step，它都同时处理整个长度为 $L$ 的 sequence。

如果继续用工厂的比喻，现在一次任务里有 $L$ 个 tokens。单步计算量确实更大，大约是

$$
O(L \cdot d^2 + L^2 \cdot d),
$$

前一项对应所有 token 的线性投影，后一项对应整个 sequence 的 global attention。

但更大的计算量，也带来了更高的 parallelism 和更充分的 **weight reuse**。同一组 weights 可以在一次较大的矩阵运算里，被 $L$ 个 tokens 共同使用。搬一次数据之后，可以做更多计算。

在一个简化的 scaling 分析里，arithmetic intensity 可以写成

$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}}
\approx
\frac{O(L \cdot d^2 + L^2 \cdot d)}{O(d^2 + L \cdot d)}
\approx O(L).
$$

具体的 scaling 还取决于实现方式，但这里的直觉很清楚：相比每次只处理一个 token，diffusion 把更多计算集中到了一次 forward 里，提高了每次数据读取所对应的计算量。

回到

$$
T_{\text{real}} \approx \max(\text{compute time}, \text{memory time}),
$$

更高的 arithmetic intensity，让这种计算方式更有可能充分利用 GPU 的计算能力。它可以更好地摊薄数据搬运的开销，也更容易提供足够多的并行任务。

如果假设这种执行方式已经进入 compute-bound 区间，那么做 $T$ 个 denoising steps 的总时间，可以粗略写成

$$
T_{\text{Diff}} \approx \sum_{t=1}^{T} \frac{O(L \cdot d^2 + L^2 \cdot d)}{\pi}
\approx O\left(\frac{T L d^2 + T L^2 d}{\pi}\right).
$$

现在比较两者的运行时间：

$$
\frac{T_{\text{Diff}}}{T_{\text{AR}}}.
$$

如果进一步假设 $d^2$ 项占主导，就有

$$
\frac{T_{\text{Diff}}}{T_{\text{AR}}}
\approx
\frac{\frac{T L d^2}{\pi}}{\frac{L d^2}{\beta}}
=
T \cdot \frac{\beta}{\pi}.
$$

在这个近似里，$L$ 和 $d$ 被约掉了，留下两个主要因素：diffusion steps 的数量 $T$，以及硬件的 bandwidth 和 compute throughput 的比例 $\beta / \pi$。

比如用 H100 的一组近似峰值做量级估算：

$$
\frac{\beta}{\pi}
\approx
\frac{3.35 \text{ TB/s}}{1000 \text{ TFLOPS}}
\approx
\frac{1}{300}\;\text{bytes/FLOP}.
$$

这个比值本身有单位，不能直接当成精确的 speedup 或 step 数阈值；前面省略的数据类型、每个元素的字节数和计算常数，也都会影响实际比较。但它提供了一个有用的直觉：GPU 的计算能力相对于数据搬运能力很强，因此多做一些 FLOPs，并不一定意味着要花更多时间。

在这些简化假设下，即使 diffusion 需要几十步，甚至几百步以内，它仍然可能在 wall-clock time 上和 AR 竞争。当然，这只是量级上的估计，不代表某个固定的 $T$ 一定更快。实际速度还取决于 batch size、kernel 实现和真正能达到的硬件利用率。

我觉得这里最有意思的是：**diffusion 可以用更多计算，换取更高的执行效率。**

## 4 Block Diffusion：用 block size 控制并行粒度

Full-sequence diffusion 一次处理整个 sequence，而 Block Diffusion 把生成粒度放在一个大小为 $B$ 的 block 上。

它不再每次只生成一个 token，也不要求每一步更新整个 sequence，而是按 block 逐步生成。假设目标长度是 $L$，那么大约需要生成 $L/B$ 个 blocks，每个 block 内部再执行 $T$ 个 denoising steps。

生成第 $i$ 个 block 时，当前需要更新的长度是 $B$，历史 context 长度大约是 $iB$。单个 denoising step 的 FLOPs 大约是

$$
O(B d^2 + i B^2 d).
$$

其中，$B d^2$ 来自当前 block 的线性投影，$i B^2 d$ 来自当前 block 对历史 context 的 attention。每个 block 做 $T$ 步，总 FLOPs 就是

$$
O\bigl(T(B d^2 + i B^2 d)\bigr).
$$

再看数据搬运。在可以缓存已确定历史的 KV 的设置下，每个 denoising step 主要需要读取两部分：大小约为 $O(d^2)$ 的 model weights，以及大小约为 $O(i B d)$ 的历史 KV cache。因此，memory I/O 可以粗略写成

$$
O(d^2 + i B d).
$$

对应的 arithmetic intensity 是

$$
I
=
\frac{\text{FLOPs}}{\text{Memory I/O}}
\approx
\frac{O(B d^2 + i B^2 d)}{O(d^2 + i B d)}
\approx O(B).
$$

这个结果比较直观。AR 每次只处理一个 token，arithmetic intensity 通常接近常数量级；full-sequence diffusion 的 intensity 随 $L$ 增长；而 Block Diffusion 可以用 block size $B$ 来控制这个量。

换句话说，BDLM 的并行粒度有了一个更直接的调节方式：不一定要把整个 sequence 都放进一次 denoising，而是选择一个足够大的 block，让单次计算能有效利用 GPU。

根据 roofline model，

$$
T_{\text{real}} \approx \max\left(\frac{\text{FLOPs}}{\pi}, \frac{\text{Memory I/O}}{\beta}\right),
$$

当

$$
I(B) > \frac{\pi}{\beta}
$$

时，计算就进入 compute-bound 区间。$\pi / \beta$ 是硬件的 ridge point，也就是从 memory-bound 转向 compute-bound 的分界。

因此，如果 $B$ 足够大，使 arithmetic intensity 越过这个分界，单步时间就主要由计算吞吐 $\pi$ 决定。

一共约有 $L/B$ 个 blocks，把历史 context 长度的增长也考虑进去，总 FLOPs 的量级可以写成

$$
O\left(
T \cdot \frac{L}{B} (B d^2 + L B d)
\right).
$$

在 compute-bound 的假设下，对应的总时间是

$$
T_{\text{Block}} \approx O\left(\frac{T (L d^2 + L^2 d)}{\pi}\right).
$$

从渐近复杂度看，它仍然可能和 full-sequence diffusion 同阶。这里 $B$ 甚至从最终的量级表达式里消失了。

但这不意味着 $B$ 对速度不重要。这个表达式已经假设硬件计算能力被充分利用，而 $B$ 正是影响这个假设能否成立的一个因素。Block size 太小，仍然可能 memory-bound；把整个长 sequence 都放进一次计算，又可能带来很大的单步计算和显存压力。

所以，Block Diffusion 的价值主要体现在 execution pattern 上：它把逐 token 的串行生成变成逐 block 的并行生成，同时给单次 denoising 的规模提供了一个可调节的粒度。

## 5 从 hardware efficiency 重新看这几种模型

从这个角度看，BDLM 在长序列生成中可能比 full-sequence DLM 更实用，主要有几个原因。

第一，**arithmetic intensity 更容易通过 $B$ 来调节**。我们不一定需要让所有 $L$ 个 tokens 一起参与更新，只需要选择合适的 block size，让单次计算有足够的并行度，并尽量跨过硬件的 ridge point。

第二，**单步更新带来的显存压力更局部**。Full-sequence diffusion 需要处理整个 sequence 的两两 attention 关系，而 BDLM 每一步只更新当前的 $B$ 个 tokens。历史 KV cache 仍然会随 context 增长，但当前 block 的更新规模更容易控制，也更有利于局部数据复用。

第三，**每个 block 已经有确定的历史作为条件**。相比从头联合生成整个 sequence，当前 block 的条件分布可能更集中，因此有机会用更少的 denoising steps 达到类似质量。如果可以减少 $T$，就能直接减少 forward passes。不过这仍然取决于模型和任务，不能仅凭 block 结构就保证。

所以，我现在更倾向于从两个角度一起看 AR、DLM 和 BDLM：它们需要多少计算，以及这些计算以什么方式执行。

- **AR** 的依赖结构更稀疏，但 token 之间的串行依赖很强，单步 decoding 也容易 memory-bound。
- **DLM** 的理论计算量更大，但一次可以并行处理整个 sequence，更容易提高 weight reuse 和 GPU utilization。
- **BDLM** 用 block size 在串行依赖和硬件并行之间做折中，也给速度和生成质量之间的取舍提供了更多空间。

这并不意味着 diffusion 在计算量上天然更便宜。更准确地说：

> AR 的优势在于更稀疏的依赖结构，diffusion 的优势在于更高的硬件执行效率。在现代 GPU 上，后者有时足以抵消额外计算带来的开销。
