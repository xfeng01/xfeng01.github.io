---
title: "Why O(TL²) Diffusion Can Beat O(L) AR in Practice"
description: "A hardware-efficiency perspective on why diffusion models can sometimes generate faster than autoregressive models despite worse theoretical complexity."
date: 2026-03-06
tags:
  - Diffusion
  - LLM
  - Systems
---

## 1 Introduction: A Counterintuitive Complexity Paradox

When we talk about language models, one question naturally comes up. In an autoregressive (AR) model with a KV cache, computing $Q$, $K$, and $V$ for the new token is $O(1)$ with respect to sequence length, but the new query must still attend over all cached keys and values, making the attention cost $O(L)$ per decoding step. A diffusion language model (DLM), by contrast, applies full self-attention over the entire sequence at each denoising step, giving a cost of $O(L^2)$ per step, and $O(TL^2)$ over $T$ denoising steps.

From this perspective, diffusion seems much more expensive than autoregressive decoding.  
So why can diffusion models still remain competitive in real GPU inference, especially for long-context generation, and sometimes even outperform autoregressive models?

The key point is that theoretical complexity and actual GPU speed are not the same thing. This note explores that gap by comparing AR, DLM, and BDLM from a hardware-efficiency perspective.

## 2 Background: Why Are AR Models So Slow in Practice?

To understand what really determines inference time on GPUs, it helps to first look at two basic physical quantities. For intuition, think of a GPU as a large **data-processing factory**:

- **Global memory (HBM/VRAM):** a huge **storage room** deep inside the factory. It has massive capacity (for example, 80GB) and holds large model weights and the full KV cache.
- **Compute units (Tensor Cores):** the **workers** in the factory. They carry out the actual matrix computations.
- **On-chip cache (SRAM):** a small but extremely fast **workbench** right next to the workers.
- **Peak compute throughput ($\pi$, peak FLOPs):** the GPU’s theoretical maximum compute capability, like the workers’ **top processing speed**.
- **Memory bandwidth ($\beta$):** the rate at which data can move from global memory to the workbench, like the **conveyor belt speed** feeding the workers.


On modern GPUs, there is a harsh reality: the workers are much faster than the conveyor belt, that is, $\pi \gg \beta$.

With that picture in mind, the weakness of AR generation becomes easier to understand.

In AR decoding, even with a KV cache, generating the $i$-th token does not require much computation. The FLOPs are only about $O(d^2 + i \cdot d)$, where $d$ is the hidden dimension.  
What really hurts is memory traffic. The GPU still has to keep pulling model weights and past KV entries from global memory at every step. Since each step produces only one token, parallelism stays low and the same weights cannot be reused very effectively.

From the roofline perspective, this means the **arithmetic intensity** is usually very low:

$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}} \approx O(1)
$$

This leads to another important hardware idea on modern GPUs: **overlap**.

At first glance, you might think the latency of one step should be

$$
\text{data movement time} + \text{compute time}.
$$

But a well-optimized GPU pipeline does not work like that. While the compute units are processing the current work, data for the next work can already be moving through memory at the same time. Because these two parts overlap, the real runtime is usually not their sum. It is better approximated by the slower one:

$$
T_{\text{real}} = \max\left(\text{compute time}, \text{memory time}\right)
= \max\left(\frac{\text{FLOPs}}{\pi}, \frac{\text{Memory I/O}}{\beta}\right)
$$

So what does an $O(1)$ arithmetic intensity mean here?

It means computation grows only in the same order as memory traffic. On modern GPUs, where compute is much stronger than memory bandwidth, this is usually not enough to keep the compute units busy. In the factory picture, the workers finish quickly and then wait for the next data to arrive. The system therefore falls into a strongly **memory-bound** regime: the large compute capability $\pi$ cannot be fully used, and runtime is dominated by the slower memory bandwidth $\beta$.

As a result, AR decoding suffers twice. Each step has low arithmetic intensity, and the whole process must also run sequentially, one token at a time. To generate $L$ tokens, this inefficient pipeline is launched $L$ times in a row. Under a memory-bound approximation, the total wall-clock time can be written roughly as

$$
T_{\text{AR}} \approx \sum_{i=1}^{L} \frac{O(d^2 + i \cdot d)}{\beta}
\approx O\left(\frac{L d^2 + L^2 d}{\beta}\right)
$$


## 3 The Key Advantage of Diffusion Models: Better Weight Reuse Under High Parallelism

Compared with the token-by-token serial generation of AR models, diffusion language models (DLMs) follow a very different computation pattern. At each denoising step, the model processes the **entire sequence of length $L$ at once**.

In the factory analogy, this means the workbench now holds all $L$ tokens at the same time. The amount of computation in one denoising step therefore becomes much larger, roughly
$$
O(L \cdot d^2 + L^2 \cdot d),
$$
including the linear projections for all tokens and the global attention over the full sequence.

The key advantage is that this larger computation also brings much better **parallelism** and **weight reuse**. Instead of generating one token at a time, the model applies the same weights across all $L$ tokens within one large computation. As a result, each weight load supports much more actual computation, which increases arithmetic intensity.

Under a simplified scaling view, the arithmetic intensity becomes
$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}}
\approx
\frac{O(L \cdot d^2 + L^2 \cdot d)}{O(d^2 + L \cdot d)}
\approx O(L).
$$

The exact scaling depends on implementation details, but the main point is clear: as sequence length grows, diffusion-style computation typically achieves much higher arithmetic intensity than AR decoding. This makes it far more likely to utilize GPU compute well, rather than spending most of its time waiting on memory.

Under the runtime model
$$
T = \max(\text{compute time}, \text{memory time}),
$$
this higher-intensity execution is much more favorable to modern GPUs. Memory traffic can be better amortized, parallelism is much higher, and the hardware can use more of its peak compute throughput $\pi$. If the model performs $T$ denoising steps, its total runtime can be roughly written as
$$
T_{\text{Diff}} \approx \sum_{t=1}^{T} \frac{O(L \cdot d^2 + L^2 \cdot d)}{\pi}
\approx O\left(\frac{T L d^2 + T L^2 d}{\pi}\right).
$$

To compare diffusion and AR more directly, consider the ratio
$$
\frac{T_{\text{Diff}}}{T_{\text{AR}}}.
$$
Under a first-order approximation in which the $d^2$ term dominates, we get
$$
\frac{T_{\text{Diff}}}{T_{\text{AR}}}
\approx
\frac{\frac{T L d^2}{\pi}}{\frac{L d^2}{\beta}}
=
T \cdot \frac{\beta}{\pi}.
$$

This gives a useful hardware-level intuition. In this approximation, the sequence length $L$ and hidden size $d$ cancel out, leaving two main factors: the number of diffusion steps $T$ and the hardware ratio $\beta / \pi$. For a modern GPU such as the H100, this ratio is roughly
$$
\frac{\beta}{\pi}
\approx
\frac{3.35 \text{ TB/s}}{1000 \text{ TFLOPS}}
\approx
\frac{1}{300}.
$$

So, as a rough back-of-the-envelope estimate, if the number of diffusion steps is small enough, for example on the order of a few hundred or less, diffusion can remain competitive with AR in wall-clock time, and may even be faster in practice despite doing more FLOPs on paper. The reason is simple: big-O counts operations, but real GPU speed depends on how efficiently those operations map onto hardware.

## 4 The Hardware Advantage of Block Diffusion

The key idea of Block Diffusion is to increase the generation granularity from a single token to a block of size $B$. Instead of predicting one token at a time, each step generates $B$ tokens together. If the target sequence has length $L$, then the generation process consists of about $\frac{L}{B}$ blocks, and each block internally runs $T$ diffusion denoising steps.

When generating the $i$-th block, the current block length is $B$, and the historical context length is about $iB$. The FLOPs of a single diffusion step are roughly $O(B d^2 + i B^2 d)$. Since each block runs $T$ denoising steps, the total FLOPs per block are

$$
O\bigl(T(B d^2 + i B^2 d)\bigr)
$$

Now consider memory traffic. At each diffusion step, the GPU must read model weights $W$ of about $O(d^2)$ and historical KV cache of about $O(i B d)$. So the memory I/O is approximately

$$
O(d^2 + i B d)
$$

This gives a rough estimate of the arithmetic intensity of Block Diffusion:

$$
I
=
\frac{\text{FLOPs}}{\text{Memory I/O}}
=
\frac{O(B d^2 + i B^2 d)}{O(d^2 + i B d)}
\approx O(B)
$$

This is very different from AR models. With AR plus KV cache, arithmetic intensity is typically close to a constant. For full-sequence diffusion, it grows with the sequence length. For Block Diffusion, it is controlled more directly by the block size $B$. In other words, the hardware utilization of Block Diffusion depends less on the full sequence length and more on the chosen block size.

Under the GPU runtime model,

$$
T_{\text{real}} = \max\left(\frac{\text{FLOPs}}{\pi}, \frac{\text{Memory I/O}}{\beta}\right),
$$

the system becomes compute-bound once

$$
I(B) > \frac{\pi}{\beta}.
$$

That is, if the block size is large enough for the arithmetic intensity to exceed the hardware ridge point $\frac{\pi}{\beta}$, runtime will be dominated by compute throughput $\pi$ rather than memory bandwidth $\beta$.

There are $\frac{L}{B}$ blocks in total, so the overall FLOPs are approximately

$$
O\left(
T \cdot \frac{L}{B} (B d^2 + L B d)
\right).
$$

This simplifies to

$$
T_{\text{Block}} \approx O\left(\frac{T (L d^2 + L^2 d)}{\pi}\right).
$$

Asymptotically, Block Diffusion may still have the same order as standard diffusion. But its advantage is not mainly about changing the complexity class. Instead, it changes the **execution pattern**: it replaces extremely serial token-by-token generation with block-by-block parallel generation, while also avoiding the excessively large instantaneous computation and memory pressure of full-sequence diffusion on very long inputs.

Put differently, Block Diffusion improves arithmetic intensity at the algorithmic level so that the computation pattern is better aligned with the roofline characteristics of modern GPUs.

## 5 Conclusion: Choosing Architectures from a Hardware-Efficiency Perspective

From the analysis above, we can see why Block Diffusion (BDLM) often has better practical behavior than full-sequence Diffusion (DLM) in long-sequence generation. The main reasons are roughly:

- **More stable arithmetic intensity:** although BDLM and DLM both face the challenge of $O(L^2)$ complexity, BDLM keeps its arithmetic intensity roughly in the range of $I \approx O(B)$. As long as the block size is large enough to cross the hardware ridge point, the model can more reliably stay compute-bound.
- **Localized memory pressure:** global attention needs to handle $L \times L$ intermediate structures, which can easily become a memory bottleneck for long sequences. BDLM updates only $B$ tokens at a time, turning global memory pressure into local block-level pressure. This not only reduces the risk of OOM, but also improves locality for on-chip caches.
- **Better marginal utility of the diffusion step count $T$:** in BDLM, each block is denoised conditioned on already-determined history, so the conditional distribution is often more concentrated than in full-sequence generation. This means fewer diffusion steps may be enough to reach similar quality, and since total runtime is roughly proportional to $T$, reducing $T$ can lead to direct speedups.

Overall, the difference between AR, DLM, and BDLM is not just about whose theoretical complexity is smaller. It is also about whose computation pattern is better matched to modern GPUs.

- **AR:** sparse dependency structure in theory, but the strongest sequential bottleneck, and often the most likely to become memory-bound.
- **DLM:** more FLOPs in theory, but the highest parallelism, making it easier to utilize GPU compute.
- **BDLM:** a compromise between serial dependency and hardware parallelism, often offering a better balance between speed and quality.

So the real takeaway is not that diffusion models inherently require less computation. Rather:

> **The advantage of AR lies in its sparser dependency structure, while the advantage of diffusion models lies in better hardware execution efficiency. On modern GPUs, the latter can sometimes matter more.**
