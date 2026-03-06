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

In discussions of generative AI, we often encounter a puzzling phenomenon. When generating a sequence of length $L$, the per-step time complexity of a traditional autoregressive model (AR model with KV cache) is usually viewed as $O(L)$ along the attention path. By contrast, a text diffusion model performs full self-attention over the entire sequence for $T$ denoising steps, leading to a theoretical complexity of $O(TL^2)$. From a purely algorithmic standpoint, $O(L) \ll O(TL^2)$ seems decisive. So why, in real modern GPU inference, especially for long-context generation, do diffusion models often keep up with AR models and sometimes even outperform them?

The answer lies in the large gap between algorithmic complexity and **low-level hardware execution efficiency**. This note examines that gap by comparing AR, DLM, and BDLM from a hardware-efficiency perspective.

## 2 Background: Why Are AR Models So Slow in Practice?

To understand what really determines inference time on GPUs, it helps to introduce two core physical quantities. For intuition, think of a GPU as a large **data-processing factory**:

- **Global memory (HBM/VRAM):** a huge **storage room** deep inside the factory. It has enormous capacity (for example, 80GB) and stores large model weights and the full KV cache.
- **Compute units (Tensor Cores):** the **workers** in the factory. They perform the actual matrix computations.
- **On-chip cache (SRAM):** a tiny but extremely convenient **workbench** right next to the workers.
- **Peak compute throughput ($\pi$, Peak FLOPs):** the theoretical compute peak of the GPU, corresponding to the workers' **maximum processing speed**.
- **Memory bandwidth ($\beta$):** the bandwidth between global memory and the workbench, corresponding to the **conveyor belt** that moves data to the workers.

On modern GPUs, there is a brutal reality: the workers are much faster than the conveyor belt, i.e., $\pi \gg \beta$.

Once this hardware picture is clear, the main weakness of AR generation becomes easier to see.

In AR decoding, even with KV cache, generating the $i$-th token does not require that much actual computation. The FLOPs are only about $O(d^2 + i \cdot d)$, where $d$ is the hidden dimension. The problem is that the GPU still has to repeatedly fetch model weights and historical KV entries from global memory. In other words, **each step computes only one token, which means very low parallelism and poor weight reuse**. From the roofline perspective, the resulting **arithmetic intensity** is typically very low:

$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}} \approx O(1)
$$

This connects to a key hardware mechanism on modern GPUs: **asynchronous overlap**.

Naively, one might think the total latency for processing a batch of data should be "data movement time + compute time." But in an efficient pipeline, while the workers are processing the current batch, the conveyor belt can already be moving the next one. Since these two processes overlap in physical time, the actual runtime is not their simple sum. Instead, it is better approximated by the slower of the two:

$$
T_{\text{real}} = \max\left(\text{compute time}, \text{memory time}\right)
= \max\left(\frac{\text{FLOPs}}{\pi}, \frac{\text{Memory I/O}}{\beta}\right)
$$

What does an $O(1)$ arithmetic intensity imply?

It means that in this factory, **the workers often finish their current computation quickly, then sit idle waiting for the next batch of data to arrive from memory**. As a result, the system easily falls into a strongly **memory-bound** regime. The large compute capability $\pi$ cannot be fully utilized, and actual runtime is dominated by the slower memory bandwidth $\beta$.

Therefore, to generate $L$ tokens, this inefficient pipeline must be launched serially $L$ times. The overall wall-clock time can be roughly approximated as

$$
T_{\text{AR}} \approx \sum_{i=1}^{L} \frac{O(d^2 + i \cdot d)}{\beta}
\approx O\left(\frac{L d^2 + L^2 d}{\beta}\right)
$$

## 3 The Key Advantage of Diffusion Models: Better Weight Reuse Under High Parallelism

Compared with the token-by-token serial generation of AR models, DLMs have a fundamentally different computation pattern. At each denoising step, the model computes **the entire sequence of length $L$ at once**.

Using the same factory analogy, this means the workers' workbench now holds all $L$ tokens simultaneously. As a result, the FLOPs for one denoising step increase to about $O(L \cdot d^2 + L^2 \cdot d)$, including the linear projections for all tokens and the $L \times L$ global attention computation.

**But the crucial change is that both weights and computation are reused much more effectively.** To process all $L$ tokens, the model weights only need to be fetched into on-chip memory once, after which the entire sequence can use them together. This is a classic form of **weight reuse**. So although the FLOPs per step grow substantially, the arithmetic intensity of diffusion models also increases with sequence length:

$$
I = \frac{\text{FLOPs}}{\text{Memory I/O}}
= \frac{O(L \cdot d^2 + L^2 \cdot d)}{O(d^2 + L \cdot d)}
\approx O(L)
$$

This means that as the sequence length $L$ grows, diffusion models become increasingly likely to escape the low-intensity, low-utilization execution mode of AR models and move into a **compute-bound** regime.

Under the runtime model

$$
T = \max(\text{compute time}, \text{memory time}),
$$

the computation is now large enough that the memory bottleneck can be partially hidden in the background. As a result, the GPU can utilize much more of its massive compute capability $\pi$. If the model uses $T$ denoising steps, the total runtime can be approximated as

$$
T_{\text{Diff}} \approx \sum_{t=1}^{T} \frac{O(L \cdot d^2 + L^2 \cdot d)}{\pi}
\approx O\left(\frac{T \cdot L \cdot d^2 + T \cdot L^2 \cdot d}{\pi}\right)
$$

**The critical threshold.** To compare the actual runtimes of diffusion and AR more directly, consider the ratio $\frac{T_{\text{Diff}}}{T_{\text{AR}}}$. Under a first-order approximation where the feature-dimension term $d^2$ dominates, we have

$$
\frac{T_{\text{Diff}}}{T_{\text{AR}}}
\approx
\frac{\frac{T \cdot L \cdot d^2}{\pi}}{\frac{L \cdot d^2}{\beta}}
=
T \cdot \frac{\beta}{\pi}
$$

This simple expression gives a very clear physical intuition. Under this approximation, the effects of sequence length $L$ and model dimension $d$ cancel out. The main factors are the algorithmic number of diffusion steps $T$ and the hardware constant $\frac{\beta}{\pi}$. For a modern GPU such as the H100, this ratio is roughly

$$
\frac{\beta}{\pi}
\approx
\frac{3.35 \text{ TB/s}}{1000 \text{ TFLOPS}}
\approx
\frac{1}{300}
$$

> This suggests that as long as the number of diffusion steps $T$ is small enough, for example $T < 300$, a diffusion model may still run faster than an AR model in practice, even if it performs more FLOPs in theory. Big-O notation hides real hardware utilization, and hardware efficiency is often what ultimately determines wall-clock speed.

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
