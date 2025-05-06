---
title: Storytelling in Research Writing
published: 2025-05-07
description: Here are some of my thoughts on how to write a research paper. 
tags:
  - writing
draft: false 
---

## 1. Why Does "Storytelling" Matter in Research Writing?

A collaborator once told me: *“Your writing feels like a technical report—it lacks a story.”* That got me thinking: **what does “story” really mean in the context of an academic paper?** Here, I'll break down what storytelling means in research writing and share a simple method—*self-questioning*—that helps you build a strong, logical flow.



## 2. What Is “Storytelling” in Scientific Writing?

Based on my understanding, as well as advice from my advisor Ian, *storytelling means guiding the reader along a clear logical chain.*  For example, when a reader is going through your paper, they should feel that each sentence flows naturally and anticipates their questions. In my view, that's a sign of a well-constructed logical chain. It's a bit like solving a math problem step by step.

> Good storytelling answers the reader’s questions before they even ask them. It feels natural and persuasive.


## 3. A Common Mistake: Listing Facts Without Flow

Here’s an example of poor writing:

```text
Diffusion models are widely used to represent multi-modal data.  
While expressive, they are slow.  
Consistency models accelerate inference.  
We propose a new method.
```

**What’s the problem?**
- These are just isolated facts.
- There’s no natural connection or sense of progression.



## 4. The Self-Questioning Method

One simple trick to improve storytelling is to *“self-question”* as you write. For each major point, ask yourself:
- Why?
- So what?
- What next?

Usually, you can apply this in two ways:

- **Implicit self-questioning** (using guiding phrases):  
  - *However, this raises a critical question: …*  
  - *To address this issue, we propose…*

- **Explicit self-questioning** (writing actual questions):  
  - *But why is this the case?*  
  - *The reason lies in...*



## 5. Example: Before and After

**❌ Original (no story flow):**

```text
Diffusion models are widely used to represent multi-modal data, including policies in offline RL.  
While expressive, these models rely on multi-step denoising, resulting in slow inference.  
Consistency models use a one-step mapping, improving speed.  
We propose ODE-Trajectory Policies.
```

**✅ Improved (storytelling style):**

```text
Diffusion models have become popular tools for modeling multi-modal policies in offline RL due to their expressiveness.  
However, a key challenge remains: why are these models often impractical in RL applications?  
The problem lies in their reliance on multi-step denoising, which leads to slow inference.  
To overcome this, consistency models replace multi-step denoising with a single-step mapping.  
Yet, this raises another question: can we maintain expressiveness while achieving fast inference?  
In this work, we address this challenge with Continuous-Time Diffusion Policies, which explicitly model the full trajectory of the data generation process.
```

**Commentary:**
- Notice how each sentence naturally leads to the next.
- Each problem sets up the need for the next solution.



## 6. Takeaways

- Storytelling in research = building a clear logical chain.
- The self-questioning method is a practical way to create this chain.
- Always ask yourself: what might the reader wonder next? Answer that in your writing.



## Bonus Phrases You Can Use

- *This raises a key question: …*  
- *To address this challenge, we propose …*  
- *However, a limitation remains: …*  
- *Our method aims to balance …*  
- *We demonstrate that …*
