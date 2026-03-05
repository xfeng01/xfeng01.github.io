---
title: "Some Basic Thoughts on Attention Mechanisms and the KV-Cache"
description: "Some notes on attention machanisms."
date: 2024-05-15
tags:
  - LLM
  - Notes
---

## 1 Fundamentals of the Attention Mechanism and KV-Cache

### 1.1 Self-Attention vs. Cross-Attention

- Self-Attention: The Query $Q$, Key $K$, and Value $V$ matrices are all derived from the **same input sequence**. This structure enables each token to observe and integrate information from all other tokens within the sequence, effectively capturing internal semantic dependencies.
- Cross-Attention: The $Q$ matrix originates from one sequence (typically the Decoder), while the $K$ and $V$ matrices originate from another (typically the Encoder). This mechanism allows a target sequence to "query" information from a source sequence. A classic example is machine translation: when generating a target word (acting as $Q$), the model queries the source sentence (acting as $K$ and $V$) to identify the most relevant context.

### 1.2 The Roles of $Q$, $K$, and $V$

The interaction within the attention mechanism can be intuitively understood as an information retrieval process from an "archive":

- **Key $K$ and Value $V$ - The Archive:**  These represent the deterministic, historical context. Specifically, $K$ acts as the **label/index** (determining *who* is important), while $V$ contains the **actual content** (determining *what* information to extract).
- **Query $Q$ - The Search Request:**  $Q$ encodes the current step's intent. It matches against all available $K$ matrices to compute attention weights, subsequently retrieving the most relevant combination of $V$ matrices.

### 1.3 Autoregressive Generation and the KV-Cache

During inference, standard Transformers operate autoregressively. Generating the $t$-th token requires computing its attention relationship with all preceding $t$ tokens. Without caching, the Attention computation is formulated as:

$$
\mathrm{Attention}(Q_t, K_{1:t}, V_{1:t}) = \mathrm{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}
$$

In this naive approach, to compute the output at position $t$, the model must redundantly recompute the keys $K_{1:t-1}$ and values $V_{1:t-1}$ for all historical positions.

Since $Q$ drives the active generation while $K$ and $V$ provide static historical information, models store historical keys and values in the ​**KV-Cache**. Consequently, the memory capacity allocated for the KV-Cache directly dictates the maximum context window the model can retain.

### 1.4 Semantic Compression via Hidden States $h_t$

Throughout the sequential generation pipeline:

- The query $Q_t$ used to predict the $t$-th token is intrinsically derived from the output embedding of the $(t-1)$-th token.
- As the representation processes through the Transformer layers, the final layer's output, $h_t$, serves as the **semantic compression** (or summarization) of all sequential information spanning from position $1$ to position $t$.

‍

## 2 An Example to Understand the Generation Process

Suppose we have a simplified one-layer Transformer. Our target is to make the model autoregressively generate the sequence: "To be or not to be."

Let us define the key notations and their corresponding dimensions. Let $d$ denote the model's hidden dimension, and $|V|$ denote the vocabulary size:

- **Input Representation $x_t \in \mathbb{R}^{1 \times d}$:** The input vector at step $t$, which is the sum of the current token embedding and the positional embedding.
- $Q_t, K_t, V_t \in \mathbb{R}^{1 \times d}$: Obtained by multiplying $x_t$ with three respective weight matrices ($W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$). Here, $Q_t$ acts as the query, while $K_t$ and $V_t$ are the keys and values to be stored in the KV-Cache.
- **Context Vector $z_t \in \mathbb{R}^{1 \times d}$:** The output of the Attention layer. It is a weighted sum of the historical values ($V_{1:t} \in \mathbb{R}^{t \times d}$) and represents the features incorporating context information.
- **Hidden State $h_t \in \mathbb{R}^{1 \times d}$:** The final output of the Transformer layer after passing $z_t$ through a FFN/MLP. It represents the accumulated semantic information up to step $t$.

Now, let's examine how the model generates this sequence step-by-step.

Step 1: Initialization (Input: `[BOS]` $\rightarrow$ Predict: `"To"`)

We feed a special token `[BOS]` (Begin of Sequence) to prompt the model to start generating.

- **Input Layer:**  Compute the token embedding for `[BOS]`, add the positional encoding, and obtain $x_1 \in \mathbb{R}^{1 \times d}$.
- **Generate Q, K, V:**  Apply linear projections to $x_1$ to obtain $Q_1, K_1, V_1 \in \mathbb{R}^{1 \times d}$.
- **Update KV-Cache:**  Store $K_1$ and $V_1$ into the cache. The current cache size for both K and V is $1 \times d$.
- **Attention ($z_1$):**  Since there is no prior context, $Q_1$ only attends to $K_1$ (the attention weight is trivially $1$). Thus, $z_1 = V_1 \in \mathbb{R}^{1 \times d}$.
- **Hidden State & Prediction:**  Pass $z_1$ through the MLP to obtain $h_1 \in \mathbb{R}^{1 \times d}$. Finally, $h_1$ is projected by the Language Modeling (LM) head (using a weight matrix $W_{LM} \in \mathbb{R}^{d \times |V|}$) to yield a logits vector $\in \mathbb{R}^{1 \times |V|}$, predicting the highest-probability next token: `"To"`.

Step 2: Incorporating Context (Input: `"To"` $\rightarrow$ Predict: `"be"`)

In the autoregressive setting, the token predicted in the previous step (`"To"`) becomes the input for the current step.

- **Input Layer:**  Compute the token embedding for `"To"`, add the positional encoding for position 2, and obtain $x_2 \in \mathbb{R}^{1 \times d}$.
- **Generate Q, K, V:**  Apply the linear projections to $x_2$ to generate $Q_2, K_2, V_2 \in \mathbb{R}^{1 \times d}$. Here, $Q_2$ encodes the semantic query: "Given the context so far, what should follow the word 'To'?"
- **Update KV-Cache:**  Append the newly computed $(K_2, V_2)$ row vectors to the existing KV-Cache. The cache now holds the accumulated history matrices $K_{1:2}, V_{1:2} \in \mathbb{R}^{2 \times d}$.
- **Attention (**​**$z_2$**​ **):**  The query $Q_2 \in \mathbb{R}^{1 \times d}$ attends to all available keys in the cache $K_{1:2} \in \mathbb{R}^{2 \times d}$ via dot-product, resulting in attention scores $\in \mathbb{R}^{1 \times 2}$. After applying the softmax function to these scores, the resulting attention weights are used to compute a weighted sum of $V_{1:2}$. The output, $z_2 \in \mathbb{R}^{1 \times d}$, is no longer the isolated representation of `"To"`; it is now a context-aware feature vector.
- **Hidden State & Prediction:**  The context vector $z_2$ is processed by the MLP to yield the hidden state $h_2 \in \mathbb{R}^{1 \times d}$. Finally, the LM head maps $h_2$ to the vocabulary distribution $\in \mathbb{R}^{1 \times |V|}$, predicting the highest-probability next token: `"be"`.

The subsequent steps follow the exact same autoregressive pattern, appending new $K$ and $V$ matrices to the cache and generating the next sequence token until a termination condition is met.

‍

‍

‍

‍

‍

‍

‍

‍
