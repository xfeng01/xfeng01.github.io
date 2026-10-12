---
title: "Steering Is Easy, Routing Is Not: Dual-Stream Provenance Routing for Multi-Turn Constraint Adherence"
authors: ["Chenan Wang", "Xinsong Feng", "Yuanhe Li", "Johnny Liu", "Da Li", "Dong Li", "Ayan Mukhopadhyay", "Haipeng Chen"]
pub: "Submitted to AAAI 2027"
image: "/support.png"
date: 2026-07-28
description: "We frame multi-turn constraint adherence as a routing problem between user constraints and model-generated history, and introduce SUPPORT, a training-free dual-stream method. A constraint-focused stream provides both corrective representations and an adaptive routing gate, improving adherence while preserving useful conversation history."
---

Large language models (LLMs) can accurately restate a constraint even while violating it.
Across multi-turn conversations, they may ignore active constraints or continue to follow the withdrawn ones, despite correctly recalling both when asked.
This Knowing-But-Violating (KBV) gap can hardly be attributed to memory failure because the constraints remain present and retrievable; we instead interpret the gap as a routing failure: generation increasingly relies on previous model responses rather than active user constraints.
Strengthening user constraints can reduce such violations, but previous responses may also contain intermediate results required by later turns.
Fixed-strength steering can therefore improve KBV while severely degrading conflict-free Lost in Conversation (LIC) tasks.
The central challenge is not how to steer generation, but when and how strongly each source should influence it.
A natural low-cost gate based on prediction divergence instead measures dependence on previous responses rather than the presence of a constraint conflict.
Among the gates we tested, only those using a second contextual view are effective on both axes.

We propose *SUPPORT*, a training-free dual-stream method that uses this second view for both correction and routing.
The same model maintains two independently KV-cached streams: a full stream over the complete conversation and a clean stream that removes previous assistant responses while retaining system and user messages.
At one intermediate layer, the clean hidden state is injected into the full hidden state using an adaptive weight derived from the predictions of the two streams.
The method requires only one model-specific choice: the injection layer.

Across three models, four benchmarks, and two independent LLM judges, SUPPORT approximately halves violations under heavy drift while matching or improving the baseline on 17 of 18 LIC model-task cells.
Among the evaluated methods, SUPPORT is the only approach that improves constraint adherence without systematically degrading multi-turn capability.
