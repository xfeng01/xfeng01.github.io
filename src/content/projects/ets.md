---
title: "Keeping Is Not Enough: Separating Selection and Use in Long-Conversation Context Management"
authors: ["Chenan Wang", "Xinsong Feng", "Yuanhe Li", "Johnny Liu", "Da Li", "Dong Li", "Ayan Mukhopadhyay", "Haipeng Chen"]
pub: "Submitted to ACL ARR 2026"
image: "/ets.webp"
date: 2026-10-12
description: "We introduce a selection–use decomposition that separates context-management errors on absent text from errors on text retained verbatim, showing that retaining information does not ensure its use. We propose Explicit Task State (ETS), which keeps requirements verbatim near the final query and presents values in their original sentences."
---

Long conversations require models to follow instructions and reuse values from earlier turns. Context managers select and organize conversation history, yet retaining required information does not ensure that models use it correctly. We introduce an item-level selection–use decomposition that separates errors on absent text from errors on text retained verbatim, using exact string matching before generation. Our analysis finds that rewriting- and retrieval-based managers often fail to retain required text verbatim, while errors persist even with the full context. Controlled experiments further show that successful use depends not only on what is kept, but also on where and in what form it appears. To improve both retention and use, we propose Explicit Task State (ETS), which preserves requirements verbatim near the final query, presents values in their original sentences, and verifies model-selected text against the conversation. We evaluate ETS on Self-Context Bench, a judge-free benchmark with five new evaluation blocks and a common protocol for two public benchmarks. Across seven blocks and three open models, ETS significantly outperforms eight baselines and the full context in 162 of 189 comparisons, with only four significant losses, while its final input contains a median of 14% of full-context tokens.
