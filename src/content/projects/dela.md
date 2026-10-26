---
title: "DELA: Learning Controllable Edits for Diffusion Language Models"
authors: ["Xinsong Feng", "Chenan Wang", "Leshu Tang", "Johnny Liu", "Dong Li", "Da Li", "Haipeng Chen"]
pub: "Submitted to ICLR 2027"
image: "/dela.png"
date: 2026-09-26
description: "We introduce Diffusion Editing Language Adapter (DELA), a plug-in edit controller that separates draft-level control from token-level denoising in frozen diffusion language models. Explicit INSERT, DELETE, REMASK, and STOP actions enable controllable variable-length refinement, with cache-friendly drafting followed by targeted global edits."
paper: "https://arxiv.org/abs/2610.32096"
---

Diffusion language models (DLMs) support bidirectional denoising and iterative refinement, but generation remains largely controlled by token-level masking schedules.
This leaves draft-level refinement implicit and limits explicit control over when to insert missing content, delete redundant spans, revisit uncertain regions, or stop editing.
We introduce Diffusion Editing Language Adapter (DELA), a plug-in framework that enables controllable draft refinement with a frozen DLM.
DELA operates on clean, variable-length drafts using four explicit editing actions: INSERT, DELETE, REMASK, and STOP.
The controller determines when and where to edit the draft, while the frozen DLM fills masked regions through denoising, separating high-level edit control from token-level generation.
A naive single-phase editor would be computationally expensive because arbitrary interior edits can disrupt prefix stability and limit reliable KV-cache reuse.
To address this, DELA uses a two-phase decoding procedure: a fast, append-only drafting phase that supports KV-cache reuse, followed by a global refinement phase that enables targeted non-monotonic edits to the full draft.
We train the controller using a staged curriculum that begins with learning local edit priors from synthetic recovery tasks and mined actions that improve draft quality.
Subsequent stages refine edit selection through rollout-based outcome preferences and alignment with downstream task feedback.
Experiments on mathematical reasoning and code generation show that DELA achieves gains of up to 5.5 percentage points over direct diffusion decoding while keeping the backbone frozen.
DELA thus provides a modular approach to explicit draft-level control in diffusion language models.
