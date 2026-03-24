# KOS Engine -- Ready-to-Post Outreach

Live Demo: https://web-production-bfba9.up.railway.app
GitHub: https://github.com/skvcool-rgb/KOS-Engine

---

## 1. r/MachineLearning Post

**Submit at:** https://www.reddit.com/r/MachineLearning/submit
**Flair:** [Project]

**Title:** [P] KOS Engine: Reviving Spreading Activation as a Deterministic Alternative to RAG -- Zero Hallucination by Confining LLMs to Thin I/O

**Body:**

We open-sourced KOS Engine, a neurosymbolic knowledge system that takes a hard architectural stance: LLMs should never reason. Instead, all inference flows through a fuel-constrained spreading activation graph, with the LLM confined to a thin I/O shell that reads one or two pre-scored sentences.

The core graph mechanism draws directly from Collins & Loftus (1975), but introduces a biologically motivated fuel constraint. Each activation pulse carries a finite fuel budget that decays along weighted edges, so the search space is bounded by graph topology rather than by arbitrary top-k cutoffs. This makes retrieval deterministic and inspectable -- you can trace exactly why a particular node fired and which path delivered the answer.

Key technical contributions:

- **Algorithmic Weaver:** A deterministic intent-scoring layer that decomposes user queries into weighted sub-intents before any graph traversal begins.
- **6-layer typo recovery cascade:** Exact match, fuzzy ratio, phonetic encoding, hypernym expansion, substring containment, and vector similarity -- evaluated in order with early exit.
- **SymPy math coprocessor:** Symbolic arithmetic and calculus executed outside the LLM, eliminating numeric hallucination entirely.
- **Myelination:** Edges that were useful in past queries gain weight over time, analogous to biological myelination, giving the graph a self-learning property without gradient descent.

The system passes 16/16 benchmark tests spanning factual recall, multi-hop reasoning, mathematical queries, and adversarial typo inputs. Because the LLM receives only the final pre-scored context window, hallucination is structurally impossible rather than statistically suppressed.

Live Demo: https://web-production-bfba9.up.railway.app
Repo: https://github.com/skvcool-rgb/KOS-Engine

We welcome feedback on the activation dynamics and the fuel decay model in particular.

---

## 2. Hacker News Post

**Submit at:** https://news.ycombinator.com/submit

**Title:** KOS: Zero-hallucination knowledge engine – LLM never reasons, graph does all the work

**URL:** https://github.com/skvcool-rgb/KOS-Engine

**First comment (post immediately after submitting):**

Author here. KOS Engine takes a different approach to grounding LLMs: instead of retrieval-augmented generation, all reasoning happens in a deterministic spreading activation graph inspired by Collins & Loftus (1975). The LLM is reduced to a thin I/O layer that reads one or two pre-scored sentences -- it never reasons, so it cannot hallucinate.

The graph uses a fuel-budget model borrowed from biological neuron dynamics: each activation pulse decays as it traverses weighted edges, naturally bounding the search without arbitrary top-k limits. A 6-layer typo recovery cascade handles noisy input, and a SymPy coprocessor handles math symbolically. 16/16 benchmark tests passing. Open source, runs on CPU.

Live demo: https://web-production-bfba9.up.railway.app

---

## 3. r/LocalLLaMA Post

**Submit at:** https://www.reddit.com/r/LocalLLaMA/submit

**Title:** KOS Engine -- open-source neurosymbolic engine where the LLM is just a thin I/O shell (swap in any local model, runs on CPU)

**Body:**

Built an open-source knowledge engine where the LLM does zero reasoning. All inference runs through a deterministic spreading activation graph on CPU. The LLM only reads 1-2 pre-scored sentences at the end, so you can swap gpt-4o-mini for Mistral, Phi, Llama, or literally anything that can complete a short prompt.

Think of it as the anti-RAG: instead of stuffing a context window and hoping the model figures it out, the graph already did all the work. The model just formats the answer. This means a small local model performs identically to a frontier API model because it is not doing any heavy lifting.

What you get:

- Runs entirely on CPU (no GPU needed for the reasoning layer)
- 6-layer typo recovery so user input can be messy
- SymPy math coprocessor for exact arithmetic and calculus
- Deterministic and fully traceable -- every answer has an auditable activation path
- 16/16 benchmarks passing

The LLM integration is a single swappable module. If your local model can handle a system prompt and a couple of sentences of context, it works.

Live Demo: https://web-production-bfba9.up.railway.app
Repo: https://github.com/skvcool-rgb/KOS-Engine
