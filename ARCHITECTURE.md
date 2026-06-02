# Architecture and design decisions

This document explains *why* the system is built the way it is. It is the
reference for defending the design in an interview.

## Why a multi-agent pipeline rather than one function

Auditing an agent is several distinct jobs: deciding what to probe,
running the probes, judging the answers, and sanity-checking the judge.
Each is a separate concern with its own failure modes, so each is a
separate agent behind its own interface. The orchestrator is the only
component that knows the sequence; the agents do not call each other. This
is hierarchical delegation, and it keeps every stage independently
testable and swappable.

## The agents

- **Generator** decides *what* to test based on a description of the
  target, rather than replaying a fixed script. That decision-making is
  what makes it an agent. The live version uses a ReAct loop (reason about
  weaknesses, then emit probes). The stub uses templated probes across the
  same categories so the pipeline is testable without model calls.
- **Executor** is the only component that touches the target. It measures
  latency, tokens and cost at the call site, because those production
  metrics cannot be reconstructed later.
- **Judge** scores accuracy, safety and instruction-following, each in
  [0, 1], and attaches a confidence. Safety is a hard gate in
  `Verdict.passed`: an accurate but unsafe answer still fails.
- **Critic** is the self-reflection layer. It re-examines *only*
  low-confidence verdicts (below `CRITIC_THRESHOLD`). The premise is that a
  single judge pass is noisy on borderline cases; a second focused look
  catches some of the judge's own mistakes. Running it only on
  low-confidence verdicts keeps cost down while targeting the cases most
  likely to be wrong.

## Why the stub/live provider pattern

Every external dependency (the judge model, the target, tracing) sits
behind a base class with a `stub` and a real implementation, chosen by a
config flag. This gives three things: the pipeline runs and is testable
offline with zero cloud cost; the architecture is provable without
spending model calls; and swapping to production is a config change, not a
rewrite. The adapter boundary is also where the fragile parts live (LLM
JSON parsing in `parse_judge_json`), isolated so they can be unit-tested
against malformed output.

## Why a vector store (roadmap)

Test cases and run results are stored with embeddings so that, given a new
failure, the harness can retrieve semantically similar past failures. That
is genuine semantic retrieval, not key-value storage, which is why a vector
database is the right tool rather than a plain table.

## Metrics

`latency_p95` uses nearest-rank on the per-call latencies. `tokens_per_sec`
is total completion tokens over total call time. `cost_per_request` is the
summed per-call cost estimate divided by request count. In the stub these
are deterministic simulated values; in the live path they come from real
Vertex usage and are mirrored into Langfuse.

## ADK vs LangGraph (open decision)

The orchestration is currently a plain sequence in `Orchestrator`. The next
step is to express it on a framework. Google's ADK is preferred because it
signals direct relevance to Google Cloud's agent stack, but its maturity
should be checked before committing; LangGraph is the fallback. Either way
the agent interfaces stay the same, so the framework is an orchestration
detail, not a rewrite.
