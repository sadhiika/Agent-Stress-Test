# Agent Stress-Test Harness

A tool that audits any deployed AI agent and emits a production-readiness
report. Point it at an agent endpoint and it runs a multi-agent pipeline
that probes the target with edge cases, adversarial prompts, prompt
injections and instruction-following checks, grades each response for
accuracy and safety, re-examines its own low-confidence judgments, and
reports pass rates, safety failures, latency p95, throughput and
cost-per-request.

## Architecture

```
                 +------------------+
                 |   Orchestrator   |   (owns run lifecycle, delegates)
                 +--------+---------+
                          |
   +----------+-----------+-----------+------------+
   |          |           |           |            |
Generator  Executor     Judge       Critic      Report
 (probes)  (calls    (scores acc/  (self-      compiler
  ReAct)   target)   safety/instr) reflection
                                    on low-conf
                                    verdicts)
```

No agent calls another directly; everything flows through the
orchestrator. That is what makes the multi-agent structure real.

## Providers: offline first, live by config

Every external dependency sits behind an adapter with a `stub` and a real
implementation, switched by environment variable. With the defaults the
whole pipeline runs offline against a built-in toy target, so you can see
it work in seconds without any cloud setup. Flip the flags to use real
Gemini, a real target endpoint, and Langfuse tracing.

| Setting | `stub` (default) | live |
|---|---|---|
| `GENERATOR_PROVIDER` | templated probes | Gemini (ReAct) |
| `JUDGE_PROVIDER` | heuristic scoring | Gemini-as-judge |
| `TARGET_MODE` | built-in toy bot | HTTP endpoint (`TARGET_URL`) |
| `ENABLE_TRACING` | off | Langfuse |

## Run the demo (under 60 seconds, no cloud needed)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python main.py --target-description "a customer support bot for a retail bank"
```

This writes `samples/sample_report.json` and `samples/sample_report.md`.
A committed sample report is in `samples/`.

## Go live (real Gemini on Vertex AI)

1. `cp .env.example .env` and fill in `GCP_PROJECT`.
2. Set `GENERATOR_PROVIDER=vertex` and `JUDGE_PROVIDER=vertex`.
3. Authenticate: `gcloud auth application-default login`.
4. Point at a real target: `TARGET_MODE=http` and `TARGET_URL=...`.
5. Optional tracing: `ENABLE_TRACING=1` plus Langfuse keys.

## Tests

```bash
pytest -q
```

## Roadmap

- OAuth-secured MCP connection layer for registering target agents as tools.
- Vector store (Vertex AI Vector Search / pgvector) for test-case reuse and
  retrieval of similar past failures.
- Cloud Run deployment as a callable service.
