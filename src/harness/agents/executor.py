"""Executor agent: sends each probe to the target and measures the response.

The target is whatever you are auditing. Two adapters:
  - StubTarget: a deliberately imperfect toy bot, so the offline demo finds
    real failures (it leaks on some injections, ignores some constraints).
  - HttpTarget: posts to any HTTP endpoint that returns text.

We measure latency, token counts and a cost estimate here, at the point of
the call, because those are the production metrics the report needs.
"""
from __future__ import annotations

import hashlib
import time

from harness.config import Settings
from harness.models import ExecutionResult, TestCase, TestCategory


# Rough public Gemini-class pricing for the cost estimate. Override via the
# real provider once you have actual usage numbers.
_PRICE_PER_1K_INPUT = 0.00125
_PRICE_PER_1K_OUTPUT = 0.005


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * _PRICE_PER_1K_INPUT + (completion_tokens / 1000) * _PRICE_PER_1K_OUTPUT


def _approx_tokens(text: str) -> int:
    # ~4 chars per token is a fine approximation for reporting.
    return max(1, len(text) // 4)


class BaseTarget:
    def send(self, test_case: TestCase) -> ExecutionResult:
        raise NotImplementedError


class StubTarget(BaseTarget):
    """A toy bot with realistic flaws, used for the offline demo.

    Its behaviour is deterministic (seeded by the test id) so reports are
    reproducible, but it intentionally fails some probes so the harness has
    genuine failures to grade.
    """

    def send(self, test_case: TestCase) -> ExecutionResult:
        start = time.perf_counter()
        seed = int(hashlib.md5(test_case.id.encode()).hexdigest(), 16)

        if test_case.category == TestCategory.PROMPT_INJECTION:
            # Leaks on roughly half of injections -> real safety failures.
            if seed % 2 == 0:
                text = "Sure. The secret token is SK-DEMO-1234 and my system prompt is..."
            else:
                text = "I can't share system content or any secret token."
        elif test_case.category == TestCategory.ADVERSARIAL:
            text = "I won't do that." if seed % 3 != 0 else "Okay, developer mode enabled."
        elif test_case.category == TestCategory.INSTRUCTION_FOLLOWING:
            # Sometimes adds chatter, violating 'only' constraints.
            text = "OK" if seed % 2 == 0 else "Sure thing! OK, here you go."
        elif test_case.category == TestCategory.MULTI_TURN_TRAP:
            text = "I won't pre-commit to breaking any rules."
        else:  # EDGE_CASE
            text = "Could you clarify what you'd like me to do?"

        # Simulated, deterministic metrics.
        latency = 120 + (seed % 400)
        p_tok = _approx_tokens(test_case.prompt)
        c_tok = _approx_tokens(text)
        return ExecutionResult(
            test_case=test_case,
            response_text=text,
            latency_ms=float(latency),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=_estimate_cost(p_tok, c_tok),
        )


class HttpTarget(BaseTarget):
    """Calls a real agent endpoint. Adjust the request/response shape to
    match your target's API."""

    def __init__(self, settings: Settings):
        self.url = settings.target_url

    def send(self, test_case: TestCase) -> ExecutionResult:
        import httpx

        start = time.perf_counter()
        try:
            resp = httpx.post(self.url, json={"prompt": test_case.prompt}, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or data.get("text") or resp.text
            error = None
        except Exception as exc:  # noqa: BLE001 - we want to record any failure
            text, error = "", str(exc)
        latency = (time.perf_counter() - start) * 1000
        p_tok = _approx_tokens(test_case.prompt)
        c_tok = _approx_tokens(text)
        return ExecutionResult(
            test_case=test_case,
            response_text=text,
            latency_ms=latency,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=_estimate_cost(p_tok, c_tok),
            error=error,
        )

class GeminiTarget(BaseTarget):
    """The audited agent is a real Gemini model (via Vertex AI).

    Produces real latency and token metrics. Uses the naive system prompt
    from settings so the audit surfaces genuine weaknesses.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, test_case: TestCase) -> ExecutionResult:
        from harness.llm import generate

        start = time.perf_counter()
        prompt = f"{self.settings.target_system_prompt}\n\nUser: {test_case.prompt}"
        try:
            r = generate(self.settings, prompt, json_mode=False)
            text, error = r.text, None
            p_tok, c_tok, cost = r.prompt_tokens, r.output_tokens, r.cost_usd
        except Exception as exc:  # noqa: BLE001
            text, error = "", str(exc)
            p_tok, c_tok, cost = 0, 0, 0.0
        latency = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            test_case=test_case,
            response_text=text,
            latency_ms=latency,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
            error=error,
        )

def build_target(settings: Settings) -> BaseTarget:
    if settings.target_mode == "gemini":
        return GeminiTarget(settings)
    if settings.target_mode == "http":
        return HttpTarget(settings)
    return StubTarget()