"""Shared Google Gen AI (Vertex AI) client and a thin generate() helper.

Why this module exists: every agent that calls Gemini goes through one
authenticated client and one generate() function. That gives a single place
to control the model name, force JSON output, and do token/cost accounting.
Swapping the model, or moving off Vertex entirely, is a change here, not in
four different agent files. This is the adapter boundary.

Uses the google-genai SDK (the vertexai.generative_models module is
deprecated and removed June 2026).
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.config import Settings

# Cost estimate for the report. The token COUNTS below are real (read from
# the API response). The per-token PRICE is an estimate; update these two
# constants with the current published gemini-2.5-flash rates if you want
# the cost line to be exact. Input/output are USD per 1,000 tokens.
_PRICE_PER_1K_INPUT = 0.000075
_PRICE_PER_1K_OUTPUT = 0.0003

# One process-wide client, built lazily on first use.
_client = None


@dataclass
class LlmResult:
    text: str
    prompt_tokens: int
    output_tokens: int  # candidates + thinking tokens (both billed as output)
    cost_usd: float


def get_client(settings: Settings):
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )
    return _client


def generate(settings: Settings, prompt: str, json_mode: bool = False) -> LlmResult:
    """Call Gemini once and return text plus real token/cost metrics."""
    from google.genai import types

    client = get_client(settings)
    config = None
    if json_mode:
        # Forces the model to return parseable JSON, which makes the judge's
        # output reliable instead of hoping it doesn't wrap things in prose.
        config = types.GenerateContentConfig(response_mime_type="application/json")

    resp = client.models.generate_content(
        model=settings.gemini_model, contents=prompt, config=config
    )

    um = resp.usage_metadata
    prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
    candidates = getattr(um, "candidates_token_count", 0) or 0
    thoughts = getattr(um, "thoughts_token_count", 0) or 0
    output_tokens = candidates + thoughts
    cost = (prompt_tokens / 1000) * _PRICE_PER_1K_INPUT + (
        output_tokens / 1000
    ) * _PRICE_PER_1K_OUTPUT

    return LlmResult(
        text=resp.text or "",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )