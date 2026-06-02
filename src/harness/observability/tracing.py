"""Tracing wrapper. No-op offline; Langfuse when ENABLE_TRACING=1.

Every agent call is wrapped in `trace()` so that flipping tracing on adds
observability without touching agent code. The Langfuse path records
latency, token counts and cost so the same numbers appear both in the
report and in the Langfuse dashboard.
"""
from __future__ import annotations

from contextlib import contextmanager

from harness.config import Settings


class _NoOpTracer:
    @contextmanager
    def trace(self, name: str, **kwargs):
        yield None

    def flush(self):
        pass


class _LangfuseTracer:
    def __init__(self, settings: Settings):
        from langfuse import Langfuse  # imported only when enabled

        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    @contextmanager
    def trace(self, name: str, **kwargs):
        span = self._client.span(name=name, metadata=kwargs)
        try:
            yield span
        finally:
            span.end()

    def flush(self):
        self._client.flush()


def build_tracer(settings: Settings):
    if settings.enable_tracing:
        return _LangfuseTracer(settings)
    return _NoOpTracer()
