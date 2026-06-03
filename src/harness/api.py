"""FastAPI wrapper exposing the harness as an HTTP service.

This is what runs on Cloud Run. It turns the orchestrator into a callable
endpoint so the harness is a deployed service, not just a local script.

Endpoints:
  GET  /health  -> liveness check (used by Cloud Run)
  GET  /        -> short usage blurb
  POST /audit   -> run an audit, return the report as JSON
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from harness.config import load_settings
from harness.orchestrator import Orchestrator

app = FastAPI(title="Agent Stress-Test Harness")


class AuditRequest(BaseModel):
    target_description: str = "a generic demo assistant"
    # Optional override so a deployed request can stay small and fast.
    cases_per_category: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "Agent Stress-Test Harness",
        "usage": "POST /audit with JSON {\"target_description\": \"...\"}",
    }


@app.post("/audit")
def audit(req: AuditRequest) -> dict:
    settings = load_settings()
    if req.cases_per_category is not None:
        settings.cases_per_category = req.cases_per_category
    orchestrator = Orchestrator(settings)
    report, _results, _verdicts = orchestrator.run(req.target_description)
    return report