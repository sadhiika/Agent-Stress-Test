"""Report compiler: turns graded results into the production-readiness report.

Produces a structured dict (serialised to JSON) and a human-readable
Markdown summary. All metrics are computed from real run data; nothing is
fabricated.
"""
from __future__ import annotations

import statistics
from typing import Dict, List

from harness.models import ExecutionResult, Verdict


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # nearest-rank p95
    idx = max(0, int(round(0.95 * len(ordered))) - 1)
    return ordered[idx]


def compile_report(
    results: List[ExecutionResult],
    verdicts: List[Verdict],
    target_description: str,
) -> Dict:
    by_id = {v.test_case_id: v for v in verdicts}

    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.passed)
    overall_pass_rate = round(passed / total, 3) if total else 0.0

    # Pass rate by category
    cats: Dict[str, Dict[str, int]] = {}
    for v in verdicts:
        c = v.category.value
        cats.setdefault(c, {"passed": 0, "total": 0})
        cats[c]["total"] += 1
        if v.passed:
            cats[c]["passed"] += 1
    by_category = {
        c: {"pass_rate": round(d["passed"] / d["total"], 3), **d} for c, d in cats.items()
    }

    # Safety failures with examples
    safety_failures = []
    for r in results:
        v = by_id.get(r.test_case.id)
        if v and v.safety < 0.5:
            safety_failures.append(
                {
                    "test_case_id": r.test_case.id,
                    "category": r.test_case.category.value,
                    "prompt": r.test_case.prompt,
                    "response": r.response_text,
                    "rationale": v.rationale,
                }
            )

    # Latency / cost / throughput
    latencies = [r.latency_ms for r in results]
    total_completion_tokens = sum(r.completion_tokens for r in results)
    total_time_s = sum(r.latency_ms for r in results) / 1000.0
    tokens_per_sec = round(total_completion_tokens / total_time_s, 2) if total_time_s else 0.0
    total_cost = sum(r.cost_usd for r in results)

    overturned = sum(1 for v in verdicts if v.overturned_by_critic)

    return {
        "target_description": target_description,
        "summary": {
            "total_tests": total,
            "passed": passed,
            "overall_pass_rate": overall_pass_rate,
            "safety_failures": len(safety_failures),
            "verdicts_overturned_by_critic": overturned,
        },
        "by_category": by_category,
        "performance": {
            "latency_p95_ms": round(_p95(latencies), 2),
            "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "tokens_per_sec": tokens_per_sec,
            "total_cost_usd": round(total_cost, 6),
            "cost_per_request_usd": round(total_cost / len(results), 6) if results else 0.0,
        },
        "safety_failure_examples": safety_failures,
    }


def to_markdown(report: Dict) -> str:
    s = report["summary"]
    p = report["performance"]
    lines = [
        "# Agent Production-Readiness Report",
        "",
        f"**Target:** {report['target_description']}",
        "",
        "## Summary",
        f"- Tests run: {s['total_tests']}",
        f"- Passed: {s['passed']} ({s['overall_pass_rate'] * 100:.1f}%)",
        f"- Safety failures: {s['safety_failures']}",
        f"- Verdicts overturned by critic: {s['verdicts_overturned_by_critic']}",
        "",
        "## Performance",
        f"- Latency p95: {p['latency_p95_ms']} ms",
        f"- Latency mean: {p['latency_mean_ms']} ms",
        f"- Throughput: {p['tokens_per_sec']} tokens/sec",
        f"- Cost per request: ${p['cost_per_request_usd']}",
        f"- Total cost: ${p['total_cost_usd']}",
        "",
        "## Pass rate by category",
    ]
    for cat, d in report["by_category"].items():
        lines.append(f"- {cat}: {d['pass_rate'] * 100:.1f}% ({d['passed']}/{d['total']})")
    if report["safety_failure_examples"]:
        lines += ["", "## Safety failures (examples)"]
        for f in report["safety_failure_examples"]:
            lines += [
                f"- **[{f['category']}]** prompt: `{f['prompt'][:80]}`",
                f"  - response: `{f['response'][:120]}`",
                f"  - {f['rationale']}",
            ]
    return "\n".join(lines)
