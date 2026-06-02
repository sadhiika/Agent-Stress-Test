import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.models import ExecutionResult, TestCase, TestCategory, Verdict
from harness.reporting.compiler import compile_report, _p95


def _result(tid, latency, c_tok=10):
    tc = TestCase(id=tid, category=TestCategory.EDGE_CASE, prompt="p", expectation="e")
    return ExecutionResult(test_case=tc, response_text="r", latency_ms=latency,
                           prompt_tokens=5, completion_tokens=c_tok, cost_usd=0.001)


def _verdict(tid, safety=1.0):
    return Verdict(test_case_id=tid, category=TestCategory.EDGE_CASE, accuracy=0.9,
                   safety=safety, instruction_following=0.9, confidence=0.9, rationale="r")


def test_p95_basic():
    assert _p95([100, 200, 300, 400, 500]) == 500


def test_pass_rate_and_safety_failures():
    results = [_result("a", 100), _result("b", 200)]
    verdicts = [_verdict("a", safety=1.0), _verdict("b", safety=0.0)]
    report = compile_report(results, verdicts, "demo")
    assert report["summary"]["total_tests"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["safety_failures"] == 1
    assert report["performance"]["latency_p95_ms"] > 0
