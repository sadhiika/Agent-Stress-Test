"""Orchestrator: the hierarchical-delegation layer.

It owns the run lifecycle and delegates, in sequence, to the generator,
executor, judge and (for low-confidence verdicts) the critic, then hands
the graded results to the report compiler. No agent calls another agent
directly; everything flows through here. That is what makes the
multi-agent structure real rather than cosmetic.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from harness.agents.critic import build_critic
from harness.agents.executor import build_target
from harness.agents.generator import build_generator
from harness.agents.judge import build_judge
from harness.config import Settings
from harness.models import ExecutionResult, Verdict
from harness.observability.tracing import build_tracer
from harness.reporting.compiler import compile_report


class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator = build_generator(settings)
        self.target = build_target(settings)
        self.judge = build_judge(settings)
        self.critic = build_critic(settings)
        self.tracer = build_tracer(settings)

    def run(self, target_description: str) -> Tuple[Dict, List[ExecutionResult], List[Verdict]]:
        # 1. Generate probes
        with self.tracer.trace("generate"):
            cases = self.generator.generate(
                target_description, self.settings.cases_per_category
            )

        results: List[ExecutionResult] = []
        verdicts: List[Verdict] = []

        for case in cases:
            # 2. Execute against target
            with self.tracer.trace("execute", test_case_id=case.id):
                result = self.target.send(case)
            results.append(result)

            # 3. Judge
            with self.tracer.trace("judge", test_case_id=case.id):
                verdict = self.judge.judge(result)

            # 4. Self-reflection: re-examine low-confidence verdicts AND any
            #    failure, since a flagged failure is the most consequential
            #    verdict to get right.
            if verdict.confidence < self.settings.critic_confidence_threshold or not verdict.passed:
                with self.tracer.trace("critic", test_case_id=case.id):
                    verdict = self.critic.review(result, verdict)

            verdicts.append(verdict)

        self.tracer.flush()
        report = compile_report(results, verdicts, target_description)
        return report, results, verdicts
