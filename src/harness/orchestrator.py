"""Orchestrator: the multi-agent pipeline, wired with LangGraph.

The pipeline is expressed as a LangGraph StateGraph. Each agent is a node,
and the graph owns the control flow between them. This is the
hierarchical-delegation layer: the graph delegates to the generator,
executor, judge and (conditionally) the critic, then compiles the report.

A conditional edge after the judge routes to the critic only when at least
one verdict needs review, and skips straight to the report otherwise, which
avoids unnecessary model calls. The agents themselves are unchanged;
LangGraph replaces the sequencing that previously lived in a plain loop.
"""
from __future__ import annotations

from typing import Dict, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from harness.agents.critic import build_critic
from harness.agents.executor import build_target
from harness.agents.generator import build_generator
from harness.agents.judge import build_judge
from harness.config import Settings
from harness.models import ExecutionResult, Verdict
from harness.observability.tracing import build_tracer
from harness.reporting.compiler import compile_report


class HarnessState(TypedDict):
    """State passed between graph nodes. Each node fills in its own field."""
    target_description: str
    cases_per_category: int
    cases: List
    results: List
    verdicts: List
    report: Dict


class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator = build_generator(settings)
        self.target = build_target(settings)
        self.judge = build_judge(settings)
        self.critic = build_critic(settings)
        self.tracer = build_tracer(settings)
        self.graph = self._build_graph()

    # --- graph nodes ---

    def _generate_node(self, state: HarnessState) -> Dict:
        with self.tracer.trace("generate"):
            cases = self.generator.generate(
                state["target_description"], state["cases_per_category"]
            )
        return {"cases": cases}

    def _execute_node(self, state: HarnessState) -> Dict:
        results: List[ExecutionResult] = []
        for case in state["cases"]:
            with self.tracer.trace("execute", test_case_id=case.id):
                results.append(self.target.send(case))
        return {"results": results}

    def _judge_node(self, state: HarnessState) -> Dict:
        verdicts: List[Verdict] = []
        for result in state["results"]:
            with self.tracer.trace("judge", test_case_id=result.test_case.id):
                verdicts.append(self.judge.judge(result))
        return {"verdicts": verdicts}

    def _critic_node(self, state: HarnessState) -> Dict:
        results_by_id = {r.test_case.id: r for r in state["results"]}
        reviewed: List[Verdict] = []
        for verdict in state["verdicts"]:
            if self._needs_review(verdict):
                result = results_by_id[verdict.test_case_id]
                with self.tracer.trace("critic", test_case_id=verdict.test_case_id):
                    verdict = self.critic.review(result, verdict)
            reviewed.append(verdict)
        return {"verdicts": reviewed}

    def _report_node(self, state: HarnessState) -> Dict:
        self.tracer.flush()
        report = compile_report(
            state["results"], state["verdicts"], state["target_description"]
        )
        return {"report": report}

    # --- routing ---

    def _needs_review(self, verdict: Verdict) -> bool:
        """A verdict is re-examined if the judge was unsure OR it is a failure
        (the most consequential verdicts to get right)."""
        return (
            verdict.confidence < self.settings.critic_confidence_threshold
            or not verdict.passed
        )

    def _route_after_judge(self, state: HarnessState) -> str:
        if any(self._needs_review(v) for v in state["verdicts"]):
            return "critic"
        return "report"

    def _build_graph(self):
        g = StateGraph(HarnessState)
        g.add_node("generate", self._generate_node)
        g.add_node("execute", self._execute_node)
        g.add_node("judge", self._judge_node)
        g.add_node("critic", self._critic_node)
        g.add_node("report", self._report_node)

        g.add_edge(START, "generate")
        g.add_edge("generate", "execute")
        g.add_edge("execute", "judge")
        # Conditional delegation: only run the critic if something needs review.
        g.add_conditional_edges(
            "judge", self._route_after_judge, {"critic": "critic", "report": "report"}
        )
        g.add_edge("critic", "report")
        g.add_edge("report", END)
        return g.compile()

    def run(self, target_description: str):
        """Run the pipeline. Same return signature as before so the FastAPI
        service and CLI are unaffected."""
        initial: HarnessState = {
            "target_description": target_description,
            "cases_per_category": self.settings.cases_per_category,
            "cases": [],
            "results": [],
            "verdicts": [],
            "report": {},
        }
        final = self.graph.invoke(initial)
        return final["report"], final["results"], final["verdicts"]
