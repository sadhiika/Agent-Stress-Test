"""CLI entry point for the Agent Stress-Test Harness.

Usage:
    python main.py --target-description "a customer support bot for a bank"

Reads providers from environment (stub by default), runs the pipeline,
writes samples/sample_report.json and samples/sample_report.md, and prints
a short summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `src` importable without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from harness.config import load_settings  # noqa: E402
from harness.orchestrator import Orchestrator  # noqa: E402
from harness.reporting.compiler import to_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a deployed AI agent.")
    parser.add_argument(
        "--target-description",
        default="a generic demo assistant",
        help="Short description of the agent being audited.",
    )
    parser.add_argument("--out-dir", default="samples")
    args = parser.parse_args()

    settings = load_settings()
    print(f"Providers -> generator: {settings.generator_provider}, "
          f"judge: {settings.judge_provider}, target: {settings.target_mode}")

    orch = Orchestrator(settings)
    report, _results, _verdicts = orch.run(args.target_description)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sample_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = to_markdown(report)
    with open(os.path.join(args.out_dir, "sample_report.md"), "w") as f:
        f.write(md)

    s = report["summary"]
    print("\n=== Run complete ===")
    print(f"Tests: {s['total_tests']}  Passed: {s['passed']}  "
          f"Pass rate: {s['overall_pass_rate'] * 100:.1f}%")
    print(f"Safety failures: {s['safety_failures']}  "
          f"Critic overturns: {s['verdicts_overturned_by_critic']}")
    print(f"Report written to {args.out_dir}/sample_report.md")


if __name__ == "__main__":
    main()
