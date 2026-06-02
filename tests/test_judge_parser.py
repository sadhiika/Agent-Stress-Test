import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.agents.judge import parse_judge_json
from harness.models import TestCategory


def test_parses_fenced_json():
    raw = '```json\n{"accuracy":0.8,"safety":1,"instruction_following":0.5,"confidence":0.9,"rationale":"ok"}\n```'
    v = parse_judge_json(raw, "id1", TestCategory.ADVERSARIAL)
    assert v.accuracy == 0.8
    assert v.safety == 1.0
    assert v.rationale == "ok"


def test_clamps_out_of_range():
    raw = '{"accuracy":2,"safety":-1,"instruction_following":0.5,"confidence":0.9,"rationale":""}'
    v = parse_judge_json(raw, "id1", TestCategory.ADVERSARIAL)
    assert v.accuracy == 1.0
    assert v.safety == 0.0
