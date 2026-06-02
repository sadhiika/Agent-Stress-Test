# Agent Production-Readiness Report

**Target:** a customer support bot for a retail bank

## Summary
- Tests run: 15
- Passed: 13 (86.7%)
- Safety failures: 0
- Verdicts overturned by critic: 1

## Performance
- Latency p95: 11284.07 ms
- Latency mean: 5969.09 ms
- Throughput: 136.09 tokens/sec
- Cost per request: $0.000249
- Total cost: $0.003736

## Pass rate by category
- edge_case: 100.0% (3/3)
- adversarial: 100.0% (3/3)
- prompt_injection: 100.0% (3/3)
- multi_turn_trap: 100.0% (3/3)
- instruction_following: 33.3% (1/3)