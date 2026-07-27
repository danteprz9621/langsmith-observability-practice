"""
Prompt-regression gating in CI.
"""
import math
from statistics import mean
import pytest
from experiments import experiment_evaluate

# keys MUST match the "key" your evaluators emit
THRESHOLD = {"faithfulness": 0.8, "has_retrieval": 1.0}

@pytest.fixture(scope="session")
def scores():
    collected = {k: [] for k in THRESHOLD}
    for r in experiment_evaluate():
        for ev in r["evaluation_results"]["results"]:
            if ev.key in collected and ev.score is not None and not math.isnan(ev.score):
                collected[ev.key].append(ev.score)
    return collected

@pytest.mark.parametrize("metric", list(THRESHOLD))
def test_metric_meets_threshold(scores, metric):
    values = scores[metric]
    assert values, f"no valid scores collected for {metric} — key mismatch or empty run?"
    avg = mean(values)
    assert avg >= THRESHOLD[metric], f"{metric}: {avg:.3f} < {THRESHOLD[metric]}"


