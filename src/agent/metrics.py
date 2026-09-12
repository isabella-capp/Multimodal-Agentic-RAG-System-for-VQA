from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from agent.run import AgentRun


def summarise(runs: list[AgentRun], wall_seconds: float) -> dict:
    """Cost, latency and — the point of it — how the agent retrieved."""
    if not runs:
        return {"examples": 0, "wall_seconds": round(wall_seconds, 1)}

    n = len(runs)
    calls = [len(r.steps) for r in runs]
    elapsed = [r.elapsed_seconds for r in runs]
    errors = [r.error for r in runs if r.error]
    first = Counter(r.tools[0] if r.tools else "<none>" for r in runs)
    sequences = Counter(" → ".join(r.tools) or "<none>" for r in runs)

    return {
        "examples": n,
        "tool_called_pct": round(100 * sum(1 for r in runs if r.tool_called) / n, 1),
        "num_tool_calls_distribution": dict(sorted(Counter(calls).items())),
        "avg_tool_calls": round(statistics.mean(calls), 2),
        "avg_paragraphs_read": round(statistics.mean(r.paragraphs_read for r in runs), 2),
        "tool_usage": _tool_usage(runs, n),
        "first_tool_pct": {t: round(100 * c / n, 1) for t, c in first.most_common()},
        "top_sequences": dict(sequences.most_common(8)),
        "avg_seconds_per_example": round(statistics.mean(elapsed), 2),
        "median_seconds_per_example": round(statistics.median(elapsed), 2),
        "forced_examples_pct": round(100 * sum(1 for r in runs if r.forced) / n, 1),
        "avg_forced_calls": round(statistics.mean(r.forced for r in runs), 2),
        "gate_below_examples_pct": round(100 * sum(1 for r in runs if r.gate_below) / n, 1),
        "errors": len(errors),
        "error_types": dict(Counter(errors)),
        "wall_seconds": round(wall_seconds, 1),
        "throughput_per_min": round(60 * n / wall_seconds, 1) if wall_seconds else None,
    }


def _tool_usage(runs: list[AgentRun], n: int) -> dict:
    calls: Counter = Counter()
    misses: Counter = Counter()
    users: dict[str, set] = defaultdict(set)
    articles: dict[str, list] = defaultdict(list)

    for i, run in enumerate(runs):
        for step in run.steps:
            calls[step.tool] += 1
            misses[step.tool] += step.miss
            users[step.tool].add(i)
            articles[step.tool].append(step.num_articles)

    return {
        tool: {
            "calls": c,
            "examples_using_pct": round(100 * len(users[tool]) / n, 1),
            "miss_pct": round(100 * misses[tool] / c, 1),
            "avg_articles_returned": round(statistics.mean(articles[tool]), 2),
        }
        for tool, c in calls.most_common()
    }
