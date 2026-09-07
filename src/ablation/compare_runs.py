"""Confidence intervals for a run, and paired tests between runs.

A summary number on 1000 examples carries a sampling error of about +/-3 points
at 95%, which is ten times the +/-0.3 we see between two identical runs. Most of
our decisions turned on 1-3 point gaps, so they need the paired test: the two
systems answer the SAME examples, and the interval on their difference is far
tighter than the interval on either score.

Needs the per-example scores written by ``evqa_eval/score_evqa.py`` next to each
results file (``*.scores.jsonl``).
"""

import argparse
import json
import random
import statistics


def load(path):
    return {r["unique_id"]: r["score"] for r in
            (json.loads(l) for l in open(path, encoding="utf-8"))}


def boot(values, n=5000, seed=0):
    """Percentile bootstrap interval for the mean."""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=k)) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores", nargs="+", help="*.scores.jsonl files, first is the reference")
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()

    runs = [(f.split("/")[-2] + "/" + f.split("/")[-1].replace(".scores.jsonl", ""), load(f))
            for f in args.scores]

    print(f"{'run':46s} {'BEM':>7} {'IC 95%':>18}")
    for name, s in runs:
        v = list(s.values())
        lo, hi = boot(v, args.bootstrap)
        print(f"  {name[-44:]:44s} {statistics.mean(v):7.4f} [{lo:.4f}, {hi:.4f}]")

    if len(runs) < 2:
        return
    ref_name, ref = runs[0]
    print(f"\ndifferenza rispetto a {ref_name[-44:]}, sugli stessi esempi:")
    print(f"{'run':46s} {'delta':>8} {'IC 95%':>18} {'p':>8}")
    for name, s in runs[1:]:
        ids = [u for u in ref if u in s]
        diff = [s[u] - ref[u] for u in ids]
        lo, hi = boot(diff, args.bootstrap)
        # two-sided sign test on the examples where the two disagree
        wins = sum(1 for d in diff if d > 0)
        ties = sum(1 for d in diff if d == 0)
        n = len(diff) - ties
        pval = _sign_test(wins, n) if n else 1.0
        print(f"  {name[-44:]:44s} {statistics.mean(diff):+8.4f} [{lo:+.4f}, {hi:+.4f}] {pval:8.3f}"
              f"   ({wins} meglio / {n - wins} peggio / {ties} pari)")


def _sign_test(wins, n):
    from math import comb
    tail = sum(comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2 * tail / 2 ** n)


if __name__ == "__main__":
    main()
