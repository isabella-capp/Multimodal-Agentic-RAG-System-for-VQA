import argparse
import glob
import json
import os
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate ablation results.")
    p.add_argument(
        "--results-dir",
        default="outputs/ablation",
        help="Directory containing result JSON files.",
    )
    p.add_argument(
        "--output",
        default="outputs/ablation/ablation_summary_BEM.json",
        help="Path for the aggregated summary JSON.",
    )
    p.add_argument(
        "--pattern",
        default="results_BEM_cross_topK*_rerankN*.json",
        help="Glob pattern for result files. Use 'results_cross_*' for proxy scores.",
    )
    return p.parse_args()


def extract_config(filename):
    """Extract top_k and rerank_top_n from filename."""
    match = re.search(r"topK(\d+)_rerankN(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    args = parse_args()

    pattern = os.path.join(args.results_dir, args.pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        pattern_proxy = os.path.join(args.results_dir, "results_cross_topK*_rerankN*.json")
        files = sorted(glob.glob(pattern_proxy))
        if files:
            print("BEM results not found, using proxy scores.")
        else:
            print(f"No result files found matching {pattern}")
            return

    print(f"Found {len(files)} result files.\n")

    results = []
    all_qtypes = set()

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        top_k, rerank_n = extract_config(os.path.basename(filepath))
        if top_k is None:
            print(f"  [skip] Cannot parse config from {filepath}")
            continue

        entry = {
            "top_k": top_k,
            "rerank_top_n": rerank_n,
            "accuracy_overall": data.get("accuracy_overall", 0.0),
            "accuracy_by_type": data.get("accuracy_by_type", {}),
            "num_examples": data.get("num_examples", 0),
            "file": os.path.basename(filepath),
        }
        results.append(entry)
        all_qtypes.update(entry["accuracy_by_type"].keys())

    results.sort(key=lambda x: x["accuracy_overall"], reverse=True)

    qtypes_sorted = sorted(all_qtypes)
    header_qtypes = "".join(f"{qt[:12]:>14s}" for qt in qtypes_sorted)
    header = f"{'Rank':>4s}  {'top_k':>5s}  {'rerank_n':>8s}  {'Overall':>8s}{header_qtypes}"
    print(header)
    print("─" * len(header))

    for rank, entry in enumerate(results, 1):
        qtypes_str = "".join(
            f"{entry['accuracy_by_type'].get(qt, 0.0):>14.4f}" for qt in qtypes_sorted
        )
        print(
            f"{rank:>4d}  {entry['top_k']:>5d}  {entry['rerank_top_n']:>8d}"
            f"  {entry['accuracy_overall']:>8.4f}{qtypes_str}"
        )

    best = results[0]
    print(f"\n★ Best configuration:")
    print(f"  top_k={best['top_k']}, rerank_top_n={best['rerank_top_n']}")
    print(f"  Overall accuracy: {best['accuracy_overall']:.4f}")
    for qt in qtypes_sorted:
        acc = best["accuracy_by_type"].get(qt, 0.0)
        print(f"    {qt:20s}: {acc:.4f}")

    pivot = defaultdict(dict)
    for entry in results:
        pivot[entry["top_k"]][entry["rerank_top_n"]] = entry["accuracy_overall"]

    top_k_values = sorted(pivot.keys())
    rerank_n_values = sorted({e["rerank_top_n"] for e in results})

    print(f"\n\nPivot table (Overall Accuracy):")
    print(f"{'top_k \\ rerank_n':>18s}", end="")
    for rn in rerank_n_values:
        print(f"  {rn:>8d}", end="")
    print()
    print("─" * (18 + 10 * len(rerank_n_values)))
    for tk in top_k_values:
        print(f"{tk:>18d}", end="")
        for rn in rerank_n_values:
            val = pivot[tk].get(rn)
            if val is not None:
                print(f"  {val:>8.4f}", end="")
            else:
                print(f"  {'—':>8s}", end="")
        print()

    summary = {
        "best_config": {"top_k": best["top_k"], "rerank_top_n": best["rerank_top_n"]},
        "best_accuracy": best["accuracy_overall"],
        "pivot_table": {str(tk): {str(rn): pivot[tk].get(rn) for rn in rerank_n_values} for tk in top_k_values},
        "ranking": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()
