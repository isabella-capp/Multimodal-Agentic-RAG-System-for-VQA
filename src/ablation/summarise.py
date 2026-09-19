import argparse
import glob
import json
import os


def row(results_path: str) -> dict:
    name = os.path.basename(results_path)[len("results_"):-len(".json")]
    out = {"name": name, "bem": json.load(open(results_path))["accuracy_overall"]}
    metrics = results_path.replace("results_", "predictions_").replace(".json", ".metrics.json")
    if os.path.exists(metrics):
        m = json.load(open(metrics))
        out["tools"] = m.get("avg_tool_calls")
        out["seconds"] = m.get("avg_seconds_per_example")
    pool = os.path.join(os.path.dirname(results_path), f"pool_{name}.json")
    if os.path.exists(pool):
        out["coverage"] = json.load(open(pool))["percent"]["union"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.directory, "results_*.json"))):
        if f.endswith(".scores.jsonl"):
            continue
        try:
            rows.append(row(f))
        except Exception as e:
            print(f"  {os.path.basename(f)}: {e}")
    if not rows:
        return
    rows.sort(key=lambda r: -r["bem"])

    head = f"  {'config':22} {'BEM':>7}"
    for key, label in (("coverage", "cov"), ("tools", "tools"), ("seconds", "s/ex")):
        if any(r.get(key) is not None for r in rows):
            head += f" {label:>7}"
    print(head)
    for r in rows:
        line = f"  {r['name']:22} {r['bem']:7.4f}"
        for key in ("coverage", "tools", "seconds"):
            if any(x.get(key) is not None for x in rows):
                v = r.get(key)
                line += f" {v:7.2f}" if v is not None else f" {'--':>7}"
        print(line)


if __name__ == "__main__":
    main()
