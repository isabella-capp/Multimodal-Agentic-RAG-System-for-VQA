import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import paths
from vlm.dataset import load_dataset


def norm(url):
    return (url or "").rstrip("/").lower().replace("http://", "https://")


def main():
    p = argparse.ArgumentParser(description="Channel composition of the candidate pool.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    gt = {it["unique_id"]: norm(it["wikipedia_url"])
          for it in load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)}

    stats = {"examples": 0, "image": 0, "name": 0, "text": 0, "union": 0,
             "name_only": 0, "text_only": 0, "image_only": 0, "neither": 0,
             "named": 0, "name_resolved_to_something": 0}
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            target = gt.get(rec["unique_id"])
            ctx = rec.get("retrieved_context") or {}
            cands = ctx.get("candidates") or []
            if target is None:
                continue
            stats["examples"] += 1
            image_hit = any(norm(c["wiki_url"]) == target
                            for c in cands if c.get("source", "image") == "image")
            named = [c for c in cands if c.get("source") == "name"]
            name_hit = any(norm(c["wiki_url"]) == target for c in named)
            text_hit = any(norm(c["wiki_url"]) == target
                           for c in cands if c.get("source") == "text")
            stats["named"] += bool(ctx.get("predicted_name"))
            stats["name_resolved_to_something"] += bool(named)
            stats["image"] += image_hit
            stats["name"] += name_hit
            stats["text"] += text_hit
            stats["union"] += image_hit or name_hit or text_hit
            stats["image_only"] += image_hit and not (name_hit or text_hit)
            stats["name_only"] += name_hit and not (image_hit or text_hit)
            stats["text_only"] += text_hit and not (image_hit or name_hit)
            stats["neither"] += not (image_hit or name_hit or text_hit)

    n = stats["examples"] or 1
    pct = {k: round(100 * v / n, 1) for k, v in stats.items() if k != "examples"}
    print("=" * 62)
    print(f"{args.predictions}   ({stats['examples']} examples)")
    print(f"  right article in the pool via IMAGE   : {pct['image']:5.1f}%")
    print(f"  right article in the pool via NAME    : {pct['name']:5.1f}%")
    print(f"  right article in the pool via TEXT    : {pct['text']:5.1f}%")
    print(f"  UNION                                 : {pct['union']:5.1f}%   <- what the model sees")
    print(f"    only the image found it             : {pct['image_only']:5.1f}%")
    print(f"    only the name found it              : {pct['name_only']:5.1f}%")
    print(f"    only the text found it              : {pct['text_only']:5.1f}%")
    print(f"    neither                             : {pct['neither']:5.1f}%")
    print(f"  a name was produced                   : {pct['named']:5.1f}%")
    print(f"  that name resolved to some article    : {pct['name_resolved_to_something']:5.1f}%")
    print("=" * 62)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"counts": stats, "percent": pct}, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
