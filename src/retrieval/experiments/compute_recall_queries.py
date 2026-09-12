import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
from PIL import Image
from tqdm import tqdm

from retrieval.retriever import Retriever
from vlm.dataset import load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
QUERY_FIELDS = ["question", "question_original", "wikipedia_title"]
ALPHA_FUSED = 0.7
REPORT_KS = [1, 5, 10, 20, 50, 100]


def parse_args():
    p = argparse.ArgumentParser(description="Recall@k by query source")
    p.add_argument("--json-path", default=f"{BASE_FOLDER}/encyclopedic_test_subset.json")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output", default="outputs/retrieval/recall_queries.jsonl")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    p.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}


def norm_url(u):
    if not u:
        return u
    u = re.sub(r"^https?://", "", u.strip().lower())
    return u.replace("en.m.wikipedia.org", "en.wikipedia.org").rstrip("/")


def normalize(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def gt_rank(results, gt):
    g = norm_url(gt)
    for i, r in enumerate(results, 1):
        if norm_url(r["wiki_url"]) == g:
            return i
    return None


def report(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("wikipedia_url")]
    configs = ["image"] + [f"{f}.{m}" for f in QUERY_FIELDS for m in ("text", "fused")]
    print(f"\nRecall@k over {len(recs)} examples:")
    print(f"{'config':22} " + "  ".join(f"@{k:<4}" for k in REPORT_KS))
    for c in configs:
        ranks = [r["ranks"].get(c) for r in recs]
        row = [f"{100 * sum(1 for x in ranks if x is not None and x <= k) / len(ranks):5.1f}"
               for k in REPORT_KS]
        print(f"{c:22} " + "  ".join(row))


def main():
    args = parse_args()

    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = load_done_ids(args.output)
    if done:
        print(f"Skipping {len(done)} already-processed examples")

    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=args.top_k,
        device=args.device,
    )

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Query-source recall"):
            if item["unique_id"] in done:
                continue
            gt = item.get("wikipedia_url")
            ranks = {}
            if os.path.exists(item["image_path"]):
                try:
                    image = Image.open(item["image_path"]).convert("RGB")
                    img_emb = retriever.encode_image(image)
                    ranks["image"] = gt_rank(retriever.search_index(img_emb, args.top_k), gt)
                    for field in QUERY_FIELDS:
                        text = item.get(field) or ""
                        if not text:
                            continue
                        t_emb = retriever.encode_text(text)
                        ranks[f"{field}.text"] = gt_rank(
                            retriever.search_index(t_emb, args.top_k), gt)
                        fused = normalize(ALPHA_FUSED * img_emb + (1 - ALPHA_FUSED) * t_emb)
                        ranks[f"{field}.fused"] = gt_rank(
                            retriever.search_index(fused, args.top_k), gt)
                except Exception as e:
                    tqdm.write(f"failed for {item['unique_id']}: {e}")
            else:
                tqdm.write(f"missing image: {item['image_path']}")

            out.write(json.dumps({
                "unique_id": item["unique_id"],
                "wikipedia_url": gt,
                "ranks": ranks,
            }, ensure_ascii=False) + "\n")
            out.flush()

    report(args.output)


if __name__ == "__main__":
    main()
