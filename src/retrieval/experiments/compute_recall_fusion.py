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
ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
REPORT_KS = [1, 5, 10, 20, 50, 100]


def parse_args():
    p = argparse.ArgumentParser(description="Recall@k for fused image+text retrieval")
    p.add_argument("--json-path", default=f"{BASE_FOLDER}/encyclopedic_test_subset.json")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output", default="outputs/retrieval/recall_fusion_topk100.jsonl")
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


def fuse(img_emb, txt_emb, alpha):
    v = alpha * img_emb + (1 - alpha) * txt_emb
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def norm_url(u):
    if not u:
        return u
    u = re.sub(r"^https?://", "", u.strip().lower())
    return u.replace("en.m.wikipedia.org", "en.wikipedia.org").rstrip("/")


def report(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("wikipedia_url")]
    print(f"\nRecall@k over {len(recs)} examples with a ground-truth URL:")
    header = "alpha  " + "  ".join(f"@{k:<4}" for k in REPORT_KS)
    print(header)
    for a in ALPHAS:
        key = f"{a:.1f}"
        row = []
        for k in REPORT_KS:
            hit = sum(
                1 for r in recs
                if norm_url(r["wikipedia_url"]) in {norm_url(u) for u in r["by_alpha"][key][:k]}
            )
            row.append(f"{100 * hit / len(recs):5.1f}")
        tag = "img" if a == 1.0 else ("txt" if a == 0.0 else "")
        print(f"{key:5} {tag:3} " + "  ".join(row))


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
        for item in tqdm(dataset, desc="Fusion recall"):
            if item["unique_id"] in done:
                continue
            by_alpha = {}
            if os.path.exists(item["image_path"]):
                try:
                    image = Image.open(item["image_path"]).convert("RGB")
                    img_emb = retriever.encode_image(image)
                    txt_emb = retriever.encode_text(item["question"])
                    for a in ALPHAS:
                        results = retriever.search_index(fuse(img_emb, txt_emb, a), args.top_k)
                        by_alpha[f"{a:.1f}"] = [r["wiki_url"] for r in results]
                except Exception as e:
                    tqdm.write(f"failed for {item['unique_id']}: {e}")
            else:
                tqdm.write(f"missing image: {item['image_path']}")

            out.write(json.dumps({
                "unique_id": item["unique_id"],
                "wikipedia_url": item.get("wikipedia_url"),
                "by_alpha": by_alpha,
            }, ensure_ascii=False) + "\n")
            out.flush()

    report(args.output)


if __name__ == "__main__":
    main()
