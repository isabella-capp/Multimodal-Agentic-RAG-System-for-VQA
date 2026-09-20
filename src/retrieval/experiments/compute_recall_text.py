import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tqdm import tqdm

import paths
from retrieval.knowledge_base import KnowledgeBase, load_df_cache
from vlm.dataset import load_dataset

KS = (1, 5, 10, 20, 50)


def norm(u):
    return (u or "").rstrip("/").lower().replace("http://", "https://")


def main():
    p = argparse.ArgumentParser(description="Recall of the full-text channel.")
    p.add_argument("--image-recall", default="outputs/retrieval/retrieval_topk100.jsonl")
    p.add_argument("--output", default="outputs/retrieval/recall_text.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--image-k", type=int, default=20, help="image cut-off for the union")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    load_df_cache(paths.TERM_DF_PATH)
    kb = KnowledgeBase(paths.KB_PATH)

    image = {}
    for line in open(args.image_recall):
        r = json.loads(line)
        image[r["unique_id"]] = [norm(c["wiki_url"]) for c in r["candidates"]]

    dataset = load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)[: args.limit]
    hits = {k: 0 for k in KS}
    img_hit = union = text_only = image_only = neither = tot = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for it in tqdm(dataset, desc="text recall"):
            uid, gt = it["unique_id"], norm(it["wikipedia_url"])
            if uid not in image:
                continue
            tot += 1
            found = [norm(a["wiki_url"])
                     for a in kb.search_articles_by_text(it["question"], limit=args.top_k)]
            rank = found.index(gt) + 1 if gt in found else None
            for k in KS:
                hits[k] += bool(rank and rank <= k)

            in_img = gt in image[uid][: args.image_k]
            in_txt = bool(rank and rank <= args.image_k)
            img_hit += in_img
            union += in_img or in_txt
            text_only += in_txt and not in_img
            image_only += in_img and not in_txt
            neither += not (in_img or in_txt)
            out.write(json.dumps({"unique_id": uid, "text_rank": rank,
                                  "image_hit": in_img, "titles": found[:5]}) + "\n")

    n = tot or 1
    print(f"text channel (question -> BM25 over 14.1M paragraphs), n={tot}")
    for k in KS:
        print(f"  recall@{k:<3}: {100 * hits[k] / n:5.1f}%")
    print(f"\nagainst the image index at k={args.image_k}:")
    print(f"  image                 : {100 * img_hit / n:5.1f}%")
    print(f"  UNION                 : {100 * union / n:5.1f}%")
    print(f"    only the image      : {100 * image_only / n:5.1f}%")
    print(f"    only the text       : {100 * text_only / n:5.1f}%")
    print(f"    neither             : {100 * neither / n:5.1f}%")
    print(f"Per-example detail: {args.output}")


if __name__ == "__main__":
    main()
