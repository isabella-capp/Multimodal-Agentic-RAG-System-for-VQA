import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
from tqdm import tqdm

import paths
from retrieval.knowledge_base import KnowledgeBase, load_df_cache
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import Retriever
from vlm.dataset import load_dataset


def norm(u):
    return (u or "").rstrip("/").lower().replace("http://", "https://")


def main():
    p = argparse.ArgumentParser(description="Cross-encoder score of the best pooled passage, per example.")
    p.add_argument("--output", default="outputs/retrieval/gate.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--naming-limit", type=int, default=3)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    load_df_cache(paths.TERM_DF_PATH)
    kb = KnowledgeBase(paths.KB_PATH)
    retriever = Retriever(paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH,
                          top_k=args.top_k, device=paths.RETRIEVER_DEVICE,
                          ef_search=paths.EF_SEARCH)
    retriever._ensure_index(); retriever._ensure_model()
    reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL,
                                    device=paths.RETRIEVER_DEVICE)

    with open(args.output, "w", encoding="utf-8") as out:
        for it in tqdm(load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)[: args.limit],
                       desc="gate"):
            if not os.path.exists(it["image_path"]):
                continue
            gt = norm(it["wikipedia_url"])
            image = Image.open(it["image_path"]).convert("RGB")
            articles = retriever.retrieve(image, it["question"])
            pooled = [p for a in articles for p in kb.get_paragraphs_by_url(a["wiki_url"])]
            score = reranker.score_of_best(it["question"], pooled) if pooled else None

            text_hits = [norm(a["wiki_url"])
                         for a in kb.search_articles_by_text(it["question"], limit=5)]
            out.write(json.dumps({
                "unique_id": it["unique_id"],
                "top_score": score,
                "image_hit": gt in [norm(a["wiki_url"]) for a in articles],
                "text_would_add": gt in text_hits,
                "pool_size": len(pooled),
            }) + "\n")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
