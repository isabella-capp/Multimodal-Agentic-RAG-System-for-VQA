import argparse
import json
import os
import random
import re
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, SRC_ROOT)

from tqdm import tqdm

import paths
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker

MODELS = ["BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"]
TOP_NS = [5, 10, 20, 30]


def _norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())


def _gold(item):
    return [g.strip() for alt in (item["answer"] or "").split("|")
            for g in alt.split("&&") if g.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Does the reranker surface the answer paragraph? No generation involved.")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--top-k", type=int, default=20, help="candidate articles pooled per example")
    p.add_argument("--dump", default="outputs/retrieval/retrieval_topk50.jsonl")
    args = p.parse_args()

    kb = KnowledgeBase(paths.KB_PATH)
    dump = {r["unique_id"]: r for r in (json.loads(l) for l in open(args.dump))}
    data = {r["unique_id"]: r for r in json.load(open(paths.JSON_PATH))}

    def nu(u):
        return (u or "").rstrip("/").lower().replace("http://", "https://")

    random.seed(0)
    cases = []
    for uid, d in dump.items():
        item = data.get(uid)
        if not item:
            continue
        cands = d["candidates"][: args.top_k]
        if nu(item["wikipedia_url"]) not in {nu(c["wiki_url"]) for c in cands}:
            continue
        pool = [(c.get("title", ""), para)
                for c in cands
                for para in kb.get_paragraphs_by_url(wiki_url=c["wiki_url"])]
        gold = _gold(item)
        bearing = {i for i, (_, para) in enumerate(pool)
                   if any(_norm(g) in _norm(para) for g in gold)}
        if not bearing:
            continue
        cases.append((item["question"], pool, bearing))
        if len(cases) >= args.limit:
            break

    print(f"usable cases: {len(cases)}  (right article pooled AND a paragraph states the answer)")
    print(f"pool size: median {sorted(len(p) for _, p, _ in cases)[len(cases)//2]} paragraphs\n")

    for name in MODELS:
        try:
            rr = CrossEncoderReranker(name, device=paths.RETRIEVER_DEVICE)
        except Exception as e:
            print(f"{name}: unavailable ({str(e)[:80]})")
            continue
        hits = {n: 0 for n in TOP_NS}
        for question, pool, bearing in tqdm(cases, desc=name.split("/")[-1], leave=False):
            texts = [para for _, para in pool]
            ranked = rr.rerank(question, texts, top_n=max(TOP_NS))
            order = [texts.index(t) for t in ranked]
            for n in TOP_NS:
                if bearing & set(order[:n]):
                    hits[n] += 1
        print(f"  {name}")
        for n in TOP_NS:
            print(f"    answer paragraph in top-{n:<3}: {hits[n]/len(cases):6.1%}")
        del rr

if __name__ == "__main__":
    main()
