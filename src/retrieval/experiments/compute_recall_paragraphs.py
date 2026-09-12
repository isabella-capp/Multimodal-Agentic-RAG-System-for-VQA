from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from PIL import Image
from tqdm import tqdm

from retrieval.bm25 import BM25Ranker
from retrieval.knowledge_base import KnowledgeBase
from retrieval.fusion import rrf_score
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import Retriever
from vlm.dataset import load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
REPORT_KS = [1, 3, 5, 10, 20, 50]

_LEGACY_MODES   = ["bm25", "reranker", "bm25+reranker"]
_BM25_MODES     = ["bm25_top5", "bm25_top10", "bm25_top20", "bm25_top50"]
_BGE_MODES      = ["bge_top5", "bge_top10", "bge_top20"]
_BM25_BGE_MODES = [
    "bm25_20_bge_5",  "bm25_20_bge_10",  "bm25_20_bge_20",
    "bm25_50_bge_5",  "bm25_50_bge_10",  "bm25_50_bge_20",
]
_RRF_MODES      = ["rrf_top5", "rrf_top10", "rrf_top20"]

ALL_MODES = _LEGACY_MODES + _BM25_MODES + _BGE_MODES + _BM25_BGE_MODES + _RRF_MODES


def parse_args():
    p = argparse.ArgumentParser(
        description="Recall@k at the paragraph level: BM25 / BGE / BM25→BGE / RRF"
    )
    p.add_argument("--json-path", default=f"{BASE_FOLDER}/encyclopedic_test_subset.json")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output", default="outputs/retrieval/recall_paragraphs.jsonl")
    p.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    p.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    p.add_argument("--kb-path", default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.db")
    p.add_argument("--cross-encoder-model", default="BAAI/bge-reranker-base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of articles retrieved by EVA-CLIP/FAISS.")
    p.add_argument("--bm25-top-m", type=int, default=50)
    p.add_argument("--rerank-top-n", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)

    p.add_argument("--modes", nargs="+", choices=ALL_MODES,
                   default=["bge_top20", "bm25_50_bge_20", "bm25_top50"],
                   help="Retrieval modes to evaluate.")

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report-only", action="store_true",
                   help="Skip inference; print recall table from an existing output file.")
    return p.parse_args()


def _parse_mode(mode: str, default_bm25_top_m: int, default_rerank_top_n: int) -> dict:
    if mode == "bm25":
        return dict(strategy="bm25", top_k=default_bm25_top_m, bm25_top_m=default_bm25_top_m)
    if mode == "reranker":
        return dict(strategy="bge", top_k=default_rerank_top_n, bm25_top_m=None)
    if mode == "bm25+reranker":
        return dict(strategy="bm25_bge", top_k=default_rerank_top_n, bm25_top_m=default_bm25_top_m)

    m = re.fullmatch(r"bm25_top(\d+)", mode)
    if m:
        k = int(m.group(1))
        return dict(strategy="bm25", top_k=k, bm25_top_m=k)

    m = re.fullmatch(r"bge_top(\d+)", mode)
    if m:
        k = int(m.group(1))
        return dict(strategy="bge", top_k=k, bm25_top_m=None)

    m = re.fullmatch(r"bm25_(\d+)_bge_(\d+)", mode)
    if m:
        return dict(strategy="bm25_bge", bm25_top_m=int(m.group(1)), top_k=int(m.group(2)))

    m = re.fullmatch(r"rrf_top(\d+)", mode)
    if m:
        return dict(strategy="rrf", top_k=int(m.group(1)), bm25_top_m=None)

    raise ValueError(f"Unknown mode: {mode!r}")


def _gold_ranks(
    query: str,
    answer: str,
    paragraphs: list[str],
    modes: list[str],
    bm25_ranker: BM25Ranker | None,
    reranker: CrossEncoderReranker | None,
    default_bm25_top_m: int,
    default_rerank_top_n: int,
    rrf_k: int,
) -> dict[str, int | None]:

    results: dict[str, int | None] = {}
    if not paragraphs:
        return {m: None for m in modes}

    def is_gold(p_text: str) -> bool:
        p_lower = p_text.lower()
        alternatives = answer.split("|")
        for alt in alternatives:
            parts = [part.strip().lower() for part in alt.split("&&") if part.strip()]
            if parts and all(part in p_lower for part in parts):
                return True
        return False

    def first_gold(ranked: list[str]) -> int | None:
        for i, p in enumerate(ranked, 1):
            if is_gold(p):
                return i
        return None

    parsed = {m: _parse_mode(m, default_bm25_top_m, default_rerank_top_n) for m in modes}

    need_bm25 = any(p["strategy"] in ("bm25", "bm25_bge", "rrf") for p in parsed.values())
    need_bge_full = any(p["strategy"] in ("bge", "rrf") for p in parsed.values())

    bm25_full: list[str] = []
    if need_bm25 and bm25_ranker is not None:
        bm25_full = bm25_ranker.rank(
            query, paragraphs, top_m=len(paragraphs), force_sort=True
        )

    bge_full: list[str] = []
    if need_bge_full and reranker is not None:
        bge_full = reranker.rerank(
            query, paragraphs, top_n=len(paragraphs), force_sort=True
        )

    bge_of_bm25: dict[int, list[str]] = {}
    for p in parsed.values():
        if p["strategy"] == "bm25_bge" and p["bm25_top_m"] not in bge_of_bm25:
            m_val: int = p["bm25_top_m"]
            bm25_slice = bm25_full[:m_val]
            if bm25_slice and reranker is not None:
                bge_of_bm25[m_val] = reranker.rerank(
                    query, bm25_slice, top_n=len(bm25_slice), force_sort=True
                )
            else:
                bge_of_bm25[m_val] = bm25_slice

    for mode, spec in parsed.items():
        strategy, top_k, bm25_top_m = spec["strategy"], spec["top_k"], spec["bm25_top_m"]

        if strategy == "bm25":
            results[mode] = first_gold(bm25_full[:top_k])

        elif strategy == "bge":
            results[mode] = first_gold(bge_full[:top_k])

        elif strategy == "bm25_bge":
            bm25_slice = bm25_full[:bm25_top_m]
            results[f"{mode}_bm25_int"] = first_gold(bm25_slice)
            bge_ranked = bge_of_bm25.get(bm25_top_m, [])
            results[mode] = first_gold(bge_ranked[:top_k])

        elif strategy == "rrf":
            if bm25_full and bge_full:
                fused = rrf_score([bm25_full, bge_full], rrf_k=rrf_k)
                results[mode] = first_gold(fused[:top_k])
            else:
                results[mode] = None

    return results


def report(path: str, modes: list[str]) -> None:
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(records)
    print(f"\n{'='*72}")
    print(f"Paragraph Recall@k  ({n} examples | {path})")
    print(f"{'='*72}")

    col_w = 8
    header = f"{'mode':<24}" + "".join(f"@{k:<{col_w-1}}" for k in REPORT_KS)
    print(header)
    print("-" * len(header))

    for mode in modes:
        ranks = [r["ranks"].get(mode) for r in records]
        row = f"{mode:<24}"
        for k in REPORT_KS:
            hits = sum(1 for r in ranks if r is not None and r <= k)
            row += f"{100*hits/n:6.1f}% "
        print(row)

        int_key = f"{mode}_bm25_int"
        if any(r["ranks"].get(int_key) is not None for r in records):
            int_ranks = [r["ranks"].get(int_key) for r in records]
            row2 = f"  └ BM25 pre-filter    "
            for k in REPORT_KS:
                hits = sum(1 for r in int_ranks if r is not None and r <= k)
                row2 += f"{100*hits/n:6.1f}% "
            print(row2)

    no_gold = sum(1 for r in records if all(r["ranks"].get(m) is None for m in modes))
    print(f"\n  No gold paragraph in pool: {no_gold}/{n} ({100*no_gold/n:.1f}%)")
    print(f"{'='*72}\n")


def _load_done(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["unique_id"] for l in f if l.strip()}


def main():
    args = parse_args()

    if args.report_only:
        report(args.output, args.modes)
        return

    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    done = _load_done(args.output)
    if done:
        print(f"Resuming: skipping {len(done)} already-processed examples.")

    print("Loading EVA-CLIP retriever ...")
    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=args.top_k,
        device=args.device,
    )
    retriever._ensure_index()
    retriever._ensure_model()

    print("Loading Knowledge Base ...")
    kb = KnowledgeBase(args.kb_path)

    parsed_modes = [_parse_mode(m, args.bm25_top_m, args.rerank_top_n) for m in args.modes]
    need_bm25    = any(p["strategy"] in ("bm25", "bm25_bge", "rrf") for p in parsed_modes)
    need_reranker = any(p["strategy"] in ("bge",  "bm25_bge", "rrf") for p in parsed_modes)

    bm25_ranker = BM25Ranker() if need_bm25 else None

    reranker: CrossEncoderReranker | None = None
    if need_reranker:
        print(f"Loading cross-encoder reranker ({args.cross_encoder_model}) ...")
        reranker = CrossEncoderReranker(args.cross_encoder_model, device=args.device)

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Paragraph recall"):
            uid = item["unique_id"]
            if uid in done:
                continue

            question = item["question"]
            answer   = item.get("answer", "")
            paragraphs: list[str] = []
            ranks: dict[str, int | None] = {}

            if not os.path.exists(item["image_path"]):
                tqdm.write(f"missing image: {item['image_path']}")
            else:
                try:
                    image   = Image.open(item["image_path"]).convert("RGB")
                    img_emb = retriever.encode_image(image)
                    articles = retriever.search_index(img_emb, top_k=args.top_k)
                    for art in articles:
                        paragraphs.extend(kb.get_paragraphs_by_url(art["wiki_url"]))

                    if paragraphs and answer:
                        ranks = _gold_ranks(
                            query=question,
                            answer=answer,
                            paragraphs=paragraphs,
                            modes=args.modes,
                            bm25_ranker=bm25_ranker,
                            reranker=reranker,
                            default_bm25_top_m=args.bm25_top_m,
                            default_rerank_top_n=args.rerank_top_n,
                            rrf_k=args.rrf_k,
                        )
                except Exception as e:
                    tqdm.write(f"failed for {uid}: {e}")

            out.write(json.dumps({
                "unique_id":           uid,
                "question_type":       item.get("question_type", ""),
                "answer":              answer,
                "num_pool_paragraphs": len(paragraphs),
                "ranks":               ranks,
            }, ensure_ascii=False) + "\n")
            out.flush()

    report(args.output, args.modes)

if __name__ == "__main__":
    main()