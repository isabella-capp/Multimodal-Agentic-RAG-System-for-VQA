import argparse
import itertools
import json
import os
import sys
import time

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from llm import VLMClient
from vlm.dataset import load_dataset
from prompts import NO_RAG_PROMPT
from runner import run_batch
from vlm.run_inference import build_rag_prompt, build_record

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
DATA_DIR = os.path.abspath(os.path.join(SRC_ROOT, "..", "data"))


def parse_args():
    p = argparse.ArgumentParser(description="Ablation study: top-k × rerank-top-n (cross-encoder)")

    p.add_argument("--val-json", default=f"{DATA_DIR}/encyclopedic_val_split.json",
                    help="Path to the validation split JSON.")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output-dir", default="outputs/ablation",
                    help="Directory for per-config prediction & result files.")

    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")

    p.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    p.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    p.add_argument("--kb-path", default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.db")
    p.add_argument("--retriever-device", default="cuda")

    p.add_argument("--cross-encoder-model", default="BAAI/bge-reranker-base")

    p.add_argument("--top-k-values", type=int, nargs="+", default=[5, 10, 20, 50, 80],
                    help="List of top-k values to test.")
    p.add_argument("--rerank-top-n-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35],
                    help="List of rerank-top-n values to test.")

    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--debug-samples", type=int, default=1,
                    help="Print detailed trace for first N examples of each config.")
    p.add_argument("--limit", type=int, default=None,
                    help="Limit the number of validation examples (useful for testing).")

    return p.parse_args()


def _truncate(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def run_single_config(
    dataset, model, retriever, kb, reranker, top_k, rerank_top_n,
    output_path, debug_samples=1, concurrency=8,
):
    """Run inference for a single (top_k, rerank_top_n) configuration."""
    retriever.top_k = top_k

    records = []
    shown = []

    def predict(item):
        image_path = item["image_path"]
        retrieved_context = None
        top_paragraphs = None
        prompt = NO_RAG_PROMPT.format(question=item["question"])

        try:
            user_image = Image.open(image_path).convert("RGB")
            results = retriever.retrieve(user_image, item["question"])

            if results:
                pooled = []
                for r in results:
                    pooled.extend(kb.get_paragraphs_by_url(r["wiki_url"]))

                if pooled:
                    top_paragraphs = reranker.rerank(
                        item["question"], pooled, top_n=rerank_top_n
                    )
                    prompt = build_rag_prompt(item["question"], top_paragraphs)
                    retrieved_context = {
                        "wiki_url": results[0]["wiki_url"],
                        "title": results[0].get("title", ""),
                        "score": results[0].get("score"),
                        "candidates": [
                            {
                                "wiki_url": r["wiki_url"],
                                "title": r.get("title", ""),
                                "score": r.get("score"),
                            }
                            for r in results
                        ],
                        "num_paragraphs_total": len(pooled),
                        "num_paragraphs_used": len(top_paragraphs) if top_paragraphs else 0,
                    }
        except Exception as e:
            tqdm.write(f"  retrieval failed for {item['unique_id']}: {e}")

        prediction = model.generate_response(image_path, prompt)

        if len(shown) < debug_samples:
            tqdm.write(f"\n  [DEBUG] {item['unique_id']} ({item['question_type']})")
            tqdm.write(f"    Q : {item['question']}")
            tqdm.write(f"    GT: {item['answer']}")
            if top_paragraphs:
                tqdm.write(f"    Paragraphs used: {len(top_paragraphs)}")
                for i, p in enumerate(top_paragraphs, 1):
                    tqdm.write(f"      [{i}] {_truncate(p)}")
            tqdm.write(f"    Pred: {prediction}")
            shown.append(item["unique_id"])

        record = build_record(item, prediction, retrieved_context)
        records.append(record)
        return record

    run_batch(dataset, predict, output_path, concurrency=concurrency,
              setting="B-ablation", model=model.llm.model_name,
              top_k=top_k, rerank_top_n=rerank_top_n)
    return records


def compute_scores_simple(records):
    """Compute accuracy using simple exact-match (case-insensitive substring)."""
    scores_by_type = defaultdict(list)

    for rec in records:
        pred = (rec.get("prediction") or "").strip().lower()
        gt = rec.get("answer", "").strip().lower()

        score = 1.0 if (gt in pred or pred in gt) and pred else 0.0
        scores_by_type[rec["question_type"]].append(score)

    all_scores = [s for scores in scores_by_type.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    accuracy_by_type = {
        q: sum(s) / len(s) for q, s in scores_by_type.items()
    }

    return {
        "num_examples": len(all_scores),
        "accuracy_overall": overall,
        "accuracy_by_type": accuracy_by_type,
    }


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading validation dataset …")
    dataset = load_dataset(args.val_json, args.base_folder)
    if args.limit is not None:
        dataset = dataset[:args.limit]
    print(f"Validation set: {len(dataset)} examples")

    qtypes = defaultdict(int)
    for item in dataset:
        qtypes[item["question_type"]] += 1
    for qt in sorted(qtypes):
        print(f"  {qt:20s}: {qtypes[qt]}")

    print("Loading Qwen VLM …")
    model = VLMClient(model_name=args.model_name, base_url=args.base_url)

    print("Loading EVA-CLIP retriever …")
    max_top_k = max(args.top_k_values)
    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=max_top_k,
        device=args.retriever_device,
    )
    retriever._ensure_index()
    retriever._ensure_model()

    print("Loading Knowledge Base …")
    kb = KnowledgeBase(args.kb_path)

    print("Loading Cross-Encoder reranker …")
    reranker = CrossEncoderReranker(
        args.cross_encoder_model, device=args.retriever_device
    )

    grid = list(itertools.product(args.top_k_values, args.rerank_top_n_values))
    print(f"Ablation grid: {len(grid)} configurations")
    for top_k, rerank_n in grid:
        print(f"  top_k={top_k:2d}  rerank_top_n={rerank_n}")

    all_results = {}

    for i, (top_k, rerank_n) in enumerate(grid, 1):
        config_name = f"cross_topK{top_k}_rerankN{rerank_n}"
        pred_path = os.path.join(args.output_dir, f"predictions_{config_name}.jsonl")
        result_path = os.path.join(args.output_dir, f"results_{config_name}.json")

        print(f"\n[{i}/{len(grid)}] {config_name}")

        if os.path.exists(result_path):
            print("  already scored, skipping")
            with open(result_path, encoding="utf-8") as f:
                all_results[config_name] = json.load(f)
            continue

        if os.path.exists(pred_path):
            print(f"  discarding partial {os.path.basename(pred_path)}")
            os.remove(pred_path)

        print(f"  top_k={top_k}, rerank_top_n={rerank_n}")
        t0 = time.time()

        records = run_single_config(
            dataset=dataset,
            model=model,
            retriever=retriever,
            kb=kb,
            reranker=reranker,
            top_k=top_k,
            rerank_top_n=rerank_n,
            output_path=pred_path,
            concurrency=args.concurrency,
            debug_samples=args.debug_samples,
        )

        elapsed = time.time() - t0
        print(f"  Inference done in {elapsed:.1f}s")

        scores = compute_scores_simple(records)
        scores["config"] = {"top_k": top_k, "rerank_top_n": rerank_n}
        scores["elapsed_seconds"] = round(elapsed, 1)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        all_results[config_name] = scores

        print(f"  Proxy accuracy: {scores['accuracy_overall']:.4f}")
        for qt in sorted(scores["accuracy_by_type"]):
            print(f"    {qt:20s}: {scores['accuracy_by_type'][qt]:.4f}")


    ranking = sorted(all_results.items(), key=lambda x: x[1]["accuracy_overall"], reverse=True)

    print(f"\n{'Config':<30s} {'Accuracy':>10s} {'Time (s)':>10s}")
    for name, res in ranking:
        cfg = res["config"]
        print(
            f"  top_k={cfg['top_k']:<2d} rerank_n={cfg['rerank_top_n']:<2d}"
            f"        {res['accuracy_overall']:>8.4f}"
            f"  {res['elapsed_seconds']:>8.1f}"
        )

    best_name, best_res = ranking[0]
    best_cfg = best_res["config"]
    print(f"\nBest config: top_k={best_cfg['top_k']}, rerank_top_n={best_cfg['rerank_top_n']}")
    print(f"  Proxy accuracy: {best_res['accuracy_overall']:.4f}")

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    summary = {
        "grid": {"top_k_values": args.top_k_values, "rerank_top_n_values": args.rerank_top_n_values},
        "best_config": best_cfg,
        "best_accuracy": best_res["accuracy_overall"],
        "results": {name: res for name, res in ranking},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")

    print("Accuracies above are exact-match proxies; score the predictions "
          "files with evqa_eval/score_evqa.py for BEM.")

if __name__ == "__main__":
    main()
