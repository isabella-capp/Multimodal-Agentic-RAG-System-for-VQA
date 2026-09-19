import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from tqdm import tqdm

import paths
from agent.metrics import summarise
from agent.rag import AgenticRAG
from llm import chat_model
from retrieval.fusion import STRATEGIES, Ranking
from retrieval.knowledge_base import KnowledgeBase, load_df_cache
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.dataset import build_record

MAX_TOKENS = 512
TRACE_SAMPLES = 1   # a full trace is hundreds of lines; --debug-samples 0 turns it off


def ranking_from_args(args) -> Ranking:
    """One object out of the ranking flags, validated by Ranking itself."""
    return Ranking(tools=args.tools_strategy, preview=args.preview_strategy,
                   final=args.final_strategy, top_n=args.rerank_top_n,
                   bm25_top_m=args.bm25_top_m, rrf_k=args.rrf_k)


def parse_args():
    p = argparse.ArgumentParser(description="Agentic RAG evaluation on Encyclopedic-VQA")
    p.add_argument("--output", default="outputs/predictions_agentic.jsonl")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rerank-top-n", type=int, default=5)
    p.add_argument("--bm25-top-m", type=int, default=50,
                   help="BM25 candidate pool size before BGE reranking.")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--debug-samples", type=int, default=3)
    p.add_argument("--text-limit", type=int, default=5,
                   help="Articles kept from the full-text side of each search.")
    p.add_argument("--max-names", type=int, default=4,
                   help="Titles the agent may open per search call.")
    p.add_argument("--lookup-limit", type=int, default=3,
                   help="Articles kept per title looked up.")
    p.add_argument("--preview", type=int, default=0,
                   help="Passages search_by_image shows from the image pool (0: titles only).")
    p.add_argument("--final-pass", action="store_true",
                   help="Answer from the pool the agent assembled with one call, "
                        "the way baseline B does, instead of from the loop.")
    p.add_argument("--text-gate", type=float, default=None,
                   help="Run a second search only where the best pooled paragraph "
                        "scores below this. Evidence-driven iteration: the model's "
                        "own sense of whether it has enough has failed every test.")
    p.add_argument("--tool-set", default="two", choices=["two", "four"],
                   help="Ablation: 'two' is the interface every reported run "
                        "uses; 'four' splits the textual operations back into "
                        "separate tools.")
    p.add_argument("--tools-strategy", default=Ranking.tools, choices=STRATEGIES,
                   help="How the tools rank what the agent reads mid-loop.")
    p.add_argument("--preview-strategy", default=Ranking.preview, choices=STRATEGIES,
                   help="How the passages shown with the image candidates are "
                        "picked. Was hard-coded and unreachable until now, so no "
                        "run has ever varied it.")
    p.add_argument("--final-strategy", default=Ranking.final, choices=STRATEGIES,
                   help="How --final-pass ranks the pool it answers from.")
    p.add_argument("--rrf-k", type=int, default=Ranking.rrf_k,
                   help="RRF smoothing constant. Only used by the rrf strategy.")
    return p.parse_args()


def build_agent(args):
    llm = chat_model(args.model_name, args.base_url, MAX_TOKENS)
    print(f"Agent model: {args.model_name} @ {args.base_url}")

    retriever = Retriever(paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH,
                          top_k=args.top_k, device=paths.RETRIEVER_DEVICE,
                          ef_search=paths.EF_SEARCH)
    retriever._ensure_index()
    retriever._ensure_model()
    kb = KnowledgeBase(paths.KB_PATH)
    n = load_df_cache(paths.TERM_DF_PATH)
    print(f"Term frequencies loaded: {n}" if n else
          "No term-frequency cache: searches will be slow "
          "(run scripts/retrieval/run_prime_df.sh)")
    reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE)

    return AgenticRAG(llm, retriever, kb, reranker,
                      ranking=ranking_from_args(args),
                      top_k=args.top_k,
                      max_iterations=args.max_iterations,
                      text_gate=args.text_gate,
                      final_pass=args.final_pass,
                      preview=args.preview, tool_set=args.tool_set,
                      text_limit=args.text_limit, max_names=args.max_names,
                      lookup_limit=args.lookup_limit)


def format_trace(messages) -> str:
    """The full loop for one example: question -> tool calls -> results -> answer."""

    def text(content):
        if isinstance(content, str):
            return content
        parts = []
        for block in content or []:
            if isinstance(block, dict):
                parts.append(
                    block["text"] if block.get("type") == "text" else "<image>"
                )
            else:
                parts.append(str(block))
        return " ".join(parts)

    lines = ["=" * 78]
    for m in messages:
        if isinstance(m, SystemMessage):
            lines += ["[SYSTEM / REMINDER]", text(m.content).strip(), ""]
        elif isinstance(m, HumanMessage):
            lines += ["[USER]", text(m.content).strip(), ""]
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                lines.append("[ASSISTANT → tool call]")
                lines += [
                    f"  {tc['name']}(args={tc.get('args', {})})"
                    for tc in m.tool_calls
                ]
            else:
                lines += [
                    "[ASSISTANT → final answer]",
                    f"  {text(m.content).strip()}",
                ]
            lines.append("")
        elif isinstance(m, ToolMessage):
            lines += [f"[TOOL RESULT] ({m.name})", text(m.content).strip(), ""]
    return "\n".join(lines + ["=" * 78])


def print_debug_example(item, run):
    steps = " | ".join(f"{s.order}:{s.tool}{s.arguments}" for s in run.steps) or "<no tool>"
    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"[DEBUG] {item['unique_id']}  ({item['question_type']})")
    tqdm.write(f"  Q : {item['question']}")
    tqdm.write(f"  GT: {item['answer']}")
    tqdm.write(f"  steps: {steps}")
    tqdm.write(f"  Prediction: {run.prediction}  ({run.elapsed_seconds}s)")
    tqdm.write("=" * 70)


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args)

    runs, shown, traced = [], [], []
    log_lock = threading.Lock()

    def predict(item):
        run = agent.run(item["image_path"], item["question"])

        with log_lock:
            runs.append(run)

            if len(shown) < args.debug_samples:
                shown.append(item["unique_id"])
                print_debug_example(item, run)

            if (args.debug_samples and len(traced) < TRACE_SAMPLES
                and run.tool_called and run.messages):
                traced.append(item["unique_id"])
                tqdm.write(format_trace(run.messages))

        record = build_record(item, run.prediction)
        record["agent"] = {
            "tool_called": run.tool_called,
            "num_tool_calls": len(run.steps),
            "elapsed_seconds": run.elapsed_seconds,
            "error": run.error,
            "steps": [s.as_record() for s in run.steps],
        }
        return record

    t0 = time.time()
    run_batch(
        load_todo(args.output, args.limit),
        predict,
        args.output,
        args.concurrency,
        setting="C",
        model=args.model_name,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,

        text_gate=args.text_gate,
        final_pass=args.final_pass,

        preview=args.preview,
        tool_set=args.tool_set,
        text_limit=args.text_limit, max_names=args.max_names,
        lookup_limit=args.lookup_limit,
        bm25_top_m=args.bm25_top_m,
        max_iterations=args.max_iterations,
        reranker=paths.CROSS_ENCODER_MODEL,
        tools_strategy=args.tools_strategy,
        preview_strategy=args.preview_strategy,
        final_strategy=args.final_strategy,
        rrf_k=args.rrf_k,
    )

    metrics_path = str(Path(args.output).with_suffix(".metrics.json"))
    metrics = summarise(runs, time.time() - t0)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics: {json.dumps(metrics)}")
    print(f"Predictions: {args.output} | Metrics: {metrics_path}")

if __name__ == "__main__":
    main()