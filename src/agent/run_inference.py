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
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.dataset import build_record

MAX_TOKENS = 512
TRACE_SAMPLES = 1   # a full trace is hundreds of lines; --debug-samples 0 turns it off


def parse_args():
    p = argparse.ArgumentParser(description="Agentic RAG evaluation on Encyclopedic-VQA")
    p.add_argument("--output", default="outputs/predictions_agentic.jsonl")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rerank-top-n", type=int, default=5)
    p.add_argument("--bm25-top-m", type=int, default=50,
                   help="BM25 candidate pool size before BGE reranking.")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--debug-samples", type=int, default=3)
    p.add_argument("--unified", action="store_true",
                   help="One way into the KB instead of three: the agent writes a "
                        "query, the tool decides how to look it up.")
    p.add_argument("--text-limit", type=int, default=5,
                   help="Articles kept from the full-text side of each search.")
    p.add_argument("--max-names", type=int, default=4,
                   help="Titles the agent may open per search call.")
    p.add_argument("--lookup-limit", type=int, default=3,
                   help="Articles kept per title looked up.")
    p.add_argument("--preview", type=int, default=0,
                   help="Passages search_by_image shows from the image pool (0: titles only).")
    p.add_argument("--no-read-article", dest="with_read", action="store_false",
                   help="Drop read_article: called on 16%% of examples and never decisive.")
    p.add_argument("--direct-prompt", action="store_true",
                   help="With --final-pass, long-form answers without preamble.")
    p.add_argument("--legacy-prompt", action="store_true",
                   help="With --final-pass, answer with the prompt that has no "
                        "answer-format block, matching baseline B's long-form arm.")
    p.add_argument("--final-pass", action="store_true",
                   help="Answer from the pool the agent assembled with one call, "
                        "the way baseline B does, instead of from the loop.")
    p.add_argument("--text-gate", type=float, default=None,
                   help="Run a second search only where the best pooled paragraph "
                        "scores below this. Evidence-driven iteration: the model's "
                        "own sense of whether it has enough has failed every test.")
    p.add_argument("--retrieval-mode", default="bm25+reranker",
                   choices=["bm25+reranker", "reranker", "rrf"],
                   help="Paragraph retrieval pipeline: 'bm25+reranker' (default) "
                        "pre-filters with BM25 then reranks; "
                        "'reranker' sends all paragraphs directly to the cross-encoder; "
                        "'rrf' fuses BM25 and BGE rankings via Reciprocal Rank Fusion.")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF smoothing constant (default 60). Only used with --retrieval-mode rrf.")
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
    reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE)

    return AgenticRAG(llm, retriever, kb, reranker, top_n=args.rerank_top_n,
                      top_k=args.top_k, bm25_top_m=args.bm25_top_m,
                      max_iterations=args.max_iterations,
                      
                      retrieval_mode=args.retrieval_mode,
                      rrf_k=args.rrf_k,

text_gate=args.text_gate,
                      final_pass=args.final_pass, legacy_prompt=args.legacy_prompt,
                      unified=args.unified,
                      direct_prompt=args.direct_prompt,

                      with_read=args.with_read, preview=args.preview,
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

        # Protezione lock per aggiornamenti e log concorrenti
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
        legacy_prompt=args.legacy_prompt,
        unified=args.unified,
        direct_prompt=args.direct_prompt,

        with_read=args.with_read,
        preview=args.preview,
        text_limit=args.text_limit, max_names=args.max_names,
        lookup_limit=args.lookup_limit,
        bm25_top_m=args.bm25_top_m,
        max_iterations=args.max_iterations,

        reranker=paths.CROSS_ENCODER_MODEL,
        retrieval_mode=args.retrieval_mode,
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