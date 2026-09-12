import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

import paths
from agent.messages import image_to_data_uri
from agent.prompts import MULTI_NAMING_PROMPTS, NAMING_PROMPT
from llm import VLMClient
from prompts import (NO_RAG_PROMPT, NO_RAG_PROMPT_LEGACY, RAG_PROMPT,
                     RAG_PROMPT_DIRECT, RAG_PROMPT_LEGACY, extract_answer)
from retrieval.bm25 import BM25Ranker
from retrieval.knowledge_base import KnowledgeBase, load_df_cache
from retrieval.fusion import rank_paragraphs
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.arg_parser import parse_args
from vlm.dataset import build_record


def build_rag_prompt(question, paragraphs, legacy=False, direct=False):
    template = (RAG_PROMPT_DIRECT if direct else
                RAG_PROMPT_LEGACY if legacy else RAG_PROMPT)
    return template.format(context="\n\n".join(paragraphs), question=question)


def setup_retrieval(top_k, retrieval_strategy, no_rerank):
    """Load retriever, KB, and only the ranking components the strategy needs."""
    retriever = Retriever(
        paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH, top_k=top_k,
        device=paths.RETRIEVER_DEVICE, ef_search=paths.EF_SEARCH
    )
    retriever._ensure_index()
    retriever._ensure_model()
    kb = KnowledgeBase(paths.KB_PATH)

    reranker = bm25 = None
    if not no_rerank:
        need_bm25 = retrieval_strategy in ("bm25", "bm25_bge", "rrf")
        need_bge  = retrieval_strategy in ("bge",  "bm25_bge", "rrf")
        if need_bm25:
            bm25 = BM25Ranker()
        if need_bge:
            from retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(
                paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE
            )
    return retriever, kb, reranker, bm25


def name_entity(model, image_path, guesses=1):
    """What the model thinks the image shows, as bare Wikipedia-style names."""
    prompt = (NAMING_PROMPT if guesses == 1
              else MULTI_NAMING_PROMPTS["diverse"].format(n=guesses))
    resp = model.llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
        ]),
    ])
    text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
    if guesses == 1:
        return [text] if text else []
    names = []
    for line in text.splitlines():
        n = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip()
        if n and n not in names:
            names.append(n)
    return names[:guesses]


def name_articles(kb, names, limit):
    """Articles the predicted names resolve to, deduplicated, best guess first."""
    out, seen = [], set()
    for name in ([names] if isinstance(names, str) else names or []):
        for h in kb.lookup_articles(name, limit=limit):
            if h["wiki_url"] in seen:
                continue
            seen.add(h["wiki_url"])
            out.append({"wiki_url": h["wiki_url"], "title": h["title"], "score": None,
                        "source": "name", "match": h["match"]})
    return out


def text_articles(kb, question, limit):
    """Articles whose text matches the question, as retrieval results."""
    return [{"wiki_url": h["wiki_url"], "title": h["title"], "score": None,
             "source": "text"}
            for h in kb.search_articles_by_text(question, limit=limit)]


def build_context(
    retriever, kb, reranker, bm25, question, image_path, rerank_top_n,
    bm25_top_m, no_rerank, extra_articles=(),
    retrieval_strategy: str = "bm25_bge", rrf_k: int = 60,
    text_gate=None, text_articles_fn=None,
):
    """Retrieve articles for the image, pool and rank their paragraphs."""
    user_image = Image.open(image_path).convert("RGB")
    results = retriever.retrieve(user_image, question)

    seen = {r["wiki_url"] for r in results}
    results = [a for a in extra_articles if a["wiki_url"] not in seen] + results
    if not results:
        return None

    pooled = []
    for r in results:
        pooled.extend(kb.get_paragraphs_by_url(r["wiki_url"]))
    if not pooled:
        return None

    def rank(pool):
        if no_rerank:
            return pool if rerank_top_n <= 0 else pool[:rerank_top_n]
        return rank_paragraphs(
            question, pool, strategy=retrieval_strategy, top_k=rerank_top_n,
            bm25_top_m=bm25_top_m, bm25_ranker=bm25, reranker=reranker, rrf_k=rrf_k)

    top_paragraphs = rank(pooled)

    gate_open = False
    if text_gate is not None and text_articles_fn is not None:
        score = getattr(reranker, "last_top_score", None)
        gate_open = score is None or score < text_gate
        if gate_open:
            extra = [a for a in text_articles_fn()
                     if a["wiki_url"] not in {r["wiki_url"] for r in results}]
            if extra:
                results = results + extra
                pooled = pooled + [p for a in extra
                                   for p in kb.get_paragraphs_by_url(a["wiki_url"])]
                top_paragraphs = rank(pooled)

    retrieved_context = {
        "wiki_url": results[0]["wiki_url"],
        "title": results[0].get("title", ""),
        "score": results[0].get("score"),
        "candidates": [
            {
                "wiki_url": r["wiki_url"],
                "title": r.get("title", ""),
                "score": r.get("score"),
                "source": r.get("source", "image"),
            }
            for r in results
        ],
        # None where no gate is configured, so a closed gate is distinguishable
        "text_gate_open": gate_open if text_gate is not None else None,
        "num_paragraphs_total": len(pooled),
        "num_paragraphs_used": len(top_paragraphs),
    }
    return top_paragraphs, retrieved_context


def _truncate(text, n=200):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def print_debug_example(item, retrieved_context, top_paragraphs, prediction):
    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"[DEBUG] {item['unique_id']}  ({item['question_type']})")
    tqdm.write(f"  Q : {item['question']}")
    tqdm.write(f"  GT: {item['answer']}")
    if retrieved_context is not None:
        if retrieved_context.get("predicted_name") is not None:
            named = [c["title"] for c in retrieved_context["candidates"]
                     if c.get("source") == "name"]
            tqdm.write(f"  Named: {retrieved_context['predicted_name']!r} -> {named}")
        tqdm.write(
            f"  Retrieved: {retrieved_context['title']!r} "
            f"(score={retrieved_context['score']}, "
            f"paragraphs {retrieved_context['num_paragraphs_used']}/"
            f"{retrieved_context['num_paragraphs_total']})"
        )
        tqdm.write(f"             {retrieved_context['wiki_url']}")
        for i, p in enumerate(top_paragraphs or [], 1):
            tqdm.write(f"    [{i}] {_truncate(p)}")
    else:
        tqdm.write("  Retrieved: <none> (baseline prompt, question only)")
    tqdm.write(f"  Prediction: {prediction}")
    tqdm.write("=" * 70)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if (args.use_naming or args.use_text) and not args.use_retrieval:
        raise SystemExit("--use-naming/--use-text widen the retrieved pool; "
                         "they need --use-retrieval")
    if args.use_text:
        n = load_df_cache(paths.TERM_DF_PATH)
        print(f"Term frequencies loaded: {n}" if n else
              "No term-frequency cache: the first questions will be slow "
              "(run scripts/retrieval/run_prime_df.sh)")

    model = VLMClient(args.model_name, args.base_url)
    retriever = kb = reranker = bm25 = None
    if args.use_retrieval:
        retriever, kb, reranker, bm25 = setup_retrieval(
            args.top_k, args.retrieval_strategy, args.no_rerank
        )

    shown = []

    def predict(item):
        base = NO_RAG_PROMPT_LEGACY if args.legacy_prompt else NO_RAG_PROMPT
        prompt = base.format(question=item["question"])
        paragraphs = retrieved = None
        if retriever is not None:
            try:
                name, extra = None, []
                if args.oracle:
                    extra = [{"wiki_url": item["wikipedia_url"],
                              "title": item.get("wikipedia_title", ""),
                              "score": None, "source": "oracle"}]
                if args.use_naming:
                    name = name_entity(model, item["image_path"], args.naming_guesses)
                    extra = extra + name_articles(kb, name, args.naming_limit)
                if args.use_text and args.text_gate is None:
                    extra = extra + text_articles(kb, item["question"], args.text_limit)
                context = build_context(retriever, kb, reranker, bm25,
                                        item["question"], item["image_path"],
                                        args.rerank_top_n, args.bm25_top_m,
                                        args.no_rerank, extra,
                                        retrieval_strategy=args.retrieval_strategy, oracle=args.oracle,
                                        rrf_k=args.rrf_k,
                                        text_gate=args.text_gate if args.use_text else None,
                                        text_articles_fn=lambda: text_articles(
                                            kb, item["question"], args.text_limit))
                if context is not None:
                    paragraphs, retrieved = context
                    if args.use_naming:
                        retrieved["predicted_name"] = name
                    prompt = build_rag_prompt(item["question"], paragraphs,
                                              args.legacy_prompt, args.direct_prompt)
            except Exception as e:
                tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

        prediction = extract_answer(
            model.generate_response(item["image_path"], prompt))
        if len(shown) < args.debug_samples:
            shown.append(item["unique_id"])
            print_debug_example(item, retrieved, paragraphs, prediction)
        return build_record(item, prediction, retrieved)

    run_batch(load_todo(args.output, args.limit), predict, args.output,
              args.concurrency, setting="B" if args.use_retrieval else "A",
              model=args.model_name, top_k=args.top_k, rerank_top_n=args.rerank_top_n,
              bm25_top_m=args.bm25_top_m,
              retrieval_strategy=args.retrieval_strategy,
              use_naming=args.use_naming, naming_limit=args.naming_limit,
              naming_guesses=args.naming_guesses,
              use_text=args.use_text, text_limit=args.text_limit,
              text_gate=args.text_gate,
              reranker=paths.CROSS_ENCODER_MODEL,
              legacy_prompt=args.legacy_prompt, direct_prompt=args.direct_prompt)
    print(f"Done. Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
