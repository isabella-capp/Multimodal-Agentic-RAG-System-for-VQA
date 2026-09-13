import argparse

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL inference on Encyclopedic-VQA"
    )
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--direct-prompt", action="store_true",
                        help="Long-form answers without the preamble, and full ranges.")
    parser.add_argument("--legacy-prompt", action="store_true",
                        help="Prompts without the answer-format block, as before it existed.")
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="Enable visual retrieval + KB context augmentation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of FAISS nearest neighbours to retrieve.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=3,
        help="Paragraphs to keep after reranking (or the first N with --no-rerank; <=0 keeps all).",
    )
    parser.add_argument(
        "--bm25-top-m",
        type=int,
        default=50,
        help="BM25 candidate pool size before BGE reranking (only used with reranking enabled).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip paragraph reranking; use the first --rerank-top-n paragraphs directly.",
    )
    parser.add_argument(
        "--retrieval-strategy",
        default="rrf",
        choices=["bm25", "bge", "bm25_bge", "rrf"],
        help="Paragraph retrieval strategy when --use-retrieval is active. "
             "'rrf' fuses BM25 and cross-encoder ranks and is the best measured "
             "for this pipeline (0.4760 against 0.4660 for 'bm25_bge'). "
             "'bm25_bge' pre-filters with BM25 then reranks; 'bm25' and 'bge' "
             "use one signal. Overridden by --no-rerank.",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF smoothing constant (default 60). Only used with --retrieval-strategy rrf.",
    )
    parser.add_argument(
        "--oracle", action="store_true",
        help="Upper bound: put the gold article in the pool. Uses the label, so "
             "it is not a system — it measures what retrieval still costs.")
    parser.add_argument(
        "--use-naming",
        action="store_true",
        help="Also enter the KB by name: ask the model what the image shows and "
             "add the articles that name resolves to. Requires --use-retrieval.",
    )
    parser.add_argument(
        "--naming-limit",
        type=int,
        default=3,
        help="Articles to keep from the name lookup.",
    )
    parser.add_argument(
        "--naming-guesses",
        type=int,
        default=1,
        help="Names to ask the model for. One resolves to the right article "
             "11.8%% of the time, three 17.1%%; the curve is flat past five.",
    )
    parser.add_argument(
        "--use-text",
        action="store_true",
        help="Also enter the KB by text: search the question against the paragraph "
             "index and add the articles it finds. Requires --use-retrieval.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=5,
        help="Articles to keep from the full-text search.",
    )
    parser.add_argument(
        "--text-gate",
        type=float,
        default=None,
        help="Only search the KB by text when the best paragraph of the image pool "
             "scores below this (cross-encoder logit). Below -1 the pool holds the "
             "gold article 4-40%% of the time and the text channel supplies it; "
             "above, it holds it 46-62%% and the channel only adds noise.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Print a detailed pipeline trace for the first N processed examples.",
    )
    return parser.parse_args()
