from __future__ import annotations

from dataclasses import dataclass

from retrieval.bm25 import BM25Ranker
from retrieval.reranker import CrossEncoderReranker

STRATEGIES = ("bm25", "bge", "bm25_bge", "rrf")


@dataclass(frozen=True)
class Ranking:
    """Which strategy each ranking operation uses, and the numbers behind it."""

    tools: str = "rrf"
    preview: str = "rrf"
    final: str = "bge"
    top_n: int = 20
    bm25_top_m: int = 50
    rrf_k: int = 60

    def __post_init__(self):
        for field in ("tools", "preview", "final"):
            value = getattr(self, field)
            if value not in STRATEGIES:
                raise ValueError(
                    f"Ranking.{field}={value!r} is not one of {STRATEGIES}")


def rrf_score(rankings: list[list[str]], rrf_k: int = 60) -> list[str]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, p in enumerate(ranking, 1):
            scores[p] = scores.get(p, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda p: scores[p], reverse=True)


def rank_paragraphs(
    query: str,
    paragraphs: list[str],
    strategy: str,
    top_k: int,
    *,
    bm25_top_m: int = 50,
    bm25_ranker: BM25Ranker | None = None,
    reranker: CrossEncoderReranker | None = None,
    rrf_k: int = 60,
) -> list[str]:
    """Return the *top_k* most relevant paragraphs using the chosen strategy."""
    if not paragraphs:
        return []
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; valid: {STRATEGIES}")

    if strategy == "bm25":
        assert bm25_ranker is not None, \
            "bm25_ranker required for strategy='bm25'"
        return bm25_ranker.rank(query, paragraphs, top_m=top_k)

    if strategy == "bge":
        assert reranker is not None, \
            "reranker required for strategy='bge'"
        return reranker.rerank(query, paragraphs, top_n=top_k)

    if strategy == "bm25_bge":
        assert bm25_ranker is not None and reranker is not None, \
            "both bm25_ranker and reranker required for strategy='bm25_bge'"
        bm25_pool = bm25_ranker.rank(query, paragraphs, top_m=bm25_top_m)
        return reranker.rerank(query, bm25_pool, top_n=top_k)

    assert bm25_ranker is not None and reranker is not None, \
        "both bm25_ranker and reranker required for strategy='rrf'"
    bm25_all = bm25_ranker.rank(
        query, paragraphs, top_m=len(paragraphs), force_sort=True
    )
    bge_all = reranker.rerank(
        query, paragraphs, top_n=len(paragraphs), force_sort=True
    )
    fused = rrf_score([bm25_all, bge_all], rrf_k=rrf_k)
    return fused[:top_k]
