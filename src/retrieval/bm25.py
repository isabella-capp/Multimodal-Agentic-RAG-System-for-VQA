from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, split on non-alphanumeric runs."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25Ranker:
    """Lexical ranking over an in-memory paragraph pool; one of the two lists RRF fuses."""

    def rank(
        self,
        query: str,
        paragraphs: list[str],
        top_m: int,
        force_sort: bool = False,
    ) -> list[str]:
        """Return up to *top_m* paragraphs ranked by BM25 score."""
        if not paragraphs:
            return []
        if len(paragraphs) <= top_m and not force_sort:
            return paragraphs

        tokenized = [_tokenize(p) for p in paragraphs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        return [paragraphs[i] for i in order[:top_m]]
