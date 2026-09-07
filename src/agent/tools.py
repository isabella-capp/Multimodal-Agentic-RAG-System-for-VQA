from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from retrieval.knowledge_base import normalize
from retrieval.fusion import rank_paragraphs

# Maps the agent-facing retrieval_mode names to fusion strategies.
_MODE_TO_STRATEGY = {
    "bm25+reranker": "bm25_bge",   # current default — behaviour unchanged
    "reranker":      "bge",
    "rrf":           "rrf",
}

class LookupArticleInput(BaseModel):
    name: str = Field(..., description="The exact name of the entity, person, or object to look up on Wikipedia.")

class SearchParagraphsInput(BaseModel):
    query: str = Field(..., description="A short, highly focused keyword phrase (e.g. 'Arabidopsis lyrata outcrossing') to find specific information. Do not use full sentences or questions.")

class SearchInput(BaseModel):
    query: str = Field(..., description="Keywords describing what you need to know: what the question asks about, plus any distinctive term. Rare words find things, generic ones ('large', 'population', 'typically') do not.")
    names: list[str] = Field(default_factory=list, description="Titles of articles to open as well, exactly as they appeared in a previous tool result. Give the ones that could plausibly be the subject; leave empty if none look right.")

class ReadArticleInput(BaseModel):
    title: str = Field(..., description="The EXACT title of the Wikipedia article, exactly as it appeared in previous tool results.")
    query: str = Field(..., description="The keyword phrase to search for inside this specific article.")

@dataclass
class Candidate:
    """One article in the working set, with its provenance."""

    title: str
    wiki_url: str
    sources: set[str] = field(default_factory=set)   # "image" | "lookup"
    image_score: float | None = None


def _format(paragraphs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1} — {title}] {text}"
                       for i, (title, text) in enumerate(paragraphs))


def build_tools(retriever, kb, reranker, bm25, image,
                top_n=5, top_k=20, bm25_top_m=50, lookup_limit=3,
                retrieval_mode: str = "reranker", rrf_k: int = 60,
                text_limit: int = 5, state=None,
                unified: bool = False, max_names: int = 4, with_read: bool = True,
                preview: int = 0, question: str = ""):
    """Retrieval tools for one query image, over a working set the agent grows.

    Three ways into the KB — by image (EVA-CLIP/FAISS), by name, and by what
    the articles say — and two ways to read: search across all candidates, or
    read one article deeply.

    The text entry is the one the pipeline cannot use well. Given the question
    verbatim it lifts article coverage from 46.6% to 56.1%, but stacking its
    articles onto the pool costs more in noise than it wins: measured, it gains
    55 points where it alone finds the article and loses 9 on the five times as
    many where the article was already there. An agent can call it only after
    seeing that what it read is about the wrong entity, and can put the rare
    word the question lacks into the query — which is the whole reason to try
    this in an agent rather than a fixed pipeline.
    The working set tracks how each article was found (provenance) but this
    information is kept internal; tool outputs are unchanged.

    retrieval_mode controls the paragraph-level pipeline:
      "reranker"       — all paragraphs go directly to the cross-encoder (default, current behaviour)
      "bm25+reranker"  — BM25 pre-filter → cross-encoder 
      "rrf"            — BM25 + BGE independent rankings → Reciprocal Rank Fusion
    """
    candidates: dict[str, Candidate] = {}   # keyed by wiki_url
    state = {} if state is None else state   # per-example, never shared
    tried: set[str] = set()
    tried_raw: set[str] = set()
    cache: dict = {}

    def _register_image(articles: list[dict]) -> None:
        for a in articles:
            url = a["wiki_url"]
            if url in candidates:
                candidates[url].sources.add("image")
            else:
                candidates[url] = Candidate(
                    title=a["title"],
                    wiki_url=url,
                    sources={"image"},
                    image_score=a.get("score"),
                )

    def _register_text(articles: list[dict]) -> None:
        for a in articles:
            url = a["wiki_url"]
            if url in candidates:
                candidates[url].sources.add("text")
            else:
                candidates[url] = Candidate(title=a["title"], wiki_url=url,
                                            sources={"text"}, image_score=None)

    def _register_lookup(articles: list[dict]) -> None:
        for a in articles:
            url = a["wiki_url"]
            if url in candidates:
                candidates[url].sources.add("lookup")
            else:
                candidates[url] = Candidate(
                    title=a["title"],
                    wiki_url=url,
                    sources={"lookup"},
                    image_score=None,
                )

    def _image_candidates() -> list[dict]:
        if "articles" not in cache:
            cache["articles"] = retriever.search_index(
                retriever.encode_image(image), top_k=top_k
            )
            _register_image(cache["articles"])
        return cache["articles"]

    state["candidates"] = candidates   # what the agent assembled, for a final pass

    def _pool() -> list[tuple[str, str]]:
        """Every paragraph of every article in the working set, tagged with title."""
        _image_candidates()
        key = tuple(sorted((c.wiki_url, c.title) for c in candidates.values()))
        if cache.get("pool_key") != key:
            cache["pool"] = [
                (c.title, p)
                for c in candidates.values()
                for p in kb.get_paragraphs_by_url(wiki_url=c.wiki_url)
            ]
            cache["pool_key"] = key
        return cache["pool"]

    @tool(args_schema=LookupArticleInput)
    def lookup_article(name: str) -> str:
        """Add Wikipedia articles matching an entity name to the candidate pool.
        
        Use this when you want to explore a specific entity by name, or if the 
        image search didn't give good results. Calling this multiple times with 
        DIFFERENT names expands your search pool.
        """
        if normalize(name) in tried:
            return (f"You already tried '{name}'. Names tried so far: "
                    f"{', '.join(sorted(tried_raw))}. Give a different one — a "
                    f"more specific name, the common name instead of the "
                    f"scientific one, or another candidate entirely.")
        tried.add(normalize(name)); tried_raw.add(name)
        hits = kb.lookup_articles(name, limit=lookup_limit)
        if not hits:
            return (f"No article found for '{name}'. Try a different name, "
                    f"or use search_by_image to see what the image matches.")
        _register_lookup(hits)
        return ("Added to the search pool:\n"
                + "\n".join(f"- {h['title']}" for h in hits))

    @tool
    def search_by_image() -> str:
        """Find candidate Wikipedia articles whose reference images are visually similar to the input image.

        Use this to generate candidates; do not assume the top result is correct.
        All returned articles are added to the working set automatically.
        """
        articles = _image_candidates()
        if not articles:
            return "No articles found for this image."
        listing = "\n".join(f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                            for i, a in enumerate(articles, 1))
        if not preview or not question:
            return listing

        # Show the passages, not just the titles: the agent has to write the
        # keywords for the next search, and it writes better ones after reading
        # text than after reading a list of names. All 20 candidates stay in the
        # pool — only how much of it the agent sees is cut.
        pool = _pool()
        if not pool:
            return listing
        by_text = {text: title for title, text in pool}
        best = rank_paragraphs(question, [t for _, t in pool], strategy="bge",
                               top_k=preview, reranker=reranker,
                               bm25_ranker=bm25, bm25_top_m=bm25_top_m, rrf_k=rrf_k)
        state["top_score"] = getattr(reranker, "last_top_score", None)
        return (f"Articles this image matches:\n{listing}\n\n"
                f"Passages from them, most relevant first:\n"
                + _format([(by_text.get(p, "?"), p) for p in best]))

    @tool(args_schema=ReadArticleInput)
    def read_article(title: str, query: str) -> str:
        """Extract relevant text passages from one specific candidate article.
        
        Use this ONLY when you want to deeply inspect a single article that you 
        have already discovered. It returns the most relevant paragraphs from that article.
        """
        cand = next((c for c in candidates.values() if c.title == title), None)
        url = cand.wiki_url if cand else None
        if url is None:
            hits = kb.lookup_articles(title, limit=1)
            if not hits:
                return (f"Unknown article '{title}'. Find it with lookup_article "
                        f"or search_by_image first.")
            url = hits[0]["wiki_url"]
            _register_lookup(hits)
        paragraphs = kb.get_paragraphs_by_url(wiki_url=url)
        if not paragraphs:
            return f"No text available for '{title}'."
        strategy = _MODE_TO_STRATEGY.get(retrieval_mode, "bge")
        results = rank_paragraphs(
            query, paragraphs, strategy=strategy, top_k=top_n,
            bm25_top_m=bm25_top_m, bm25_ranker=bm25, reranker=reranker,
            rrf_k=rrf_k,
        )
        return _format([(title, p) for p in results]) if results else \
            "No relevant paragraphs found."

    @tool(args_schema=SearchParagraphsInput)
    def search_paragraphs(query: str) -> str:
        """Search for relevant text passages across ALL currently loaded candidate articles.
        
        Use this to quickly gather facts and compare information across all the articles 
        in your pool simultaneously. Each returned passage will tell you which article it came from.
        """
        pool = _pool()
        if not pool:
            return "No candidate articles available for this image."
        by_text = {text: title for title, text in pool}
        texts = [text for _, text in pool]
        strategy = _MODE_TO_STRATEGY.get(retrieval_mode, "bge")
        best = rank_paragraphs(
            query, texts, strategy=strategy, top_k=top_n,
            bm25_top_m=bm25_top_m, bm25_ranker=bm25, reranker=reranker,
            rrf_k=rrf_k,
        )
        
        state["top_score"] = getattr(reranker, "last_top_score", None)
        return _format([(by_text.get(p, "?"), p) for p in best]) if best else \
            "No relevant paragraphs found."

    @tool(args_schema=SearchInput)
    def search(query: str, names: list[str] | None = None) -> str:
        """Find and read passages about what you are looking for.

        `query` is what you want to know, in keywords. `names` are articles to
        open by title, taken from what an earlier tool listed.

        Splitting them matters: a title is looked up as a title, keywords are
        matched against the text of every article. One string cannot do both —
        "Aeschynomene as a plant in the US" finds nothing as a title, and
        "Aeschynomene" alone finds little as a query.
        """
        for name in (names or [])[:max_names]:
            _register_lookup(kb.lookup_articles(name, limit=lookup_limit))
        if not names:
            _register_lookup(kb.lookup_articles(query, limit=lookup_limit))
        _register_text(kb.search_articles_by_text(query, limit=text_limit))
        return search_paragraphs.func(query)

    if unified:
        return [search_by_image, search] + ([read_article] if with_read else [])

    return [lookup_article, search_by_image, search_paragraphs, read_article]
