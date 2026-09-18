from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from retrieval.fusion import Ranking, rank_paragraphs


class LookupArticleInput(BaseModel):
    name: str = Field(..., description="The exact name of the entity, person, or object to look up on Wikipedia.")


class SearchParagraphsInput(BaseModel):
    query: str = Field(..., description="A short, highly focused keyword phrase (e.g. 'Arabidopsis lyrata outcrossing') to find specific information. Do not use full sentences or questions.")


class ReadArticleInput(BaseModel):
    title: str = Field(..., description="The EXACT title of the Wikipedia article, exactly as it appeared in previous tool results.")
    query: str = Field(..., description="The keyword phrase to search for inside this specific article.")


class SearchInput(BaseModel):
    query: str = Field(..., description="Keywords describing what you need to know: what the question asks about, plus any distinctive term. Rare words find things, generic ones ('large', 'population', 'typically') do not.")
    names: list[str] = Field(default_factory=list, description="Wikipedia article titles to open. Include your best visual guess of what the entity is, and/or any plausible titles that appeared in a previous tool result.")


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
                ranking: Ranking = Ranking(),
                top_k=20, lookup_limit=3,
                text_limit: int = 5, state=None,
                max_names: int = 4, tool_set: str = "minimal",
                preview: int = 0, question: str = ""):
    """Retrieval tools for one query image, over a working set the agent grows."""
    candidates: dict[str, Candidate] = {}   # keyed by wiki_url
    state = {} if state is None else state   # per-example, never shared
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

        pool = _pool()
        if not pool:
            return listing
        by_text = {text: title for title, text in pool}
        best = rank_paragraphs(question, [t for _, t in pool],
                               strategy=ranking.preview, top_k=preview,
                               reranker=reranker, bm25_ranker=bm25,
                               bm25_top_m=ranking.bm25_top_m, rrf_k=ranking.rrf_k)
        state["top_score"] = getattr(reranker, "last_top_score", None)
        return (f"Articles this image matches:\n{listing}\n\n"
                f"Passages from them, most relevant first:\n"
                + _format([(by_text.get(p, "?"), p) for p in best]))

    def _rank_pool(query: str) -> str:
        """Rank every paragraph in the working set against `query`."""
        pool = _pool()
        if not pool:
            return "No candidate articles available for this image."
        by_text = {text: title for title, text in pool}
        texts = [text for _, text in pool]
        best = rank_paragraphs(
            query, texts, strategy=ranking.tools, top_k=ranking.top_n,
            bm25_top_m=ranking.bm25_top_m, bm25_ranker=bm25, reranker=reranker,
            rrf_k=ranking.rrf_k,
        )

        state["top_score"] = getattr(reranker, "last_top_score", None)
        return _format([(by_text.get(p, "?"), p) for p in best]) if best else \
            "No relevant paragraphs found."

    @tool(args_schema=SearchInput)
    def search(query: str, names: list[str] | None = None) -> str:
        """Find and read passages about what you are looking for.

        `query` is what you want to know, in keywords. `names` are Wikipedia
        articles to open by title: your best visual guess of the entity, and/or
        plausible titles from an earlier tool. A schema description is prompt —
        telling the agent to propose its own name, rather than to pick from the
        list it was shown, moved `search` from 60.3% of examples to 67.3%.

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
        return _rank_pool(query)

    if tool_set == "legacy":
        @tool(args_schema=LookupArticleInput)
        def lookup_article(name: str) -> str:
            """Add Wikipedia articles matching an entity name to the candidate pool."""
            hits = kb.lookup_articles(name, limit=lookup_limit)
            if not hits:
                return f"No article found for '{name}'."
            _register_lookup(hits)
            return "Added to the pool:\n" + "\n".join(
                f"- {h['title']}" for h in hits)

        @tool(args_schema=ReadArticleInput)
        def read_article(title: str, query: str) -> str:
            """Extract relevant text passages from one specific candidate article."""
            cand = next((c for c in candidates.values() if c.title == title), None)
            url = cand.wiki_url if cand else None
            if url is None:
                hits = kb.lookup_articles(title, limit=1)
                if not hits:
                    return f"Unknown article '{title}'."
                url = hits[0]["wiki_url"]
                _register_lookup(hits)
            paragraphs = kb.get_paragraphs_by_url(wiki_url=url)
            if not paragraphs:
                return f"No text available for '{title}'."
            results = rank_paragraphs(
                query, paragraphs, strategy=ranking.tools, top_k=ranking.top_n,
                bm25_top_m=ranking.bm25_top_m, bm25_ranker=bm25,
                reranker=reranker, rrf_k=ranking.rrf_k)
            return _format([(title, p) for p in results]) if results else \
                "No relevant paragraphs found."

        @tool(args_schema=SearchParagraphsInput)
        def search_paragraphs(query: str) -> str:
            """Search for relevant passages across ALL currently loaded articles."""
            return _rank_pool(query)

        return [lookup_article, search_by_image, search_paragraphs, read_article]

    return [search_by_image, search]
