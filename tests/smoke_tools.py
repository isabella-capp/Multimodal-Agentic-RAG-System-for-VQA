"""Build and call the agent's tools with stubs, on the CPU, in a second.

Written after `@dataclass` was lost from `Candidate` in a refactor: the module
still imported and `--help` still exited 0, so every check we had passed, and
two 1000-example runs came back with 1000 identical `Candidate() takes no
arguments`. Parsing a module does not construct anything; this does.

    uv run python tests/smoke_tools.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from agent.tools import Candidate, build_tools  # noqa: E402
from retrieval.fusion import Ranking  # noqa: E402


class FakeRetriever:
    def encode_image(self, image):
        return None

    def search_index(self, _vec, top_k=20):
        return [{"wiki_url": f"u{i}", "title": f"Article {i}", "score": 1.0 - i / 10}
                for i in range(3)]


class FakeKB:
    def get_paragraphs_by_url(self, wiki_url):
        return [f"paragraph one of {wiki_url}", f"paragraph two of {wiki_url}"]

    def lookup_articles(self, name, limit=1):
        return [{"wiki_url": "named", "title": name, "match": "title"}]

    def search_articles_by_text(self, query, limit=5):
        return [{"wiki_url": "texty", "title": f"Found by {query}"}]


class FakeReranker:
    last_top_score = None

    def rerank(self, query, paragraphs, top_n=3, force_sort=False, **kw):
        self.last_top_score = 0.5
        return list(paragraphs)[:top_n]


class FakeBM25:
    def rank(self, query, paragraphs, top_m=50, force_sort=False):
        return list(paragraphs)[:top_m]


def check(tool_set: str, strategy: str) -> None:
    state: dict = {}
    tools = build_tools(
        FakeRetriever(), FakeKB(), FakeReranker(), FakeBM25(), image=None,
        ranking=Ranking(tools=strategy, preview=strategy, final=strategy),
        state=state, preview=2, question="what is this?", tool_set=tool_set,
    )
    names = [t.name for t in tools]
    expected = (["search_by_image", "search"] if tool_set == "minimal"
                else ["lookup_article", "search_by_image", "search_paragraphs",
                      "read_article"])
    assert names == expected, f"{tool_set}: got {names}, want {expected}"

    by_name = {t.name: t for t in tools}
    out = by_name["search_by_image"].func()
    assert "Article 0" in out, f"{tool_set}: image tool returned {out[:80]!r}"
    assert state.get("top_score") is not None, f"{tool_set}: the gate got no score"

    if tool_set == "minimal":
        out = by_name["search"].func("rare term", ["Article 1"])
    else:
        out = by_name["search_paragraphs"].func("rare term")
    assert "Paragraph" in out, f"{tool_set}: search returned {out[:80]!r}"

    assert state.get("candidates"), f"{tool_set}: nothing reached the pool"
    for cand in state["candidates"].values():
        assert isinstance(cand, Candidate)
    print(f"  {tool_set:8} {strategy:9} {len(names)} tools, "
          f"{len(state['candidates'])} candidates, gate score {state['top_score']}")


def main() -> int:
    for tool_set in ("minimal", "legacy"):
        for strategy in ("bge", "rrf"):
            check(tool_set, strategy)
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
