import json
import os
import re
import sqlite3
import threading
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9 ]")
_SPACE = re.compile(r"\s+")
_PAREN = re.compile(r"\s*\([^)]*\)")
_STOP = {"the", "a", "an", "of", "in", "and", "on", "at"}

# Question words carry no content but appear in millions of paragraphs, so they
# dominate the cost of a full-text search without narrowing it: "this" is in
# 2.46M paragraphs and "where" in 1.08M, against 50k for "indies".
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "has", "have", "had", "can", "could", "would", "will",
    "it", "its", "they", "them", "their", "there", "here", "also", "else",
    "besides", "other", "another", "many", "much", "some", "any", "for", "to",
    "from", "with", "by", "as", "or", "not", "but", "than", "then", "into",
}
_TEXT_TERMS = 4          # most selective query terms kept for the search
_AND_MIN_HITS = 5        # below this, the AND was too strict: widen to OR
_DF_CACHE: dict[str, int] = {}   # term -> paragraphs containing it


def load_df_cache(path: str) -> int:
    """Prime the term-frequency cache from disk.

    Counting how many paragraphs hold a term is one query over 14.1M rows, and
    a cold run pays it for every new word: warming up on the 1000 test questions
    took an hour on the cluster against six minutes locally, purely in first
    touches. The counts never change unless the KB is rebuilt, so they belong on
    disk. ``scripts/retrieval/run_prime_df.sh`` fills the file once on CPU.
    """
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        _DF_CACHE.update(json.load(f))
    return len(_DF_CACHE)


def save_df_cache(path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_DF_CACHE, f)
    return len(_DF_CACHE)

_MIN_JACCARD = 0.5
_FUZZY_CANDIDATES = 200  # BM25 shortlist, re-scored exactly below


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return _SPACE.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def alias_forms(title: str) -> list[str]:
    """Surface forms for a title, most specific first.

    Ordered, not a set: lookup tries them in turn, and set iteration order over
    strings varies between processes, which would make matches non-reproducible.
    """
    base = normalize(title)
    forms = [base, normalize(_PAREN.sub("", title))]
    forms += [f[4:] for f in forms if f.startswith("the ")]
    return list(dict.fromkeys(f for f in forms if f))


def title_tokens(title: str) -> set[str]:
    tokens: set[str] = set()
    for alias in alias_forms(title):
        tokens |= set(alias.split())
    return tokens - _STOP


class KnowledgeBase:
    """Read-only encyclopedic KB backed by a SQLite file.

    Build the file once with ``src/retrieval/build_kb_sqlite.py``, then add the
    name-lookup tables with ``src/retrieval/build_title_index.py``. Lookups hit
    the disk on demand, so the 20 GB KB is never loaded into memory.

    ``lookup_articles`` resolves an entity NAME to its article by string matching
    — deliberately not by embedding: the EVA-CLIP text tower is misaligned with
    the image index (0% recall@50 even given the ground-truth title), and exact
    matching is anyway more precise than similarity for near-identical names.

    Three ways in: ``lookup_articles`` by name, the image index (elsewhere), and
    ``search_articles_by_text`` by what the question asks about. Only the last
    does not depend on the model recognising or naming the subject.

    Thread-safe: each thread gets its own connection (SQLite connections cannot
    be shared across threads for concurrent queries). The DB is opened read-only,
    so concurrent readers are fine.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # read-only but NOT immutable: immutable=1 promises SQLite the bytes
        # never change, and the KB gains indexes over time (paragraphs_fts was
        # added after the first runs). A reader holding that promise while the
        # file changes gets wrong rows, not an error.
        self._uri = f"file:{db_path}?mode=ro"
        self._local = threading.local()
        self._has_text_index = bool(self._conn().execute(
            "SELECT 1 FROM sqlite_master WHERE name='paragraphs_fts'").fetchone())
        print(f"Knowledge Base ready (SQLite): {db_path}"
              + ("" if self._has_text_index else "  [no paragraphs_fts: text search off]"))

    def get_paragraphs_by_url(self, wiki_url: str) -> list[str]:
        """Return the non-empty section texts for a Wikipedia URL, in order.

        Boilerplate is deliberately not filtered. References, External links and
        See also are 29.9% of every pool and look like obvious noise, but taking
        them out at query time changed nothing (0.452 against 0.455) and cost
        60% more time per example: the cross-encoder already discards them, and
        reading a section title for every article does not come free.
        """
        rows = self._conn().execute(
            "SELECT text FROM paragraphs WHERE url = ? ORDER BY section_idx",
            (wiki_url,),
        ).fetchall()
        return [r[0] for r in rows]

    def lookup_articles(self, name: str, limit: int = 5) -> list[dict]:
        """Articles whose title matches ``name``; an exact match returns one."""
        query = normalize(name)
        if not query:
            return []
        conn = self._conn()
        for form in alias_forms(name):
            row = conn.execute(
                "SELECT a.title, a.url FROM aliases al "
                "JOIN articles a ON a.url = al.url "
                "WHERE al.alias = ? ORDER BY al.rowid LIMIT 1",
                (form,),
            ).fetchone()
            if row:
                return [{"title": row[0], "wiki_url": row[1], "match": "exact"}]
        return self._fuzzy(query, limit)

    def __contains__(self, wiki_url: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM articles WHERE url = ? LIMIT 1", (wiki_url,)
        ).fetchone()
        return row is not None

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn

    def search_articles_by_text(self, query: str, limit: int = 20,
                                candidates: int = 200) -> list[dict]:
        """Articles whose paragraph text best matches ``query`` (BM25 over FTS5).

        The third way into the KB, and the only one that does not depend on the
        model: `lookup_articles` needs it to name the entity and the image index
        needs the entity to have a photograph, while here the question is the
        query. Paragraph hits are folded up to their article, best hit first.

        Needs the index built by ``build_kb_sqlite.py --paragraphs-fts``;
        returns nothing when it is absent.
        """
        if not self._has_text_index:
            return []
        conn = self._conn()
        tokens = self._select_terms(conn, query)
        if not tokens:
            return []

        hits = self._match(conn, " AND ".join(tokens), candidates)
        if len(hits) < _AND_MIN_HITS:
            hits = self._match(conn, " OR ".join(tokens), candidates)
        if not hits:
            return []

        scores = {pid: s for pid, s in hits}
        rows = conn.execute(
            "SELECT p.id, p.url, a.title FROM paragraphs p JOIN articles a ON a.url = p.url "
            f"WHERE p.id IN ({','.join('?' * len(scores))})",
            tuple(scores),
        ).fetchall()

        best: dict[str, tuple[float, str]] = {}
        for pid, url, title in rows:
            score = scores[pid]
            if url not in best or score < best[url][0]:   # fts5 bm25: lower is better
                best[url] = (score, title)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0])
        return [{"title": t, "wiki_url": u, "score": sc, "match": "text"}
                for u, (sc, t) in ranked[:limit]]

    @staticmethod
    def _match(conn, expr: str, candidates: int) -> list[tuple[int, float]]:
        return conn.execute(
            "SELECT rowid, bm25(paragraphs_fts) AS s FROM paragraphs_fts "
            "WHERE paragraphs_fts MATCH ? ORDER BY s LIMIT ?",
            (expr, candidates),
        ).fetchall()

    def _select_terms(self, conn, query: str) -> list[str]:
        """The rarest content words of the query — what makes the search work.

        FTS5 scores every row in the union of the terms' posting lists, so one
        common word costs more than the rest of the query put together. Two
        filters: drop the question words, which carry no content and sit in
        millions of paragraphs ("this" in 2.46M, "where" in 1.08M against 50k
        for "indies"), then keep the rarest of what is left.

        Rarest, not longest. Length looks like a free proxy and is a bad one: it
        keeps `populations` (608k paragraphs) over `plant` (291k) and `northern`
        (369k) over `breeds` (56k) — long common words in, short rare ones out,
        which is backwards. Measured, it costs 3 points of recall@20 (26.7%
        against 30.0%). Six terms are worse than four: past the rarest few, the
        extra words only add noise.

        Document frequencies are cached per process because a run shares
        vocabulary heavily across questions.
        """
        tokens = {t for t in normalize(query).split()
                  if len(t) > 2 and t not in _STOP and t not in _QUESTION_WORDS}
        for t in tokens - _DF_CACHE.keys():
            _DF_CACHE[t] = conn.execute(
                "SELECT count(*) FROM paragraphs_fts WHERE paragraphs_fts MATCH ?",
                (t,)).fetchone()[0]
        return sorted((t for t in tokens if _DF_CACHE[t]),
                      key=lambda t: _DF_CACHE[t])[:_TEXT_TERMS]

    def _fuzzy(self, query: str, limit: int) -> list[dict]:
        qtokens = set(query.split()) - _STOP
        if not qtokens:
            return []
        rows = self._conn().execute(
            "SELECT a.title, f.url, f.tokens FROM titles_fts f "
            "JOIN articles a ON a.url = f.url "
            "WHERE titles_fts MATCH ? ORDER BY bm25(titles_fts) LIMIT ?",
            (" OR ".join(sorted(qtokens)), _FUZZY_CANDIDATES),
        ).fetchall()

        ranked = []
        for title, url, tokens in rows:
            ttokens = set(tokens.split())
            shared = len(qtokens & ttokens)
            if shared < 2:
                continue
            union = len(qtokens) + len(ttokens) - shared
            score = shared / union if union else 0.0
            if score >= _MIN_JACCARD or (shared == len(ttokens) and len(ttokens) >= 2):
                ranked.append((score, len(ttokens), title, url))

        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [{"title": t, "wiki_url": u, "match": "fuzzy"} for _, _, t, u in ranked[:limit]]
