import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tqdm import tqdm

import paths
from retrieval.knowledge_base import (KnowledgeBase, load_df_cache, save_df_cache,
                                      _QUESTION_WORDS, _STOP, normalize)
from vlm.dataset import load_dataset


def main():
    p = argparse.ArgumentParser(description="Precompute document frequencies for BM25.")
    p.add_argument("--output", default=paths.TERM_DF_PATH)
    args = p.parse_args()

    kb = KnowledgeBase(paths.KB_PATH)
    known = load_df_cache(args.output)
    print(f"already cached: {known}")

    terms = set()
    for it in load_dataset(paths.JSON_PATH, paths.BASE_FOLDER):
        terms |= {t for t in normalize(it["question"]).split()
                  if len(t) > 2 and t not in _STOP and t not in _QUESTION_WORDS}
    from retrieval.knowledge_base import _DF_CACHE
    todo = sorted(terms - _DF_CACHE.keys())
    print(f"distinct terms: {len(terms)}, still to measure: {len(todo)}")

    conn = kb._conn()
    for t in tqdm(todo, desc="term df"):
        kb._select_terms(conn, t)      # fills _DF_CACHE for this term
    print(f"cached now: {save_df_cache(args.output)} -> {args.output}")


if __name__ == "__main__":
    main()
