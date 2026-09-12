from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import paths
import provenance
from vlm.dataset import build_record, done_ids, load_dataset


def load_todo(output: str, limit: int | None = None) -> list[dict]:
    """Examples still to predict. Output files are appended to, so a killed run"""
    dataset = load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)
    if limit is not None:
        dataset = dataset[:limit]
    done = done_ids(output)
    todo = [it for it in dataset if it["unique_id"] not in done]
    print(f"Dataset: {len(dataset)} | already done: {len(done)} | to do: {len(todo)}")
    return todo


def run_batch(items: list[dict], predict, output: str, concurrency: int = 8,
              **meta) -> None:
    """Run ``predict(item) -> record`` over items, appending JSONL to ``output``."""
    failures: list[str] = []
    elapsed: list[float] = []
    wall_t0 = time.time()

    def work(item):
        t0 = time.time()
        try:
            return _work(item)
        finally:
            elapsed.append(time.time() - t0)

    def _work(item):
        if not os.path.exists(item["image_path"]):
            tqdm.write(f"missing image: {item['image_path']}")
            return build_record(item, None)
        try:
            return predict(item)
        except Exception as e:
            failures.append(str(e))
            tqdm.write(f"failed on {item['unique_id']}: {str(e)[:140]}")
            return build_record(item, None)

    with open(output, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(work, it) for it in items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()

    if failures:
        from collections import Counter
        print(f"{len(failures)}/{len(items)} examples failed:")
        for msg, n in Counter(failures).most_common(3):
            print(f"  [{n}x] {msg[:160]}")
    wall = time.time() - wall_t0
    meta.update(concurrency=concurrency, wall_seconds=round(wall, 1),
                throughput_per_min=round(60 * len(items) / wall, 1) if wall else None)
    if elapsed:
        from statistics import mean, median
        print(f"Seconds per example: mean {mean(elapsed):.2f}, median {median(elapsed):.2f}"
              f"  (concurrency {concurrency}, {60 * len(items) / wall:.1f}/min)")
        meta.update(avg_seconds_per_example=round(mean(elapsed), 2),
                    median_seconds_per_example=round(median(elapsed), 2))
    provenance.stamp(output, examples=len(items), failures=len(failures), **meta)
