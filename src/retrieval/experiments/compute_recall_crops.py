import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
from PIL import Image
from tqdm import tqdm

from retrieval.retriever import Retriever
from vlm.dataset import load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
POOL = 100                       # candidate articles fed to the RAG (recall ceiling)
RAW_K = 500                      # raw neighbours fetched before dedup-by-article
EF_SWEEP = [50, 100, 200, 400, 800]
EF_HIGH = 800                    # near-exact search for the query-side comparison
AQE_N = 10                       # neighbours averaged back into the query
REPORT_KS = [1, 5, 10, 20, 50, 100]


def parse_args():
    p = argparse.ArgumentParser(description="Recall@k: can query-side image tricks raise the pool ceiling?")
    p.add_argument("--json-path", default=f"{BASE_FOLDER}/encyclopedic_test_subset.json")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output", default="outputs/retrieval/recall_crops.jsonl")
    p.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    p.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}


def norm_url(u):
    if not u:
        return u
    u = re.sub(r"^https?://", "", u.strip().lower())
    return u.replace("en.m.wikipedia.org", "en.wikipedia.org").rstrip("/")


def normalize(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def center_crop(img, ratio):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


def five_crops(img, ratio=0.6):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    cl, ct = (w - cw) // 2, (h - ch) // 2
    boxes = [
        (0, 0, cw, ch), (w - cw, 0, w, ch),
        (0, h - ch, cw, h), (w - cw, h - ch, w, h),
        (cl, ct, cl + cw, ct + ch),
    ]
    return [img.crop(b) for b in boxes]


def ranked_articles(retriever, embedding, ef, raw_k=RAW_K, top=POOL):
    """Search FAISS at a given efSearch and return top unique article urls (order kept)."""
    retriever.img_index.hnsw.efSearch = max(ef, raw_k)
    _, indices = retriever.img_index.search(embedding, k=raw_k)
    urls, seen = [], set()
    for idx in indices[0]:
        if idx == -1 or idx >= len(retriever.img_values):
            continue
        u = retriever.img_values[idx][0]
        if u in seen:
            continue
        seen.add(u)
        urls.append(u)
        if len(urls) >= top:
            break
    return urls


def sweep_articles(retriever, embedding, ef):
    """Like ranked_articles but efSearch == raw_k == ef (honest ANN-quality probe)."""
    retriever.img_index.hnsw.efSearch = ef
    _, indices = retriever.img_index.search(embedding, k=ef)
    urls, seen = [], set()
    for idx in indices[0]:
        if idx == -1 or idx >= len(retriever.img_values):
            continue
        u = retriever.img_values[idx][0]
        if u not in seen:
            seen.add(u)
            urls.append(u)
        if len(urls) >= POOL:
            break
    return urls


def rrf(ranked_lists, k0=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, u in enumerate(ranked, 1):
            scores[u] = scores.get(u, 0.0) + 1.0 / (k0 + rank)
    return [u for u, _ in sorted(scores.items(), key=lambda x: -x[1])][:POOL]


def gt_rank(urls, gt):
    g = norm_url(gt)
    for i, u in enumerate(urls, 1):
        if norm_url(u) == g:
            return i
    return None


def aqe_embedding(retriever, q_emb, ef, n=AQE_N):
    """Average query expansion: fold the top-n neighbour vectors back into the query."""
    retriever.img_index.hnsw.efSearch = max(ef, n)
    _, indices = retriever.img_index.search(q_emb, k=n)
    vecs = [q_emb[0]]
    for idx in indices[0]:
        if idx != -1:
            vecs.append(retriever.img_index.reconstruct(int(idx)))
    return normalize(np.mean(vecs, axis=0, keepdims=True))


def strategies(retriever, image):
    """Return {name: ranked article urls} for one query image."""
    full = retriever.encode_image(image)
    out = {}
    for ef in EF_SWEEP:
        out[f"full_ef{ef}"] = sweep_articles(retriever, full, ef)
    out["center80"] = ranked_articles(retriever, retriever.encode_image(center_crop(image, 0.8)), EF_HIGH)
    out["center60"] = ranked_articles(retriever, retriever.encode_image(center_crop(image, 0.6)), EF_HIGH)
    crop_lists = [ranked_articles(retriever, full, EF_HIGH)]
    crop_lists += [ranked_articles(retriever, retriever.encode_image(c), EF_HIGH) for c in five_crops(image)]
    out["fivecrop_rrf"] = rrf(crop_lists)
    out["aqe"] = ranked_articles(retriever, aqe_embedding(retriever, full, EF_HIGH), EF_HIGH)
    return out


def report(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("wikipedia_url")]
    names = [f"full_ef{ef}" for ef in EF_SWEEP] + ["center80", "center60", "fivecrop_rrf", "aqe"]
    print(f"\nRecall@k over {len(recs)} examples:")
    print(f"{'strategy':16} " + "  ".join(f"@{k:<4}" for k in REPORT_KS))
    for name in names:
        ranks = [r["ranks"].get(name) for r in recs]
        row = [f"{100 * sum(1 for x in ranks if x is not None and x <= k) / len(ranks):5.1f}"
               for k in REPORT_KS]
        print(f"{name:16} " + "  ".join(row))


def main():
    args = parse_args()

    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = load_done_ids(args.output)
    if done:
        print(f"Skipping {len(done)} already-processed examples")

    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=POOL,
        device=args.device,
    )
    retriever._ensure_index()

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Crop recall"):
            if item["unique_id"] in done:
                continue
            gt = item.get("wikipedia_url")
            ranks = {}
            if os.path.exists(item["image_path"]):
                try:
                    image = Image.open(item["image_path"]).convert("RGB")
                    ranks = {name: gt_rank(urls, gt) for name, urls in strategies(retriever, image).items()}
                except Exception as e:
                    tqdm.write(f"failed for {item['unique_id']}: {e}")
            else:
                tqdm.write(f"missing image: {item['image_path']}")

            out.write(json.dumps({
                "unique_id": item["unique_id"],
                "wikipedia_url": gt,
                "ranks": ranks,
            }, ensure_ascii=False) + "\n")
            out.flush()

    report(args.output)

if __name__ == "__main__":
    main()
