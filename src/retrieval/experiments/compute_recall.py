import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from PIL import Image
from tqdm import tqdm

from retrieval.retriever import Retriever
from vlm.dataset import load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}


def parse_args():
    parser = argparse.ArgumentParser(description="Top-k retrieval dump for recall@k")
    parser.add_argument("--json-path", default=f"{BASE_FOLDER}/encyclopedic_test_subset.json")
    parser.add_argument("--base-folder", default=BASE_FOLDER)
    parser.add_argument("--output", default="outputs/retrieval/retrieval_topk50.jsonl")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    parser.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"Skipping {len(done_ids)} already-processed examples")

    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=args.top_k,
        device=args.device,
    )

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Retrieval"):
            if item["unique_id"] in done_ids:
                continue

            candidates = []
            image_path = item["image_path"]
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    results = retriever.retrieve(image, item["question"])
                    candidates = [
                        {
                            "wiki_url": r["wiki_url"],
                            "title": r.get("title", ""),
                            "score": r.get("score"),
                        }
                        for r in results
                    ]
                except Exception as e:
                    tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")
            else:
                tqdm.write(f"missing image: {image_path}")

            out.write(
                json.dumps(
                    {
                        "unique_id": item["unique_id"],
                        "wikipedia_url": item.get("wikipedia_url"),
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    print(f"Done. Retrieval dump saved to {args.output}")


if __name__ == "__main__":
    main()
