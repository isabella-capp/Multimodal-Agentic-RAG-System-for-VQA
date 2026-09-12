import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a balanced val/test split from the EVQA test subset."
    )
    parser.add_argument(
        "--input",
        default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json",
        help="Path to the full test subset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where val/test split files will be saved.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=200,
        help="Target number of validation examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def stratified_split(dataset, val_size, seed):
    """Split dataset into val and test, stratified by question_type."""
    rng = random.Random(seed)

    by_type = defaultdict(list)
    for item in dataset:
        by_type[item["question_type"]].append(item)

    total = len(dataset)
    val_items = []
    test_items = []

    for qtype, items in sorted(by_type.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) / total * val_size))
        n_val = min(n_val, len(items) - 1) if len(items) > 1 else 0
        val_items.extend(items[:n_val])
        test_items.extend(items[n_val:])

    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return val_items, test_items


def print_stats(name, items):
    counts = Counter(item["question_type"] for item in items)
    print(f"\n{name}: {len(items)} examples")
    for qtype in sorted(counts):
        print(f"  {qtype:20s}: {counts[qtype]:4d}")


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples from {args.input}")
    print_stats("Full dataset", dataset)

    val_items, test_items = stratified_split(dataset, args.val_size, args.seed)

    print_stats("Validation split", val_items)
    print_stats("Test split", test_items)

    val_ids = {item["unique_id"] for item in val_items}
    test_ids = {item["unique_id"] for item in test_items}
    assert len(val_ids & test_ids) == 0, "Overlap between val and test!"
    assert len(val_ids) + len(test_ids) == len(dataset), "Items lost in split!"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_path = output_dir / "encyclopedic_val_split.json"
    test_path = output_dir / "encyclopedic_test_split.json"

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_items, f, indent=2, ensure_ascii=False)
    print(f"\nValidation split saved to {val_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2, ensure_ascii=False)
    print(f"Test split saved to {test_path}")

if __name__ == "__main__":
    main()
