import os

# Must be set before TensorFlow is imported (through evaluation_utils).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

import evaluation_utils

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DEFAULT_PREDICTIONS = os.path.join(OUTPUTS_DIR, "predictions.jsonl")
DEFAULT_OUTPUT = os.path.join(OUTPUTS_DIR, "results.json")


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_prediction(record, scoring_function):
    if record["prediction"] is None:
        return 0.0
    example = {
        "question": record["question"],
        "reference": record["answer"],
        "candidate": record["prediction"],
        "question_type": record["question_type"],
    }
    return scoring_function(example)


def summarize(scores_by_type):
    all_scores = [s for scores in scores_by_type.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "num_examples": len(all_scores),
        "accuracy_overall": overall,
        "accuracy_by_type": {q: sum(s) / len(s) for q, s in scores_by_type.items()},
    }


def print_report(summary, scores_by_type):
    print(
        f"\nOverall accuracy: {summary['accuracy_overall']:.4f} "
        f"(n={summary['num_examples']})"
    )
    for qtype in sorted(scores_by_type):
        scores = scores_by_type[qtype]
        print(f"  {qtype:15s}: {sum(scores) / len(scores):.4f} (n={len(scores)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Encyclopedic-VQA evaluation")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()

    records = read_predictions(args.predictions)
    print(f"Loaded {len(records)} predictions from {args.predictions}")

    scoring_function = evaluation_utils.initialize_encyclopedic_vqa_evaluation_function()

    scores_by_type = defaultdict(list)
    per_example = []
    for record in tqdm(records, desc="Scoring"):
        score = score_prediction(record, scoring_function)
        scores_by_type[record["question_type"]].append(score)
        per_example.append({"unique_id": record["unique_id"],
                            "question_type": record["question_type"],
                            "score": score})

    summary = summarize(scores_by_type)
    print_report(summary, scores_by_type)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.output}")

    scores_path = args.output.replace(".json", ".scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row) + "\n")
    print(f"Per-example scores written to {scores_path}")


if __name__ == "__main__":
    main()
