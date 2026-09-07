"""Can the model name the entity in the image, and does cropping help?

The name is the key to the only channel with real headroom: 84.2% of the gold
titles are resolvable by name, against 40.6% recall@20 for the image index. But
Qwen3-VL-8B produces a name that resolves to the right article only 11.6% of the
time, so that ceiling is out of reach.

Looking at where it fails, the model is not confusing the subject with the
background — it answers `Gila monster` for a `Tiliqua rugosa`, `Schloss
Nordkirchen` for a `Grasten Palace`. Right kind of thing, wrong instance. So a
crop cannot help by removing background; it can only help by *magnifying*, since
telling those apart depends on details a downsampled image no longer carries.

That is what the variants test, and why each crop is scaled back to the original
size: the VLM allocates tokens by pixel dimensions, so a crop fed as-is would be
seen at *fewer* tokens rather than higher magnification, testing the opposite of
the hypothesis. `--no-upscale` runs it the naive way, to show the difference.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, SRC_ROOT)

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

import paths
from agent.prompts import MULTI_NAMING_PROMPTS, NAMING_PROMPT
from llm import chat_model
from retrieval.knowledge_base import KnowledgeBase, normalize
from vlm.dataset import load_dataset


def center_crop(img, ratio):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    l, t = (w - cw) // 2, (h - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


def box_crop(img, box, pad=0.08):
    """Crop to a detector box, with a small margin so context is not lost."""
    w, h = img.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((max(0, x0 - mx), max(0, y0 - my),
                     min(w, x1 + mx), min(h, y1 + my)))


def variant_image(path, variant, boxes, upscale):
    img = Image.open(path).convert("RGB")
    size = img.size
    if variant == "full":
        return img
    if variant.startswith("center"):
        img = center_crop(img, int(variant[len("center"):]) / 100)
    elif variant == "box":
        box = boxes.get(path)
        if not box:
            return None       # no detection: excluded from this variant's rate
        img = box_crop(img, box)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return img.resize(size, Image.BICUBIC) if upscale else img


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def name_one(llm, path, variant, boxes, upscale, guesses=1, style="diverse"):
    """The model's name for the subject, or its ``guesses`` best names."""
    try:
        img = variant_image(path, variant, boxes, upscale)
        if img is None:
            return None
        prompt = (NAMING_PROMPT if guesses == 1
                  else MULTI_NAMING_PROMPTS[style].format(n=guesses))
        resp = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=[{"type": "image_url",
                                   "image_url": {"url": data_uri(img)}}]),
        ])
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        return None


def split_guesses(reply, limit):
    """One name per line, minus the numbering the model adds anyway."""
    out = []
    for line in (reply or "").splitlines():
        name = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip()
        if name and name not in out:
            out.append(name)
    return out[:limit]


def norm_url(url):
    return (url or "").rstrip("/").lower().replace("http://", "https://")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--output", default="outputs/agentic/naming_probe.jsonl")
    p.add_argument("--variants", default="full,center80,center60",
                   help="comma-separated: full, center<pct>, box")
    p.add_argument("--boxes", default=None,
                   help="JSONL of {image_path, box:[x0,y0,x1,y1]} for the 'box' variant")
    p.add_argument("--no-upscale", dest="upscale", action="store_false",
                   help="feed crops at their cropped size instead of rescaling up")
    p.add_argument("--style", default="diverse",
                   choices=sorted(MULTI_NAMING_PROMPTS),
                   help="How to ask for several guesses.")
    p.add_argument("--guesses", type=int, default=1,
                   help="Names to ask for. The model resolves one guess to the "
                        "right article 11.6%% of the time and its wrong guess is "
                        "usually the right kind of thing, so more may pay.")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kb = KnowledgeBase(paths.KB_PATH)

    boxes = {}
    if args.boxes and os.path.exists(args.boxes):
        for line in open(args.boxes):
            r = json.loads(line)
            boxes[r["image_path"]] = r.get("box")
        print(f"Boxes loaded: {len(boxes)}")

    dataset = [it for it in load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)
               if os.path.exists(it["image_path"])][: args.limit]
    llm = chat_model(args.model_name, args.base_url, max_tokens=32)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"Naming {len(dataset)} images, variants={variants}, upscale={args.upscale}")

    resolved_cache = {}

    def resolves(name, gt_url):
        if name not in resolved_cache:
            resolved_cache[name] = [norm_url(h["wiki_url"])
                                    for h in kb.lookup_articles(name, limit=3)]
        return gt_url in resolved_cache[name]

    results = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for variant in variants:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                names = list(tqdm(
                    pool.map(lambda it: name_one(llm, it["image_path"], variant,
                                                 boxes, args.upscale, args.guesses,
                                                 args.style), dataset),
                    total=len(dataset), desc=variant))
            hit = attempted = exact = 0
            for item, reply in zip(dataset, names):
                gt_url = norm_url(item["wikipedia_url"])
                guesses = split_guesses(reply, args.guesses) if reply else []
                ok = any(resolves(g, gt_url) for g in guesses)
                name = guesses[0] if guesses else None
                attempted += bool(name)
                hit += ok
                exact += bool(name) and normalize(name) == normalize(item["wikipedia_title"])
                out.write(json.dumps({
                    "unique_id": item["unique_id"], "variant": variant,
                    "gt_title": item["wikipedia_title"], "predicted_name": name,
                    "guesses": guesses, "resolved": ok,
                }, ensure_ascii=False) + "\n")
            results[variant] = (hit, exact, attempted, len(dataset))

    n = len(dataset)
    print("=" * 66)
    print(f"model: {args.model_name}   examples: {n}   upscale: {args.upscale}")
    print(f"{'variant':12s} {'name -> right article':>22s} {'exact title':>13s} {'answered':>10s}")
    for v, (hit, exact, attempted, tot) in results.items():
        print(f"{v:12s} {100 * hit / tot:21.1f}% {100 * exact / tot:12.1f}% {100 * attempted / tot:9.1f}%")
    print(f"\nreference: image index recall@20 = 40.6%, name-lookup ceiling = 84.2%")
    print("=" * 66)
    print(f"Per-example detail: {args.output}")


if __name__ == "__main__":
    main()
