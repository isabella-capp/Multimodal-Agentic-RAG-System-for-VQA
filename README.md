# Encyclopedic-VQA — Agentic RAG (Phase C)

Agentic retrieval-augmented VQA on Encyclopedic-VQA. A vision-language model,
served on **vLLM** or reached over any OpenAI-compatible endpoint, drives a
tool-using loop over a visual retrieval stack (EVA-CLIP + FAISS), a SQLite
knowledge base, and a cross-encoder reranker. Four tools, and the agent grows
its own working set of articles:

1. **`lookup_article(name)`** — resolve the entity the model recognises in the
   image to its Wikipedia article, by name. This is the only channel that can
   beat the ~47% recall ceiling of the image embedding.
2. **`search_by_image()`** — list the articles whose reference images match the
   query image; the fallback when the model cannot name what it sees.
3. **`read_article(title, query)`** — cross-encoder over *all* the paragraphs of
   one chosen article.
4. **`search_paragraphs(query)`** — cross-encoder over every candidate article at
   once, each passage labelled with where it came from. Committing to a single
   article is where the agent lost to plain RAG: it read the right one 15.6% of
   the time against 41.3% for a pooled rerank, while being *more* accurate than
   RAG whenever it did (0.656 vs 0.528).

The model may use its own knowledge to **name** what it sees (that is only a
search key), never to **answer**: every fact must come from a retrieved passage.

Three settings are compared, always at a fixed model:

| | |
|---|---|
| **A** | no retrieval — what the model answers on its own |
| **B** | retrieval + cross-encoder, the standard RAG baseline |
| **C** | the agentic loop above |

## Requirements

- SLURM cluster with the `cvcs2026` account and an A40/L40S (45 GB) GPU.
- [`uv`](https://docs.astral.sh/uv/) on `PATH` (project + scoring envs are synced
  automatically on first run).
- Shared assets already present under `/work/cvcs2026/` for the account:
  - HuggingFace cache (`recursive_retrievers/hf_cache`) with Qwen2.5-VL-3B/7B,
    Qwen3-VL-8B, EVA-CLIP-8B, `bge-reranker-base`, `clip-vit-large-patch14`
    (add more with `scripts/setup/download_model.sh <hf-id>`).
  - `encyclopedic/`: `knn.index`, `knn.json`, `encyclopedic_kb_wiki.db`,
    `encyclopedic_test_subset.json`.
- Clone the repo to your home (e.g. `/homes/$USER/cvcs2026`) and submit from its root.

## Running experiments

**Always launch through `scripts/submit.sh`.** It snapshots the code before
submitting, which is what makes a queued job run what you actually submitted:

```bash
scripts/submit.sh scripts/run_abc.sh                       # A, B and C — the reference table
SMOKE=1 scripts/submit.sh scripts/agentic/run_c.sh --time=00:40:00   # 5 examples first
```

Every knob is an environment variable, so a variant is a submit line rather than
an edit. The two best configurations at the time of writing:

```bash
# B — 0.464, 9.9 s/example
ARMS=Bgated LEGACY=1 CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3 \
  VARIANT=b-best scripts/submit.sh scripts/baselines/run_b.sh

# C — 0.462, 1.8 tool calls, 15.4 s/example
UNIFIED=1 FINAL_PASS=1 LEGACY=1 PREVIEW=8 TEXT_GATE=-1 \
  CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3 \
  VARIANT=c-best scripts/submit.sh scripts/agentic/run_c.sh
```

`LEGACY=1` is the answer prompt without a length constraint. It looks like it
only games BEM, which rewards longer answers, and that is what we assumed for
weeks — but it also holds the correct answer more often (36.0% against 33.3%),
because gold answers are ranges and lists that a four-word reply cannot carry.

`TEXT_GATE` is the one piece of orchestration that pays: below that score on the
best pooled paragraph, a second retrieval round runs. Asking the model to make
the same call from the prompt instead is worth 1.1 points less.

Run C twice before believing a difference: two runs of the identical
configuration came out 0.454 and 0.462, so its noise is ~0.8 points against
~0.1 for B. `src/ablation/compare_runs.py` gives the paired interval, which is
the only honest way to read gaps that small.

Anything after the script path is passed through to `sbatch`.

`run_abc.sh` runs the three settings against **one vLLM server in one job**, so
they differ only in method — same weights, same endpoint, same prompt format,
same examples. Run it once per model to get the reference table, then iterate
with `agentic/run_c.sh`: A and B never touch the agent's code, so re-running them
per variant would spend two thirds of a job reproducing numbers you already have.
`run_sweep.sh` answers the other question, C across model sizes, and skips any
model already scored so re-submitting only fills the gaps.

Jobs serve their model on a port derived from the SLURM job id and verify
`/v1/models` before running: `localhost` is per node, and with a fixed port a
colleague's vLLM on the same node silently answers your requests — which once
cost a full run of empty predictions.

Use `SMOKE=1` when switching model. Bad image preprocessing degrades answers
without raising anything, and five examples show it immediately.

## Working on variants

`sbatch` copies the `.sh` at submission but reads the `.py` **when the job
starts**. Editing a strategy while an earlier job is queued therefore changes
what that job runs, and loses the version you meant to test. `submit.sh` avoids
both by copying `src/`, `scripts/` and the scorer into `runs/<id>/` and pointing
the job there — 460 KB per run, and the working tree is yours again the moment
the job is submitted.

```
runs/20260815-104401-lookup-first/
├── src/  scripts/  evqa_eval/   the exact code that ran
├── RUN_INFO                     run id, variant, commit, branch, dirty count
├── uncommitted.diff             changes not in git at submit time
└── JOB_ID
```

So the loop is: edit → `VARIANT=name scripts/submit.sh …` → edit again for the
next idea, without waiting. Each run keeps its own code, its own
`logs/<run-id>/` and its own `outputs/abc/<model>/<run-id>/`, so attempts never
overwrite each other.

Every predictions file also gets a `.meta.json` recording the run id, variant,
commit, dirty flag and the exact command. Under `submit.sh` those come from the
snapshot's `RUN_INFO`, captured at **submit** time — reading git when the job
starts would report whatever the tree holds by then, which is exactly what
changes while a job waits in the queue.

Git branches are still the right tool once a variant *wins* and you want to keep
it; `runs/` tells you which one that was.

## Knowledge base and name index (SQLite)

`encyclopedic_kb_wiki.db` (~20 GB) holds the ~2.0M articles in `articles`
(url, title) and their text in `paragraphs`, plus two derived tables that turn an
entity *name* into an article:

- **`aliases`** (`alias`, `url`) — normalised surface forms of every title
  (accents and punctuation stripped, `(disambiguation)` suffix dropped, leading
  `the` removed), indexed for exact lookup.
- **`titles_fts`** — an FTS5 index over the title tokens, used only as a fuzzy
  fallback (BM25 shortlist, re-scored by token overlap).

Matching is pure string matching, deliberately **not** embeddings: the EVA-CLIP
text tower is misaligned with the image index (0% recall@50 even when given the
ground-truth title), and exact matching is anyway more precise than similarity
for near-identical names. Ceiling on the test subset: **83.5%** of articles are
reachable by name, against **46.7%** for the image embedding.

The article set never changes, so the tables are precomputed once. A full build
produces them automatically; `--index-only` rebuilds just them on an existing KB
(seconds, instead of re-ingesting the 15 GB source JSON):

```bash
sbatch scripts/setup/build_kb_sqlite.sh                 # full KB, name index included
sbatch scripts/setup/build_kb_sqlite.sh --index-only    # only the name tables
```

Use it through `KnowledgeBase`, the single point of access to the DB:

```python
kb = KnowledgeBase("/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.db")
kb.lookup_articles("Northern cardinal")   # [{'title': ..., 'wiki_url': ..., 'match': 'exact'}]
kb.get_paragraphs_by_url(wiki_url=...)    # the article text
```

Normalisation at build time and at query time must stay identical — a mismatch
fails silently — so both use the helpers in `src/retrieval/knowledge_base.py`.

## Layout

```
src/
  paths.py prompts.py llm.py runner.py provenance.py   shared by every setting
  vlm/        arg_parser  dataset  run_inference       A and B
  agent/      prompts messages tools rag run metrics   C
              run_inference
  retrieval/  retriever  knowledge_base  reranker  build_kb_sqlite
  ablation/   parameter sweeps, and compare_runs for confidence intervals

  Probes live in <pkg>/experiments/ and are never on the inference path:
  agent/experiments/      naming_probe       can the model name what it sees
  retrieval/experiments/  compute_recall*    recall@k of the image index, and query variants
                          compute_recall_text  what the full-text channel adds
                          compare_rerankers  which cross-encoder surfaces the answer
                          probe_gate         is the cross-encoder score a usable trigger
                          analyse_pool       image, name or text: where the article came from
                          prime_df_cache     term frequencies for the text channel
scripts/
  submit.sh                     snapshot + submit — the way to launch
  run_abc.sh                    A, B and C for one model — the reference table
  lib/vllm.sh                   serving lifecycle shared by every experiment

  baselines/run_b.sh            B, B+, Btext, Bgated — the arms in one job
  baselines/run_score.sh        score predictions a killed job never got to
  baselines/run_ablation_cross.sh   top-k / top-n grid over B

  agentic/run_c.sh              C — one variant per run, all knobs via env
  agentic/run_ablation_c.sh     one-at-a-time sweep of C's parameters
  agentic/run_naming_crops.sh   can the model name the subject: guesses, prompts, crops
  agentic/run_sweep.sh          C across model sizes
  agentic/run_smoke.sh          does a remote model work at all

  retrieval/run_recall*.sh      recall@k of the image index, and its variants
  retrieval/run_recall_text.sh  what the full-text channel adds
  retrieval/run_probe_gate.sh   is the cross-encoder score a usable trigger
  retrieval/run_compare_rerankers.sh   which cross-encoder surfaces the answer
  retrieval/run_prime_df.sh     fill the term-frequency cache the text channel needs

  setup/build_kb_sqlite.sh      build the KB, and its name index
  setup/build_paragraph_index.sh   add the full-text index over paragraph text
  setup/download_model.sh  setup/setup_vllm_venv.sh
```

`runner.run_batch` owns the loop, the thread pool, the resume and the writing;
each script supplies only its `predict`. `llm.chat_model` is the only place a
chat client is built. Keeping those single means A, B and C cannot drift apart
without someone noticing.

## Outputs

Under `outputs/` (git-ignored):

- `abc/<model>/<run-id>/` — the reference table: `predictions_A|B|C.jsonl`,
  `results_A|B|C.json`, and a `.meta.json` per prediction file
- `agentic/<model>/<run-id>/` — one agent variant: `predictions_C.jsonl`,
  `predictions_C.metrics.json`, `results_C.json`
- `agentic/sweep/` — the model sweep, per tag: `naming_<tag>.jsonl` (predicted
  entity name and whether it resolved), `predictions_<tag>.jsonl`,
  `predictions_<tag>.metrics.json` (tool usage, miss rate per tool, entry tool,
  call sequences) and `results_<tag>.json`
- `final_test/`, `ablation*/`, `retrieval/` — the phase-B record, kept as-is
- `_archive/` — results from pipelines that no longer exist

Logs go to `logs/<run-id>/`, one directory per run: the SLURM `.out`/`.err` and
the vLLM server log together.
