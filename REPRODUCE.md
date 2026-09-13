# Reproducing the numbers in the paper

One row per number we report. The rule this table exists to enforce:

> **A number in the paper means its code path stays.**

Removing a branch makes the line that cites it unreproducible, and the failure is
silent — nothing breaks until someone reruns a table row months later. Anything
*not* reachable from this table is either evidence for a claim we no longer make
(remove it) or an internal helper (keep it, undocumented).

All runs go through `scripts/submit.sh`, which snapshots the whole tree into
`runs/<id>/` and points `CODE_DIR` at that snapshot. A past run therefore
reproduces from **its own** code, not from HEAD, which is what makes renaming
flags safe. Every `outputs/**/*.meta.json` records the exact command line.

Unless stated otherwise: Qwen3-VL-8B-Instruct, the same 1000-example
Encyclopedic-VQA test subset in the same order, greedy decoding.

---

## Verification status

Checked against the raw output files on 2026-09-12. `OK` means recomputed from
the file and matching the paper to the printed precision.

| cluster | status |
|---|---|
| Image index recall@k | OK |
| Name channel, guess count | OK |
| Name channel, prompt wording | OK |
| Text channel recall@k | **corrected** — the paper had 15.9/19.7/23.0; the file gives 9.9/17.3/20.5/23.1 |
| Pool composition (union, per-channel) | OK |
| Ablation deltas | OK |
| Ladder deltas | OK (from the same job) |

---

## §4.4 Retrieval channels in isolation — `tab:retrieval`

| number | command | output read |
|---|---|---|
| Image recall@k = 12.9 / 27.8 / 34.4 / 40.6 / 46.7 / 52.4 | `scripts/submit.sh scripts/retrieval/run_recall.sh` | `outputs/retrieval/retrieval_topk100.jsonl` — gold hit iff `wikipedia_url` appears in `candidates[:k][*].wiki_url` |
| Name recall by guess count = 11.8 / 15.7 / 17.5 / 18.7 / 18.8 | `scripts/submit.sh scripts/agentic/run_naming_probe.sh` (`GUESSES` = 1,2,3,5,8) | `outputs/agentic/qwen3vl8b/20260906-185010-naming-sweep/naming_g{1,2,3,5,8}.jsonl` — fraction with `resolved` truthy |
| Name recall by prompt wording = 17.1 diverse / 16.5 plain / 15.3 registers / 14.4 siblings / 13.8 example | same script, `STYLE` varied at `GUESSES=3` | `outputs/agentic/qwen3vl8b/20260906-211403-naming-styles/naming_g3_*.jsonl` |
| Text recall@k = 9.9 / 17.3 / 20.5 / 23.1 | `scripts/submit.sh scripts/retrieval/run_recall_text.sh` | `outputs/retrieval/recall_text.jsonl` — fraction with `text_rank <= k` |
| Union 58.0; naming adds 7.8; text adds 9.1 | falls out of the ladder run below | `pool_Bplus.json` (`name_only`), `pool_Btext.json` (`text_only`, `union`) |

Note these are **channel** recalls, measured standalone. They are not the same
quantity as the pool contributions: the pipeline admits fewer text results than
the standalone top-20, so `pool_Btext.json` shows `text` = 10.5 against the
channel's 23.1. Do not conflate them.

## §3.4 Gate figure — `fig:gate`

| number | command | output read |
|---|---|---|
| Score distributions, medians +1.89 / −0.66, 45.9% vs 19.4% below τ | `scripts/submit.sh scripts/retrieval/run_probe_gate.sh` | `outputs/retrieval/gate.jsonl` (`top_score`, `image_hit`) |
| The figure itself | `uv run python src/ablation/make_figures.py --out outputs/paper/figures` | reads the file above; writes PDF + PNG |

## §4.5 Static progression — `tab:ladder`, `fig:coverage`

| number | command | output read |
|---|---|---|
| A / B / Bplus / Btext / Bgated = 28.2 / 41.6 / 43.4 / 43.8 / 47.8 | `scripts/submit.sh scripts/baselines/run_b.sh` with `ARMS="A B Bplus Btext Bgated"` | `outputs/baselines/qwen3vl8b/20260910-210504-b-ladder-best/results_*.json` |
| Coverage 41.2 / 49.0 / 58.0 / 56.1 | same run | `pool_*.json`, field `percent.union` |
| Paired deltas and p-values | `uv run python src/ablation/compare_runs.py <dir>/results_{A,B,Bplus,Btext,Bgated}.scores.jsonl` | prints BEM, 95% CI, paired delta, exact sign test |
| 104 improved / 100 degraded (Btext vs Bplus); 75 / 31 (Bgated vs Bplus) | same command, the discordant counts in the delta table | |
| The coverage figure | `make_figures.py` (same invocation as above) | reads `results_*.json` + `pool_*.json` |

## §4.6 Static versus agentic — `tab:main_results`

A, B and C against **one** vLLM server in one job: same weights, same endpoint,
same examples in the same order, at equal concurrency, so they differ only in
method. Two runs of one configuration land within ~0.3 points and the gaps we
care about are that size, so anything measured across separate jobs is noise.

```
scripts/submit.sh scripts/run_abc.sh --time=08:00:00
```

`C_CONCURRENCY` defaults to `CONCURRENCY`, which is what makes the latency
column comparable; lower it alone if C runs out of KV cache.

**On a larger model** the weights are split over two cards with tensor
parallelism and a **third card is left for EVA-CLIP and the cross-encoder**.
This is not optional: at `GPU_UTIL=0.90` on two cards vLLM takes 42.6 of the
44.4 GiB and the retriever OOMs the moment B starts, after A has already been
scored — which looks like a mid-job crash rather than a sizing error.

```
MODEL=Qwen/Qwen3-VL-32B-Instruct TAG=qwen3vl32b GPU_UTIL=0.85 \
  TP=2 VLLM_GPU=0,1 RETRIEVER_GPU=2 NEED_GB=70 \
  scripts/submit.sh scripts/run_abc.sh \
    --constraint=gpu_L40S_45G --gres=gpu:3 --time=20:00:00
```

Not the 96 GB RTXPro6000B nodes, even though one card would hold the model: they
are Blackwell (sm_120) and the project's torch is pinned to cu124, which stops at
sm_90. vLLM serves there once `scripts/setup/warm_flashinfer.sh` has run, but
EVA-CLIP dies with "no kernel image is available for execution on the device" as
soon as B or C starts. L40S is Ada, where the stack is proven.

`NEED_GB` is node-local **disk** for staging the weights, not GPU memory. If the
node has less, the run falls back to serving from `$HF_HOME` over BEEGFS, which
works but is slow under contention.

## §4.7 Ablation — `tab:ablation`

One job, one server, nine single-variable configurations against a fixed
reference.

```
SWEEP=20260912 scripts/submit.sh scripts/agentic/run_ablation_c.sh
```

Outputs land in `outputs/agentic/qwen3vl8b/ablation-$SWEEP/`. `OUT_DIR` is keyed
by `SWEEP` so a rerun never silently reuses an old result.

| row | config name in the script | BEM | delta |
|---|---|---|---|
| Reference | `reference` | 47.2 | — |
| Preview: RRF | `preview_rrf` | 49.4 | +2.2 (p=0.018) |
| Preview: BM25→CE | `preview_bm25` | 49.3 | +2.1 (p=0.001) |
| Final: RRF | `final_rrf` | 45.6 | −1.6 (p=0.167) |
| Final: BM25→CE | `final_bm25` | 46.4 | −0.8 (p=0.484) |
| Mid-loop: CE alone | `tools_bge` | 47.0 | −0.2 (p=0.839) |
| Four-tool interface | `fourtools` | 41.7 | −5.5 (p<0.001) |
| No final pass | `nofinalpass` | 42.1 | −5.1 (p<0.001) |
| RRF everywhere | `all_rrf` | 46.2 | −1.0 (p=0.463) |

Deltas from
`uv run python src/ablation/compare_runs.py <dir>/results_reference.scores.jsonl <dir>/results_<config>.scores.jsonl`.

Tool-use statistics quoted in the four-tool paragraph (1.64 vs 2.44 calls per
example, 975/1000 first-tool, 3 vs 18 failed episodes) come from
`predictions_<config>.metrics.json`, **not** from the per-example prediction
records — `forced` and `gate_below` are run-level and are not serialised per
example.

## §3.2 / §3.4 Agent behaviour

| number | source |
|---|---|
| `search_by_image` chosen first on 98.6–99.0% of examples | `first_tool_pct` in `predictions_*.metrics.json` across the five C repetitions |
| Gate: 47.6% of examples forced, `avg_forced_calls` 0.48 | `forced_examples_pct`, `avg_forced_calls` in the same files |
| Below-τ observed but not forced: +0.6 to +1.0 pt (two tools), +5.8 (four) | `gate_below_examples_pct` − `forced_examples_pct` |
| Agent stops after one step on 71.8% with the gold article vs 29.9% without | gate probe joined to the C run's step counts |

## Earlier Qwen2.5-VL measurements (recovered from the notebooks, 2026-09-13)

These predate the provenance machinery, so they carry no `meta.json` and the
model attribution comes from a notebook label rather than a recorded command
line. Treat them as indicative, not citable. They are kept because they are the
only record of where the project started, and because the first of them is a
calibration point against published work.

| | BEM | source |
|---|---|---|
| A, no retrieval, Qwen2.5-VL-3B | **0.245** (n=1000) | `outputs/baselines/results_A.json` |
| B, top-20 image + cross-encoder top-20 | 0.401 | `outputs/final_test/results_cross_topK20_rerankN20.json` |
| B, best of the sweep (B5) | 0.403 | `outputs/baselines/results_B5.json` |

Two caveats that the notebook recorded and that matter if these are quoted:

- The 0.401 uses the free-form prompt with no answer-format block. Under the
  shared answer format the comparable figure is **0.359**. The notebook's own
  warning: *"The historical 0.401 is not a target."*
- ReAG reports Qwen2.5-VL-3B zero-shot at **21.9** on E-VQA single-hop against
  our 24.5. On n=1000 with p≈0.22 the 95% sampling interval is about ±2.7, so
  the two are consistent rather than identical --- which is the most we can ask
  of two samples of the same split. Jobs 105429/105430 re-measure this with
  provenance on both 3B and 7B.

## §4.2 Qwen2.5-VL history

Cited, not reproduced. The runs predate the current tree and the middlewares
they used (`force_first_tool`, `remind_original_question`,
`require_tool_before_answer`) were removed in `ad8c6dc`. They survive only in
the `runs/` snapshots of the jobs that used them. We report these as
observations that motivated the model choice, with no table.

---

## What this table licenses us to remove

Paths reachable from no row above, with the evidence:

| what | evidence |
|---|---|
| `main.py` | first-day scratch; prints the first two KB entries |
| `scripts/bash.sh` | a personal `srun --pty bash` one-liner |
| `.vscode/settings.json` | editor config |
| `--no-rerank`, `--retrieval-strategy`, `--rrf-k` | 0 uses across all runs with a complete result — **re-verify before deleting** |
| `scripts/agentic/run_smoke.sh` | OpenRouter smoke; OpenRouter is diagnostic only and no reported number rests on it |

Paths that look dead and **stay**, because a row above needs them:

- the four-tool interface (`TOOL_SET=legacy`) — it is the `fourtools` row
- `--final-pass` off — it is the `nofinalpass` row
- every strategy in `fusion.STRATEGIES` — each is an ablation row
- the ten modules under `src/retrieval/experiments/` — each produces a different
  channel number; they are the retrieval half of the paper
