# Reproducing the paper

One row per number we report. The rule this file exists to enforce:

> A number in the paper means its code path stays.

Every command below was run and checked against the paper. Deviations are given
as measured, not as claims.

## How to run anything

Each script under `scripts/` is a SLURM job and runs under plain `sbatch`. The
commands below use `scripts/submit.sh`, our wrapper, which copies the tree into
`runs/<id>/` first so that a job already in the queue keeps running the code it
was submitted with. It is a convenience, not a requirement.

What matters for reading results is the other half: every
`outputs/**/*.meta.json` records the exact command, every knob, the node and the
GPU that produced the file beside it. The timings in the paper come from there.

Paired tests come from the per-example scores written beside every result:

```bash
uv run python src/ablation/compare_runs.py <ref>.scores.jsonl <other>.scores.jsonl
```

## The table

`✓` means the command was re-run and the numbers matched. Deterministic commands
are expected to match exactly; anything that passes through a generation is
expected to agree within about a point, which is the run-to-run spread we measure
(sd 0.43 agentic, 0.82 static over five repetitions).

| paper | command | reproduces |
|---|---|---|
| Tab. 1, image recall | `scripts/retrieval/run_recall.sh` | ✓ exact: 12.9 / 22.9 / 27.8 / 34.4 / 40.6 / 46.7 |
| Tab. 1, text recall | `scripts/retrieval/run_recall_text.sh` | ✓ exact: 9.9 / 14.8 / 17.3 / 20.5 / 23.1 / 26.0 |
| Tab. 1, name recall | `GUESSES="1 2 3 5 8" scripts/agentic/run_naming_crops.sh` | ✓ within 0.6: 11.6 / 16.1 / 16.9 / 18.8 / 18.6 |
| §3.1, CLIP text tower 0% | `scripts/retrieval/run_recall_queries.sh` | ✓ `wikipedia_title.text` resolves 0.0% against 52.4% for the image |
| §5.2, top-$k$ sweep | `SPLIT=test scripts/baselines/run_ablation_cross.sh` | not re-run: measured with Qwen2.5-VL-3B, which the paper states |
| §3.1, name ceiling 83.3% | `kb.lookup_articles(gold_title)` over the 1000 test titles | ✓ |
| Tab. 2, static rows | `ARMS="A B Bplus Btext Bgated" scripts/baselines/run_b.sh` | ✓ within 1.0: 28.1 / 41.5 / 42.4 / 43.8 / 47.6 |
| Tab. 2, Full | `scripts/run_abc.sh` | ✓ 49.3 exact; Gate only 47.2 against 47.8 |
| Tab. 2, Agent only | `TEXT_GATE= scripts/run_abc.sh` | ✓ five repetitions, mean $\Delta$ +5.08 |
| §5.3, oracle 65.1 | `ORACLE=1 ARMS="Bgated" scripts/baselines/run_b.sh` | ✓ 64.9 |
| Tab. 3 (a),(b) | `SWEEP=<tag> scripts/agentic/run_ablation_c.sh` | ✓ every sign and ordering; see caveats |
| Tab. 3 (c) | `SWEEP=<tag> scripts/agentic/run_gate_sweep.sh` | ✓ smaller effects; see caveats |
| §3.4, gate score distribution | `scripts/retrieval/run_probe_gate.sh` | ✓ medians +1.04 / −2.28 |
| every CI and p | `src/ablation/compare_runs.py` | ✓ exact |
| Fig. 3 | `src/ablation/make_figures.py` | ✓ exact |

`scripts/setup/*` build the knowledge base and the serving venv. They are
documented but not re-verified: both already exist and take hours to rebuild.

## Paths

Cluster-specific locations sit in two places and nowhere else:

- `src/paths.py` reads `EVQA_DATA` for the dataset, knowledge base and visual
  index, defaulting to our cluster's `/work/cvcs2026/encyclopedic`.
- The launchers export `HF_HOME` for the model cache and expect the vLLM
  environment at `$HOME/vllm_venv`, built by `scripts/setup/setup_vllm_venv.sh`.

Running elsewhere means setting `EVQA_DATA` and `HF_HOME`; no file needs editing.

## Caveats a reproducer needs

**The knowledge base and the visual index are not redistributable.** The index is
released by ReAG; the knowledge base is built from the Encyclopedic-VQA release
with `scripts/setup/build_kb_sqlite.sh`.

**`run_recall.sh` resumes.** It skips examples already in its output file, so
delete `outputs/retrieval/retrieval_topk50.jsonl` before re-running or it will
verify nothing.

**`run_recall_text.sh` needs its full runtime** (~90 minutes). Its writes are
buffered, so a job killed at its time limit leaves an empty file, not a partial
one.

**`LIMIT` is not a knob** in `run_recall.sh`, `run_recall_text.sh` or
`run_probe_gate.sh`; they always run all 1000 examples.

## Where replication is weaker than the paper

Two claims were measured a second time by an independent sweep.

**The preview ranking** gives +1.7 (p=0.075) on replication against the reported
+2.2 (p=0.018). Same sign, same ordering, still the largest of the three ranking
effects, but the significance rests on one measurement.

**The gate inside the agentic pipeline** gives −0.3 (p=0.780) against −1.1
(p=0.126). Neither is distinguishable from zero, so the paper's reading holds.
This does not touch the headline gate result, +4.0 (p=0.001) in the *static*
pipeline, which reproduces at 47.6 against 47.8.

Choosing τ=−1 over τ=0 was right: the first sweep put τ=0 nominally ahead by
+0.1, the replication puts it 0.6 behind.

The four-tool and final-pass effects replicate strongly: −5.3 and −5.8 against
−5.5 and −5.1, p<0.001 in both sweeps.