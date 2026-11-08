# Summary of the autoresearch-mlx session

Branch: `autoresearch/nov6` (HEAD `d128832`). Hardware: Apple Silicon, MLX. Each run is a 5-minute training run.

## Setup
- Read `program.md` and `train.py`, then created branch `autoresearch/nov6` from `main`.
- The data and tokenizer were already in `~/.cache/autoresearch/`.
- `prepare.py` and the dependencies were not touched.
- Python 3.15 upgrade: steps were given only, nothing was changed. A dry-run of the dependency wheels was offered but never requested.

## Phase 1: single-run keep/discard results

The first four rows were on a machine that was swapping heavily.

| # | Experiment | val_bpb | Steps | Status |
|---|---|---|---|---|
| 1 | Baseline (batch 2^16, depth 4, matrix LR 0.04) | 2.2768 | 68 | keep |
| 2 | Batch 2^15, first try | 2.7054 | 4 | discard (invalid, machine swapping) |
| 3 | Batch 2^15, second try | 2.6689 | 10 | discard (invalid, still swapping) |
| 4 | Batch 2^15, clean rerun after closing apps | 2.0830 | 147 | keep |
| 5 | Batch 2^14 (device batch 8) | 1.7386 | 483 | keep |
| 6 | Batch 2^13 (device batch 4) | 1.6787 | 923 | keep |
| 7 | Batch 2^12 (device batch 2) | 1.6987 | 1738 | discard |
| 8 | Matrix LR 0.02 | 1.6578 | 907 | keep (within noise) |
| 9 | Matrix LR 0.01 | 1.5362 | 914 | keep |
| 10 | Matrix LR 0.005 | 1.5476 | 909 | discard |
| 11 | Embedding LR 0.3 | 1.6121 | 911 | discard |
| 12 | Embedding LR 1.0 | 1.5258 | 912 | keep (marginal, within noise) |
| 13 | Unembedding LR 0.008 | 1.6014 | 904 | discard |
| 14 | Unembedding LR 0.002 | 1.5750 | 908 | discard |
| 15 | Weight decay 0.0 | 1.5445 | 910 | discard |
| 16 | Warmdown ratio 0.7 | 1.5476 | 904 | discard |
| 17 | Adam beta1 0.9 | 1.5788 | 902 | discard |
| 18 | Depth 6 (26.3M params) | 1.7750 | 437 | discard |
| 19 | Depth 3 (10.7M params) | 1.6112 | 1061 | discard |
| 20 | Window pattern "L" | 1.5744 | 902 | discard |
| 21 | Scalar LR 0.25 | 1.5697 | 901 | discard |
| 22 | Warmup ratio 0.05 | 1.6104 | 902 | discard |
| 23 | Noise check: unchanged best config rerun | 1.5521 | 904 | logged as keep |

## What worked
- **Smaller batches.** The budget allows only a few dozen optimizer steps at 2^16 tokens, so shrinking the batch gave many more steps. Going from 2^16 to 2^13 took val_bpb from 2.28 to 1.68 and cut memory from 8.3 GB to 3.6 GB.
- **Lower matrix LR.** Dropping it from 0.04 to 0.01 gave another gain (1.658 to 1.536).

## What didn't work
- **Batch 2^12:** more steps (1738), but each step was too noisy.
- **Depth 6:** it halved the step count, which cost more than the extra capacity gained.
- **Depth 3:** too small a model.
- **Every other tweak** was worse than the config it was tested against: embedding and unembedding LRs, weight decay, warmdown, beta1, window pattern, scalar LR and warmup.

## Problems and caveats
- **Swapping.** Runs 2 and 3 were invalid because the machine was swapping, so their numbers say nothing about the idea. Run 4 repeated the experiment after apps were closed.
- **Noise.** `rigor.py` says repeat runs vary by about 0.03. A rerun of the best config gave 1.552 against the recorded 1.526. So the matrix LR 0.02 and embedding LR 1.0 "keeps" are probably noise, and the 1.526 best is probably a lucky draw. The near-misses (weight decay 0.0, warmdown 0.7, matrix LR 0.005) may also be noise. The big discards (depth 6, batch 2^12, beta1 0.9, warmup) are real.
- **Odd training time.** In run 4, `training_seconds` read 398.9 instead of about 300. The cause was not investigated.
- **Lost log rows.** Twice, a reset onto a kept commit dropped results rows committed on top of it: the LR 0.005 row and the unembedding 0.008 row. Both were re-added. Several hashes in `results.tsv` point to commits that were later reset away.
- **`rigor.py`.** `program.md` doesn't mention it. Phase 1 (above) used the plain keep/discard rule; phase 2 (below) used `rigor.py`.

## Phase 2: significance-gated loop with `rigor.py`

`rigor.py` scores `train.py` over repeated runs and keeps a change only if it beats the best mean with at least 95% confidence (bootstrap). A candidate whose first run is no better than the best mean is dropped after that one run. Scores live in `rigor_ledger.jsonl`, which is gitignored.

| Experiment | val_bpb | Verdict |
|---|---|---|
| Baseline (batch 2^13, matrix LR 0.01, embedding LR 1.0) | 1.5512 +/- 0.024 (1.542, 1.528, 1.584) | baseline |
| `mx.compile` on the loss/grad function | 1.4675 (1.475, 1.462, 1.465) | **keep** (P(better) = 1.00) |
| Batch 2^12 (device batch 2) with compile | 1.4917 (1 run) | discard |
| Adam beta2 0.99 | 1.4910 (1 run) | discard |
| Head dim 64 | 1.5929 (1 run) | discard |
| Warmdown ratio 0.3 | 1.4765 (1.466, 1.481, 1.482) | discard (P = 0.06) |
| Compile the AdamW update | not scored | discard (about 5% faster, +8 net lines) |

- **The `mx.compile` win:** it cut step time from about 330 ms to about 255 ms, which gives about 30% more steps in the budget. That is worth about 0.08 val_bpb, far more than any hyperparameter tweak.
- **First attempt crashed:** `mx.compile` flattens its arguments, so the model arrived as a dict (`TypeError: 'dict' object is not callable`). The fix was to compile a lambda that closes over the model, call it as `loss_grad_fn(x, y)`, and build the mask cache eagerly so the graph does not capture tracers. The crash is also in the ledger.
- **The earlier "best" was luck:** the old 1.5258 figure was a lucky single run. Scored over 3 runs, the same config is 1.551.

### Bookkeeping notes
- `rigor.py` does not report peak memory, so the memory column in `results.tsv` is `-` for these rows.
- The hashes in `results.tsv` for discards are `rigor.py` file-content hashes, not git commits. The `mx.compile` row uses a git hash.

## Current state
Branch `autoresearch/nov6` is at `0d2728e`. `train.py` is the kept config: batch 2^13 (device batch 4), matrix LR 0.01, embedding LR 1.0, and the compiled loss/grad function. Its val_bpb is 1.4675 (3-run mean), against the 2.277 baseline.

The loop is paused, not finished. The remaining small hyperparameter tweaks are mostly inside the noise, so bigger ideas are the better use of runs: removing value embeddings, a different logit softcap, or retuning the learning rates for the faster compiled step.
