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
| Disable value embeddings | 1.6369 (1 run) | discard |
| Value embeddings on all 4 layers (instead of alternating) | 1.4886 (1 run) | discard |
| Remove logit softcap | 1.5457 (1 run) | discard |
| `x0_lambdas` init 0.3 (from 0.1) | 1.4710 (1 run) | discard |
| bf16 logits (drop fp32 cast) | 1.4777 (1 run) | discard (no speedup, 268 ms/step) |

- **The `mx.compile` win:** it cut step time from about 330 ms to about 255 ms, which gives about 30% more steps in the budget. That is worth about 0.08 val_bpb, far more than any hyperparameter tweak.
- **First attempt crashed:** `mx.compile` flattens its arguments, so the model arrived as a dict (`TypeError: 'dict' object is not callable`). The fix was to compile a lambda that closes over the model, call it as `loss_grad_fn(x, y)`, and build the mask cache eagerly so the graph does not capture tracers. The crash is also in the ledger.
- **The earlier "best" was luck:** the old 1.5258 figure was a lucky single run. Scored over 3 runs, the same config is 1.551.

### Bookkeeping notes
- `rigor.py` does not report peak memory, so the memory column in `results.tsv` is `-` for these rows.
- The hashes in `results.tsv` for discards are `rigor.py` file-content hashes, not git commits. The `mx.compile` row uses a git hash.

### Bigger ideas, second batch
After the `mx.compile` win, I tested larger changes instead of learning-rate tweaks. None beat the best:
- **Value embeddings matter.** Removing them cost about 0.17 val_bpb (1.637 against 1.467) with no step-time change. Adding them to all layers did not help (1.489), so alternating layers is a good setting.
- **The logit softcap matters.** Removing it cost about 0.08 (1.546).
- **Near-misses are noise.** `x0_lambdas` init 0.3 (1.471) and bf16 logits (1.478) are within the roughly 0.01-0.02 spread between single runs of the best config, and they were dropped on the first-run rule, not proven worse. bf16 logits gave no speedup, so there was no upside.
- **Speed is the big lever.** The only large win in phase 2 was the 30% step-time cut from `mx.compile`. The compiled AdamW gave only about 5%. The 32768-token vocabulary logits are probably the remaining cost per step, and `prepare.py` fixes the vocabulary.

## Phase 3: Muon optimizer port

`train.py` said "AdamW only. Muon port is future work." I ported Muon for the 2-D block matrices (attention and MLP weights, plus the small value-embedding gates). Embeddings, `lm_head` and the scalars stay on AdamW. The optimizer class is now `MuonAdamW`.

- **The Muon update:** Nesterov momentum, then a 5-step quintic Newton-Schulz orthogonalization (`newton_schulz`, compiled with `mx.compile`), scaled by sqrt(max(1, rows/cols)), with the same decoupled weight decay as before.
- **Cost:** step time is unchanged (about 255-275 ms), and the change is about 46 added lines.
- **Settings:** `MATRIX_LR` is now the Muon LR, and `MUON_MOMENTUM` is a new hyperparameter.

| Experiment | val_bpb | Verdict |
|---|---|---|
| Muon, LR 0.02, momentum 0.95 | 1.4162 (1.417, 1.413, 1.418) | **keep** (-0.051, P = 1.00) |
| Muon LR 0.04 | 1.4328 (1 run) | discard |
| Muon LR 0.01 | 1.4451 (1 run) | discard |
| Weight decay 0.0 under Muon | 1.4348 (1 run) | discard |
| Muon momentum 0.9 | 1.4122 (1.412, 1.413, 1.412) | **keep** (-0.004, P = 1.00) |

- **Muon is the second-biggest win** after the batch-size changes. It gave -0.051 at the same step time.
- **LR 0.02 is the sweet spot** of 0.01, 0.02 and 0.04. Weight decay still helps under Muon.
- **The momentum 0.9 gain is small.** It cleared the 95% gate because the runs were very tight, but a bootstrap over 3 runs can be overconfident on a -0.004 difference. Treat it as probable, not certain.
- **Run-to-run noise dropped.** With Muon and `mx.compile`, the three runs of a config agree within about 0.005, against about 0.024 for the phase 2 baseline. That makes smaller effects measurable.

## Phase 4: working through the untried ideas

| Experiment | val_bpb | Verdict |
|---|---|---|
| Muon momentum 0.85 | 1.4037 (1.402, 1.400, 1.409) | **keep** (-0.008, P = 1.00) |
| Muon momentum 0.8 | 1.4102 (1 run) | discard |
| Batch 2^12 (device batch 2) under Muon | 1.4571 (1 run) | discard |
| Embedding LR 0.6 under Muon | 1.4050 (1 run) | discard (essentially a tie) |
| Muon LR 0.03 at momentum 0.85 | 1.4158 (1 run) | discard |
| MLP width 3x | 1.4165 (1 run) | discard (no speedup: 283 ms/step) |

- **Momentum:** 0.95 -> 0.9 -> 0.85 each helped, and 0.8 was worse, so 0.85 is the optimum. With only about 1,200 steps, a shorter momentum memory pays off.
- **Muon LR:** 0.02 is still best after the momentum change (0.03 was worse).
- **Batch size:** 2^13 is still right. 2^12 lost by about 0.05 under Muon too.
- **Embedding LR:** 0.6 and 1.0 are within noise of each other. 1.0 was kept because the change didn't clear the gate.
- **MLP width:** 3x was worse and gave no speedup. The MLP isn't the bottleneck, because the 32768-token vocabulary logits dominate step time.
- **Skipped on purpose:**
  - *Depth 5:* with `ASPECT_RATIO` 64 it also widens the model from 256 to 384. That roughly halves the step count, which is what sank depth 6 in phase 1.
  - *Gradient clipping:* Muon normalizes the matrix updates and Adam is close to scale-invariant, so global-norm clipping would barely change the updates.

## Phase 5: overnight run (profiling, block-local attention, schedule)

### Profiling
I profiled the training step on this machine, an Apple M4 (base). The scratch scripts are not in the repo.
- **Vocabulary head:** the matmuls against the 32768-token vocabulary are compute-bound at about 3.2 TFLOPS. Rewriting the loss (manual logsumexp, chunking, bf16 softcap, no softcap) saved at most about 5%. fp16 matmuls are no faster than bf16. There is no cheap win left here.
- **Attention:** about 25 ms per layer (forward and backward), roughly 100 ms of each ~255 ms step. The three short-window "S" layers computed the full 2048 x 2048 attention and masked half of it away, so the window saved nothing. Passing `mask="causal"` instead of an array mask made no difference.

### Block-local attention
`block_local_attention` splits the sequence into blocks the size of the window. Each query block attends only to its own block and the previous one, through `scaled_dot_product_attention` with a small (window x 2 window) mask. The mask's shape picks the path: 4-D means block-local, 2-D means full attention. `create_block_local_mask` replaces `create_sliding_window_mask`.
- **Correctness:** the output matches masked full attention exactly (max difference 0.0) for window 256.
- **Speed:** each short-window layer is about 2.7x faster at window 256, and steps went from about 255 ms to about 225-235 ms.
- **Size:** 43 added and 9 removed lines in `train.py`.

| Experiment | val_bpb | Verdict |
|---|---|---|
| Depth 6 at width 256 (`ASPECT_RATIO` 42), before block attention | 1.4546 (1 run) | discard (370 ms/step) |
| Block-local attention, short window 256 (was 1024) | 1.3767 (1.375, 1.377, 1.378) | **keep** (-0.027) |
| Short window 128 | 1.3751 (1.375, 1.375, 1.375) | **keep** (-0.0016) |
| Depth 5 at width 256, pattern SSSSL | 1.3848 (1 run) | discard (251 ms/step) |
| Warmdown 0.6 | 1.3730 | **keep** (-0.002) |
| Warmdown 0.7 | 1.3709 | **keep** (-0.002) |
| Warmdown 0.85 | 1.3700 | **keep** (-0.0009) |
| Warmdown 1.0 | 1.3697 | discard (tie, P = 0.79) |
| Muon LR 0.025 | 1.3709 (1 run) | discard |

- **Block-local attention was the big win of the night.** It made the step faster and the shorter window cost nothing in quality.
- **Noise is now tiny.** The three runs of a config agree within about 0.001, so gains of 0.001-0.002 are real. The small warmdown keeps are trustworthy for that reason, though each is small.
- **Deeper models still lose.** Even at width 256 and with cheap attention, each extra layer costs more steps than it gains in quality.

## Current state
`train.py` is the kept config:
- depth 4, width 256, pattern SSSL with short window 128 (block-local) and one full-attention layer
- batch 2^13 (device batch 4)
- Muon for the block matrices (LR 0.02, momentum 0.85, weight decay 0.2)
- AdamW for the rest (embedding LR 1.0)
- warmdown 0.85
- the compiled loss/grad function

Its val_bpb is 1.3700 (3-run mean), against the 2.277 baseline.

| Milestone | val_bpb |
|---|---|
| Baseline (single run) | 2.277 |
| End of phase 1 (3-run rescore) | 1.551 |
| `mx.compile` | 1.467 |
| Muon | 1.416 |
| Muon momentum 0.85 | 1.404 |
| Block-local attention (window 256, then 128) | 1.375 |
| Warmdown 0.85 | 1.370 |

The loop is still running overnight. Next ideas in the queue: unembedding and embedding LR retunes, weight decay, logit softcap value, and Muon momentum under the new schedule.
