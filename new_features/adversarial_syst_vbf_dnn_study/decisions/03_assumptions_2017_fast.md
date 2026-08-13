# Assumptions and decisions — `03-adversarial-syst-vbf-dnn-2017-fast`

Every entry is flagged `[measured]` or `[assumption]`. Decisions taken by the
analyst are marked as such; everything else is an agent decision and is open to
being overruled.

---

## D-1 Warm start reuses ONE lambda=0 run for every lambda `[assumption]`

The instruction reads "train dnn at lambda=0 until early stop condition is met,
then train again with lambda=lambda_i until early stopping". Taken per lambda,
that repeats the identical lambda=0 phase N times: same config, same seed, same
fold parquets, same data order. So the sweep trains it **once** and uses it as
the initialiser for every lambda.

This is not only cheaper, it is the better experiment: every lambda then starts
from a *bit-identical* point, so a difference between two lambdas cannot be a
difference in their phase-1 draw. It also makes the lambda=0 reference chain and
the warm-start source the same object, which is what the phrase "re-run lambda=0
to extract the reference" implies.

**Risk if wrong:** if the analyst wanted an independent lambda=0 phase per point
(e.g. to average over phase-1 noise), this under-samples that noise. Re-running
is one training per lambda and needs no code change — drop `--init-from`'s
shared source and give each lambda its own.

## D-2 Only the weights cross the warm-start boundary `[assumption]`

`--init-from` loads `fold<i>/best.pt`'s `model_state`. The optimizer, the
`ReduceLROnPlateau` schedule and the early-stopping counter are all rebuilt.

The instruction says "train **again** ... until early stopping", not "resume".
Carrying the optimizer state would mean the adversarial phase inherits Adam
moments accumulated under a different objective, and carrying the ES counter
could stop phase 2 before it takes a step. A fresh optimizer at the configured
LR is the standard fine-tuning setup.

Consequence to keep in mind: phase 2 restarts at the **full** learning rate, not
at whatever `ReduceLROnPlateau` had decayed it to. The adversarial phase can
therefore move a long way from its initialisation in its first epochs; the
warm start biases where it starts, not how far it can travel.

Covered by `tests/test_warm_start.py` (weights load per fold; a real LR still
trains; a missing checkpoint raises rather than silently starting from random).

## D-3 Stop-gradient consistency term `[analyst decision 2026-08-11]`

Variants A + B with `--adversarial-detach-nominal`, `--adversarial-high-score-cut
2.0`, `--adversarial-consistency-smoothing none`.

The predecessor task diagnosed the undetached form as a confidence-inflation
pressure rather than a decorrelation one: with gradient in both arms,
`BCE(q; p) >= H(p)` makes the optimum `p -> 0` or `p -> 1`, not `q == p`. It
closed with the corrected run stopped at fold 0 of 4, so the corrected
configuration has never produced a significance number. This task is that run,
on an axis cheap enough to also scan lambda around it.

The undetached configuration is **not** re-scanned here. See
`../02-adversarial-syst-vbf-dnn-loss-variants/final-summary.md` §3 and its D-8.

## D-4 Parity calibration is measured, not inherited `[measured]`

Lambda grid = 0.1x / 0.3x / 1.0x of the parity lambda, where parity is where the
penalty contributes as much as the nominal loss: `lambda = nominal_loss /
penalty`. It is measured by a one-epoch probe at lambda = 0, warm-started from
the phase-1 model, so it describes the point where the adversarial phase actually
begins (`scripts/parity_from_log.py`, written to `calibration.json`).

Parity must be re-measured rather than inherited from the predecessor's 0.816:
that number was Run2, cold-start, undetached. Year, warm start and detaching all
move the penalty's natural scale.

## D-5 The datacard stripper removes shape nuisances only `[analyst decision 2026-08-11]`

`scripts/strip_datacard_variations.py` drops the 11 JEC shape rows other than
`Total`, keeping `Total` and `mu_roccor2017`. `lumi2017` (lnN) and the
`autoMCStats` line stay: they are not shape variations and the DNN cannot
decorrelate against them.

Rationale: the training is only ever shown 4 variations, while Stage-3 builds
templates for 13 shape nuisances. Leaving the other 11 in the fit dilutes the
effect being measured with systematics the network was never exposed to.

Two consequences to state plainly when reporting:

1. **These numbers are not comparable to the predecessor tasks' numbers.**
   Different year, different binning, and a much smaller nuisance set. The
   reference for this task is its own lambda = 0 run and nothing else.
2. The stripped card keeps `Total` while dropping `Absolute`, `BBEC1`, `EC2`,
   `FlavorQCD`, `HF`, `RelativeBal`, `RelativeSample` and their year-decorrelated
   twins. In the full card `Total` and the reduced-scheme components coexist,
   which double-counts the JEC uncertainty; stripping to `Total` alone removes
   that double count as a side effect. It is a *different* systematic model, not
   a subset of the same one.

The stripper writes `<card>.full` once and always re-derives from it, so the
untouched card survives and `--restore` returns the directory to what Stage-3
produced.

## D-6 Binning is re-derived per lambda, over a known objection `[analyst decision 2026-08-11]`

`scan_bins_for_dnn.py` chooses edges from its own background estimate, which its
own TODO records as **2-9x off** from what Stage-2 produces. In the predecessor
task that mismatch left a 2016postVFP bin holding 0.00576 background events from
a single DYVBF MC event (`n_eff` = 0.03), contributing `S/sqrt(B)` comparable to
the whole year and inflating the reference by ~7.7%; it was fixed by rebinning
from the Stage-3 templates instead. Re-deriving per lambda reopens that door.

Proceeding as instructed, with a guard: `--min-background-per-bin 0.05` stays
active, and `scripts/check_template_occupancy.py` runs after every Stage-3 and
records per-bin `B` and `n_eff = B^2/var(B)` against the floor the predecessor
settled on (`n_eff >= 10`, `B >= 0.5`). It never aborts a chain — it makes a
recurrence impossible to miss rather than silent. Reports land in
`occupancy_<postfix>.json`.

Second-order effect, worth stating: with per-lambda binning, two lambdas are
compared **at different binnings**. Part of any difference is then the binning,
not the model.

**Analyst decision 2026-08-11, after seeing the first two points:** this is
correct and intended. The analysis is sensitive to the DNN model, so rebinning
is required for each different DNN; a fixed binning would penalise a model whose
score distribution has moved, which is precisely what the adversarial term does.
A fixed-binning cross-check was offered and declined. Each model is judged with
the binning it would actually be deployed with.

**The occupancy problem is separate and remains open.** It is not about
comparability across lambdas but about whether any single point's number is
trustworthy. Measured on the first two chains: `scan_bins_for_dnn.py` chose 27
bins (lambda = 0.0884) and 26 (lambda = 0.2651), and in both cases **10 tail
bins fall below the floor** — together holding ~3.5 background events, several
with `n_eff` between 1.2 and 3, i.e. built from one or two MC events, while
carrying ~1.06 signal events. Barlow-Beeston cannot constrain such bins, so they
contribute `S/sqrt(B)` the fit cannot support, and the absolute pre-fit values
are inflated by an amount not yet quantified.

The root cause is upstream of the binning philosophy: the scanner's
`--min-background-per-bin` is applied in *scan* units, which its own TODO records
as 2-9x off from what Stage-2 produces, so a bin that clears the floor during the
scan can be near-empty in the template. Raising the scan-unit floor, or moving
the guard onto the Stage-3 templates, would fix it. Carried into
`final-summary.md` as a caveat on every number in this task.

## D-7 The binning file is global mutable state `[measured]`

`configs/MVA/VBF/dnn_binning.yaml` is a single repo-level file, read by
`modules/selection.py` **at import time** (`binning = load_dnn_binning()`), and
`scan_bins_for_dnn.py` overwrites it in place.

Two consequences the chain has to handle:

- **Chains cannot overlap.** `chain_2017.sh` takes a directory lock and refuses
  to start rather than scoring one model against another's edges.
- **The repo's working tree is modified by running this task.** The file is
  already dirty on this branch. Each run's edges are archived to
  `binning/dnn_binning_<postfix>.yaml`, and `binning/dnn_binning_ORIGINAL.yaml`
  holds the pre-task content so it can be put back.

## D-8 Dask cluster lifecycle `[assumption]`

The instruction says to shut the cluster down after the bin scan and "proceed in
running stage2" — but Stage-2 *attaches* to an existing gateway cluster
(`modules/dask_utils.get_dask_client` raises `RuntimeError("No Dask Gateway
clusters available")` if none exists) and cannot create one. So the shutdown is
implemented as shutdown **and recreate for Stage-2**, at the same
`worker_cores=2, worker_memory=25`.

That reading also matches the evident intent: the scan runs its own 64-process
*local* client, so holding gateway workers across it is both wasteful and a
memory risk on the same node.

Worker count is 40 (80 cores, ~1 TB) — `dask_cluster_params.txt` fixes the
worker size but not the count; the predecessor used 100 x 1-core/8 GB, and 40 x
2-core/25 GB is a comparable core count with 2017-only inputs to read.

## D-9 Run2-tuned hyperparameters on 2017-only data `[assumption]`

`--optuna-best-json` still points at the Run2 study
(`100Trials_w_VBF_filterFoldAll`). No 2017-only HPO exists and running one would
defeat the purpose of a fast axis. Batch size is left at its tuned value, as
instructed.

This means the 2017 models are **not** optimal 2017 models — they are Run2-tuned
models trained on a quarter of the data. Acceptable because every point in the
sweep, including the lambda = 0 reference, carries the identical handicap, so
the lambda comparison is still internally valid. It is not a basis for any claim
about absolute 2017 sensitivity.

## D-10 The four one-hot year features go constant `[measured]`

`configs/dnn_run2_vbf.yaml` carries `year_2018`/`year_2017`/`year_2016postVFP`/
`year_2016preVFP`. Trained on 2017 alone, one is constant 1 and three are
constant 0. `preprocess_dnn.py` guards the scaler (`std = np.where(std < 1e-6,
1.0, std)`), so this is numerically safe, and after centring all four arrive at
the network as zeros — four dead inputs.

Left in place deliberately: removing them would change the feature list, and
therefore the architecture, away from the configuration Stage-2 expects. The
cost is four unused input weights.

## D-11 Degenerate QvG columns are KEPT `[assumption, inherited]`

`--variation-drop-degenerate` is **off**, matching the predecessor's Phase-1
choice: this is what Stage-2 actually feeds the DNN at inference, so training and
inference agree by construction.

Neither setting is physically correct — REV-001 in the phase-1 task remains open:
Stage-1 should fill `jet1/jet2_btagUParTAK4QvG_<JEC>_up/down` with the QvG of
whichever jet leads under that variation, and a JEC shift changes the leading jet
in ~20% of events. That fix is upstream of this task and independent of it.
