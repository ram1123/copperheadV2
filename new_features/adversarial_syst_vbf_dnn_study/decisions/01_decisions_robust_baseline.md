# Decision log — `01-adversarial-syst-robust-vbf-dnn`

Running record of every decision that shapes the result, who made it, and why.
Physics/scope decisions belong to the human; the rest are operational choices made
by the agent and are listed so they can be overridden.

---

## A. Human decisions (2026-08-10)

Mirrored into `task.json.constraints`.

| # | Decision | Consequence |
|---|---|---|
| **H-1** (REV-001) | The constant `-1` JEC-varied QvG branches: **run both ways and compare**. | Two sweep-phase input sets are built — `Run2_advSweep_keepDegenerate_Aug10_2026` (matches Stage-2 inference) and `Run2_advSweep_dropDegenerate_Aug10_2026`. No lambda is selected until the comparison exists. |
| **H-2** (REV-002) | The pre-existing unweighted training loss: **proceed knowingly**, do not fix here. | Every term of the new objective is an *unweighted* mean BCE. Section 3.4's "which weights enter the adversarial terms" answer is "none". Must appear in `final-summary.md` under known issues. |
| **H-3** (REV-003) | Production **granted**, but **baselines first**. | If the three no-adversarial seeds do not reproduce Run2 = 1.52839 within `sigma_noise`, stop and report — do not sweep lambda. |

| **H-4** (REV-007) | Early stopping and best-checkpoint selection **keep monitoring nominal `val_auc_weighted`**, unchanged from the baseline, for comparability — **but make note of it**. | No code or config change. The known consequence stands: an adversarial run can stop, or select a checkpoint, while `train_adv_penalty` is still falling, so the reported model may be less decorrelated than the objective was driving toward. Every review report and `final-summary.md` must state this and quote the per-epoch penalty *at the selected epoch* for each lambda, so the cost of the choice is visible rather than implicit. |
| **H-5** (REV-009) | **Acceptance floor lowered from >5% to >=2%**, provided the gain is above the noise band. The **>=3-iteration floor still applies** — do not stop early even if 2% is reached sooner. | The +5% target (Run2 pre-fit > **1.6048**) remains the aim; the *acceptance* threshold is now >=2% (Run2 pre-fit > **1.5590**). Between 2% and 5% is acceptable only if the gain exceeds `sigma_noise` **and** comes from a full-26-variation run. Below 2%, or within noise, is not acceptable at any iteration. This is an **explicit human redefinition of the gate**, recorded as such — not a Reviewer silently reviewing against a different bar, which `agents/reviewer.md` forbids. |

All reviewer questions are now answered. Nothing is blocking.

### Amended gate, for quick reference

| | Run2 pre-fit | note |
|---|---|---|
| reference | 1.52839 | section 2, re-read from `significance.txt` |
| **acceptance floor (+2%)** | **> 1.5590** | human-set, must also exceed `sigma_noise` |
| original target (+5%) | > 1.6048 | still the aim |
| stat-only ceiling | 1.6903 | 10.6% total headroom; +2% consumes ~19% of all systematic impact, +5% would consume ~47% |

## B. Agent operational decisions

| # | Decision | Rationale |
|---|---|---|
| **A-1** | Dask cluster `cms.b30273b8bbcc467a9bab5093465ff9e2` on the **k8s** gateway, `worker_cores=1`, `worker_memory=8 GiB`, `n_workers=100`, pixi project `/cvmfs/cms-af.opensciencegrid.org/paf/pixi/copperheadV2` env `default`. | `dask_cluster_params.txt` says `2/25`; the task constraint overrides to `1/8` and the constraint wins. k8s rather than Slurm because the workers must see `/work`. 100 workers chosen as a middle ground against the 200-400 used elsewhere in the repo on Slurm. No stale cluster existed. |
| **A-2** | Never raise `worker_memory` to fix a worker death; lower `--files-per-chunk` instead and report it. | Explicit task constraint. |
| **A-3** | Added `--files-per-chunk` rather than hardcoding a smaller chunk for augmented runs. | Chunking bounds only the transient — every chunk's frame is retained until the final concat either way — so a blind shrink buys round-trips, not safety. Default 50 nominal / 25 augmented. |
| **A-4** | Run the baseline trainings (GPU) and the augmented preprocessing (Dask) **concurrently**. | Disjoint resources; the alternative is serialising ~3.5 h of GPU work behind many hours of Dask work for no benefit. |
| **A-5** | Baseline seeds: config default (12345), 20260810, 777001. | Seed 1 is the config default so it anchors exactly on the reference training (confirmed bitwise identical — see C-1). Seeds 2 and 3 are arbitrary and fixed for reproducibility. |
| **A-6** | Defer the **full**-variation drop-degenerate input set until the sweep comparison decides. | ~21 GB per full set. H-1 asks for the comparison at the sweep phase; building both full sets before knowing the answer writes data nobody has asked for. |
| **A-7** | Tag scheme: `Run2_adv{Sweep,Full}_{keep,drop}Degenerate_Aug10_2026` for inputs, `trained_adv{Baseline,Lambda}_<id>_Aug10_2026` for models, matching `<save_postfix>` for Stage-1. Each run gets a unique tag; the drivers refuse to reuse one. | Task constraint: "Each lambda uses a unique dated_training_tag and dated_save_postfix." |
| **A-8** | The full-variation phase will pass `--adversarial-variation-chunk`. | The GPU is a **MIG 1g.5gb slice** (4864 MiB, 14 SMs), not a whole A100. Nominal training uses ~188 MiB; a 27x wider forward lands at ~5 GB, at or over the ceiling. Gradient equivalence of the chunked path is unit-tested. |
| **A-9** | Scale the Dask cluster to 0 workers between phases rather than holding 100 idle, and scale back up before each Dask step. | The Stage-1 chain is hours away from the preprocessing finishing; idle workers are pure waste on a shared facility. |
| **A-10** | Coarse lambda grid: **{0.003, 0.03, 0.3}**, plus `lambda = 0` **with** `--use_adversarial` as the section-4.2 consistency check. | Not guessed. Measured on the smoke run: the sweep-phase penalty starts ~7.1 and settles ~1.6, against a nominal loss of ~0.11-0.5. Parity between the two terms is therefore around `lambda ~ 0.06`, so the grid spans two orders of magnitude bracketing it. Refined from Reviewer feedback in later iterations. |
| **A-11** | Sweep the three lambdas on the **keep**-degenerate inputs, then re-run only the best lambda on the **drop**-degenerate inputs. | Answers H-1's comparison at the same lambda while costing 4 full workflows instead of 8. |
| **A-12** | Run baseline seed 1's Stage-1/2/3 + combine chain rather than reusing the reference `significance.txt`. | Its training is bitwise identical to the reference model, so the chain *must* return 1.52839 exactly. Any deviation localises a bug in the Stage-1/2/3 or combine invocation rather than the training — a sharp check on the workflow commands, worth one chain. |
| **A-13** | On discovering C-7: **abort the running drop-degenerate pass**, fix the detector to a modal-share test at 0.99 with a nominal-side guard, add two regression tests, and restart passes 2-3. Do **not** re-run the completed keep pass. | The drop pass had run for ~30 s and written nothing, so aborting cost nothing; letting it finish would have produced the contaminated comparison set that H-1 exists to avoid. The keep pass is not re-run because it *keeps* every column by construction — its manifest's under-reported list is a documentation defect, not a data one, and re-running would mean deleting 3.4 GB (a destructive op needing approval) or stranding it under a `_v2` tag. Instead `scripts/check_degenerate_columns.py` regenerates the list read-only into `degenerate_report_<tag>.json`, so all three input sets are comparable on equal terms. Detector chosen O(n) via the median (a value holding >50% of entries is necessarily the median), not an O(n log n) `np.unique` count, because it runs over ~3M rows x 372 columns. |

## C. Findings that changed the plan

| # | Finding | Effect |
|---|---|---|
| **C-1** | `trained_advBaseline_seedCfg12345_Aug10_2026/fold0/best.pt` is **bitwise identical** (6/6 tensors, `torch.equal`) to the pre-existing `trained_best_optuna_100Trials_w_oneHotEncodeYear_w_MiNNLO_DY/fold0/best.pt`, which produced the section-2 reference at commit `115c8db` on 2026-07-20. | AC3 ("switch-off is unchanged") is proven at **production scale** against the real reference model, not just at unit scale. Closes the main iteration-001 limitation. |
| **C-2** | The GPU is a MIG 1g.5gb slice; CPU time is ~3.1 cores against 1 GPU slice, so the bottleneck is data handling and the two per-epoch `evaluate()` passes, not the forward pass. | Drives A-8. Also means the measured 1.26x/1.97x adversarial slowdown is *not* GPU-bound and should not be "optimised" by shrinking the variation set. |
| **C-3** | Early stopping fires around epoch 25 of 100, so a baseline fold takes ~17 min and a full 4-fold training ~70 min. | Three baselines ~3.5 h, not the ~1.5 days first estimated. |
| **C-4** | `get_dask_worker_count` reported `0/0` while the scheduler reported 100 connected workers. | Trust the scheduler directly for worker counts; the helper's Prometheus scrape lags. |
| **C-5** | VOMS proxy has 181 h left, and Stage-1 inputs are on `/work` rather than xrootd. | Proxy expiry is not a risk for this task. |
| **C-6** | The `combine` pixi environment resolves `combineTool.py`, `combineCards.py` and `text2workspace.py`. Note `pixi run -e combine bash -lc ...` resets `PATH` via the login profile — use `bash -c`. | Section 5.4 will work; avoid `-l` in wrapper invocations. |
| **C-7** | **The degeneracy detector was too strict and was corrected mid-run.** The original test was `n_unique <= 1`. On the real inputs 11 of the 48 full-set QvG columns (and 1 of the 4 sweep-set ones) reach `n_unique` in the hundreds or thousands while still being >99.4% sentinel — the extra values come from the missing/non-finite fallback handing a few rows their real nominal value back. A uniqueness test therefore leaves them in the "degenerate dropped" set and hollows out the H-1 comparison the user asked for. | See A-13. |

| **C-8** | **Stage-2 is not bit-reproducible, even from a bitwise-identical model.** Re-running Stage-2 with seed 1 (whose weights are byte-identical to the reference model) gives 47 of 68 histogram files bitwise identical and 21 differing by <= 4e-5 relative. Every sample's **integral is identical to 0.00e+00 relative** — only the distribution across bins moves. | See below. |

### C-8 in detail — and why it is useful rather than alarming

Per-sample check on 2018 (all 17 samples): integral relative difference `0.00e+00`
for every one, including the two that differ per-bin:

| sample | integral new | integral ref | integral rel. diff | max bin rel. diff |
|---|---|---|---|---|
| `dy_VBF_filter` | 93230.2 | 93230.2 | 0.00e+00 | 7.4e-06 |
| `vbf_powheg_dipole` | 252.587 | 252.587 | 0.00e+00 | 1.6e-05 |
| the other 15 | — | — | 0.00e+00 | **0.00e+00** |

Conserved yield with redistributed bins is the signature of **bin migration**: a
few events sit within float32 rounding of a bin edge, and DNN scoring on CPU is
not bit-reproducible across runs — float32 GEMM is non-associative and its
blocking depends on batch size, which depends on the Dask partition boundaries,
which depend on worker count and file chunking. Those events land one bin over.
It shows up only in `dy_VBF_filter` and `vbf_powheg_dipole` because those are the
samples with real population in the high-score region where the edges are dense.

Two consequences:

1. **The section-5.2b reproduction test must be judged on significance with a
   tolerance, never bitwise.** That is what `sigma_noise` is for; nothing changes
   procedurally, but "identical" was never an achievable bar downstream of
   training.
2. **A free decomposition of the noise.** Baseline seed 1's *model* is bitwise
   identical to the reference, so the gap between its final Run2 significance and
   1.52839 is a pure measurement of Stage-2 + Stage-3 + combine non-determinism,
   with zero training-seed contribution. Seeds 2 and 3 then add the training-seed
   term on top. So `sigma_noise` can be reported as an irreducible pipeline floor
   plus a seed term, rather than as one opaque number — which matters when
   deciding whether a 2% gain "exceeds the noise".

Expected magnitude: a 1.6e-5 relative shift in the signal template should move the
significance in the fifth decimal place, i.e. 1.52839 +/- ~0.00002. If seed 1's
chain returns something materially different from that, the cause is in the
Stage-1/2/3 or combine invocation, not the training — which is precisely why A-12
runs this chain instead of reusing the reference file.

| **C-9** | **STOP (section 5.2b). The reference Run2 = 1.52839 is inflated by a single nearly-empty background bin, and the resulting seed-to-seed noise (2.25%) exceeds the human-set 2% acceptance floor.** | See below. Blocks the lambda sweep; needs a human decision. |

### C-9 in detail — the section-5.2b stop condition has triggered

Baseline points so far (third still running):

| seed | Run2 pre-fit | vs reference |
|---|---|---|
| Cfg12345 (model bitwise identical to the reference) | 1.52839 | +0.00% |
| 20260810 | 1.4797 | **-3.19%** |
| 777001 | *pending* | |

`sigma_noise` from two points = **0.0344 = 2.25% of the reference**, already larger
than the 2% floor the human set for acceptance. Per-year, the swing is
concentrated entirely in one place: 2016postVFP moves -17.0% while 2018, 2017 and
2016preVFP move by +0.5%, -1.7% and +0.8%.

**Root cause, isolated.** In the reference/seed-1 templates, 2016postVFP bin 19
has a total background of **0.00576 events**, and all of it is a *single*
`DYVBF` MC event — `DY`, `EWK`, `TT+ST`, `VV` and `VVV` are all exactly 0:

```
bin 19, 2016postVFP:  B_total = 0.00576201   S = 0.06   S/sqrt(B) = 0.790
   DY 0    DYVBF 0.00576201    EWK 0    TT+ST 0    VV 0    VVV 0
```

That one bin contributes S/sqrt(B) = 0.79, comparable to the *entire year's*
significance. It is an empty-MC fluctuation, not sensitivity.

**Proof that this is the whole effect.** Asimov S/sqrt(B) in quadrature:

| | all bins | dropping bins with B < 0.1 |
|---|---|---|
| seed 1 (= reference), Run2 | 1.8771 | **1.7025** |
| seed 2, Run2 | 1.7192 | **1.7022** |
| seed 1, 2016postVFP | 0.9896 | 0.5954 |
| seed 2, 2016postVFP | 0.5905 | 0.5389 |

With all bins the two seeds differ by 9.2%. Excluding bins with B < 0.1 they agree
to **0.02%**. Essentially the entire measured "seed noise" is one nearly-empty bin.

**Why it happened.** `configs/MVA/VBF/dnn_binning.yaml` says in its own header:
`HAND-EDITED 2026-08-10 -- these edges were set by hand, NOT by a scan.` The
occupancy guards that exist for exactly this purpose in
`MVA_training/VBF_run3/scan_bins_for_dnn.py` (`min_background_per_bin`,
`min_total_events_per_bin`, `enforce_min_per_year`) were therefore never applied
to the active binning. This is also a *known open issue*: repo HEAD is
`a46bae7 "add enforce min per year. However, we still see some 2016postVFP bins
that lack bkg events, leading to crazy high significance for that year only.
Addresses ram112#202"`.

**Why this blocks the task rather than being a side note.**

1. The gate's denominator (1.52839) is partly an artefact, so "+2% over the
   reference" is not a well-defined physics target.
2. `sigma_noise` (2.25%) exceeds the 2% acceptance floor, so **no result could be
   shown to clear the bar**, however good the adversarial term turns out to be.
3. The adversarial term acts on the DNN score distribution, which is exactly what
   determines whether a bin lands nearly empty — so it would *interact* with the
   artefact, and any apparent gain or loss could be attributable to bin
   occupancy rather than to decorrelation.

**Not resolvable inside this task.** `task.json.constraints` carries "Do not modify
Stage-1 physics selection, datacard generation, or binning configs", so the fix
(rebin with a background-occupancy floor) is out of scope by construction. Escalated
to the human.

### C-9 resolution — H-6, and the rebin

**H-6 (human, 2026-08-10):** lift the binning constraint and re-derive the binning
with an occupancy floor, then redo the baselines against it. `task.json` updated:
the blanket "do not modify ... binning configs" constraint is replaced by one
scoped to `configs/MVA/VBF/dnn_binning.yaml` only (Stage-1 selection and datacard
generation stay out of scope), `max_iterations` raised 6 -> 8 to pay for the
detour.

**C-10 — the complete section-5.2b measurement on the OLD binning.** All three
points, for the record:

| seed | Run2 pre-fit | vs reference |
|---|---|---|
| Cfg12345 (model bitwise identical to the reference) | 1.52839 | +0.000% |
| 20260810 | 1.4797 | -3.186% |
| 777001 | 1.48933 | -2.556% |

mean 1.49914, `sigma_noise` = 0.0258 (**1.687%**), full spread 0.0487, mean
**-1.914%** below the reference. **Read this honestly:** the reference is the
*maximum* of the three and sits 1.13 sigma above the mean, and the only point that
lands on it is the one whose model is the reference model by construction. My
`compute_sigma_noise.py` returns `baseline_reproduces_reference: True`, but that
verdict is weak here — with seed 1 pinned to the reference, the test "reference
lies within [min, max]" is close to tautological. The two genuinely independent
seeds both land ~3% low.

**C-11 — why the rebin was NOT done with `scan_bins_for_dnn.py`.** The obvious
move is to re-run the repo's scanner with a larger `--min-background-per-bin`.
That would have repeated the original mistake. The scanner's own TODO block (next
to `bkg_globs`) documents that the background it sees disagrees with Stage-2 by
**2-9x**, lists four candidate causes, and states the consequence explicitly:
*"bins that comfortably clear the guards here come out (nearly) empty in the
stage2 h-peak plots, worst in 2016postVFP."* Its guard is a knob in scan units, not
a guarantee about the fit. So the binning was instead derived from the **Stage-3
template ROOT files** — the exact yields the fit consumes — via
`scripts/rebin_from_templates.py`. Merging bins is only summing, so candidate
binnings were evaluated on the existing baseline templates without re-running
anything.

**C-12 — the rebin, and its effect.** Occupancy metric is effective MC entries
`n_eff = B^2 / var(B)` per year (what `autoMCStats` actually responds to), floors
`n_eff >= 10` and `B >= 0.5`, evaluated as the worst case over all three seeds.
The pathological bin scored `n_eff = 0.03`.

The fix is a **single merge**: old bins 16-20 collapse into one (1.869 -> 7.354).
Everything below is untouched, so low-score resolution is preserved. 21 -> 17 bins.

| floor | bins | seed spread of Asimov Z |
|---|---|---|
| current (none) | 21 | **3.51%** |
| n_eff>=5, B>=0.2 | 18 | 1.12% |
| **n_eff>=10, B>=0.5 (chosen)** | **17** | **0.11%** |
| n_eff>=20, B>=1.0 | 13 | 0.34% |
| n_eff>=30, B>=2.0 | 11 | 0.41% |

Seed-to-seed spread collapses **3.51% -> 0.11%**, a factor of ~32, which is what
makes a 2% effect measurable at all. Asimov Z drops 1.69 -> 1.52, i.e. the
inflation disappears — as it should, it was never sensitivity. Chosen floor is the
sweet spot: minimal merging *and* the best stability; tighter floors cost four more
bins and get slightly worse.

**Consequence for the gate.** The section-2 reference 1.52839 was computed with the
old binning and is inflated. It is no longer a valid denominator. The +2% floor and
+5% target must be re-based on the rebinned baseline before any lambda result is
judged; this is recorded as a constraint in `task.json`. The previous binning is
preserved at `dnn_binning_PREVIOUS_handEdited21bins.yaml`.

**C-13 — the Gateway cluster is reaped when idle.** The first rebinned chain launch
failed in 9 s with `RuntimeError: No Dask Gateway clusters available`: the cluster
had been culled during the idle gap after the baseline chains. All three chains
aborted cleanly at Stage-2 with nothing written (verified: no `*rebin17*` output
directories existed afterwards), which is the abort-on-Stage-2-failure behaviour
working as intended. A new cluster was created with identical settings and
`run_workflow_chain.sh` gained a pre-flight worker-count check so a reaped cluster
is caught in seconds with a distinct message rather than inside Stage-2.

### C-14 — the rebinned band, and the re-based gate (section 5.2b, closed)

| seed | Run2 pre-fit (17-bin) | Run2 pre-fit (old 21-bin) |
|---|---|---|
| Cfg12345 | 1.37746 | 1.52839 |
| 20260810 | 1.36091 | 1.4797 |
| 777001 | 1.37443 | 1.48933 |

| | old 21-bin | rebinned 17-bin |
|---|---|---|
| mean | 1.49914 | **1.37093** |
| `sigma_noise` | 0.0258 (**1.687%**) | 0.00881 (**0.643%**) |
| full spread | 0.0487 | 0.0166 |
| a +2% gain is | 1.2 sigma | **3.1 sigma** |

`sigma_noise` falls by a factor of **2.6**, and the human-set 2% floor moves from
*inside* the noise to 3.1 sigma clear of it. That is what makes the rest of the
task meaningful. Gate (`gate.json`, read automatically by
`extract_significance.py`): reference **1.37093**, +2% floor **1.3984**, +5% target
**1.4395**, stat-only 1.5239.

Note the Asimov proxy predicted a 0.11% seed spread but the real chains give
~0.6%. The proxy is nominal-template `S/sqrt(B)` only; the real pre-fit
significance also profiles the JEC/`mu_roccor` shape nuisances and the per-bin
`autoMCStats` Barlow-Beeston nuisances, whose constraints themselves move with bin
occupancy. The proxy is therefore a *lower bound* on spread -- fine for ranking
binning candidates, which is all it was used for, but the band must come from the
real chains.

| **A-14** | The sweep phase is baselined on an `--use_adversarial`-**off** run against the *augmented* data directory, not on the original-fold baselines. | Original rationale (**since shown to be wrong**, see C-15): the augmented preprocessing used a different background chunk size, which I expected to change the per-fold concat order and hence each scaler's weighted mean/std. Retained rationale: it is the run that the section-4.2 `lambda=0` check must be bit-identical *to* on the same data directory. |

| **C-15** | **A-14's premise was wrong, and measurement said so.** The augmented fold parquets are bit-identical to the original ones in every shared column, and the OFF training is bit-identical to the original-dir baseline. | See below. |

### C-15 in detail — a flagged risk that measurement retired

| check | result |
|---|---|
| OFF (augmented dir) vs baseline seed 1 (original dir), fold0 `best.pt` | **0 / 6 tensors differ** |
| fold-0 parquet row order (`event` column) | identical |
| shared numeric columns differing | **0 / 39** |

The chunk-size change does not perturb anything: chunks are contiguous slices of
the same ordered file list, so `df_total` comes out in the same row order whatever
the chunk size, the weighted mean/std are unchanged, and the scaled features are
bit-identical. The concern was reasonable to act on before it could be measured --
it would have silently contaminated every lambda comparison had it been real --
but it was not real, and it should not be carried forward as though it were.

Consequence worth keeping: because OFF is bit-identical to rebinned baseline seed
1, its chain must return **1.37746** to within the ~1e-5 Stage-2 bin-migration
noise (C-8). Any larger deviation localises a fault in the chain, not the training.
Cost of the check: one training (62 min) plus one chain (~18 min), and it still
serves the section-4.2 requirement.

### C-16 — the sweep phase degrades significance monotonically

| lambda | pre-fit | vs ref | stat-only | vs ref | syst headroom | sigma |
|---|---|---|---|---|---|---|
| baseline mean | 1.37093 | — | 1.52348 | — | 11.13% | — |
| off / 0 | 1.37746 | +0.48% | 1.52449 | +0.07% | 10.67% | +0.74 |
| 0.003 | 1.35899 | -0.87% | 1.51377 | -0.64% | 11.39% | -1.36 |
| 0.03 | 1.30085 | **-5.11%** | 1.45704 | -4.36% | **12.01%** | **-7.95** |

Two things make this a real negative result rather than noise. The trend is
monotonic with no sign of an interior optimum, and the **systematic headroom
rises** (11.13% -> 12.01%) -- the term is not reducing the systematic impact at
all, so the pre-fit loss is not a robustness trade.

Against the analyst's criterion (pre-fit UP while stat-only DOWN, section H-5
discussion), every point reads "both down -- cost without benefit": raw
discrimination is being paid away and nothing is bought.

The penalty *is* being minimised as designed -- absolute level at the selected
epoch falls 3.616 (lambda=0) -> 3.516 (0.003) -> 3.232 (0.03), i.e. -10.6% -- so
the machinery works. It is the physics of what it optimises that is wrong.

**Not flattening.** The hypothesis recorded in `assumptions.md` §1.2 (inherited
label smoothing biasing variation predictions toward 0.5) predicts scores
collapsing to mid-range. Measured on the 2018 templates, the opposite happens:
signal moves *up* (mean bin 7.868 -> 8.267, top-3 fraction 24.95% -> 30.78%) but
background moves further (top-3 yield 23.9 -> 38.9, +63% against signal's +23%),
so S/B in the top three bins falls 0.1786 -> 0.1352, a 24% loss of separation.
The prediction was wrong and is recorded as such.

**Leading explanation: the degenerate QvG columns (REV-001) contaminate the sweep.**
The sweep's four variations are `Total_up/down` + `mu_roccor_up/down`, and the two
`Total` ones carry the constant `-1` `jet1/jet2_btagUParTAK4QvG` columns whose fake
shift is ~1.0-1.4 sigma against 0.03-0.09 sigma for every genuine feature. The
consistency term is therefore dominated by an artefact, and the cheapest way for
the network to satisfy it is to stop using the QvG features -- precisely a loss of
discrimination with no systematic benefit. This is the failure mode REV-001
predicted before any of it ran.

**Test queued (human decision, 2026-08-10):** let lambda=0.3 finish (it completes
the AC5 three-lambda requirement), then re-run lambda=0.03 on the
drop-degenerate inputs. Those differ from the keep inputs in exactly the four QvG
columns and in nothing else (verified: identical shape, identical event order, 4
columns differ, each equal to its nominal counterpart bit-for-bit), so it is a
controlled A/B on this single hypothesis at the lambda where the effect is largest.
Driver: `scripts/run_drop_degenerate_test.sh`.

**Second candidate explanation, not yet tested.** The sweep decorrelates against
`Total` and `mu_roccor`, which are 2 of the 13 shape nuisances in the datacards,
while paying the discrimination cost against all of them. That alone could make the
reduced sweep set an invalid proxy -- exactly the situation section 6.3 anticipates
("the sweep-phase optimum fails to reproduce at full variation count"). If the
drop-degenerate test does not recover the loss, this is the next thing to separate.

### H-7 — the analyst's fallback ladder for the loss (2026-08-10)

If the drop-degenerate A/B does **not** recover the loss (i.e. the QvG artefact was
not the root cause), change the loss itself, in this order:

**Variant A — consistency only.** Drop the variation-vs-label term and the factor
of 2:

```text
new_loss = original_loss(pred_nominal, label)
         + lambda * sum_i original_loss(pred_i, pred_nominal)
```

Rationale: `original_loss(pred_i, label)` pulls the variation predictions toward
the *hard label*, which is a discrimination objective, not a decorrelation one. It
is plausibly the term doing the damage -- it is a third of the penalty weight and
it fights the nominal task rather than aligning variations with nominal.
Flag: `--adversarial-consistency-only`.

**Variant B — consistency only, restricted to the high-score region.** As A, but
the consistency term runs only over events with

```text
high_score_filter = arctanh(pred_nominal) > 2      # i.e. pred_nominal > 0.96403
```

Rationale: the significance is generated in the top score bins; decorrelating
across the bulk of the distribution spends discrimination where it cannot pay
back. Flag: `--adversarial-high-score-cut 2.0`.

Both implemented and unit-tested ahead of need (20 tests pass). Notes:

- The default formula is **unchanged** and re-asserted by a dedicated test, so
  nothing already measured is invalidated.
- The cut is implemented as `p_nominal > tanh(cut)` rather than
  `arctanh(p_nominal) > cut`: mathematically identical, but `arctanh` diverges
  exactly where the cut selects. Equivalence is unit-tested over 5000 random
  points; the two can only disagree for `p` within one float32 ULP of the
  threshold, where both candidates round to the same float32 and the difference is
  not representable at the precision the model runs at.
- An empty selection in a batch contributes exactly 0, not NaN (tested).
- Masking changes the effective lambda scale (the penalty is a mean over far fewer
  events), so lambda must be re-scanned when Variant B is enabled -- carrying over
  the Variant A optimum would be wrong.

### C-7 in detail

Measured separation on the real inputs — this is why the threshold is not a tuned knob:

| group | modal share of the varied column |
|---|---|
| the 48 JEC-varied QvG columns | **0.9941 – 1.0000** |
| all 324 other varied columns | **<= 0.0022** |
| nominal QvG itself (guard side) | <= 0.5723 |

Any threshold in `[0.90, 0.99]` flags exactly 48/48 QvG and 0/324 genuine columns.

Verified on the completed 4-year sweep set: **4/4** QvG columns degenerate, against
**3** found by the old test. The one it missed,
`jet2_btagUParTAK4QvG_nominal__var__Total_down`, has `n_unique = 3053` and a modal
share of 0.9955, with mean `|delta|` vs nominal of 0.96 sigma — i.e. it is exactly
the artefact the whole H-1 comparison is about.

Confirmed again on the full 4-year, 26-variation set: **48/48** QvG columns flagged
(2 jets x 24 JEC variations), out of 372 augmented columns. The old test would have
found 37 and let 11 through. Both sweep sets and the full set are therefore
consistent, and the drop-variant comparison is uncontaminated.

Verified A/B on the two sweep sets: 1,544,010 rows x 83 columns each, identical
column order and `event` alignment, **exactly 4 columns differ**, and they are
exactly the 4 flagged ones. In the drop set each equals its nominal feature
bit-for-bit, so it contributes precisely zero to the consistency term.
