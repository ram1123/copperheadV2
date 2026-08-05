# Stage1 Input Compatibility

Generated: `2026-08-03T21:43:44Z`

- Stage1 root: `/work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_METphiXYCorr_Jul26_2026/stage1_output/2017`
- Dataset subdir: `compacted`
- Resolved samples: `21`
- Excluded entries: `6`

## Resolved Samples

| sample | class | group | files | inspected_files | inspected_rows | missing_required |
| --- | --- | --- | --- | --- | --- | --- |
| ggh_powhegPS | ggH | signal/GGH | 10 | 1 | 63718 | none |
| vbf_powheg_dipole | VBF | signal/VBF | 20 | 1 | 60546 | none |
| dyTo2Mu_M-50_MiNNLO | bkg | background/DY | 1551 | 1 | 32811 | none |
| dyTo2Mu_M-100to200_MiNNLO | bkg | background/DY | 34 | 1 | 40243 | none |
| ttjets_dl | bkg | background/TT | 1037 | 1 | 6397 | none |
| ttjets_sl | bkg | background/TT | 206 | 1 | 927 | none |
| st_tW_top | bkg | background/ST | 42 | 1 | 761 | none |
| st_tW_antitop | bkg | background/ST | 42 | 1 | 785 | none |
| st_tchannel_top | bkg | background/ST | 23 | 1 | 953 | none |
| st_tchannel_antitop | bkg | background/ST | 13 | 1 | 955 | none |
| ewk_zlljj | bkg | background/EWK | 3 | 1 | 30681 | none |
| ww_2l2nu | bkg | background/VV | 12 | 1 | 24518 | none |
| wz_1l1nu2q | bkg | background/VV | 1 | 1 | 199 | none |
| wz_2l2q | bkg | background/VV | 87 | 1 | 27492 | none |
| wz_3lnu | bkg | background/VV | 15 | 1 | 25629 | none |
| zz_2l2q | bkg | background/VV | 99 | 1 | 20620 | none |
| zz_4l | bkg | background/VV | 207 | 1 | 25763 | none |
| www | bkg | background/VVV | 4 | 1 | 26231 | none |
| wwz | bkg | background/VVV | 7 | 1 | 27280 | none |
| wzz | bkg | background/VVV | 10 | 1 | 26238 | none |
| zzz | bkg | background/VVV | 15 | 1 | 27059 | none |

## Exclusions and Overlap Guards

| yaml_name | source_group | reason |
| --- | --- | --- |
| dy_VBF_filter | background/DYVBF | excluded overlap sample for this inclusive-DY smoke study |
| data_B | data | data excluded from supervised MC training |
| data_C | data | data excluded from supervised MC training |
| data_D | data | data excluded from supervised MC training |
| data_E | data | data excluded from supervised MC training |
| data_F | data | data excluded from supervised MC training |

## Final Ordered Feature List

- Object token order: `mu1, mu2, jet1, jet2, dimuon`
- Object feature order: `physObj_ln_pt, physObj_ln_e, physObj_eta, physObj_sin_phi, physObj_cos_phi, physObj_id, physObj_pt_4v, physObj_eta_4v, physObj_phi_4v, physObj_energy_4v`
- Global feature order: `global_ln_htsoft2, global_ln_htsoft5, global_MET_ln_pt, global_MET_ln_e, global_MET_sin_phi, global_MET_cos_phi, global_n_jets`

## Token, Padding, and Alias Rules

- Five object tokens are built in the fixed order `mu1`, `mu2`, `jet1`, `jet2`, `dimuon`.
- Padding is indicated by `physObj_pt_4v <= 0`; padded object feature rows are zeroed and masked from attention.
- Pairwise attention uses raw `pt`, `eta`, `phi`, and `energy` slots from each object token.
- Continuous object slots and all global features are standardized from the training split only during training.
- Non-finite feature values are replaced with zero after bounded log/energy transformations.
- `PuppiMET_sumEt` is used as the Stage1 proxy for `global_MET_ln_e`.

## Inspected Weight Summary

| sample | class | events | negative | neg_frac | signed_sum | abs_sum | nan | inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ggh_powhegPS | ggH | 63718 | 93 | 0.00146 | 27.025232 | 27.103661 | 0 | 0 |
| vbf_powheg_dipole | VBF | 60546 | 47 | 0.000776 | 0.959066 | 0.9606 | 0 | 0 |
| dyTo2Mu_M-50_MiNNLO | bkg | 32811 | 1688 | 0.051446 | 17337.574219 | 19321.738281 | 0 | 0 |
| dyTo2Mu_M-100to200_MiNNLO | bkg | 40243 | 2007 | 0.049872 | 38045.03125 | 42240.859375 | 0 | 0 |
| ttjets_dl | bkg | 6397 | 20 | 0.003126 | 204.853973 | 206.130798 | 0 | 0 |
| ttjets_sl | bkg | 927 | 2 | 0.002157 | 37.092525 | 37.215912 | 0 | 0 |
| st_tW_top | bkg | 761 | 0 | 0.0 | 173.009949 | 173.009949 | 0 | 0 |
| st_tW_antitop | bkg | 785 | 0 | 0.0 | 177.676208 | 177.676208 | 0 | 0 |
| st_tchannel_top | bkg | 953 | 32 | 0.033578 | 39.064468 | 41.907299 | 0 | 0 |
| st_tchannel_antitop | bkg | 955 | 27 | 0.028272 | 43.802834 | 46.252926 | 0 | 0 |
| ewk_zlljj | bkg | 30681 | 0 | 0.0 | 8247.459961 | 8247.459961 | 0 | 0 |
| ww_2l2nu | bkg | 24518 | 30 | 0.001224 | 1530.095459 | 1533.875977 | 0 | 0 |
| wz_1l1nu2q | bkg | 199 | 37 | 0.18593 | 10.520578 | 20.216084 | 0 | 0 |
| wz_2l2q | bkg | 27492 | 5512 | 0.200495 | 230.546844 | 385.664551 | 0 | 0 |
| wz_3lnu | bkg | 25629 | 5173 | 0.201842 | 459.233185 | 771.788513 | 0 | 0 |
| zz_2l2q | bkg | 20620 | 3820 | 0.185257 | 99.418289 | 158.535034 | 0 | 0 |
| zz_4l | bkg | 25763 | 75 | 0.002911 | 13.70245 | 13.78164 | 0 | 0 |
| www | bkg | 26231 | 1219 | 0.046472 | 21.354418 | 23.505239 | 0 | 0 |
| wwz | bkg | 27280 | 1218 | 0.044648 | 17.392559 | 19.068249 | 0 | 0 |
| wzz | bkg | 26238 | 1223 | 0.046612 | 5.538704 | 6.105301 | 0 | 0 |
| zzz | bkg | 27059 | 1450 | 0.053587 | 1.528945 | 1.713313 | 0 | 0 |

## Compatibility Result

All inspected files contained the required transformer columns after alias resolution.

DY samples with `separate_wgt_zpt` will use `wgt_nominal / separate_wgt_zpt`; other samples use `wgt_nominal`.
The available Copperhead MET energy-like branch is `PuppiMET_sumEt`, used as the global MET energy proxy.
