---
title: HemVetoRatio Calculation
---

# Setup

Set `do_HemVetoStudy` in `configs/parameters/switches.yaml` as `true` (it should false by default) and make sure `do_HemVeto` in `configs/parameters/switches.yaml` is also `true` (it should true by default).

# Skimming

Run stage1 as normal over only `data_*` samples using label `stage1_example_label`.

# Ratio calculation

Run
```
python ./validation/hem_veto/hemveto_calculate_ratio4MC.py --label stage1_example_label
```

It should print the ratio value within the print statement `The proportion of 2018UL data events vetoed....`, suppose that values is `0.x`, then manually set that value over `HemVeto_ratio` config in `configs/parameters/jet.yaml`.

# Cleanup

Go back to `configs/parameters/switches.yaml` and set `do_HemVetoStudy` config back to `false.`
