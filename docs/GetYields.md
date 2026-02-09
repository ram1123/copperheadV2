---
title: Get Yields
---

# Get yield

For this there is script named [get_yields.py](../scripts/get_yields.py) in the scripts directory. This script can be used to get the yields from the stage1 output parquet files.

This uses the selection defined in the `modules/selection.py` file.


Also, there is another script named [compare_yield_csv.py](../scripts/compare_yield_csv.py) which can be used to compare the yields obtained and saved as csv files using the script [get_yields.py](../scripts/get_yields.py). This script will compare the yields saved in two csv files and print the differences. This can be useful to check if the yields obtained from different runs are consistent or if there are any discrepancies.

This will print the summary like:

```bash
====================================================================================================
YEAR: 2022postEE
====================================================================================================
           sample category      region       year  raw_events_A  raw_events_B  raw_events_diff_pct     yield_A     yield_B  yield_diff_pct
            data*      vbf      h-peak 2022postEE          1326          1307            -1.432881 1326.000000 1307.000000       -1.432881
 dyTo2L_M-50_incl      vbf      h-peak 2022postEE         11471         10821            -5.666463 2252.661380 2046.038848       -9.172374
            ewk_*      vbf      h-peak 2022postEE          3835          3799            -0.938722   51.667347   51.237093       -0.832739
     ggh_powhegPS      vbf      h-peak 2022postEE         10953         10278            -6.162695    6.210484    5.763953       -7.189949
         ttjets_*      vbf      h-peak 2022postEE         15632         15207            -2.718782  250.612726  242.916114       -3.071118
vbf_powheg_dipole      vbf      h-peak 2022postEE        326802        321480            -1.628509    6.550290    6.436373       -1.739113
             w*_*      vbf      h-peak 2022postEE          6491          6111            -5.854260   44.101458   41.194833       -6.590770
             zz_*      vbf      h-peak 2022postEE          5232          4892            -6.498471    5.439032    5.075334       -6.686812
            data*      vbf h-sidebands 2022postEE          1139          1120            -1.668130 1139.000000 1120.000000       -1.668130
 dyTo2L_M-50_incl      vbf h-sidebands 2022postEE         10064          9522            -5.385533 1840.600305 1681.217659       -8.659275
            ewk_*      vbf h-sidebands 2022postEE          3278          3244            -1.037218   44.612385   44.102603       -1.142693
     ggh_powhegPS      vbf h-sidebands 2022postEE           182           169            -7.142857    0.107269    0.098133       -8.517135
         ttjets_*      vbf h-sidebands 2022postEE         13901         13532            -2.654485  222.928150  216.068470       -3.077081
vbf_powheg_dipole      vbf h-sidebands 2022postEE          4611          4536            -1.626545    0.094564    0.092808       -1.857152
             w*_*      vbf h-sidebands 2022postEE          5711          5398            -5.480651   38.706532   36.228572       -6.401917
             zz_*      vbf h-sidebands 2022postEE          4533          4248            -6.287227    4.616150    4.339529       -5.992463

====================================================================================================
```

