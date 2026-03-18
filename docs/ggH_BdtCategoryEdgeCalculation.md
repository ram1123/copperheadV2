# Introduction

# Technical Details

## Obtaining new BDT Score edges for BDT cateogories

Run stage2 with full sample list (but dy has to be the M100To200 samples only, otherwise memory overload):
```
sh stage2_sh.sh
```

then go to 

```
cd stage2/ggH/score_edge_generation/
```

update the label and BDT model that you want to use for optimization, and then run the script

```
sh run_script.sh
```


Then update the new BDT edges in the `/stage2/ggH/target_yields.yaml`

# References/Important links
