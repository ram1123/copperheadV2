# Introduction

# Technical Details

## Obtaining new BDT Score edges for BDT cateogories

We assume you have run stage2 with all the JEC uncertainties (for signal samples only for ggH channel). This is done by adding `--do_jecUnc` flag when runing `run_stage2.py` i.e.

```
year="2018"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_jecUnc

year="2017"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_jecUnc


year="2016postVFP"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_jecUnc

year="2016preVFP"
sample_l="ggh vbf" 
python run_stage2.py -load $stage2_load_path -save $stage2_save_path --samples $sample_l -cat $category --fraction 1.0 --year $year --model_name $model --do_jecUnc
```


then go to 

```
cd stage2/ggH_datacard/
```

update the label and BDT model that you want to use for the datacard, and then run the script

```
sh run_script.sh
```


# References/Important links
