#!/bin/bash
set -e


label="Run3_nanoAODv15_24Jan2025"

category="ggh"

model="Run3PrelimResultsJan25_2026_NoAnnhilateWgts"



# stage2_save_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/${model}_${category}_recreate1_87SigOct31_2025_newEdgeTarget/stage2_output" 
stage2_save_path="/depot/cms/hmm/yun79/hmm_ntuples/$label/${model}_${category}/stage2_output" 


# years="2016preVFP 2016postVFP 2017 2018"
years="2022preEE 2022postEE 2023 2023BPix 2024"
python determine_score_edge.py -load $stage2_save_path --years ${years}

# despite its name, this plots the AMS values from saved .csv output to pngs
python validation.py