#!/usr/bin/env bash
set -euo pipefail

sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP
sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2

bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2 
bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018 
bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017 
bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP 
bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun11_2026_50nTrialsFoldsAll_Max70bins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP 

# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 3 -k)


# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m compact -k)


# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016 

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP 


# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m compact -k)


# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_20nTrialsFoldsAll_Max40bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun08_2026_50nTrialsFoldsAll/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 


# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 3 -k)

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_21Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 


# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun07_2026_17Bins/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 

#-------------------------------------

# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m compact -k)


# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_Jun05_2026_RamMay2025Binning/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc Run2 

# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 3 -k)


# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun04_2026_50startNBins/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc Run2 

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun03_2026_oldDnnBinning/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2 


# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 3 -k)

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2018 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2017 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016postVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ 2016preVFP 
# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/stage3_datacards_Jun01_2026/score_Run2_NanoV12_forVBFChannel_May15_2026_jetUnc/ Run2 



# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m compact -k)

# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2017" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2017" -m 2p -k)


# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved-Copy1.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_May15_2026_jetUnc -y "2018,2017,2016postVFP,2016preVFP" -m 3 -k)




# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2 

# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ Run2

# bash produce_significance.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May12_2026/score_Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc/ Run2 

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 3 -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 15 -l  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 15 -l  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 15 -l  Run2_NanoV15_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 3 -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m compact -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018" -m compact -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2016postVFP" -m compact -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2016preVFP" -m compact -k)


# VBF stage2 and 3
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k)
# # time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 3 -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017 2016postVFP 2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017 2016postVFP 2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 3 -k)
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2018
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2017
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016postVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ 2016preVFP
# sh produce_combine_cards.sh /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/stage3_datacards_May10_2026/score_Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc/ Run2




# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 2p -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2018 2017 2016postVFP 2016preVFP" -m 3 -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026_jetUnc -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2017" -m compact -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2018" -m compact -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016postVFP" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_forVBFChannel_Apr29_2026 -y "2016preVFP" -m compact -k)

# python run_plotter.py


# cp -r /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2018/ /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output/
# cp -r /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016postVFP/ /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output/
# cp -r /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl/stage1_output/2016preVFP/ /work/projects/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_forVBFChannel_Apr29_2026/stage1_output/

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2018" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2018" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2018" -m compact -k)




# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016postVFP" -m compact -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2016preVFP" -m compact -k)



# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016postVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016postVFP" -m compact -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016preVFP" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2016preVFP" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr24_2026_JetHornPuId_JerStrat3_UpdatedBtagWp_wQgl -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr14_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr14_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr14_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr14_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Apr14_2026_UpdatedBtagWp -y "2017" -m compact -k)

# python run_plotter.py


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr09_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr09_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr09_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Apr09_2026_JetHornPuId_JerStrat3_UpdatedBtagWp -y "2017" -m compact -k)




# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_4XudongTransformer_Ap06_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_4XudongTransformer_Ap06_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_4XudongTransformer_Ap06_2026 -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap04_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap04_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap04_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap04_2026 -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap04_2026_JetHornPuId_JerStrat3 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap04_2026_JetHornPuId_JerStrat3 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap04_2026_JetHornPuId_JerStrat3 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap04_2026_JetHornPuId_JerStrat3 -y "2017" -m compact -k)

# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l  Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3 -y "2017" -m compact -k)


# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap01_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap01_2026 -y "2017" -m 1 -k)
# time(bash stage1_loop_Improved.sh -c configs/datasets/dataset_nanoAODv15_run2.yaml -v 15 -l  Run2_NanoV15_Ap01_2026 -y "2017" -m compact -k)


# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Ap01_2026/stage1_output/2017/compacted/dyTo2Mu_M-100to200_MiNNLO/0/part0.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV15_Ap01_2026/dy_100To200_MiNNLO

# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Ap01_2026/stage1_output/2017/compacted/ggh_powhegPS/0/part0.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV15_Ap01_2026/ggh_powhegPS
    
# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV15_Ap01_2026/stage1_output/2017/compacted/vbf_powheg_dipole/0/part00.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV15_Ap01_2026/vbf_powheg_dipole

# #--------------------------------
# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/stage1_output/2017/compacted/ggh_powhegPS/0/part0.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/ggh_powhegPS

# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/stage1_output/2017/compacted/vbf_powheg_dipole/0/part00.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/vbf_powheg_dipole

# python plotter/plot_real_fake_jets_stack.py \
#     -i /depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/stage1_output/2017/compacted/dy_M-100To200_MiNNLO/0/part0.parquet  \
#     -o validation/compare_real_fake/Run2_NanoV12_Ap01_2026_JetHornPuId_JerStrat3/dy_100To200_MiNNLO