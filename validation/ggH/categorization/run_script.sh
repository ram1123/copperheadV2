#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# sample_l="data ggh vbf dy ewk tt ww wz zz" 
# sample_l="data ggh vbf dy ewk tt ww wz" 
sample_l="data ggh vbf dy tt ww wz" 

# label="Run3_nanoAODv15_24Jan2025"
# label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"
label="Run3_nanoAODv12_02Feb_FilterJetsHorn30GeV"

base_path="/depot/cms/hmm/yun79/hmm_ntuples/copperheadV1clean"

# model_name="Run3PrelimResultsJan25_2026_NoAnnhilateWgts"
# model_name="Run3PrelimResultsJan29_2026_reducedInput"
# model_name="Run3PrelimResultsJan29_2026_reducedInput2"
# model_name="Run3PrelimResultsFeb07_2026_jecjer"
model_name="Run3PrelimResultsFeb09_2026_jecjer"
# model_name="Run3PrelimResultsFeb10_2026_jecjer_flatDimuMass"

category="ggh"
stage2_save_path="${model_name}_${category}/" 
# stage2_save_path="${model_name}_${category}_memoryRefactor/" 



# region="z-peak"
region="h-sidebands"

# year="all"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}

# year="2024"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}
# year="2023BPix"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}
# year="2023"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}
# year="2022postEE"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}
# year="2022preEE"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}

# year="all"
# region="signal"
# python validation_plot.py -label $label -cat $stage2_save_path --samples $sample_l -y $year --region ${region} --base_path ${base_path}


# # plot Fig 6.13 from AN-19-124
year="all"
region="signal"
python plot_6_8.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# python plot_6_13.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# python plot_6_19.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# python getTable_6_2And6_12.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # -----------------------------------------------------
# plot 6.7 
# # -----------------------------------------------------
# python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path} --model_name ${model_name}


# # # # -----------------------------------------------------
# # # plot 6.7 
# # # # -----------------------------------------------------

# # model_name="V2_Aug16_2025AddIssue1To3_IssueNum2"

# # stage2_save_path="${model_name}_ggh" # stage2 ouput name
# # year="all"
# # # year="2018"
# # # year="2017"
# # label="fullRun_Jun23_2025_1n2Revised"
# # region="signal"
# # python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # python plot_6_8.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # python getTable_6_2And6_12.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # python plot_6_13.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # python plot_6_19.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}

# # # year="2018"
# # # python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # year="2017"
# # # python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # year="2016postVFP"
# # # python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}
# # # year="2016preVFP"
# # # python plot_6_7.py -label $label -cat $stage2_save_path -y ${year} --region ${region} --base_path ${base_path}

