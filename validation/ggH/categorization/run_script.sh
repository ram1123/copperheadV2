#!/bin/bash
set -e
# run stage2 twice. First to generate BDT scores (we assume that an appropriate BDT is already trained, then generate score bin edges once more, then finally run stage2 again to save both bdt scores and ggH sub-category index

# sample_l="data dy ewk tt st ww wz zz" 
sample_l="data ggh vbf dy ewk tt st ww wz zz" 

# label="V2_Jan29_JecOn_TrigMatchFixed_2016UlJetIdFix"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff"
# label="UpdatedDY_100_200_CrossSection_24Feb_jetpuidOff_newZptWgt25Mar2025"
# label="DYMiNNLO_jetpuidOff_newZptWgt25Mar2025"
label="DYMiNNLO_30Mar2025"

stage2_load_path="/depot/cms/users/yun79/hmm/copperheadV1clean/$label/stage1_output"

# category="20Mar2025_ggh"

# model_name="V2_UL_Mar24_2025_DyTtStVvEwkGghVbf_scale_pos_weight"
# model_name="V2_UL_Mar25_2025_DyGghVbf_scale_pos_weight_newZpt"
# model_name="V2_UL_Mar26_2025_DyGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar26_2025_DyTtStVvEwkGghVbf_scale_pos_weight_dyMiNNLO"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_removeAllJetVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar"
# model_name="V2_UL_Mar30_2025_DyMiNNLOGghVbf_onlyMuVar_ZeppenJjMass_DeltaVars"
model_name="V2_UL_Apr09_2025_DyMinnloTtStVvEwkGghVbf_hyperParamOnScaleWgt0_75"

category="${model_name}_ggh"
# year="2018"

# region="z-peak"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

# region="signal"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

year="2017"
region="z-peak"
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}

region="signal"
python validation_plot.py -label $label -cat $category --samples $sample_l -y $year --region ${region}
# year="2016postVFP"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year

# year="2016preVFP"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year


# year="all"
# python validation_plot.py -label $label -cat $category --samples $sample_l -y $year