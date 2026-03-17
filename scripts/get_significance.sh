#!/bin/bash
set -e

# -------------------------------------------------------
#  pre-fit signal significance
# -------------------------------------------------------
# best_idx="0" # sumExp
best_idx="1" # BWZRedux
# best_idx="2" # FEWZxBern

combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat0"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat1"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat2"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat3"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat4"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "${category} pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
category="all"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
echo "combined pre-fit Expected Significance:" >> significance.txt
combine -M Significance -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
cat comb_expSignificance.log | grep "Significance:" >> significance.txt


# -------------------------------------------------------
#  post-fit signal significance
# -------------------------------------------------------

# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat0"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat1"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat2"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat3"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt

# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat4"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "${category} post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt


# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
# category="all"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# echo "combined post-fit Expected Significance:" >> significance.txt
# combine -M Significance -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > comb_expSignificance.log
# cat comb_expSignificance.log | grep "Significance:" >> significance.txt



# -------------------------------------------------------
#  pre-fit MultiDimFit signal strength fit
# -------------------------------------------------------
# best_idx="0" # sumExp
best_idx="1" # BWZRedux
# best_idx="2" # FEWZxBern


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat0"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root --rMin=0 --rMax=2 > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat1"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat2"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat3"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt

combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
category="cat4"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
category="all"
text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=${best_idx} --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
combine -M MultiDimFit --algo singles -d datacard_comb_sig_${category}_ggh.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=${best_idx} --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root  > multidim_${category}Singles.log
echo "pre-fit combined multiDim fit using Asimov toys:"  >> signal_strengths.txt
cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# -------------------------------------------------------
#  post-fit MultiDimFit signal strength fit
# -------------------------------------------------------



# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat0_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat0"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat1_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat1"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat2_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat2"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt



# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh.txt datacard_cat4_ggh_bkg_test.txt >  datacard_comb_sig_cat3_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat3"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh_bkg_test.txt datacard_cat1_ggh_bkg_test.txt datacard_cat2_ggh_bkg_test.txt datacard_cat3_ggh_bkg_test.txt datacard_cat4_ggh.txt >  datacard_comb_sig_cat4_ggh.txt # NOTE: combine cards command is different for EACH category
# category="cat4"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit ${category} multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# combineCards.py datacard_cat0_ggh.txt datacard_cat1_ggh_test.txt datacard_cat2_ggh_test.txt datacard_cat3_ggh_test.txt datacard_cat4_ggh_test.txt >  datacard_comb_sig_all_ggh.txt # NOTE: combine cards command is different for EACH category
# category="all"
# text2workspace.py -m 125 datacard_comb_sig_${category}_ggh.txt
# combineTool.py datacard_comb_sig_${category}_ggh.root -M GenerateOnly -m 125 -t -1  --expectSignal 1 --setParameters pdf_index_ggh=0 --saveToys -n ${category}_asimovToys # NOTE: Generate asimov with pdf_index that's most representative
# combine -M MultiDimFit datacard_comb_sig_${category}_ggh.root -m 125 --freezeParameters MH --saveWorkspace -n .bestfit_${category}.with_syst -t -1 --expectSignal 1 --saveSpecifiedNuis=all --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --cminDefaultMinimizerStrategy=0
# combine -M MultiDimFit --algo singles -d higgsCombine.bestfit_${category}.with_syst.MultiDimFit.mH125.root -m 125 -n _signif_${category}_ggh --cminDefaultMinimizerStrategy=0 --snapshotName "MultiDimFit" --freezeParameters MH,allConstrainedNuisances  -t -1 --expectSignal 1 --cminRunAllDiscreteCombinations  --setParameters pdf_index_ggh=0 --toysFile higgsCombine${category}_asimovToys.GenerateOnly.mH125.123456.root > multidim_${category}Singles.log
# echo "post-fit combined multiDim fit using Asimov toys:"  >> signal_strengths.txt
# cat multidim_${category}Singles.log | grep "   r :  " >> signal_strengths.txt


# # -------------------------------------------------------
# # for printing plot1Dscans
# # -------------------------------------------------------

# index="1"
# cat="cat4"

# # observed NLL scan
# text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh.txt
# echo "combine -M MultiDimFit datacard_comb_sig_${cat}_ggh.root -m 125 --freezeParameters MH -n .scan --algo grid --points 10 --setParameterRanges r=0,3 --setParameters pdf_index_ggh=${index} --cminDefaultMinimizerStrategy=0 
# plot1DScan.py higgsCombine.scan.MultiDimFit.mH125.root -o nll_scan_${cat}_index${index}_observed"
# combine -M MultiDimFit datacard_comb_sig_${cat}_ggh.root -m 125 --freezeParameters MH -n .scan --algo grid --points 10 --setParameterRanges r=0,3 --setParameters pdf_index_ggh=${index} --cminDefaultMinimizerStrategy=0 
# plot1DScan.py higgsCombine.scan.MultiDimFit.mH125.root -o nll_scan_${cat}_index${index}_observed --main-label "Observed"

# # expected NLL scan
# text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh.txt
# combine -M MultiDimFit datacard_comb_sig_${cat}_ggh.root -m 125 --freezeParameters MH -n .scan --algo grid --points 10 --setParameterRanges r=0,3 --setParameters pdf_index_ggh=${index} --cminDefaultMinimizerStrategy=0 -t -1  --expectSignal 1
# plot1DScan.py higgsCombine.scan.MultiDimFit.mH125.root -o nll_scan_${cat}_index${index}_expected --main-label "Expected"

# # -------------------------------------------------------
# # for printing plotGof
# # -------------------------------------------------------

# index="1"
# # cat="cat4"
# cat="all"

# text2workspace.py -m 125 datacard_comb_sig_${cat}_ggh.txt

# combine -M GoodnessOfFit datacard_comb_sig_${cat}_ggh.root --algo saturated -m 125 --freezeParameters MH -n .goodnessOfFit_data

# combine -M GoodnessOfFit datacard_comb_sig_${cat}_ggh.root --algo saturated -m 125 --freezeParameters MH -n .goodnessOfFit_toys -t 100 #1050

# combineTool.py -M CollectGoodnessOfFit --input higgsCombine.goodnessOfFit_data.GoodnessOfFit.mH125.root higgsCombine.goodnessOfFit_toys.GoodnessOfFit.mH125.123456.root -m 125.0 -o gof.json

# plotGof.py gof.json --statistic saturated --mass 125.0 -o gof_plot