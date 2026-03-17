index_value="0"
index_name="sumExp"
text2workspace.py datacard_comb_sig_all_ggh.txt -m 125
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doInitialFit --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --expectSignal 1 -t -1 &> initial_fit.log 
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doFits --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --parallel 20 --expectSignal 1 -t -1 &> doFits.log
combineTool.py -M Impacts -d datacard_comb_sig_all_ggh.root -m 125 -o impacts_allyears_pdfIdx${index_value}.json
plotImpacts.py -i impacts_allyears_pdfIdx${index_value}.json -o impacts_allyears_pdfIdx_${index_name}

index_value="1"
index_name="BWZRedux"
text2workspace.py datacard_comb_sig_all_ggh.txt -m 125
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doInitialFit --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --expectSignal 1 -t -1 &> initial_fit.log 
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doFits --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --parallel 20 --expectSignal 1 -t -1 &> doFits.log
combineTool.py -M Impacts -d datacard_comb_sig_all_ggh.root -m 125 -o impacts_allyears_pdfIdx${index_value}.json
plotImpacts.py -i impacts_allyears_pdfIdx${index_value}.json -o impacts_allyears_pdfIdx_${index_name}

index_value="2"
index_name="FEWZxBern"
text2workspace.py datacard_comb_sig_all_ggh.txt -m 125
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doInitialFit --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --expectSignal 1 -t -1 &> initial_fit.log 
combineTool.py -M Impacts -d  datacard_comb_sig_all_ggh.root -m 125 --doFits --robustFit 1  --freezeParameters pdf_index_ggh --setParameters pdf_index_ggh=${index_value} --cminDefaultMinimizerStrategy=0 --X-rtd MINIMIZER_freezeDisassociatedParams  --cminRunAllDiscreteCombinations --parallel 20 --expectSignal 1 -t -1 &> doFits.log
combineTool.py -M Impacts -d datacard_comb_sig_all_ggh.root -m 125 -o impacts_allyears_pdfIdx${index_value}.json
plotImpacts.py -i impacts_allyears_pdfIdx${index_value}.json -o impacts_allyears_pdfIdx_${index_name}