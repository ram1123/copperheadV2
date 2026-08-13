# print inputs arguements
echo "========================================"
echo "Producing Significance for the following inputs:"
echo "First argument (Path of datacard): $1"
echo "Second argument (Year): $2"
echo "Third argument (Scenario, can add anything): $3"
echo "========================================"


# bash produce_significance.sh Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_AfterHPScan_03Sep2025_WithAllDY 2018
# bash produce_significance.sh Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_AfterHPScan_03Sep2025_WithAllDY 2017
# bash produce_significance.sh Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_AfterHPScan_03Sep2025_WithAllDY 2016preVFP
# bash produce_significance.sh Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt_JESVar_AfterHPScan_03Sep2025_WithAllDY 2016postVFP

echo $1/significance.txt
echo "========================================" >> $1/significance.txt
echo "" >> $1/significance.txt
echo "date: $(date)" >> $1/significance.txt
echo "First argument: $1" >> $1/significance.txt
echo "Second argument: $2" >> $1/significance.txt
echo "Third argument: $3" >> $1/significance.txt
echo "" >> $1/significance.txt
echo "Prefit Significance:" >> $1/significance.txt
combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_$3_ -t -1 --rMin -2 --rMax 5  > $1/prefitsignificance.log
cat $1/prefitsignificance.log | grep "Significance:" >> $1/significance.txt
# echo "Postfit Significance:" >> $1/significance.txt

# postfit significance is partially unblinded so we should not look at it until unblinding decision is made
# combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_$3_ -t -1 --rMin -2 --rMax 5 --toysFrequentist  > $1/postfitsignificance.log
# cat $1/postfitsignificance.log | grep "Significance:" >> $1/significance.txt



echo "Significance (Stat Only):" >> $1/significance.txt
combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_$3_ -t -1 --rMin -2 --rMax 5 --freezeParameters allConstrainedNuisances   > $1/prefitsignificance_StatOnly.log
cat $1/prefitsignificance_StatOnly.log | grep "Significance:" >> $1/significance.txt
