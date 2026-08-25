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


rate_params="XSecAndNorm2017DY01J,XSecAndNorm2017DY2J,XSecAndNorm2018DY01J,XSecAndNorm2018DY2J,XSecAndNorm2016preVFPDY01J,XSecAndNorm2016preVFPDY2J,XSecAndNorm2016postVFPDY01J,XSecAndNorm2016postVFPDY2J"
echo "Significance (Stat Only):" >> $1/significance.txt
combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_$3_ -t -1 --rMin -2 --rMax 5 --freezeParameters allConstrainedNuisances,$rate_params    > $1/prefitsignificance_StatOnly.log
cat $1/prefitsignificance_StatOnly.log | grep "Significance:" >> $1/significance.txt


# blinded post-fit expected significance. Fit nuisance parameters over SB region, save the nuisance
# parameters -> load saved nuisance parameters -> fit over full region with nuisance parameters frozen
if [[ "$2" == "Run2" ]]; then
    sh produce_combine_cards.sh $1 Run2SB # this produces HMuMu_13TeV_Run2SB.txt, which we need for MultiDimFit
    sb_datacard="HMuMu_13TeV_Run2SB.txt"    
else
    sb_datacard="datacard_vbf_SB_$2.txt"
fi

postfit_tag="${2}SBPostfit"
snapshot_name="$postfit_tag"

echo "Postfit Significance :" >> $1/significance.txt

(
    cd "$1" || exit 1
    combine -M MultiDimFit \
        -d "$sb_datacard" \
        -m 125 \
        --setParameters r=0 \
        --freezeParameters r \
        --saveWorkspace \
        --saveSpecifiedNuis=all \
        -n ".${postfit_tag}_$3" \
        > "${postfit_tag}_multidimfit_$3.log"
)

sb_fit_workspace="$1/higgsCombine.${postfit_tag}_$3.MultiDimFit.mH125.root"
orig_workspace="$1/HMuMu_13TeV_$2.root"
postfit_workspace="$1/HMuMu_13TeV_${2}_${postfit_tag}.root"
echo "sideband datacard: $sb_datacard"
echo "full workspace: $orig_workspace"
echo "post-fit workspace: $postfit_workspace"

# re-do text2workspace of orig_workspace just to be safe
text2workspace.py "$1/HMuMu_13TeV_$2.txt" \
    -m 125 \
    -o "$orig_workspace"

root -l -b -q -e \
    "TFile sb_fit_file(\"$sb_fit_workspace\"); \
     RooWorkspace *sb_workspace = (RooWorkspace *)sb_fit_file.Get(\"w\"); \
     sb_workspace->loadSnapshot(\"MultiDimFit\"); \
     RooStats::ModelConfig *sb_model = (RooStats::ModelConfig *)sb_workspace->genobj(\"ModelConfig\"); \
     TFile full_file(\"$orig_workspace\"); \
     RooWorkspace *orig_workspace = (RooWorkspace *)full_file.Get(\"w\"); \
     RooArgSet full_parameters(orig_workspace->allVars()); \
     full_parameters.assignValueOnly(*sb_model->GetNuisanceParameters()); \
     orig_workspace->saveSnapshot(\"$snapshot_name\", full_parameters); \
     orig_workspace->writeToFile(\"$postfit_workspace\");"

echo "Start GenerateOnly"
(
    cd "$1" || exit 1
    combineTool.py \
        -d "$(basename "$postfit_workspace")" \
        -M GenerateOnly \
        -m 125 \
        -t -1 \
        --expectSignal 1 \
        --snapshotName "$snapshot_name" \
        --setParameters r=1 \
        --saveToys \
        -s 123456 \
        -n ".${2}PostfitAsimov_$3"
)
echo "Start Significance"
postfit_asimov="$1/higgsCombine.${2}PostfitAsimov_$3.GenerateOnly.mH125.123456.root"
(
    cd "$1" || exit 1
    combine -M Significance \
        -d "$(basename "$postfit_workspace")" \
        -m 125 \
        -t -1 \
        --expectSignal 1 \
        --toysFile "$(basename "$postfit_asimov")" \
        --snapshotName "$snapshot_name" \
        --freezeParameters allConstrainedNuisances,$rate_params \
        --rMin -2 \
        --rMax 5 \
        -n ".${2}PostfitSignificance_$3" \
        > postfitsignificance.log
)

cat $1/postfitsignificance.log | grep "Significance:" >> $1/significance.txt
