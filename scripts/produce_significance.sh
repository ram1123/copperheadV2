# Optional arguments after the year can be a scenario label and/or
# --manual-set-parameters. The default transfers fitted nuisance values into a
# new workspace; manual mode passes the values directly to Combine.
scenario=""
nuisance_transfer_mode="workspace"
for argument in "${@:3}"; do
    case "$argument" in
        --manual-set-parameters)
            nuisance_transfer_mode="manual"
            ;;
        *)
            if [[ -n "$scenario" ]]; then
                echo "Error: more than one scenario label was provided." >&2
                exit 2
            fi
            scenario="$argument"
            ;;
    esac
done

# print inputs arguements
echo "========================================"
echo "Producing Significance for the following inputs:"
echo "First argument (Path of datacard): $1"
echo "Second argument (Year): $2"
echo "Scenario: $scenario"
echo "Nuisance transfer mode: $nuisance_transfer_mode"
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
echo "Scenario: $scenario" >> $1/significance.txt
echo "Nuisance transfer mode: $nuisance_transfer_mode" >> $1/significance.txt
echo "" >> $1/significance.txt
echo "Prefit Significance:" >> $1/significance.txt
combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_${scenario}_ -t -1 --rMin -2 --rMax 5  > $1/prefitsignificance.log
cat $1/prefitsignificance.log | grep "Significance:" >> $1/significance.txt
# echo "Postfit Significance:" >> $1/significance.txt

# postfit significance is partially unblinded so we should not look at it until unblinding decision is made
# combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_$3_ -t -1 --rMin -2 --rMax 5 --toysFrequentist  > $1/postfitsignificance.log
# cat $1/postfitsignificance.log | grep "Significance:" >> $1/significance.txt


rate_params="XSecAndNorm2017DY01J,XSecAndNorm2017DY2J,XSecAndNorm2018DY01J,XSecAndNorm2018DY2J,XSecAndNorm2016preVFPDY01J,XSecAndNorm2016preVFPDY2J,XSecAndNorm2016postVFPDY01J,XSecAndNorm2016postVFPDY2J"
echo "Significance (Stat Only):" >> $1/significance.txt
combineTool.py -d $1/HMuMu_13TeV_$2.txt -M Significance -m 125 --expectSignal=1 -n _$2_${scenario}_ -t -1 --rMin -2 --rMax 5 --freezeParameters allConstrainedNuisances,$rate_params    > $1/prefitsignificance_StatOnly.log
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
        -n ".${postfit_tag}_$scenario" \
        > "${postfit_tag}_multidimfit_$scenario.log"
)

sb_fit_workspace="$1/higgsCombine.${postfit_tag}_$scenario.MultiDimFit.mH125.root"
orig_workspace="$1/HMuMu_13TeV_$2.root"
postfit_workspace="$1/HMuMu_13TeV_${2}_${postfit_tag}.root"
echo "sideband datacard: $sb_datacard"
echo "full workspace: $orig_workspace"
echo "post-fit workspace: $postfit_workspace"

# re-do text2workspace of orig_workspace just to be safe
text2workspace.py "$1/HMuMu_13TeV_$2.txt" \
    -m 125 \
    -o "$orig_workspace"

significance_workspace="$postfit_workspace"
snapshot_options=(--snapshotName "$snapshot_name")
set_parameter_options=()

if [[ "$nuisance_transfer_mode" == "manual" ]]; then
    nuisance_output=$(root -l -b -q -e \
        "TFile sb_fit_file(\"$sb_fit_workspace\"); \
         RooWorkspace *sb_workspace = (RooWorkspace *)sb_fit_file.Get(\"w\"); \
         sb_workspace->loadSnapshot(\"MultiDimFit\"); \
         RooStats::ModelConfig *sb_model = (RooStats::ModelConfig *)sb_workspace->genobj(\"ModelConfig\"); \
         TFile full_file(\"$orig_workspace\"); \
         RooWorkspace *orig_workspace = (RooWorkspace *)full_file.Get(\"w\"); \
         for (RooAbsArg *nuisance : *sb_model->GetNuisanceParameters()) { \
             RooRealVar *fitted_parameter = dynamic_cast<RooRealVar *>(nuisance); \
             if (fitted_parameter && orig_workspace->var(fitted_parameter->GetName())) { \
                 std::cout << \"NUISANCE_PARAMETER \" << fitted_parameter->GetName() \
                           << \"=\" << Form(\"%.17g\", fitted_parameter->getVal()) << std::endl; \
             } \
         }")
    nuisance_parameters=$(printf '%s\n' "$nuisance_output" | sed -n 's/^NUISANCE_PARAMETER //p' | paste -sd, -)
    if [[ -z "$nuisance_parameters" ]]; then
        echo "Error: no fitted nuisance parameters shared by the sideband and full workspaces were found." >&2
        exit 1
    fi

    significance_workspace="$orig_workspace"
    snapshot_options=()
    set_parameter_options=(--setParameters "$nuisance_parameters,r=1")
    echo "Manual --setParameters values: $nuisance_parameters"
else
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
fi

echo "Start GenerateOnly"
(
    cd "$1" || exit 1
    combineTool.py \
        -d "$(basename "$significance_workspace")" \
        -M GenerateOnly \
        -m 125 \
        -t -1 \
        --expectSignal 1 \
        "${snapshot_options[@]}" \
        "${set_parameter_options[@]}" \
        --saveToys \
        -s 123456 \
        -n ".${2}PostfitAsimov_$scenario"
)
echo "Start Significance"
postfit_asimov="$1/higgsCombine.${2}PostfitAsimov_$scenario.GenerateOnly.mH125.123456.root"
(
    cd "$1" || exit 1
    combine -M Significance \
        -d "$(basename "$significance_workspace")" \
        -m 125 \
        -t -1 \
        --expectSignal 1 \
        --toysFile "$(basename "$postfit_asimov")" \
        "${snapshot_options[@]}" \
        "${set_parameter_options[@]}" \
        --freezeParameters allConstrainedNuisances,$rate_params \
        --rMin -2 \
        --rMax 5 \
        -n ".${2}PostfitSignificance_$scenario" \
        > postfitsignificance.log
)

cat $1/postfitsignificance.log | grep "Significance:" >> $1/significance.txt
