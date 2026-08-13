cat $1/HMuMu_13TeV_$2.txt
echo "text2 workspace"
# text2workspace.py $1/HMuMu_13TeV_$2.txt -m 125 -o HMuMu_13TeV_$2.root
text2workspace.py $1/HMuMu_13TeV_$2.txt -m 125  --channel-masks


GENERATE='n;toysFrequentist;t;;NAME_prefit_asimov,!,-1;NAME_postfit_asimov, ,-1'
DRYRUN="--dry-run"
EXTRASTRING=""

# Dmitry command
# combine -M FitDiagnostics $1/HMuMu_13TeV_$2.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -5 --rMax 5  --saveShapes --saveNormalizations --saveWithUncertainties -n _$2_$3 --X-rtd MINIMIZER_analytic --expectSignal=1  #--setParameters mask_ch1=1

# new command 10 Sep 2025
echo "Running FitDiagnostics for $2 in $1, output name suffix $3"
combine -M FitDiagnostics $1/HMuMu_13TeV_$2.root  --cminRunAllDiscreteCombinations  -m 125 --rMin -5 --rMax 5  --saveShapes --saveNormalizations --saveWithUncertainties -n _$2_$3  --expectSignal=1  --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0  --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams  --setParameters mask_SR_2017=1

#According to gitlab
#combine -M FitDiagnostics $1/HMuMu_13TeV_$2.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -5 --rMax 5  --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic_gitlab --X-rtd MINIMIZER_analytic --expectSignal=1 --skipBOnlyFit --robustHesse 1 #--setParameters mask_ch1=1

#combine -M FitDiagnostics $1/combined.root --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams -m 125 --rMin -2 --rMax 2 --saveShapes --saveNormalizations --saveWithUncertainties -n _$1_new_analytic --plots --robustFit=1 --X-rtd MINIMIZER_analytic --job-mode condor --generate "${GENERATE//NAME/$name}" --task-name FitDiagnostics_${dc##*/} --sub-opts="$JOBOPTS\n+JobBatchName=\"FitDiagnostics\"" $DRYRUN --expectSignal=1  --skipBOnlyFit --robustHesse 1
# echo 'python3 postFitPlot.py'
#python postfitplot_AP.py

echo "Preparing postfit plots for $2 in $1, output name suffix $3"

ch1="SR_$2"
ch2="SB_$2"


mkdir -p $1/plots/$2/$3
# python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_fit_s --region $ch1 --year $2 --output_dir $1/plots/$2/$3
# python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_fit_b --region $ch1 --year $2 --output_dir $1/plots/$2/$3
# python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_prefit --region $ch1 --year $2 --output_dir $1/plots/$2/$3
python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_fit_s --region $ch2 --year $2 --output_dir $1/plots/$2/$3
python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_fit_b --region $ch2 --year $2 --output_dir $1/plots/$2/$3
python3 postFitPlot.py --input_file fitDiagnostics_$2_$3.root --shape_type shapes_prefit --region $ch2 --year $2 --output_dir $1/plots/$2/$3
python3 pre_post_shape_comp.py -i fitDiagnostics_$2_$3.root -o $1/plots/$2/$3 --postfit b
