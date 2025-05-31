# if conda command is not found, load the conda module
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, loading the conda module"
    source /etc/profile.d/modules.sh
    module --force purge
    module load anaconda/2024.06-1
fi

conda activate /depot/cms/users/yun79/conda_envs/yun_coffea_latest
# conda activate /depot/cms/kernels/root632 # FIXME: Temp fix, as coffea latest not working starting from early march 2025
# conda activate /depot/cms/kernels/coffea_latest
# conda activate /depot/cms/kernels/python3
# if there is no arguments only then setup proxy
if [ "$#" -eq 0 ]; then
    echo "No arguments provided. Setting up the proxy..."
    voms-proxy-init -voms cms -rfc -valid 192:00 --out $(pwd)/voms_proxy.txt
    echo "Your proxy is here: $(pwd)/voms_proxy.txt"
    export X509_USER_PROXY=$(pwd)/voms_proxy.txt
fi
export WORKDIR=$PWD
export XRD_REQUESTTIMEOUT=2400
# Setup CMSSW environment & related commands
source /cvmfs/cms.cern.ch/cmsset_default.sh
# export PYTHONPATH="/depot/cms/users/$USER/copperheadV2:$PYTHONPATH"
export PYTHONPATH="$WORKDIR:$PYTHONPATH" # Load from the `copperheadV2` directory

