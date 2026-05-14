if [[ "$2" == "2016preVFP" ]]; then
    echo "2016preVFP"
    combineCards.py \
        SR_2016preVFP=$1/datacard_vbf_SR_2016preVFP.txt \
        SB_2016preVFP=$1/datacard_vbf_SB_2016preVFP.txt \
        > $1/HMuMu_13TeV_2016preVFP.txt
    text2workspace.py $1/HMuMu_13TeV_2016preVFP.txt -m 125
fi

if [[ "$2" == "2016postVFP" ]]; then
    echo "2016postVFP"
    combineCards.py \
        SR_2016postVFP=$1/datacard_vbf_SR_2016postVFP.txt \
        SB_2016postVFP=$1/datacard_vbf_SB_2016postVFP.txt \
        > $1/HMuMu_13TeV_2016postVFP.txt
    text2workspace.py $1/HMuMu_13TeV_2016postVFP.txt -m 125
fi

if [[ "$2" == "2016" ]]; then
    echo "2016"
    combineCards.py \
        preVFP=$1/HMuMu_13TeV_2016preVFP.txt \
        postVFP=$1/HMuMu_13TeV_2016postVFP.txt \
        > $1/HMuMu_13TeV_2016.txt
    text2workspace.py $1/HMuMu_13TeV_2016.txt -m 125
fi

if [[ "$2" == "2017" ]]; then
    echo "2017"
    combineCards.py \
        SR_2017=$1/datacard_vbf_SR_2017.txt \
        SB_2017=$1/datacard_vbf_SB_2017.txt \
        > $1/HMuMu_13TeV_2017.txt
    text2workspace.py $1/HMuMu_13TeV_2017.txt -m 125
fi

if [[ "$2" == "2018" ]]; then
    echo "2018"
    combineCards.py \
        SR_2018=$1/datacard_vbf_SR_2018.txt \
        SB_2018=$1/datacard_vbf_SB_2018.txt \
        > $1/HMuMu_13TeV_2018.txt
    text2workspace.py $1/HMuMu_13TeV_2018.txt -m 125
fi

if [[ "$2" == "2022postEE" ]]; then
    echo "2022postEE"
    combineCards.py \
        SR_2022postEE=$1/datacard_vbf_SR_2022postEE.txt \
        SB_2022postEE=$1/datacard_vbf_SB_2022postEE.txt \
        > $1/HMuMu_13TeV_2022postEE.txt
    text2workspace.py $1/HMuMu_13TeV_2022postEE.txt -m 125
fi

if [[ "$2" == "Run2" ]]; then
    echo "Run2 (2016-2018)"
    combineCards.py \
        y2016pre=$1/HMuMu_13TeV_2016preVFP.txt \
        y2016post=$1/HMuMu_13TeV_2016postVFP.txt \
        y2017=$1/HMuMu_13TeV_2017.txt \
        y2018=$1/HMuMu_13TeV_2018.txt \
        > $1/HMuMu_13TeV_Run2.txt
    # combineCards.py \
    #     SR_2016preVFP=$1/datacard_vbf_SR_2016preVFP.txt \
    #     SB_2016preVFP=$1/datacard_vbf_SB_2016preVFP.txt \
    #     SR_2016postVFP=$1/datacard_vbf_SR_2016postVFP.txt \
    #     SB_2016postVFP=$1/datacard_vbf_SB_2016postVFP.txt \
    #     SR_2017=$1/datacard_vbf_SR_2017.txt \
    #     SB_2017=$1/datacard_vbf_SB_2017.txt \
    #     SR_2018=$1/datacard_vbf_SR_2018.txt \
    #     SB_2018=$1/datacard_vbf_SB_2018.txt \
        # > $1/HMuMu_13TeV_Run2.txt
    text2workspace.py $1/HMuMu_13TeV_Run2.txt -m 125
fi
