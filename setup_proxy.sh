conda activate /depot/cms/kernels/root632
voms-proxy-init -voms cms -rfc -valid 192:00 --out /depot/cms/users/$USER/voms_proxy.txt
export X509_USER_PROXY=/depot/cms/users/$USER/voms_proxy.txt
export RUCIO_ACCOUNT=$USER
export WORKDIR=$PWD
export XRD_REQUESTTIMEOUT=2400
