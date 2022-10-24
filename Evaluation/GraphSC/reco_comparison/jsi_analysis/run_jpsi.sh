#!/bin/sh -e
export X509_USER_PROXY=/tmp/x509up_u96307
voms-proxy-info

cp -r /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/DeepSC_JPsi_12_5_0.tar.gz ./DeepSC_JPsi_12_5_0.tar.gz
tar -xzvf DeepSC_JPsi_12_5_0.tar.gz
cd DeepSC_JPsi_12_5_0/src
echo -e "evaluate"
eval `scramv1 ru -sh`

JOBID=$1;  
INPUTFILE=$2;
cd RecoSimStudies/Dumpers/crab/

cmsRun MiniAOD_fromRaw_Run3_rereco_DeepSC_algoA_Data2022_cfg.py inputFile=${INPUTFILE} outputFile=../../../output.root

cd ../../..;
cd Bmmm/Analysis/test/;
python3 inspector_bmmm_analysis.py --inputFiles ../../../output.root --filename output_${JOBID}.root

xrcdp output_${JOBID}.root /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/Jpsi_Run3/bparking_RunF;

