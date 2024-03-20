#!/bin/sh -e
export X509_USER_PROXY=/tmp/x509up_u96307
voms-proxy-info

#cp -r DeepSC_JPsi_12_5_0.tar.gz.tar.gz ./DeepSC_JPsi_12_5_0.tar.gz
#tar -xzvf DeepSC_JPsi_12_5_0.tar.gz
cd DeepSC_JPsi_12_5_0/src
echo -e "evaluate"
eval `scramv1 ru -sh`

JOBID=$1;  
INPUTFILE=$2;
cd RecoSimStudies/Dumpers/crab/

cmsRun MiniAOD_fromRaw_Run3_rereco_DeepSC_algoA_Data2022_cfg.py inputFile=${INPUTFILE} outputFile=../../../output.root maxEvents=10

cd ../../..;
cd Bmmm/Analysis/test/;
python3 inspector_kee_analysis.py --inputFiles ../../../output.root --filename output_${JOBID}.root

cp output_${JOBID}.root /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/jpsi_analysis;

