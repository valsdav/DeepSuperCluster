#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
WETA_EB=$4;
WETA_EE=$5;
WPHI_EB=$6;
WPHI_EE=$7;
MAXNOCALO=$8;
ASSOC=$9;


echo -e "Running numpy dumper.."

python cluster_tonumpy_mustache.py -i ${INPUTFILE} -o output.pkl --weta ${WETA_EB} ${WETA_EE}                     --wphi ${WPHI_EB} ${WPHI_EE} --maxnocalow ${MAXNOCALO} --assoc-strategy ${ASSOC} ;

echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.pkl root://eosuser.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.pkl;

echo -e "DONE";
