#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
ASSOC=$4;
WPFILE=$5;
MAXNOCALO=$6;
ET_SEED=$7;


echo -e "Running numpy dumper.."

python cluster_tonumpy_dynamic_global_overlap.py -i ${INPUTFILE} -o output.ndjson             -a ${ASSOC} --wp-file ${WPFILE} --min-et-seed ${ET_SEED};

tar -zcf output.ndjson.tar.xz output.ndjson
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.ndjson.tar.xz root://eosuser.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.ndjson.tar.xz;

echo -e "DONE";
