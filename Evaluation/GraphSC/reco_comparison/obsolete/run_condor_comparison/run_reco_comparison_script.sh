#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
ASSOC=$4;
WPFILE=$5;

echo -e "Running reco comparison dumper.."

python run_reco_comparison.py -i ${INPUTFILE} -o output_{type}.csv             -a ${ASSOC} --wp-file ${WPFILE} ;

tar -zcf output_seeds.csv.tar.gz output_seeds.csv && tar -zcf output_event.csv.tar.gz output_event.csv
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output_seeds.csv.tar.gz root://eosuser.cern.ch/${OUTPUTDIR}/seed_data_${JOBID}.csv.tar.gz;
xrdcp -f --nopbar  output_event.csv.tar.gz root://eosuser.cern.ch/${OUTPUTDIR}/event_data_${JOBID}.csv.tar.gz;
echo -e "DONE";
