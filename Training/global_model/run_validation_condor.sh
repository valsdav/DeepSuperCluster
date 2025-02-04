#!/bin/bash -e

echo "Starting"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

echo "Evaluation"
if [ "$6" == "True" ]; then
    python run_validation_dataset_awk.py --model-config $1 --model-weights $2 --outputdir $3 --conf-overwrite $4 --flavour $5 --diff-model
else
    python run_validation_dataset_awk.py --model-config $1 --model-weights $2 --outputdir $3 --conf-overwrite $4 --flavour $5
fi