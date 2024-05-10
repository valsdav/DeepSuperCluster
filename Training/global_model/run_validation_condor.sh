#!/bin/bash -e

echo "Starting"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc12-opt/setup.sh

echo "Evaluation"
python run_validation_dataset_awk.py --model-config $1 --model-weights $2 --outputdir $3 --conf-overwrite $4
