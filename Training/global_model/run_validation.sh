#!/bin/sh

python run_validation_dataset_enregr_jet_binary.py -d v11 --model-dir /eos/user/p/psimkina/evaluation/jet_model/binary_genpt_Et_SA_weights/run_01 \
   --model-weights weights.best.hdf5 -o /eos/user/p/psimkina/evaluation/jet_model/binary_genpt_Et_SA_weights/run_01/ -n 180000 -b 400
