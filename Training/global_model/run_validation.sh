#!/bin/sh

python run_validation_dataset.py --model-dir /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/MultiSA_v2/run_03 \
   --model-weights weights.best.hdf5 -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/MultiSA_v2/run_03/validation_data -n 1400000 -b 250
