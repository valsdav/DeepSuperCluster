#!/bin/sh

python run_validation_dataset_enregr.py -d v11 --model-dir /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v18/run_01 \
   --model-weights weights.best.hdf5 -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v18/run_01/validation_data -n 1800000 -b 400
