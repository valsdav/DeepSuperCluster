#!/bin/sh

python run_validation_dataset_enregr.py -d v10 --model-dir /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v17/run_01 \
   --model-weights weights.best.hdf5 -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v17/run_01/validation_data_v2 -n 1800000 -b 400
