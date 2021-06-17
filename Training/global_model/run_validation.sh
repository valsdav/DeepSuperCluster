#!/bin/sh

python run_validation_dataset_enregr.py -d v10 --model-dir /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v14/run_01/ \
   --model-weights weights.best.hdf5 -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/gcn_models_SA_v14/run_01/validation_data -n 1600000 -b 300
