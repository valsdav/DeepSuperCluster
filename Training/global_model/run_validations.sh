#!/bin/sh

MODELS_DIR="/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/ACAT2022/"

python submit_validation_condor.py --model-config ${MODELS_DIR}/tests/run_32_standard_datasetv10/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests/run_32_standard_datasetv10/validation_data \
       --conf-overwrite validation_config_v10.json  \
       

python submit_validation_condor.py --model-config ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/validation_data \
       --conf-overwrite validation_config_v10_norechits.json  \
       

python submit_validation_condor.py --model-config ${MODELS_DIR}/tests_onlySA/run_10_datasetv10_smaller/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests_onlySA/run_10_datasetv10_smaller/validation_data \
       --conf-overwrite validation_config_v10.json  \
       

python submit_validation_condor.py --model-config ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/validation_data \
        --conf-overwrite validation_config_v10.json  \
        
python submit_validation_condor.py --model-config ${MODELS_DIR}/tests_focal_loss/run_03_datasetv10/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests_focal_loss/run_03_datasetv10/validation_data \
        --conf-overwrite validation_config_v10.json 
        
        
python submit_validation_condor.py --model-config ${MODELS_DIR}/tests_norechits/run_24_datasetv10_supersmall_nconv2_softF1large/training_config.json \
       --model-weights weights.best.hdf5 \
       -o  ${MODELS_DIR}/tests_norechits/run_24_datasetv10_supersmall_nconv2_softF1large/validation_data \
        --conf-overwrite validation_config_v10_norechits.json 
        
