#!/bin/sh

MODELS_DIR="/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/models_archive/gcn_models/ACAT2022/"


##########################3
#Default
# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests/run_32_standard_datasetv10/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests/run_32_standard_datasetv10/model_forexport.py \
#        -o  ${MODELS_DIR}/tests/run_32_standard_datasetv10/model_smallpadding.pb \
#        -os ${MODELS_DIR}/tests/run_32_standard_datasetv10/scaler_config \
#        --conf-overwrite validation_config_v10.json  \
#        --max-ncls 15 \
#        --max-nrechits 20


# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests/run_32_standard_datasetv10/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests/run_32_standard_datasetv10/model_forexport.py \
#        -o  ${MODELS_DIR}/tests/run_32_standard_datasetv10/model_largepadding.pb \
#        -os ${MODELS_DIR}/tests/run_32_standard_datasetv10/scaler_config \
#        --conf-overwrite validation_config_v10.json  \
#        --max-ncls 60 \
#        --max-nrechits 60


# #################
# #simpler rechit

# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/model_smallpadding.pb \
#        -os ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/scaler_config \
#        --conf-overwrite validation_config_v10.json  \
#        --max-ncls 15 \
#        --max-nrechits 20


# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/model_largepadding.pb \
#        -os ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/scaler_config \
#        --conf-overwrite validation_config_v10.json  \
#        --max-ncls 60 \
#        --max-nrechits 60

## verysmall
python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_simpler_rechits/orun_41_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/model_forexport.py \
       -o  ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/model_smallpadding.pb \
       -os ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/scaler_config \
       --conf-overwrite validation_config_v10.json  \
       --max-ncls 15 \
       --max-nrechits 20


python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/model_forexport.py \
       -o  ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/model_largepadding.pb \
       -os ${MODELS_DIR}/tests_simpler_rechits/run_41_datasetv10_verysmall_nconv2_morepatience_lessL2/scaler_config \
       --conf-overwrite validation_config_v10.json  \
       --max-ncls 60 \
       --max-nrechits 60



# ################################3
# ### No rechits

# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/model_smallpadding.pb \
#        -os ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/scaler_config \
#        --conf-overwrite validation_config_v10_norechits.json  \
#        --max-ncls 15 \
#        --max-nrechits 20


# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/model_largepadding.pb \
#        -os ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/scaler_config \
#        --conf-overwrite validation_config_v10_norechits.json  \
#        --max-ncls 60 \
#        --max-nrechits 60



# # No rechits verysmall
# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/model_smallpadding.pb \
#        -os ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/scaler_config \
#        --conf-overwrite validation_config_v10_norechits.json  \
#        --max-ncls 15 \
#        --max-nrechits 20


# python3 exporter_awk.py --model-config ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
#        --model-weights weights.best.hdf5 \
#        --model-python ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/model_forexport.py \
#        -o  ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/model_largepadding.pb \
#        -os ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/scaler_config \
#        --conf-overwrite validation_config_v10_norechits.json  \
#        --max-ncls 60 \
#        --max-nrechits 60

