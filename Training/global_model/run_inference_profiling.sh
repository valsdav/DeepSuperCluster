#!/bin/sh

MODELS_DIR="/data/user/dvalsecc/Clustering/models/ACAT2022/"

# Small

python inference_profiling.py --config ${MODELS_DIR}/tests/run_32_standard_datasetv10/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl15_r20/ \
       --conf-overwrite validation_config_v10_cmsmachine_cl15_r20.json

python inference_profiling.py --config ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl15_r20/  \
       --conf-overwrite validation_config_v10_cmsmachine_norechits_cl15_r20.json

python inference_profiling.py --config ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/training_config.json  \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl15_r20/ \
       --conf-overwrite validation_config_v10_cmsmachine_cl15_r20.json


python inference_profiling.py --config ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl15_r20/  \
       --conf-overwrite validation_config_v10_cmsmachine_norechits_cl15_r20.json


# Large



python inference_profiling.py --config ${MODELS_DIR}/tests/run_32_standard_datasetv10/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl60_r60/ \
       --conf-overwrite validation_config_v10_cmsmachine_cl60_r60.json

python inference_profiling.py --config ${MODELS_DIR}/tests_norechits/run_20_datasetv10_small_l2small/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl60_r60/  \
       --conf-overwrite validation_config_v10_cmsmachine_norechits_cl60_r60.json

python inference_profiling.py --config ${MODELS_DIR}/tests_simpler_rechits/run_20_datasetv10_verysmall_nconv3/training_config.json  \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl60_r60/ \
       --conf-overwrite validation_config_v10_cmsmachine_cl60_r60.json


python inference_profiling.py --config ${MODELS_DIR}/tests_norechits/run_30_datasetv10_verysmall_nconv2_morepatience_lessL2/training_config.json \
       --weights_name weights.best.hdf5 \
       --log_folder tensorflow_logs_fixedpadding_cl60_r60/  \
       --conf-overwrite validation_config_v10_cmsmachine_norechits_cl60_r60.json
