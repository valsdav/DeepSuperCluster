#!/bin/sh

MODELS_DIR="/work/dvalsecc/Clustering/models_archive/gcn_models/PFRechitsThresholsTests/"


##########################3
# 235noise 235 thresholds
python3 exporter_awk.py --model-config ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/model_forexport.py \
       -o  ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/model_smallpadding.pb \
       -os ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/scaler_config \
       --max-ncls 15 \
       --max-nrechits 20


python3 exporter_awk.py --model-config ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/model_forexport.py \
       -o  ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/model_largepadding.pb \
       -os ${MODELS_DIR}/models_235noise_235thres/run_11_235_235_1.5M_verylarge_smallerdrop/scaler_config \
       --max-ncls 60 \
       --max-nrechits 60



##########################
# 235noise UL18 thresholds
python3 exporter_awk.py --model-config ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/model_forexport.py \
       -o  ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/model_smallpadding.pb \
       -os ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/scaler_config \
       --max-ncls 15 \
       --max-nrechits 20


python3 exporter_awk.py --model-config ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/training_config.json \
       --model-weights weights.best.hdf5 \
       --model-python ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/model_forexport.py \
       -o  ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/model_largepadding.pb \
       -os ${MODELS_DIR}/models_235noise_UL18thres/run_16_235_UL18_1.5M_verylarge_smallerdrop/scaler_config \
       --max-ncls 60 \
       --max-nrechits 60



