#!/bin/sh -e
BASEDIR="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/reco_regression_pfthresholds_studies/electrons/"
ver=$1

# python join_datasets.py -i ${BASEDIR}/Mustache_235noise_34sigma_v3/ -o ${BASEDIR}/Mustache_235noise_34sigma_v3.h5py

# python join_datasets.py -i ${BASEDIR}/Mustache_235noise_UL18_v3/ -o ${BASEDIR}/Mustache_235noise_UL18_v3.h5py


python join_datasets.py -i ${BASEDIR}/DeepSC_algoA_235noise_34sigma_v3/ -o ${BASEDIR}/DeepSC_algoA_235noise_34sigma_v3.h5py

# python join_datasets.py -i ${BASEDIR}/DeepSC_algoA_235noise_UL18_v3/ -o ${BASEDIR}/DeepSC_algoA_235noise_UL18_v3.h5py
