#!/bin/sh -e
BASEDIR="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/reco_regression_pfthresholds_studies/ttbar_genmatching/"
ver=$1

python join_datasets.py -i ${BASEDIR}/Mustache_235noise_34sigma_v${ver}/ -o ${BASEDIR}/Mustache_235noise_34sigma_v${ver}.h5py

python join_datasets.py -i ${BASEDIR}/Mustache_235noise_UL18_v${ver}/ -o ${BASEDIR}/Mustache_235noise_UL18_v${ver}.h5py


python join_datasets.py -i ${BASEDIR}/DeepSC_algoA_235noise_34sigma_v${ver}/ -o ${BASEDIR}/DeepSC_algoA_235noise_34sigma_v${ver}.h5py

python join_datasets.py -i ${BASEDIR}/DeepSC_algoA_235noise_UL18_v${ver}/ -o ${BASEDIR}/DeepSC_algoA_235noise_UL18_v${ver}.h5py
