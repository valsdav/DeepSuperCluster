#!/bin/sh -e

ver=$1

python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_DeepSC_AlgoA_${ver} -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_DeepSC_AlgoA_${ver}_{type}.h5py


python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_Mustache_${ver} -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_Mustache_${ver}_{type}.h5py



python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_DeepSC_AlgoA_${ver} -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_DeepSC_AlgoA_${ver}_{type}.h5py


python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_Mustache_${ver} -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_Mustache_${ver}_{type}.h5py
