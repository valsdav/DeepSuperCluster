#!/bin/sh -e

# python ../condor_ndjson.py -q espresso \
# -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_v10 \
# -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_FULL/hadd/ \
# -tf 0 --min-et-seed 1. --maxnocalow 2 -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg $1 --compress

python ../condor_ndjson_jet.py -q espresso \
 -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/jets/ndjson_v11 \
 -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourJetsGunPt1-100_EMEnriched_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper_SLIM_v2/hadd \
 -tf 0 --min-et-seed 1. --maxnocalow 0 -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_updated.root -nfg $1 --compress

