#!/bin/sh -e


 python ../condor_ndjson.py -q espresso \
 -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_v10 \
 -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_FULL/hadd/ \
 -tf 0 --min-et-seed 1. --maxnocalow 2 -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg $1 --compress

#  python condor_numpy.py -q espresso \
#  -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_v2 \
#  -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_withPU_withTracker_106X_mcRun3_2021_realistic_v3_RAW_StdSeedingGathering_Mustache_optimizedDeepSC_v17_joindet_elegamma_EBEE \
#  -tf 0 --min-et-seed 1. --maxnocalow 8 -a sim_fraction --wp-file simScore_Minima_PhotonsOnly.root -nfg $1 --compress
