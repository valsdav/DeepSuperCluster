#!/bin/sh -e

python run_reco_comparison.py -i  \
"/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper_AlgoA/crab_FourGammasGunPt1-100_Dumper_AlgoA/220214_094249/0000/output_*.root"  \
-o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/RecoPlots/RecoComparison_v1/dataset_photon.h5py -a sim_fraction --wp-file \
/afs/cern.ch/work/d/dvalsecc/private/Clustering_tools/DeepSuperCluster/NtuplesProduction/simScore_WP/simScore_Minima_PhotonsOnly_updated.root


python run_reco_comparison.py -i \
"/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper_AlgoA/crab_FourElectronsGunPt1-100_Dumper_AlgoA/220214_092524/0000/output_*.root" \
-o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/RecoPlots/RecoComparison_v1/dataset_electron.h5py -a sim_fraction --wp-file \
/afs/cern.ch/work/d/dvalsecc/private/Clustering_tools/DeepSuperCluster/NtuplesProduction/simScore_WP/simScore_Minima_ElectronsOnly_updated.root


