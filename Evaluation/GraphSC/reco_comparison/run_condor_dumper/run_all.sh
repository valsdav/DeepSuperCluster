#!/bin/sh -e
ver=$1
options=" --loop-on-calo"

mkdir ele_must
cd ele_must
mkdir error log output
python ../condor_run_dumper.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_SCRegression_Mustache_Dumper  -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_Mustache_v${ver}/ -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg 30 -q espresso --compress ${options}

condor_submit condor_job.txt

cd ..
mkdir ele_deepsc
cd ele_deepsc
mkdir error log output
python ../condor_run_dumper.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_SCRegression_DeepSC_AlgoA_Dumper  -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_DeepSC_AlgoA_v${ver}/ -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg 30 -q espresso --compress ${options}

condor_submit condor_job.txt

cd ..
mkdir gamma_must
cd gamma_must
mkdir error log output
python ../condor_run_dumper.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_SCRegression_Mustache_Dumper  -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_Mustache_v${ver}/ -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_updated.root -nfg 30 -q espresso --compress ${options}

condor_submit condor_job.txt

cd ..
mkdir gamma_deepsc
cd gamma_deepsc
mkdir error log output
python ../condor_run_dumper.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_SCRegression_DeepSC_AlgoA_Dumper  -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/gammas/gamma_UL18_123X_DeepSC_AlgoA_v${ver}/ -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_updated.root -nfg 30 -q espresso --compress ${options}

condor_submit condor_job.txt

cd ..
