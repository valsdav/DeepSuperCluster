#!/bin/sh -e
ver=$1
options=" --reco-collection electron"


basedir="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/"
outdir="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/reco_ichep22/"



python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_EleRegression_DeepSC_AlgoA_125X_bugFix  \
       -o ${outdir}electrons/DeepSC_algoA_ICHEP22_UL18_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_UL18.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ele_deepsc_UL18_ichep22

cd condor_dumper_ele_deepsc_UL18_ichep22;
condor_submit condor_job.txt
cd ..

.


python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_Mustache_125X_bugFix  \
       -o ${outdir}electrons/Mustache_ICHEP22_UL18_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_UL18.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ele_mustache_UL18_ichep22

cd condor_dumper_ele_mustache_UL18_ichep22;
condor_submit condor_job.txt
cd ..
