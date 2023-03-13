#!/bin/sh -e
ver=$1
options=" --reco-collection electron"


basedir="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/"
outdir="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/reco_regression_pfthresholds_studies/"


python condor_reco_dumper.py -i ${basedir}/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40_130X_mcRun3_2022_realistic_v2_235fbNoiseECAL_DeepSC_AlgoA_pfRechitThres_34sigma_235fb_Dumper \
       -o ${outdir}/ttbar/DeepSC_algoA_235noise_34sigma_v${ver}/ -a sim_fraction --overwrite-runid \
       --wp-file simfraction_electron_235_235.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ttbar_deepsc_34sigma

cd condor_dumper_ttbar_deepsc_34sigma;
condor_submit condor_job.txt
cd ..


python condor_reco_dumper.py -i ${basedir}/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40_130X_mcRun3_2022_realistic_v2_235fbNoiseECAL_DeepSC_AlgoA_pfRechitThres_UL18_2e3sigma_Dumper/ \
       -o ${outdir}/ttbar/DeepSC_algoA_235noise_UL18_v${ver}/ -a sim_fraction --overwrite-runid\
       --wp-file simfraction_electron_235_UL18.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ttbar_deepsc_UL18

cd condor_dumper_ttbar_deepsc_UL18;
condor_submit condor_job.txt
cd ..


python condor_reco_dumper.py -i ${basedir}/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40_130X_mcRun3_2022_realistic_v2_235fbNoiseECAL_Mustache_pfRechitThres_34sigma_235fb_Dumper \
       -o ${outdir}/ttbar/Mustache_235noise_34sigma_v${ver}/ -a sim_fraction --overwrite-runid\
       --wp-file simfraction_electron_235_235.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ttbar_mustache_34sigma
cd condor_dumper_ttbar_mustache_34sigma;
condor_submit condor_job.txt
cd ..


python condor_reco_dumper.py -i ${basedir}/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40_130X_mcRun3_2022_realistic_v2_235fbNoiseECAL_Mustache_pfRechitThres_UL18_2e3sigma_Dumper\
       -o ${outdir}/ttbar/Mustache_235noise_UL18_v${ver}/ -a sim_fraction --overwrite-runid \
       --wp-file simfraction_electron_235_UL18.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_ttbar_mustache_UL18

cd condor_dumper_ttbar_mustache_UL18;
condor_submit condor_job.txt
cd ..
