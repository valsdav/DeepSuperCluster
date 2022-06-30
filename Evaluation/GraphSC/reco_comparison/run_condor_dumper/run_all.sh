#!/bin/sh -e
ver=$1
options=" --reco-collection electron"


basedir="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/"
outdir="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/reco_final_regression/"


mkdir ele_must
cd ele_must
mkdir error log output
python ../condor_run_dumper.py -i ${basedir}/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_EleRegression_Mustache_125X_bugFix  -o ${outdir}electrons/ele_UL18_123X_Mustache_v${ver}/ -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg 40 -q espresso --compress ${options}

condor_submit condor_job.txt

 cd ..
mkdir ele_deepsc
cd ele_deepsc
mkdir error log output
python ../condor_run_dumper.py -i ${basedir}/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_EleRegression_DeepSC_AlgoA_125X_bugFix -o ${outdir}/electrons/ele_UL18_123X_DeepSC_AlgoA_v${ver}/ -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_updated.root -nfg 40 -q espresso --compress ${options}

condor_submit condor_job.txt




# options=" --reco-collection photon"

# # cd ..
 # mkdir gamma_must
# cd gamma_must
# mkdir error log output
# python ../condor_run_dumper.py -i ${basedir}/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_PhoRegression_Mustache_125X_bugFix -o ${outdir}/gammas/gamma_UL18_123X_Mustache_v${ver}/ -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_updated.root -nfg 40 -q espresso --compress ${options}

# condor_submit condor_job.txt

# cd ..
# mkdir gamma_deepsc
# cd gamma_deepsc
# mkdir error log output
# python ../condor_run_dumper.py -i ${basedir}/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_PhoRegression_DeepSC_AlgoA_125X_bugFix  -o ${outdir}/gammas/gamma_UL18_123X_DeepSC_AlgoA_v${ver}/ -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_updated.root -nfg 40 -q espresso --compress ${options}

# condor_submit condor_job.txt

# cd ..

