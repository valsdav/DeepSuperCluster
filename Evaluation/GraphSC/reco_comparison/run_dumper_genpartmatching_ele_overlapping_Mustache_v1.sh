#!/bin/sh -e
ver=$1
options=" --reco-collection genparticle "


basedir="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DumperTraining"
outdir="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/overlapping_objects_studies/"


python condor_reco_dumper.py -i ${basedir} \
       -o ${outdir}electrons_genmatching/Mustache_126X_mcRun3_2023_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_UL18.json -nfg 40 -q espresso --compress ${options} -cf condor_dumper_genmatching_ele_Mustache

