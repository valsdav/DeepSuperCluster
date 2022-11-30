#!/bin/sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh

BASEDIR="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering"

# ## Electrons
# # noise 235fb, pfrechit threshold UL18
python condor_ndjson.py -i ${BASEDIR}/FourElectronsGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_UL18_pfRechitThres_DumperTraining \
       -nfg 25  --maxnocalow 0 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_pfthresholds_studies/ndjson_235noise_UL18thres\
       -a sim_fraction --wp-file simfraction_electron_235_UL18.json \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron_235_UL18

# noise 235fb, pfrechit threshold 235fb
python condor_ndjson.py -i ${BASEDIR}/FourElectronsGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_235fb_pfRechitThres_DumperTraining \
       -nfg 25  --maxnocalow 0 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_pfthresholds_studies/ndjson_235noise_235thres\
       -a sim_fraction --wp-file simfraction_electron_235_235.json \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron_235_235


## Photons
# noise 235fb, pfrechit threshold UL18
python condor_ndjson.py -i ${BASEDIR}/FourGammasGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_UL18_pfRechitThres_DumperTraining \
       -nfg 25  --maxnocalow 0 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_pfthresholds_studies/ndjson_235noise_UL18thres\
       -a sim_fraction --wp-file simfraction_photon_235_UL18.json \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_photon_235_UL18

# noise 235fb, pfrechit threshold 235fb
python condor_ndjson.py -i ${BASEDIR}/FourGammasGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_235fb_pfRechitThres_DumperTraining \
       -nfg 25  --maxnocalow 0 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_pfthresholds_studies/ndjson_235noise_235thres\
       -a sim_fraction --wp-file simfraction_photon_235_235.json \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_photon_235_235

