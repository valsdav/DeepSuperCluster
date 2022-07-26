#!/bin/sh

# Non overlapping window
# python condor_ndjson.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_DeepSC_AlgoA_125X_bugFix/ \
#        -nfg 50 \
#        -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_2022_v1\
#        -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_v2.root \
#        -q espresso -e user \
#        --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron

# python condor_ndjson.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_DeepSC_AlgoA_125X_bugFix/\
#        -nfg 50 -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_2022_v1 \
#        -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_v2.root -q espresso -e user \
#        --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_gamma


# Overlapping window
python condor_ndjson.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_DeepSC_AlgoA_125X_bugFix/ \
       -nfg 50 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_2022_v1_overlapping\
       -a sim_fraction --wp-file simScore_Minima_ElectronsOnly_v2.root \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron_overlap \
       --overlap

python condor_ndjson.py -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_DeepSC_AlgoA_125X_bugFix/\
       -nfg 50 \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_2022_v1_overlapping \
       -a sim_fraction --wp-file simScore_Minima_PhotonsOnly_v2.root \
       -q espresso -e user \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_gamma_overlap \
       --overlap
