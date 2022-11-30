# python condor_run_calomatch_dataset.py \
#        -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_UL18_pfRechitThres_DumperTraining \
#        -nfg 50 \
#        -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_UL18thres \
#        -q espresso -c --condor-folder ele_235_UL



# python condor_run_calomatch_dataset.py \
#        -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_235fb_pfRechitThres_DumperTraining \
#        -nfg 50 \
#        -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_235thres \
#        -q espresso -c --condor-folder ele_235_235

# python condor_run_calomatch_dataset.py \
#        -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_UL18_pfRechitThres_DumperTraining \
#        -nfg 50 \
#        -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_UL18thres \
#        -q espresso -c --condor-folder gamma_235_UL



# python condor_run_calomatch_dataset.py \
#        -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-500_pythia8_PremixRun3_13p6TeV_125X_mcRun3_2022_realistic_v4_235fbNoise_235fb_pfRechitThres_DumperTraining \
#        -nfg 50 \
#        -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_235thres \
#        -q espresso -c --condor-folder gamma_235_235





python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_UL18thres \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_UL18thres.hdf5

python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_235thres \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/electrons_235fbnoise_235thres.hdf5

python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_UL18thres \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_UL18thres.hdf5

python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_235thres \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_235fbnoise_235thres.hdf5
