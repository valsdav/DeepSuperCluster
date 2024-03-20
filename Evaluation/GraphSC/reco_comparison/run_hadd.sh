#!/bin/sh

python hadd_files_groups.py -i \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_Mustache_thres235fb/230307_154857/0000  \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_Mustache_thres235fb/230307_154857/0001 \
       -o /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_Mustache_thres235fb/hadd \
-n 20 -c 4



python hadd_files_groups.py -i \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thres235fb/230307_155019/0000  \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thres235fb/230307_155019/0001 \
       -o /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thres235fb/hadd \
-n 20 -c 4

python hadd_files_groups.py -i \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thresUL18/230307_154940/0000  \
       /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thresUL18/230307_154940/0001 \
       -o /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/TTTo2L2Nu_powheg_pythia8_13p6TeV_PremixRun3PU40/RECO_DeepSC_algoA_thresUL18/hadd \
-n 20 -c 4
