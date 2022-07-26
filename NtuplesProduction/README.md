# Training dataset preparation

## Goal
We need a dataset formed by a list of detector "windows" (eta-phi rectangular regions) containing the clusters and the caloparticle (gen level info) creating them. This window is the unit of the dataset:  both the training and the final reconstruction algorithm will be applied on it. 

These windows can be created with a lot of flexibility. Details on how they are formed are really important for the
characteristics of the training dataset. 

### Truth labels
The first step is analyzing the input dataset with minimum requirements to understand the pfClusters-CaloParticle
interplay. The truth labels needs to be defined to obtain the best possible resolution of the SuperCluster. 

Code and plotting scripts for the truth level analysis are described in the folder [input_dataset_truth](./input_dataset_truth/).

## Procedure

- The Dumper is applied on top of RECO dataset to extract TTrees containing all the necessary informations about clusters, caloparticles, rechits, simhits. 
- The script `windows_creator_dynamic_global_overlap` is applied on the dumper output in order to extract a list of detector windows for each event. 
    - this version of the script creates all the possible windows around all the possible seeds and save the list of clusters inside each window. Windows can be overalapped and the clusters can be associated to multiple seeds. 
    - Different versions of this script generate the windows in different ways. 
- To apply the window creator algo the helper script `cluster_ndjson_dynamic_global_overlap.py` is used:
  - This script uses the dumper input tree, applies the window creation step and save text file containing 1 window for each line. The dictionary containing the information for each window is saved in json format. 
  - The txt file corresponding to each input file is saved and compressed
- In order to apply these scripts in an efficient way an helper script for condor submission has been prepared: `condor_ndjson.py`
  - An example on how to run is is in `prod_ntuple.sh`

### Tensorflow dataset format

The dataset built with the script described above is not suitable for a fast integration with the tensorflow library.  So it is converted in the TFRecord format in order to use the tf.data facilities (https://www.tensorflow.org/api_docs/python/tf/data/Dataset). 

The script to do that is: `convert_tfrecord_dataset_allinfo.py`. This script defines the information that will be part of the TFrecord dataset. 

The helper script to run the conversion on condor is `condor_tfrecords.py`


### Window creation details

- All the windows are created for all the seeds with Et> 1
- The seed is required to have:
  - at least 1% of simfraction of the matched caloparticle
  - the matched caloparticle needs to be inside the window
  - no simfraction WP is checked for the seed

- The clusters are calo_matched to the caloparticle of the seed if:
  - they pass the sim fraction threshold WP
  - it is not checked the distance from the caloparticle, the calo needs to be in the same window of the seed

- All the clusters are put in all the window

- Saved both calo_match and calo_seed flags both for the seed and for the clusters
  - For the clusters the calo is always the calo of the seed. 
  - If the seed is not associated with a calo the matching of the cluster is not checked


### Production v9

#### Electron

First step for Ndjson windows output.
```
python ../condor_ndjson.py -q espresso \
 -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/ndjson_v9 \
 -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_v2/FourElectronsGunPt1_Dumper_v2_hadd \
 -tf 0 --min-et-seed 1. --maxnocalow 4 -a sim_fraction --wp-file simScore_Minima_ElectronsOnly.root -nfg 1 --compress
```


#### Photons

First step for Ndjson windows output.
```
python ../condor_ndjson.py -q espresso \
 -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/ndjson_v9 \
 -i /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper/hadd \
 -tf 0 --min-et-seed 1. --maxnocalow 4 -a sim_fraction --wp-file simScore_Minima_PhotonsOnly.root -nfg 1 --compress
```



### Features in final dataset

seed_features = ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                     "en_true","et_true",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ]

seed_labels = [ "is_seed_calo_matched","is_seed_calo_seed","is_seed_mustach_matched"]
seed_metadata = ["nclusters_insc","max_en_cluster_insc","max_deta_cluster_insc",
                   "max_dphi_cluster_insc", "max_en_cluster","max_deta_cluster","max_dphi_cluster","seed_score" ]

cls_features = [  "cluster_ieta","cluster_iphi","cluster_iz",
                     "cluster_deta", "cluster_dphi",
                     "en_cluster","et_cluster", 
                     "en_cluster_calib", "et_cluster_calib",
                    "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                    "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                    "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                    "cl_sigmaIphiIphi","cl_swissCross",
                    "cl_nxtals", "cl_etaWidth","cl_phiWidth",
                    ]

cls_labels = ["is_seed","is_calo_matched","is_calo_seed", "in_scluster","in_geom_mustache","in_mustache"]
cls_metadata = [ "calo_score" ]

