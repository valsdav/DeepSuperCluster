# Training dataset preparation

## Goal
We need a dataset formed by a list of detector "windows" (eta-phi rectangular regions) containing the clusters and the caloparticle (gen level info) creating them. This window is the unit of the dataset:  both the training and the final reconstruction algorithm will be applied on it. 

These windows can be created with a lot of flexibility. Details on how they are formed are really important for the
characteristics of the training dataset. 

### Truth labels
The first step is analyzing the input dataset with minimum requirements to understand the pfClusters-CaloParticle
interplay. The truth labels needs to be defined to obtain the best possible resolution of the SuperCluster. 

Code and plotting scripts for the truth level analysis are described in the folder [input_dataset_truth](./input_dataset_truth/).

## Training dataset preparation

### Software environment
These scripts do not need CMSSW, but only a recent LCG environment with python>=3.8
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh
```

### First step

- The Dumper is applied on top of RECO dataset to extract TTrees containing all the necessary informations about clusters, caloparticles, rechits, simhits. 
- The script `windows_creator_general.py` is applied on the dumper output to extract a list of detector windows for each event. 
    - this script creates all the possible windows around all the possible seeds and save the list of clusters inside
 each window.
    - Windows can be overlapped (optionally)
    - All the clusters can be associated to multiple seeds. 
    
- To apply the window creator algo the helper script `cluster_ndjson_general.py` is used:
  - This script reads the input TTree, applies the window creation code and saves a text file containing 1 window for each line. The dictionary containing the information for each window is saved in json format. 
  - The txt file corresponding to each input file is saved and compressed
  
- The script `condor_ndjson.py` runs the window creation script on condor on all the files in parallel. 

```bash
 python condor_ndjson.py -h
usage: condor_ndjson.py [-h] -i INPUTDIR -nfg NFILE_GROUP -o OUTPUTDIR -a ASSOC_STRATEGY [--wp-file WP_FILE] -q QUEUE [-e EOS] [--maxnocalow MAXNOCALOW] [--min-et-seed MIN_ET_SEED] [-ov] [--pu-limit PU_LIMIT] [-c] [--redo] [-d] [-cf CONDOR_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTDIR, --inputdir INPUTDIR
                        Inputdir
  -nfg NFILE_GROUP, --nfile-group NFILE_GROUP
                        How many files per numpy file
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Outputdir
  -a ASSOC_STRATEGY, --assoc-strategy ASSOC_STRATEGY
                        Association strategy
  --wp-file WP_FILE     File with sim fraction thresholds
  -q QUEUE, --queue QUEUE
                        Condor queue
  -e EOS, --eos EOS     EOS instance user/cms
  --maxnocalow MAXNOCALOW
                        Number of no calo window per event
  --min-et-seed MIN_ET_SEED
                        Min Et of the seeds
  -ov, --overlap        Overlapping window mode
  --pu-limit PU_LIMIT   SimEnergy PU limit
  -c, --compress        Compress output
  --redo                Redo all files
  -d, --debug           debug
  -cf CONDOR_FOLDER, --condor-folder CONDOR_FOLDER
                        Condor folder
```

An example of the commands used for the main productions are documented in the folder [prod_scripts](./prod_scritps)

### Output formats


#### Awkward format
The `ndjson` dataset can also be transformed in Awkward arrays for convinient analysis. 
The script `convert_awkward_dataset.py` reads the `ndjson` files and creates parquet files.
Condor jobs are prepared by `condor_awkward_dataset.py`.

```bash
python condor_awkward_dataset.py -h
usage: condor_awkward_dataset.py [-h] -i INPUTDIR -nfg NFILE_GROUP -o OUTPUTDIR -q QUEUE [-f FEATURES_DEF] [-cf CONDOR_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTDIR, --inputdir INPUTDIR
                        Inputdir
  -nfg NFILE_GROUP, --nfile-group NFILE_GROUP
                        How many files per tfrecord file
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Outputdir
  -q QUEUE, --queue QUEUE
                        Condor queue
  -f FEATURES_DEF, --features-def FEATURES_DEF
                        Features definition file
  -cf CONDOR_FOLDER, --condor-folder CONDOR_FOLDER
                        Condor folder
```
An example of this script can be found in [prod_scripts](./prod_scritps/awkward_2022v1.sh).

The output files are saved in **parquet** format. In order to be able to use all the files in the same folder as a
single parquet dataframe the metadata of the folder must be properly updated. 
An helper script is available to do that, **to be run at the end of all the jobs**: 

```bash
python finalize_awkward_dataset.py --inputdir FOLDER
```


#### Tensorflow dataset format

The NDjson dataset can be converted to a format created for fast integration with Tensorflow libraries. 
It can be converted in the TFRecord format in order to use the tf.data facilities (https://www.tensorflow.org/api_docs/python/tf/data/Dataset). 

The script to do that is: `convert_tfrecord_dataset_allinfo.py`. This script defines the information that will be part of the TFrecord dataset. 

The helper script to run the conversion on condor is `condor_tfrecords.py`

## Window creation details

- All the windows are created for all the seeds with Et> 1
- The seed is required to have:
  - at least 1% of simfraction of the matched caloparticle
  - the matched caloparticle Gen position  needs to be inside the geometric window defined by the seed: this is needed
  to avoid the border effect in the crack
  - no simfraction WP is checked for the seed

- The clusters are calo_matched to the caloparticle of the seed if:
  - they pass the sim fraction threshold WP
  - it is not checked the distance from the caloparticle, the calo needs to be in the same window of the seed

- All the clusters are put in all the window

- Saved both calo_match and calo_seed flags both for the seed and for the clusters
  - For the clusters the calo is always the calo of the seed. 
  - If the seed is not associated with a calo the matching of the cluster is not checked

- Windows can be created in two modes: overlapping/non-overlapping:
  - Overlapping: clusters with at least 1 GeV of Et and passing the requirement for the seeds always create a window,
    also if they are already inside another window. 
  - Non-overlapping: seeds create a new window only if they are not inside the window defined by an higher energy
    cluster. 




### Features in final dataset


```python
default_features_dict = {
        "cl_features" : [ "en_cluster","et_cluster",
                        "cluster_eta", "cluster_phi", 
                        "cluster_ieta","cluster_iphi","cluster_iz",
                        "cluster_deta", "cluster_dphi",
                        "cluster_den_seed","cluster_det_seed",
                        "en_cluster_calib", "et_cluster_calib",
                        "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                        "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                        "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                        "cl_sigmaIphiIphi","cl_swissCross",
                        "cl_nxtals", "cl_etaWidth","cl_phiWidth"],


    "cl_metadata": [ "calo_score", "calo_simen_sig", "calo_simen_PU",
                     "cluster_PUfrac","calo_nxtals_PU",
                     "noise_en","noise_en_uncal","noise_en_nofrac","noise_en_uncal_nofrac" ],

    "cl_labels" : ["is_seed","is_calo_matched","is_calo_seed", "in_scluster","in_geom_mustache",],

    
    "seed_features" : ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth"
                    ],

    "seed_metadata": [ "seed_score", "seed_simen_sig", "seed_simen_PU", "seed_PUfrac"],

    "window_features" : [ "max_en_cluster","max_et_cluster","max_deta_cluster",
                           "max_dphi_cluster","max_den_cluster","max_det_cluster",
                           "min_en_cluster","min_et_cluster","min_deta_cluster",
                           "min_dphi_cluster","min_den_cluster","min_det_cluster",
                           "mean_en_cluster","mean_et_cluster","mean_deta_cluster",
                           "mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ],

    "window_metadata": ["flavour", "ncls", "nclusters_insc",
                        "nVtx", "rho", "obsPU", "truePU",
                        "sim_true_eta", "sim_true_phi",  
                        "gen_true_eta","gen_true_phi",
                        "en_true_sim","et_true_sim", "en_true_gen", "et_true_gen",
                        "en_true_sim_good", "et_true_sim_good",
                        "en_mustache_raw", "et_mustache_raw","en_mustache_calib", "et_mustache_calib",
                        "max_en_cluster_insc","max_deta_cluster_insc","max_dphi_cluster_insc",
                        "event_tot_simen_PU","wtot_simen_PU","wtot_simen_sig",
                        "is_seed_calo_matched", "is_seed_calo_seed", "is_seed_mustache_matched"],
}

```


## Datasets logs

- **ndjson_2022_v3** (12-10-2022): found a bug in `window_creator_general.py`: some clusters at low energy were
  skipped. That's why there were less events with small Et of the seed. 

- **ndjson_2022_v2**: new ndjson with optimized window dimension and re-computed sim-fraction. (N.B. the in_mustache
  label is wrong, please use only the in_geom_mustache label.)
  
   - **awkward_2022v9_onlycalomatched**:  corresponding awkward dataset. The association normalization factor files and
     reweighting files are:
     - normalization_factors_v9_onlycalomatched.json
     - total_reweighting_v9_calomatched.json
 
