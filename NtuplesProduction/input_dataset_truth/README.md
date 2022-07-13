# Truth preparation

This folder contains the code and notebook to analyze the raw dataset in order to extract the best truth information for
the seed-cluster matching. 

## Dataset preparation

A simple ntuplizer is applied on the output of the [RecoSimDumper](https://github.com/bmarzocc/RecoSimStudies).
The baseline files contain all the objects for each event: pfClusters,Rechits, caloparticle info. 

The ntuplizer `calo_match_dataset.py` performs the following steps in order to obtain a dataset useful for the truth
matching study. 

### CaloParticle - PfCluster matching

We need to identify which pfCluster are linked to which CaloParticle. To do that we compute a score for each pfCluster
and CaloParticle, called **simFraction**. 

This score is computed as:

    (sum of the simEnergy deposited by the CaloFraction in the pfCluster weighted by hits_and_fraction)/(total simEN of
    the caloparticle )

Once the score is known the following matching algorithm is applied: 
- Each pfCluster is associated to the CaloParticle which has the larger simFraction for him. A minimum fraction of 1e-4
  is considered for the association. 
- Each CaloParticle keeps a list of the associated pfCluster 
  - The pfCluster with the larger simScore for each caloParticle is considered the truth-level seed of the CaloParticle. 

This procedure is implemented in `calo_association.py`.

### Ntuplizer content

The following procedure is implementedin the ntuplized `calo_match_dataset.py`

1. PfCluster-CaloParticle association is done
2. **for each caloparticle**, the highest score pfCluster is taken as **seed**.  A `window_index` is defined to
    uniquely identify the seed.  The seed is required to have at least 1% of simFraction. 
3. All the clusters associated to the same CaloParticle are also saved, with the same `window_index` to be able to
    group them later


For each Cluster several info are saved in the ntuples
- dEta, dPhi wrt the pfCluster considered as seed. 
- energy, position
- sim energy of the signal (and PU if available)
- associated caloparticle info. 

#### Run the ntuplizer

To run the ntuplizer a condor preparation script is available: 

    python condor_run_calomatch_dataset.py -i
    /eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_DeepSC_AlgoA_125X_bugFix/
    -nfg 70 -o [...]/output_deepcluster_dumper/input_truth/gammas_v2 -q espresso -c
    --condor-folder condor_run_gamma
    
Then go to the `condor_folder` and just do `condor_submit condor_job.txt`. 
This will create an output file for each group of input files. 

To collect all the results in a single file run the script: 

    python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/input_truth/gammas_v2  -o [...]input_truth/gammas/calo_matched_clusters_dataset_v2.hdf5


## Truth computation

Two ingredients can be extracted from this basic ntuples to define the truth level of the DeepSC algorithm. 

- *Window dimension*:  the maximum dimension in deltaEta,deltaPhi around the seed needs to be defined a priory in order
  to be inclusive. The DeepSC model will be trained inside this window and the CMSSW reconstrion will work with this
  boundary. 

- *Truth label*:  simFraction thresholds are optimized in bins of (eta_seed, Et seed) in order to obtain the best
  possible resolution of the cluster, defined as (sum pfCluster_energy)/(caloParticle energy).
  These thresholds will be used to define the truth labels for the DeepSC supervised learning algo. 
  
The notebooks `Truth_definition_study_electrons.ipynb` and `Truth_definition_study_photons.ipynb` contain the code and
explanation of the truth computation. 

