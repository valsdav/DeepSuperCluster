## Code to analyze and evaluate the DeepSC performance on the RECO output

## Reco dumper
General dumper of SuperCluster info to be applied separately on DeepSC reco and Mustache reco collection. 
This dumper saves a flat pandas dataframe for further analysis of the final performance. The 
script containing the code is called `reco_dumper.py` and `run_reco_dumper.py`.

The dumper can run in two different modes: **loop-on-calo** or **loop-on-reco**. 

- When **loop-on-calo** mode is active information is saved for each PFCluster seed matched to a CaloParticle. 
- When **loop-on-reco** mode is active, the dumper works on the final SC or reco (electron/photon) collection and saves
  information for each reco object. 
  
For each object many piece of information are saved:
- calomatching information
- gen matching (by deltaR) information
- position and energy of the seed and of the object. 
- Number of selected clusters
- Gen and Sim energy of the matched calo or genparticle, if any. 
- PU info
- Number of event and run to be able to perform matching in the case the reconstruction is done two times with different
  algos. 
  
### How to run the reco dumper
In order to run the reco dumper on condor go to the subfoler `run_condor_dumper`. 

Use the script `run_reco_dumper.py` to prepare the job submission: 

```bash
python run_reco_dumper.py -h                                                 
usage: run_reco_dumper.py [-h] -i INPUTFILE [-o OUTPUTFILE] [-a ASSOC_STRATEGY] [--wp-file WP_FILE] [-n NEVENTS [NEVENTS ...]] [-d] [--maxnocalow MAXNOCALOW] [--min-et-seed MIN_ET_SEED] [--loop-on-calo] [-s SC_COLLECTION] [-r RECO_COLLECTION]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputfile INPUTFILE
                        inputfile
  -o OUTPUTFILE, --outputfile OUTPUTFILE
                        outputfile
  -a ASSOC_STRATEGY, --assoc-strategy ASSOC_STRATEGY
                        Association strategy
  --wp-file WP_FILE     File with sim fraction thresholds
  -n NEVENTS [NEVENTS ...], --nevents NEVENTS [NEVENTS ...]
                        n events iterator
  -d, --debug           debug
  --maxnocalow MAXNOCALOW
                        Number of no calo window per event
  --min-et-seed MIN_ET_SEED
                        Min Et of the seeds
  --loop-on-calo        If true, loop only on calo-seeds, not on all the SC
  -s SC_COLLECTION, --sc-collection SC_COLLECTION
                        SuperCluster collection
  -r RECO_COLLECTION, --reco-collection RECO_COLLECTION
                        Reco collection (none/electron/photon)

```

For example to analyze the electron collection looping on the reco objects. 

```
    python ../condor_run_dumper.py -i
    ${basedir}/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_123X_mcRun3_2021_realistic_v11_UL18_pfRechitThres_Dumper_SCRegression_EleRegression_Mustache_125X_bugFix 
    -o ${outdir}electrons/ele_UL18_123X_Mustache_v1/ 
    -a sim_fraction 
    --wp-file simScore_Minima_ElectronsOnly_updated.root 
    -nfg 40 -q espresso --compress 
    --reco-collection electron

```

Then run the condor submission `condor_submit condor_job.txt`. 

It is convienient to submit the jobs for different datasets in different folders: have a look at the script `run_all.sh`
for an example. 

At the end of the condor processing the dataset needs to be joint in a single pandas dataframe for convinience: 

```bash
python join_datasets.py -i /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_Mustache_v${ver} -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons/ele_UL18_123X_Mustache_v${ver}_{type}.h5py
```


## Reco comparison (obsolete)
Mustache-DeepSC comparison event by event. 
To be used only on special ntuples containing both the DeepSC and Mustache collections. 

This is obsolete, please refer to the more general reco dumper code above. 

