# Reconstruction performance evaluation

The notebooks in this folder are used to evaluate the performance of the final CMSSW reconstruction and to compare the
electrons and photons energy resolution between Mustache and DeepSC superclusters. 

The script [plotting_code.py](./plotting_code.py) contains the basic code for the final performance plots. 

An example with the evaluation of the final resolution can be found in the notebooks:

- Electrons: [Plotting_RecoRegression_electron.ipynb](./Plotting_RecoRegression_electron.ipynb)
- Photons: [Plotting_RecoRegression_electron.ipynb](./Plotting_RecoRegression_electron.ipynb). 

These notebooks have been used to produce the results approved for ICHEP2022. 

## Debugging notebooks

All the other notebooks have been used for investigation and debugging. 

- v8: keeping original linking (only lower energy seeds linked, there is not effect on calo matched objects) and analyse the regressed energy

- Now comparing the latest reco but with all the links between the seeds (new) or with only lower energy linked (old) to see the effect.

- v7: running on the latest 12_4_0_pre4 version of the PR and fixig a change that was introduced in the linking of the seeds. In fact, only lower energy seeds were linked to 
  seeds in the older version of the reconstruction. Then all the links were added. In this notebook we again setup the link as the old version (only lower energy seeds)

- v6 added number of clusters in the window to understand the differences wrt of the older reconstruction version.

- v5 Loop on calo-seed but matching both the deepSC and Mustache reconstruction on the same seed. In pratice we request that a seed is the same for both the algos, as in the reco_comparison plots

- V4 the loop is on calo-seeds only: same results as expected

- V3 the calomatching has been corrected adding the cut on the 1% simFraction for the seed and the `inWindow(seed,calo)` requirement.  This should fix the tails and result in a Sim-matched resolution similar to the one observed before. --> At the end the result is not changed much
