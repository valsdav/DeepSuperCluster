# ECAL SuperClustering with deep learning

Repository for ECAL SuperClustering with machine learning, also called **DeepSC**. 
It contains both the code to produce the training ntuples and the training and evaluation code. 

Documentation in progress...

# Setup environment

```bash
git clone https://github.com/valsdav/DeepSuperCluster.git

source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh
```
# Dataset preparation
The `NtupleProduction` folder contains the scripts used to produce the necessary ntuples for training the models. 

# Training 
The `Training` folder contains notebooks and scripts with the tensorflow/keras models and the training procedures. 
Model are saved in the `models` folder.

# Evaluation
The `Evaluation` folder contains plotting scripts and notebook. 

# Tools
The `Tools` folder will contains miscellaneous utilities scripts.
