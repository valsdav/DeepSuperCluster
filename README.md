# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh


# Numpy output (simple)
The script **cluster_to_numpy_simple.py** prepares the simplest possible input for machine learning cluster selection. 

Baseline for windows eta-phi: 30-80

# Numpy output (complex)

The script **cluster_tonumpy_v2.py** prepares information for machine learning techniques. 

* Metadata are saved for each window and for each cluster. A map of energies is created for each window. 

* Windows are created either if the seed corresponds to a calopartible or not