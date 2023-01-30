#!/bin/sh
BASEDIR=/eos/user/p/psimkina/

python condor_awkward_dataset.py -i $BASEDIR/DeepSuperCluster_jetData_19_01_23 \
       -o $BASEDIR/DeepSuperCluster_jetData_19_01_23/awkward_arrays \
       -q espresso -nfg 10 -f features_definition_jet.json -cf condor_awk_electrons_235_UL18 --flavour 0
