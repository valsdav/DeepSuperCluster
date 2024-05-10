#!/bin/sh
BASEDIR=/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/

python condor_awkward_dataset.py -i $BASEDIR/electrons/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_v4 \
       -o $BASEDIR/electrons/run3_126X_2023/awkward_126X_mcRun3_2023_forPU65_v4 \
       -q espresso -nfg 5 -f features_definition.json -cf condor_awk_electrons_run3_2023 --flavour 11

python condor_awkward_dataset.py -i $BASEDIR/gammas/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_v4 \
       -o $BASEDIR/gammas/run3_126X_2023/awkward_126X_mcRun3_2023_forPU65_v4 \
       -q espresso -nfg 5 -f features_definition.json -cf condor_awk_gammas_run3_2023 --flavour 22


