#!/bin/sh
BASEDIR=/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/

python condor_awkward_dataset.py -i $BASEDIR/electrons/ndjson_pfthresholds_studies/ndjson_235noise_UL18thres \
       -o $BASEDIR/electrons/awkward_235noise_UL18thres \
       -q espresso -nfg 10 -f features_definition.json -cf condor_awk_electrons_235_UL18 --flavour 11


python condor_awkward_dataset.py -i $BASEDIR/electrons/ndjson_pfthresholds_studies/ndjson_235noise_235thres \
       -o $BASEDIR/electrons/awkward_235noise_235thres \
       -q espresso -nfg 10 -f features_definition.json -cf condor_awk_electrons_235_235 --flavour 11


python condor_awkward_dataset.py -i $BASEDIR/gammas/ndjson_pfthresholds_studies/ndjson_235noise_UL18thres \
       -o $BASEDIR/gammas/awkward_235noise_UL18thres \
       -q espresso -nfg 10 -f features_definition.json -cf condor_awk_gammas_235_UL18 --flavour 22


python condor_awkward_dataset.py -i $BASEDIR/gammas/ndjson_pfthresholds_studies/ndjson_235noise_235thres \
       -o $BASEDIR/gammas/awkward_235noise_235thres \
       -q espresso -nfg 10 -f features_definition.json -cf condor_awk_gammas_235_235 --flavour 22

