from __future__ import print_function
import ROOT as R 
R.gROOT.SetBatch(True)
#R.PyConfig.IgnoreCommandLineOptions = True

import sys 
import os
from collections import defaultdict
from itertools import islice, chain
from numpy import mean
from array import array
from math import sqrt
from pprint import pprint as pp
import numpy as np
import argparse
import pickle
import pandas as pd
from multiprocessing import Pool
from reco_comparison import WindowCreator

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile", default="clusters_data.pkl")
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", default="sim_fraction")
parser.add_argument("--wp-file", type=str,  help="File with sim fraction thresholds")
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1.)
args = parser.parse_args()

import glob
inputfiles = glob.glob(args.inputfile)

simfraction_thresholds_file = R.TFile(args.wp_file)
simfraction_thresholds = simfraction_thresholds_file.Get("h2_Minimum_simScore_seedBins")

# Parameters controlling the creation of the window
# min simFraction for the seed with a signal caloparticle
SEED_MIN_FRACTION=1e-2
# min simFraction for the cluster to be associated with the caloparticle
CL_MIN_FRACION=1e-4
# threshold of simEnergy PU / simEnergy signal for each cluster and seed to be matched with a caloparticle
SIMENERGY_PU_LIMIT=1.0

windows_creator = WindowCreator(simfraction_thresholds, SEED_MIN_FRACTION,cl_min_fraction=CL_MIN_FRACION, simenergy_pu_limit = SIMENERGY_PU_LIMIT)

debug = args.debug
nocalowNmax = args.maxnocalow


def run(inputfile):
    f = R.TFile(inputfile);
    tree = f.Get("recosimdumper/caloTree")

    if args.nevents and len(args.nevents) >= 1:
        nevent = args.nevents[0]
        if len(args.nevents) == 2:
            nevent2 = args.nevents[1]
        else:
            nevent2 = nevent+1
        tree = islice(tree, nevent, nevent2)

    print ("Starting")
    output = []
    for iev, event in enumerate(tree):
        output += windows_creator.get_windows(event, args.assoc_strategy, 
                                    nocalowNmax= args.maxnocalow,
                                    min_et_seed= args.min_et_seed,
                                    debug= args.debug)
    f.Close()
    return output
 

p = Pool(6)

data = p.map(run, inputfiles)

data_join = pd.concat([ pd.DataFrame(data_cl) for data_cl in data ])

print(data_join)
# df_en.to_csv(args.out+"/output_PUfrac_en.txt", sep=';', index=False)
# df_cl.to_csv(args.out+"/output_PUfrac_cls.txt", sep=';', index=False)
store = pd.HDFStore(args.outputfile)
store['df'] = data_join  # save it

store.close()        
