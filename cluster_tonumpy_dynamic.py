from __future__ import print_function
import ROOT as R 
R.gROOT.SetBatch(True)
#R.PyConfig.IgnoreCommandLineOptions = True

import sys 
import os
from tqdm import tqdm
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
import windows_creator_dynamic
'''
This script analyse the overlapping of two caloparticles
'''

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile", default="clusters_data.pkl")
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", default="sim_fraction_min1")
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1.)
args = parser.parse_args()

if "#_#" in args.inputfile: 
    inputfiles = args.inputfile.split("#_#")
else:
    inputfiles = [args.inputfile]

# if args.nevents and len(args.nevents) >= 1:
#     nevent = args.nevents[0]
#     if len(args.nevents) == 2:
#         nevent2 = args.nevents[1]
#     else:
#         nevent2 = nevent+1
#     tree = islice(tree, nevent, nevent2)


debug = args.debug
nocalowNmax = args.maxnocalow

energies_maps = []
metadata = []
clusters_masks = []

for inputfile in inputfiles:
    f = R.TFile(inputfile);
    tree = f.Get("recosimdumper/caloTree")

    print ("Starting")
    for iev, event in enumerate(tree):
        if iev % 10 == 0: print(".",end="")
        windows_event, clusters_event = windows_creator.get_windows(event, args.maxnocalow, 
                                    args.assoc_strategy, args.min_et_seed, args.debug )
        clusters_masks += clusters_event
        #print(clusters_event)
    
    f.Close()
        

df_cl = pd.DataFrame(clusters_masks) 

df_cl.head()

pickle.dump(df_cl, open(args.outputfile, "wb"))
