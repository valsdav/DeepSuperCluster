import ROOT as R
from pprint import pprint as pp
import numpy as np
import argparse
import pickle
import pandas as pd
from itertools import islice, chain
import windows_creator_dynamic as windows_creator

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=0)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1.)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
args = parser.parse_args()


debug = args.debug
nocalowNmax = args.maxnocalow

energies_maps = []
metadata = []
clusters_masks = []


f = R.TFile(args.inputfile);
tree = f.Get("recosimdumper/caloTree")

if args.nevents and len(args.nevents) >= 1:
    nevent = args.nevents[0]
    if len(args.nevents) == 2:
        nevent2 = args.nevents[1]
    else:
        nevent2 = nevent+1
    tree = islice(tree, nevent, nevent2)

print ("Starting")
for iev, event in enumerate(tree):
    if iev % 10 == 0: print(".",end="")
    windows_event, clusters_event = windows_creator.get_windows(event, args.maxnocalow, 
                            args.min_et_seed, args.debug )
    clusters_masks += clusters_event
    #print(clusters_event)

    cache = []

    for windex, window in windows_event.items():
        c = R.TCanvas("c_"+windex,"Window " +windex)
        clusters = filter(lambda c: c["window_index"]==windex, clusters_event)

        en_map = R.TH2F("en_map_"+windex, "energy map", 40,-0.6,0.6, 40, -0.3,0.3)
        for cl in clusters:
            en_map.Fill(cl["cluster_dphi"], cl["cluster_deta"], cl["en_cluster"])
        en_map.Draw("COLZ")
        c.Draw()
        cache.append((c,en_map))

    a = input("next event?")
    if a != "": exit(0)
    cache.clear()


    f.Close()
        
