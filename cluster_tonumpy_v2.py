from __future__ import print_function
import ROOT as R 
R.gROOT.SetBatch(True)
#R.PyConfig.IgnoreCommandLineOptions = True

import sys 
import os
from tqdm import tqdm
from collections import defaultdict
from math import cosh
from itertools import islice, chain
from numpy import mean
from operator import itemgetter, attrgetter
from array import array
from math import sqrt
from pprint import pprint as pp
import numpy as np
import argparse
import pickle
import pandas as pd
'''
This script analyse the overlapping of two caloparticles
'''

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("--weta", type=int,  help="Window eta width", default=10)
parser.add_argument("--wphi", type=int,  help="Window phi width", default=20)
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
args = parser.parse_args()

debug = args.debug
nocalowNmax = args.maxnocalow

f = R.TFile(args.inputfile);
tree = f.Get("deepclusteringdumper/caloTree")

#pbar = tqdm(total=tree.GetEntries())


if args.nevents and len(args.nevents) >= 1:
    nevent = args.nevents[0]
    if len(args.nevents) == 2:
        nevent2 = args.nevents[1]
    else:
        nevent2 = nevent+1
    tree = islice(tree, nevent, nevent2)


def DeltaR(phi1, eta1, phi2, eta2):
        dphi = phi1 - phi2
        if dphi > R.TMath.Pi(): dphi -= 2*R.TMath.Pi()
        if dphi < -R.TMath.Pi(): dphi += 2*R.TMath.Pi()
        deta = eta1 - eta2
        deltaR = (deta*deta) + (dphi*dphi)
        return sqrt(deltaR)

def transform_ieta(ieta):
    if ieta > 0:  return ieta +84
    elif ieta < 0: return ieta + 85

def iphi_distance(iphiseed, iphi, iz):
    if iz == 0:
        if abs(iphiseed-iphi)<= 180: return iphi-iphiseed
        if iphiseed < iphi:
            return iphi-iphiseed - 360
        else:
            return iphi - iphiseed + 360
    else :
        return iphi - iphiseed

def ieta_distance(ietaseed, ieta, iz):
    if iz == 0:
        return transform_ieta(ieta) - transform_ieta(ietaseed)
    else:
        return ieta-ietaseed


# maximum baffo
window_ieta = args.weta
window_iphi = args.wphi

def in_window(seed_ieta, seed_iphi, seed_iz, ieta, iphi, iz):
    if seed_iz != iz: return False, (-1,-1)
    ietaw = ieta_distance(seed_ieta,ieta,iz)
    iphiw = iphi_distance(seed_iphi,iphi,iz)
    if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi: 
        return True, (ietaw, iphiw)
    else:
        return False,(-1,-1)

def fill_window_cluster(window, clhits_ieta, clhits_iphi, clhits_iz, clhits_energy, rechit_energy, match_calo,fill_mask=False):
    is_in = False
    mask = []
    for ieta, iphi, iz, cl_energy,rec_energy in zip(clhits_ieta, clhits_iphi, clhits_iz, clhits_energy, rechit_energy):
        hit_in_wind, (ietaw, iphiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],
                                                    ieta, iphi, iz)
        if hit_in_wind:
            mask.append((ietaw + window_ieta, iphiw + window_iphi))
            # Add rechits of clusterhits to level0 (rechits), level1(pfRechits),
            # Add clusterhits in level2(clusters)
            window["energy_map"][ietaw + window_ieta, iphiw + window_iphi, 0] = rec_energy
            window["energy_map"][ietaw + window_ieta, iphiw + window_iphi, 1] = rec_energy
            window["energy_map"][ietaw + window_ieta, iphiw + window_iphi, 2] = cl_energy 
            if match_calo != -1 and match_calo == window["calo"] :
                window["energy_map"][ietaw + window_ieta, iphiw + window_iphi, 3] = cl_energy 
            # Fill also the additional map with 1 or 0
            is_in = True
    return is_in , mask


def fill_window_hits(window, hits_ieta, hits_iphi, hits_iz, hits_energy, levels):
    for ieta,iphi, iz, energy in zip(hits_ieta, hits_iphi, hits_iz, hits_energy):
        hit_in_wind, (ietaw, iphiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],
                                                    ieta, iphi, iz)
        if hit_in_wind:
            for l in levels:
                window["energy_map"][ietaw + window_ieta, iphiw + window_iphi, l] = energy


def cluster_in_window(window, clhits_ieta, clhits_iphi, clhits_iz, clhits_energy):
    for ieta, iphi, iz, energy in zip(clhits_ieta, clhits_iphi, clhits_iz, clhits_energy):
        hit_in_wind, (ietaw, iphiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],ieta, iphi, iz)
        if hit_in_wind:
            return True
    return False

totevents = 0
 
energies_maps = []
energies_maps_nocalo = []
metadata = []
clusters_masks = []

windows_index = -1

for iev, event in enumerate(tree):
    totevents+=1
    nocalowN = 0
    #pbar.update()
    print ('---', iev)
    # if iev % 10 == 0: print(".",end="")

    # Branches
    pfCluster_energy = event.pfCluster_energy
    pfCluster_ieta = event.pfCluster_ieta
    pfCluster_iphi = event.pfCluster_iphi
    pfCluster_eta = event.pfCluster_eta
    pfCluster_phi = event.pfCluster_phi
    pfCluster_iz = event.pfCluster_iz

    simhit_ieta = event.simHit_ieta
    simhit_iphi = event.simHit_iphi
    simhit_iz = event.simHit_iz
    simhit_energy = event.simHit_energy

    clhit_ieta = event.pfClusterHit_ieta;
    clhit_iphi = event.pfClusterHit_iphi;
    clhit_iz = event.pfClusterHit_iz;
    clhit_eta = event.pfClusterHit_eta;
    clhit_phi = event.pfClusterHit_phi;
    clhit_energy = event.pfClusterHit_energy;
    clhit_rechitEnergy = event.pfClusterHit_rechitEnergy

    calo_simeta = event.caloParticle_simEta;
    calo_simphi = event.caloParticle_simPhi;
    calo_simenergy = event.caloParticle_simEnergy;
    
    rechits_noPF_energy = event.recHit_noPF_energy
    rechits_noPF_ieta = event.recHit_noPF_ieta #using recHit_noPF ieta structure
    rechits_noPF_iphi = event.recHit_noPF_iphi
    rechits_noPF_iz = event.recHit_noPF_iz
    rechits_unclustered_energy = event.pfRecHit_unClustered_energy
    rechits_unclustered_ieta = event.pfRecHit_unClustered_ieta
    rechits_unclustered_iphi = event.pfRecHit_unClustered_iphi
    rechits_unclustered_iz = event.pfRecHit_unClustered_iz

    pfcluster_calo_map = event.pfCluster_sim_fraction_min1_MatchedIndex
    calo_pfcluster_map = event.caloParticle_pfCluster_sim_fraction_min1_MatchedIndex
   
    # map of windows, key=pfCluster seed index
    windows_map = {}
    nonseed_clusters = []
    # 1) Look for highest energy cluster
    clenergies_ordered = sorted([ (ic , en) for ic, en in enumerate(pfCluster_energy)], 
                                                    key=itemgetter(1), reverse=True)
    if debug: print ("biggest cluster", clenergies_ordered)

    # Now iterate over clusters in order of energies
    for iw, (icl, clenergy) in enumerate(clenergies_ordered):
        cl_ieta = pfCluster_ieta[icl]
        cl_iphi = pfCluster_iphi[icl]
        cl_iz = pfCluster_iz[icl]
        cl_eta = pfCluster_eta[icl]
        cl_phi = pfCluster_phi[icl]
        clxtals_ieta = clhit_ieta[icl]
        clxtals_iphi = clhit_iphi[icl]
        clxtals_iz = clhit_iz[icl]
        clxtals_energy = clhit_energy[icl]
        clxtals_rechitEnergy = clhit_rechitEnergy[icl]

        is_in_window = False
        # Check if it is already in one windows
        for wkey, window in windows_map.items():
            if cluster_in_window(window, clxtals_ieta, clxtals_iphi, clxtals_iz, clxtals_energy): 
                nonseed_clusters.append(icl)
                is_in_window = True
                break

        # If is not already in some window 
        if not is_in_window: 
            caloseed = pfcluster_calo_map[icl]
            # Let's create  new window:
            new_window = {
                "seed": (cl_ieta, cl_iphi, cl_iz),
                # level: rechits, pfrechits, clusterhit, clusterhit_calo, 
                "energy_map":   np.zeros((2*window_ieta+1, 2*window_iphi+1, 5)), 
                "calo" : caloseed,
                "metadata": {
                    "seed_eta": cl_eta,
                    "seed_phi": cl_phi, 
                    "seed_iz": cl_iz,
                    "en_seed": pfCluster_energy[icl],
                    "en_true": calo_simenergy[caloseed]
                }
            }
            if new_window["calo"]==-1:
                nocalowN+=1
                # Not creating too many windows of noise
                if nocalowN> nocalowNmax: continue
            # Update index only if we save it
            windows_index += 1
            new_window["metadata"]["index"] = windows_index
            # Save the window
            windows_map[icl] = new_window
            isin, mask = fill_window_cluster(new_window, clxtals_ieta, clxtals_iphi, clxtals_iz, 
                                clxtals_energy, clxtals_rechitEnergy, pfcluster_calo_map[icl], fill_mask=True)
            # Save also seed cluster for cluster_masks
            clusters_masks.append({
                    "window_index": new_window["metadata"]["index"],
                    "cluster_eta": cl_eta,
                    "cluster_phi": cl_phi, 
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_energy[icl],
                    "is_seed": True,
                    "mask": mask, 
                })

           
    # Now that all the seeds are inside let's add the non seed
    for icl_noseed in nonseed_clusters:
        cl_ieta = pfCluster_ieta[icl_noseed]
        cl_iphi = pfCluster_iphi[icl_noseed]
        cl_iz = pfCluster_iz[icl_noseed]
        cl_eta = pfCluster_eta[icl_noseed]
        cl_phi = pfCluster_phi[icl_noseed]
        clxtals_ieta = clhit_ieta[icl_noseed]
        clxtals_iphi = clhit_iphi[icl_noseed]
        clxtals_iz = clhit_iz[icl_noseed]
        clxtals_energy = clhit_energy[icl_noseed]

        # Fill all the windows
        for wkey, window in windows_map.items():
            isin, mask = fill_window_cluster(window, clxtals_ieta, clxtals_iphi, clxtals_iz,
                                        clxtals_energy,clxtals_rechitEnergy, 
                                        pfcluster_calo_map[icl_noseed], mask)
            if isin:
                clusters_masks.append({
                    "window_index": window["metadata"]["index"],
                    "cluster_eta": cl_eta,
                    "cluster_phi": cl_phi, 
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_energy[icl_noseed],
                    "is_seed": False,
                    "mask": mask, 
                })



    ###############################
    #### Add rechits and
    
    for window in windows_map.values():
        # Save unclustered pfrechits in both rechit and pfrechits levels
        fill_window_hits(window, rechits_unclustered_ieta,rechits_unclustered_iphi,
                    rechits_unclustered_iz,rechits_unclustered_energy, levels=[0,1])
        # Save rechits NOT PF only in first level
        fill_window_hits(window, rechits_noPF_ieta,rechits_noPF_iphi,
                    rechits_noPF_iz,rechits_noPF_energy, levels=[0,1])

        # Save simhit of the calo corresponding to the seed
        fill_window_hits(window, simhit_ieta[window["calo"]], simhit_iphi[window["calo"]],
                                 simhit_iz[window["calo"]], simhit_energy[window["calo"]], levels=[4])

        ########Calculate metadata 
        
        calo_seed = window["calo"]
        # Check the type of events
        # - Number of pfcluster associated, 
        # - deltaR of the farthest cluster
        # - Energy of the pfclusters
        if calo_seed != -1:
            # Get number of associated clusters
            assoc_clusters =  calo_pfcluster_map[calo_seed]
            max_en_pfcluster = max([pfCluster_energy[i] for i in assoc_clusters])
            max_dr = max( [ DeltaR(calo_simphi[calo_seed], calo_simeta[calo_seed], 
                            pfCluster_phi[i], pfCluster_eta[i]) for i in assoc_clusters])
            window["metadata"]["nclusters"] = len(assoc_clusters)
            window["metadata"]["max_en_cluster"] = max_en_pfcluster
            window["metadata"]["max_dr_cluster"] = max_dr

        if calo_seed != -1:
            energies_maps.append(window["energy_map"])
            metadata.append(window["metadata"])
        else:
            energies_maps_nocalo.append(window["energy_map"])


        
results = np.array(energies_maps)
results_nocalo = np.array(energies_maps_nocalo)

meta= pd.DataFrame(metadata)

#results_nocalo = np.array(energies_maps_nocalo)
np.save("data_calo.npy", results)
np.save("data_nocalo.npy", results_nocalo)

meta.to_csv("output.csv", index=False)
#np.save("data_nocalo.npy", results_nocalo)

pickle.dump(clusters_masks, open("clusters_masks.pkl", "wb"))
