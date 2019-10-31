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
import association_strategies
R.gStyle.SetOptStat(0)
'''
This script analyse the overlapping of two caloparticles
'''

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputdir", type=str, help="outputdir", required=True)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
args = parser.parse_args()

debug = args.debug

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)


f = R.TFile(args.inputfile);
tree = f.Get("recosimdumper/caloTree")

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
        return deltaR

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
window_ieta = 10
window_iphi = 20

totevents = 0

energies_maps = []
energies_maps_nocalo = []


for iev, event in enumerate(tree):
    totevents+=1
    #pbar.update()
    print ('---', iev)
    # if iev % 10 == 0: print(".",end="")

    # Branches
    pfCluster_energy = event.pfCluster_energy
    pfCluster_ieta = event.pfCluster_ieta
    pfCluster_iphi = event.pfCluster_iphi
    pfCluster_iz = event.pfCluster_iz
    calo_ieta = event.caloParticle_simIeta
    calo_iphi = event.caloParticle_simIphi
    calo_iz = event.caloParticle_simIz
    calo_simE = event.caloParticle_simEnergy
    simhit_ieta = event.simHit_ieta
    simhit_iphi = event.simHit_iphi
    simhit_iz = event.simHit_iz
    simhit_en = event.simHit_energy
    
    rechits_E = event.recHit_energy
    rechits_ieta = event.simHit_ieta #using simHit ieta structure
    rechits_iphi = event.simHit_iphi
    rechits_iz = event.simHit_iz
    pfrechits_filter = event.pfRecHit_isMatched
    # pfrechits not associated with clusters and caloparticles
    rechits_unclustered_E = event.pfRecHit_unMatched_energy
    rechits_unclustered_ieta = event.pfRecHit_unMatched_ieta
    rechits_unclustered_iphi = event.pfRecHit_unMatched_iphi
    rechits_unclustered_iz = event.pfRecHit_unMatched_iz
    # pfrechits not asssociated with calo but in a cluster
    rechits_nocalo_E = event.pfClusterHit_noCaloPart_energy
    rechits_nocalo_ieta = event.pfClusterHit_noCaloPart_ieta
    rechits_nocalo_iphi = event.pfClusterHit_noCaloPart_iphi
    rechits_nocalo_iz = event.pfClusterHit_noCaloPart_iz



    # xtal_calo = map (ieta,iphi,icalo, simhit):(iclu, clhit)
    # xtal_cluster =  map (ieta, iphi, iclu, clhit):{icalo: simhit}
    # xtal_cluster_noise = list (ieta, iphi, iclu, noisehit)

    (cluster_calo, calo_cluster), (xtal_cluster, xtal_calo, xtal_cluster_noise) =   \
                association_strategies.get_association(event, "sim_fraction_min1", cluster_type="pfCluster",debug=debug)

    if debug: 
        print(">>>>> Cluster_calo association")
        pp(cluster_calo)
        print("\n\n>>> Calo cluster association")
        pp(calo_cluster)


    # 1) Look for highest energy cluster
    clenergies_ordered = sorted([ (ic , en) for ic, en in enumerate(pfCluster_energy)], 
                                                    key=itemgetter(1), reverse=True)
    if debug: print ("biggest cluster", clenergies_ordered)

    used_pfclusters = []

    # Now iterate over clusters in order of energies
    for iw, (icl_seed, clen) in enumerate(clenergies_ordered):
        if icl_seed in used_pfclusters: continue

        if pfCluster_energy[icl_seed] != clen: print("========= BIG ERROR")

        pfClHit_numpy_window = []
        energy_dict = defaultdict(float)
        pfclusters_in_window = []
        caloparticle_in_window = []

        seed_iphi = pfCluster_iphi[icl_seed]
        seed_ieta = pfCluster_ieta[icl_seed]
        seed_iz = pfCluster_iz[icl_seed]


        if debug: 
            sc_histo = R.TH2F("supercl_map_iev{}_iw{}".format(iev, iw), 
                        "window {} - seed ieta,iphi,iz({},{},{})".format(iw,seed_ieta,seed_iphi, seed_iz), 
                window_iphi*2, -window_iphi, +window_iphi, window_ieta*2, -window_ieta, window_ieta)
        
            print("------ WINDOW {} -- cluster {}".format(iw, icl_seed))
            print("seed ieta {}, seed iphi {}, seed iz {}".format(seed_ieta, seed_iphi, seed_iz))
        #print(pfCluster_eta[cli], w_ieta, pfCluster_phi[cli], w_iphi)
        #xtalclusters (ieta, iphi, iz, iclu, clhit):(icalo, simhit)
        
        # Take all clhits in the box, 
        # take w
        caloenergy_in_window = defaultdict(float)
        calofraction_in_window = defaultdict(float)    
        clhits_in_window = []
        clhits_noise_in_window = []

        # Hits with calo
        for (ieta,iphi,iz,icl,clhit), caloinfo in xtal_cluster.items():
            if icl in used_pfclusters: continue  # Exclude already useds clusters
            if iz != seed_iz: continue  # only hits from the same detector
            
            ietaw = ieta_distance(seed_ieta,ieta,iz)
            iphiw = iphi_distance(seed_iphi,iphi,iz)

            if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi:
                clhits_in_window.append( (ietaw,iphiw,iz,icl,clhit, caloinfo))
            else:
                continue
            # Save energy
            energy_dict[(ietaw, iphiw)] += clhit
            pfclusters_in_window.append(icl)
            # Check if cluster is associated
            if icl in cluster_calo:
                caloparticle_in_window.append(cluster_calo[icl][0][0])
            if debug:
                print("hit: ietaw {} iphiw {} iz {} icl {} clhit {}".format(
                       ietaw, iphiw, iz , icl, clhit))
                sc_histo.SetBinContent(sc_histo.FindBin(iphiw, ietaw), clhit)
            # filling calo fraction
            # N.B there could be two cluster in the same xtal-- > fraction not exact
            if debug: 
                for icalo , calosim in caloinfo.items():
                    caloenergy_in_window[icalo] += calosim
                    calofraction_in_window[icalo] += calosim / calo_simE[icalo]

        # Hits with no calo
        for ieta,iphi, iz, icln, clnhit in xtal_cluster_noise:
            if icln in used_pfclusters: continue
            if iz != seed_iz: continue

            ietaw = ieta_distance(seed_ieta,ieta,iz)
            iphiw = iphi_distance(seed_iphi,iphi, iz)
            if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi:
                clhits_noise_in_window.append((ietaw,iphiw, iz,icln, clnhit))
            else:
                continue
            #Save energies
            pfclusters_in_window.append(icln)
            energy_dict[(ietaw, iphiw)] += clnhit
            if debug: 
                print("hitnoise: ietaw {} iphiw {} iz {} icl {} clhit {}".format(
                       ietaw, iphiw, iz , icln, clnhit))
                sc_histo.SetBinContent(sc_histo.FindBin(iphiw, ietaw), clnhit)

        
        caloparticle_in_window = list(set(caloparticle_in_window))
        pfclusters_in_window = list(set(pfclusters_in_window))
        
        if debug:
            print("PfClusters in window:")
            for pfclw in pfclusters_in_window:
                print("\t{}: ieta: {}, iphi: {}, iz: {}, energy: {}, is_seed: {}, is_incalo: {}".format(
                    pfclw, pfCluster_ieta[pfclw], pfCluster_iphi[pfclw], pfCluster_iz[pfclw], 
                    pfCluster_energy[pfclw], pfclw == icl_seed,
                    len( filter(lambda c: c==pfclw, map(itemgetter(3),xtal_cluster)))>0 ))
            print(pfclusters_in_window)
            print("All caloparticle in window (at least one xtal in window)")
            print(caloenergy_in_window.keys())
            print("Caloparticle associated to pfclusters in window (winning the score):")
            for icalo in caloparticle_in_window:
                print("\tcalo: {} | clusters: {}".format(icalo, calo_cluster[icalo]))

            print("-- Calo fractions in this windows:")
            for icalo in caloenergy_in_window.keys():
                print("\tcalo: {} | fraction: {} | energy: {} ".format(icalo, calofraction_in_window[icalo], caloenergy_in_window[icalo], ))
            
            print("-- Calo particle scores for pfcluster")
            for icl in pfclusters_in_window:
                if icl in cluster_calo:
                    print("\tcluster: {} | calo_associated: True".format(icl))
                    for calo, caloscore in cluster_calo[icl]:
                        print("\t\tcalo: {} | score: {}".format(calo, caloscore))
                else:
                    print("\tcluster: {} | calo_associated: False".format(icl))
        
        seed_calo = None
        if icl_seed in cluster_calo:
            #main caloparticle
            # Windows with associated calo
            seed_calo = cluster_calo[icl_seed][0][0]

        used_pfclusters += pfclusters_in_window

        if debug and len(energy_dict)> 0 :
            c = R.TCanvas("c_{}_{}".format(iev, iw))
            sc_histo.Draw("COLZ")
            c.SaveAs(args.outputdir+"/c_{}_{}.png".format(iev, iw))
        # numpy output

        # output array: 
        # shape (ieta, iphi,  3 layers):
        # 0) pfrechits
        # 1) clustered pfrechits
        # 2) truth map
        energies_map = np.zeros((2*window_ieta+1, 2*window_iphi+1, 3))
        energies_map_nocalo = np.zeros((2*window_ieta+1, 2*window_iphi+1, 3))
        # first of all rechits that pass the filter and also stays in the window
        for icalo in range(len(calo_ieta)):
            for ir, (E,ieta,iphi,iz) in enumerate(zip(rechits_E[icalo],
                    rechits_ieta[icalo],rechits_iphi[icalo], rechits_iz[icalo])):
                # Filter only pfRechits
                if not pfrechits_filter[icalo][ir]: continue
                if iz != seed_iz : continue

                ietaw = ieta_distance(seed_ieta,ieta,iz)
                iphiw = iphi_distance(seed_iphi,iphi, iz)
                if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi:
                    # Save with indexes from 0 to 2window_ieta etc
                    if seed_calo: 
                        energies_map[ietaw + window_ieta, iphiw + window_iphi, 0 ]  = E
                    else:
                         energies_map_nocalo[ietaw + window_ieta, iphiw + window_iphi, 0 ]  = E
    
        # pf rechit noise
        for ir, (E,ieta,iphi,iz) in enumerate(zip(
                    chain(rechits_nocalo_E, rechits_unclustered_E),
                    chain(rechits_nocalo_ieta, rechits_unclustered_ieta),
                    chain(rechits_nocalo_iphi, rechits_unclustered_iphi), 
                    chain(rechits_nocalo_iz, rechits_unclustered_iz) )):
            # Filter onl
            if iz != seed_iz : continue
            ietaw = ieta_distance(seed_ieta,ieta,iz)
            iphiw = iphi_distance(seed_iphi,iphi, iz)
            if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi:
                # Save with indexes from 0 to 2window_ieta etc
                if seed_calo:
                    energies_map[ietaw + window_ieta, iphiw + window_iphi, 0 ]  = E
                else:
                    energies_map_nocalo[ietaw + window_ieta, iphiw + window_iphi, 0 ]  = E

        
        # Save energy map from clustering
        for (ietaw,iphiw), E in energy_dict.items():
            if seed_calo:
                energies_map[ietaw + window_ieta, iphiw + window_iphi, 1] = E
            else:
                energies_map_nocalo[ietaw + window_ieta, iphiw + window_iphi, 1] = E

        
        # Save calo simhits for the seed_calo corresponding to the caloparticle
        # associated to the central cluster seed_cluster
        if seed_calo:
            for ir, (E,ieta,iphi,iz) in enumerate(zip(simhit_en[seed_calo],
                        simhit_ieta[seed_calo],simhit_iphi[seed_calo], simhit_iz[seed_calo])):
                    if iz != seed_iz : continue

                    ietaw = ieta_distance(seed_ieta,ieta,iz)
                    iphiw = iphi_distance(seed_iphi,iphi, iz)
                    if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi:
                        # Save with indexes from 0 to 2window_ieta etc
                        if seed_calo:
                            energies_map[ietaw + window_ieta, iphiw + window_iphi, 2 ]  = E
                        else:
                            energies_map_nocalo[ietaw + window_ieta, iphiw + window_iphi, 2 ]  = E

        ##### save  maps
        if seed_calo:
            energies_maps.append(energies_map)
        else:
            energies_maps_nocalo.append(energies_map_nocalo)


results = np.array(energies_maps)
results_nocalo = np.array(energies_maps_nocalo)
np.save("data_calo.npy", results)
np.save("data_nocalo.npy", results_nocalo)

