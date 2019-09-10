import ROOT as R 
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
import argparse

'''
This script analyse the overlapping of two caloparticles
'''

parser = argparse.ArgumentParser()
parser.add_argument("--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("--outputdir", type=str, help="outputdir", required=True)
parser.add_argument("--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("--debug", action="store_true",  help="debug", default=False)
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

window_ieta = 20
window_iphi = 50

totevents = 0


pfClHit_numpy = []


for iev, event in enumerate(tree):
    totevents+=1
    #pbar.update()
    if debug: print '---', iev

    ncalo = event.caloParticle_pt.size()
    calo_genE = event.caloParticle_energy
    calo_simE = event.caloParticle_simEnergy
    calo_eta  = event.caloParticle_eta
    calo_phi  = event.caloParticle_phi
    calo_ieta = event.caloParticle_ieta
    calo_iphi = event.caloParticle_iphi
    calo_iz = event.caloParticle_iz
    pfCluster_eta = event.pfCluster_eta
    pfCluster_phi = event.pfCluster_phi
    pfCluster_ieta = event.pfCluster_ieta
    pfCluster_iphi = event.pfCluster_iphi
    pfCluster_iz = event.pfCluster_iz
    pfCluster_energy = event.pfCluster_energy


    # caloparticle (index: (ieta, iphi, iz))
    calo_pos = {}
    # map (ieta,iphi,icalo, simhit):[(iclu, clhit)]
    xtal_calo = defaultdict(list)
    # map (ieta, iphi, iclu, clhit):[(icalo, simhit)]
    xtal_cluster = defaultdict(list)
    # list (ieta, iphi, iclu, noisehit)
    xtal_cluster_noise = []
    
    for icalo in range(ncalo):
        calo_pos[icalo] = (calo_ieta[icalo], calo_iphi[icalo], calo_iz[icalo])
        # analysis cluster hit to get the cluster number
        if debug: print ("ieta iphi simhit [ pfcluster index , pfcluster hit]")
        for i, (ieta, iphi, simhit, clhit) in enumerate(zip(event.simHit_ieta[icalo],  event.simHit_iphi[icalo],
                        event.simHit_energy[icalo], event.pfClusterHit_energy[icalo])):
            if clhit.size() > 0:
                if debug:   print( ieta, iphi, "{:.5f}".format(simhit), [(hit.first, '{:.5f}'.format(hit.second)) for hit in clhit]) 
                for chit in clhit:
                    xtal_cluster[(ieta, iphi, chit.first, chit.second)].append((icalo, simhit))
                    xtal_calo[(ieta, iphi, icalo, simhit)].append((chit.first, chit.second))
            # else:
            #     print ieta, iphi, "{:.5f}".format(simhit), "!!! No Cluster hits"

    for nclus , (energys, ietas,iphis) in enumerate(zip(event.pfClusterHit_noCaloPart_energy,   
                    event.pfClusterHit_noCaloPart_ieta, event.pfClusterHit_noCaloPart_iphi)):
        
        #print nclus , [(ieta, iphi, en) for ieta,iphi, en in zip(energys, ietas, iphis)]
        for en, ieta, iphi in zip(energys, ietas, iphis):
            xtal_cluster_noise.append((ieta, iphi,nclus, en))


    
    # 1) Look for highest energy cluster
    clenergies_ordered = sorted([ (ic , en) for ic, en in enumerate(pfCluster_energy)], key=itemgetter(1), reverse=True)
    print "biggest cluster", clenergies_ordered

    used_pfclusters = []

    for iw, (cli, clen) in enumerate(clenergies_ordered):
        if cli in used_pfclusters: continue

        print("------ WINDOW ", iw)
        pfClHit_numpy_window = []
        energy_dict = defaultdict(float)
        pfclusters_in_window = []
        caloparticle_in_window = []

        w_ieta = (pfCluster_ieta[cli] - window_ieta, pfCluster_ieta[cli] + window_ieta)
        w_iphi = (pfCluster_iphi[cli] - window_iphi, pfCluster_iphi[cli] + window_iphi)
        #print(pfCluster_eta[cli], w_ieta, pfCluster_phi[cli], w_iphi)
        
        # Take all clhits in the box, 
        # take w
        clhits_in_window = filter( 
                 lambda ((ieta,iphi,icl,clhit), l): icl not in used_pfclusters and ieta>=w_ieta[0] and ieta <= w_ieta[1] and iphi>=w_iphi[0] and iphi<=w_iphi[1] ,
                    xtal_cluster.items() )

        clhits_noise_in_window = filter( lambda (ieta,iphi, icln,clnhit) : icln not in used_pfclusters and ieta>=w_ieta[0] and ieta <= w_ieta[1] and iphi>=w_iphi[0] and iphi<=w_iphi[1] ,
                 xtal_cluster_noise)

        for (ieta,iphi, icl, iclhit), caloinfo in clhits_in_window:
            energy_dict[(ieta, iphi)] += iclhit
            pfclusters_in_window.append(icl)
            for caloi in caloinfo:
                caloparticle_in_window.append(caloi[0])
            
        for (ieta,iphi, iclnoise, clnoisehit) in clhits_noise_in_window:
            pfclusters_in_window.append(iclnoise)
            energy_dict[(ieta, iphi)] += clnoisehit

        pp(dict(energy_dict))
        print("Caloparticle in window")
        print(set(caloparticle_in_window))
        print("PfClusters in window")
        print(set(pfclusters_in_window))

        used_pfclusters += list(set(pfclusters_in_window))

#    9

# sum the cluster hit associated with the clusterassociated to the gammaX caloparticle
# assing also noise
# cluster_energies[gammaX]  =  sum( 
#         map(itemgetter(1), 
#             # filter cluster index
#             filter( lambda v: v[0] == gammaX_iclu, 
#                 chain.from_iterable( 
#                     # chain clusters hits
#                     map( lambda (k,clusters): clusters,
#                         # select gammaX
#                         filter( lambda (k, v): k[2] == gammaX, xtal_calo.items() )
#     ) ) ) ) 
#     )  + sum(
#             map(itemgetter(1), #summing the noise energy in that cluster 
#                 chain.from_iterable( 
#                     map(lambda (k, v): v,
#                         filter(lambda (k,v): k==gammaX_iclu, xtal_cluster_noise.items()) 
#     ) ) ) ) 