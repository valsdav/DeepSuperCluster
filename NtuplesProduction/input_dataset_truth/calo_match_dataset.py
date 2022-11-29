import ROOT  as R
import calo_association
import sys
import os
from pprint import pprint
import pandas as pd
from math import pi, sqrt, cosh
import math
import argparse 
import random
import string
import scipy

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputfile", type=str, help="input file. Multiple files can be concatenated with #_#", required=True)
parser.add_argument("-o","--outputfile", type=str, help="Output file", default="out.csv")
parser.add_argument("--pu",action="store_true", help="Analyze PU info")
args = parser.parse_args()

# min simFraction for the cluster to be associated with the caloparticle
CL_MIN_FRACTION=1e-4


def DeltaR(phi1, eta1, phi2, eta2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    deta = eta1 - eta2
    deltaR = (deta*deta) + (dphi*dphi)
    return sqrt(deltaR)

def DeltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    return dphi


def DeltaEta(seed_eta, cl_eta):
    return (1 - 2 * (seed_eta < 0)) * (cl_eta - seed_eta)


def dynamic_window(eta):
    aeta = abs(eta)

    if aeta >= 0 and aeta < 0.1:
        deta_up = 0.075
    if aeta >= 0.1 and aeta < 1.3:
        deta_up = 0.0758929 -0.0178571* aeta + 0.0892857*(aeta**2) 
    elif aeta >= 1.3 and aeta < 1.7:
        deta_up = 0.2
    elif aeta >=1.7 and aeta < 1.9:
        deta_up = 0.625 -0.25*aeta
    elif aeta >= 1.9:
        deta_up = 0.15

    if aeta < 2.1: 
        deta_down = -0.075
    elif aeta >= 2.1 and aeta < 2.5:
        deta_down = -0.1875 *aeta + 0.31875
    elif aeta >=2.5:
        deta_down = -0.15

    if aeta < 1.9:
        dphi = 0.6
    elif aeta >= 1.9 and aeta < 2.7:
        dphi = 1.075 -0.25 * aeta
    elif aeta >= 2.7:
        dphi = 0.4

    return deta_up, deta_down, dphi

def in_window(seed_eta, seed_phi, seed_iz, eta, phi, iz, window_deta_up, windows_deta_down, window_dphi):
    if seed_iz != iz: return False, (-1,-1)
    # Delta Eta ordering
    etaw = eta - seed_eta
    if seed_eta < 0:
        etaw = -etaw
    phiw = DeltaPhi(seed_phi, phi)
    if etaw >= windows_deta_down and etaw <= window_deta_up  and abs(phiw) <= window_dphi: 
        return True, (etaw, phiw)
    else:
        return False,(-1,-1)



def analyze(file):
    scipy.random.seed()
    data_cl = [ ]
    print("Working on file: ", file)
    try:
        f = R.TFile.Open(file)
        tree = f.Get("recosimdumper/caloTree")
    except:
        return data_cl 
    for i, ev in enumerate(tree):    
        pfCluster_energy = ev.pfCluster_energy
        pfCluster_rawEnergy = ev.pfCluster_rawEnergy
        pfCluster_eta = ev.pfCluster_eta
        pfCluster_phi = ev.pfCluster_phi
        pfCluster_ieta = ev.pfCluster_ieta
        pfCluster_iphi = ev.pfCluster_iphi
        pfCluster_iz = ev.pfCluster_iz
        pfCluster_nXtals = ev.pfCluster_nXtals
        pfCluster_simen_signal = ev.pfCluster_simEnergy_sharedXtals
        calo_simenergy = ev.caloParticle_simEnergy
        calo_simenergy_good = ev.caloParticle_simEnergyGoodStatus
        calo_genenergy = ev.caloParticle_genEnergy
        calo_simeta = ev.caloParticle_simEta
        calo_simphi = ev.caloParticle_simPhi
        calo_simiz = ev.caloParticle_simIz
        calo_geneta = ev.caloParticle_genEta
        calo_genphi = ev.caloParticle_genPhi
        
        # pfcl_swissCross = ev.pfCluster_swissCross
        pfcl_nxtals = ev.pfCluster_nXtals
    
        nVtx = ev.nVtx
        # rho = ev.rho
        obsPU = ev.obsPU
        # truePU = ev.truePU

        clusters_scores = ev.pfCluster_sim_fraction
        pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                                calo_association.get_calo_association(clusters_scores, sort_calo_cl=True,
                                                                      debug=False, min_sim_fraction=CL_MIN_FRACTION)
        if args.pu:
            # CaloParticle Pileup information
            cluster_nXtalsPU = ev.pfCluster_simPU_nSharedXtals 
            cluster_PU_simenergy = ev.pfCluster_simEnergy_sharedXtalsPU
            cluster_noise = ev.pfCluster_noise
            cluster_noise_uncalib  = ev.pfCluster_noiseUncalib
            cluster_noise_nofrac = ev.pfCluster_noiseNoFractions
            cluster_noise_uncalib_uncalib = ev.pfCluster_noiseUncalibNoFractions

        ### DEBUG INFO
        # print(">>> Cluster_calo map")
        # for cluster, calo in pfcluster_calo_map.items():
        #     if calo == -1: continue
        #     print("cl: {} | calo: {} (calo Et: {:.2f}, eta {:.2f}, phi {:.2f})| score: {:.4f}, simEnPU: {:.3f}".format(cluster,calo,
        #                                 calo_simenergy[calo]/cosh(calo_simeta[calo]) ,calo_simeta[calo],calo_simphi[calo],pfcluster_calo_score[cluster],cluster_PU_simenergy[cluster]))
        # print("\n>>> Calo_cluster map")
        # for calo, clusters in calo_pfcluster_map.items():
        #     print("calo: {} | clusters: ".format(calo))
        #     for cl, sc in clusters:
        #         print("\t> cl: {}, Et: {:.2f}, eta: {:.2f}, phi:{:.2f}, score: {:.4f}, simEnPU: {:.3f}".format(cl,pfCluster_rawEnergy[cl]/ cosh(pfCluster_eta[cl]), pfCluster_eta[cl],pfCluster_phi[cl], sc,cluster_PU_simenergy[cl]))

        # Get only the seed 
        for calo, clusters in calo_pfcluster_map.items():
            seed = clusters[0][0]
            # seed_score = clusters[0][1]
            # seed_en = pfCluster_rawEnergy[seed]
            window_index = "".join([ random.choice(string.ascii_lowercase) for _ in range(8)])
            #dynamic window of the seed
            deta_up, deta_down, dphi = dynamic_window(pfCluster_eta[seed])

            for icl, score in clusters:
                # simen_signal = pfcluster_calo_score[icl] * calo_simenergy[calo]
                simen_signal = pfCluster_simen_signal[icl][calo]
                if args.pu:
                    simen_pu = cluster_PU_simenergy[icl]
                    pusimen_frac = simen_pu / simen_signal
                
                is_in_window, (detaw, dphiw) = in_window(pfCluster_eta[seed], pfCluster_phi[seed], pfCluster_iz[seed],
                                         pfCluster_eta[icl], pfCluster_phi[icl], pfCluster_iz[icl],
                                        deta_up, deta_down, dphi )
                data = {
                    "wi": window_index,
                    "en": pfCluster_rawEnergy[icl],
                    "et": pfCluster_rawEnergy[icl]/ math.cosh(pfCluster_eta[icl]),
                    "ieta" : pfCluster_ieta[icl],
                    'iphi': pfCluster_iphi[icl],
                    "eta" : pfCluster_eta[icl],
                    'phi': pfCluster_phi[icl],
                    'iz': pfCluster_iz[icl],
                    "simfrac_sig": score, 
                    "simen_sig": simen_signal,
                    "simen_sig_frac": simen_signal/pfCluster_rawEnergy[icl],
                    
                    "nxtals": pfCluster_nXtals[icl],
                    "is_seed": int(seed == icl),

                    # CHeck if it would be inside the dynamic window
                    "in_window": int(is_in_window),
                    "deta_seed": DeltaEta(pfCluster_eta[seed], pfCluster_eta[icl]) ,
                    "dphi_seed": DeltaPhi(pfCluster_phi[seed], pfCluster_phi[icl]) , 
                    "nVtx": nVtx, 
                    "obsPU":obsPU,
                    "calo_simen": calo_simenergy[calo],
                    "calo_simet": calo_simenergy[calo]/ math.cosh(calo_simeta[calo]),
                    "calo_simen_good": calo_simenergy_good[calo],
                    "calo_geneta": calo_geneta[calo],
                    "calo_genphi": calo_genphi[calo],
                    "calo_simeta": calo_simeta[calo],
                    "calo_simphi": calo_simphi[calo],
                    "calo_genen" : calo_genenergy[calo],
                    "calo_genet" : calo_genenergy[calo] / math.cosh(calo_geneta[calo])
                }
                if args.pu:
                    data.update({
                        "simen_pu": simen_pu,
                        "simen_pu_frac":  simen_pu/pfCluster_rawEnergy[icl],
                        "PUsimen_frac": pusimen_frac ,
                        "nxtals_PU": cluster_nXtalsPU[icl],
                         
                        "noise_en" : cluster_noise[icl],
                        "noise_en_uncal": cluster_noise_uncalib[icl],
                        "noise_en_nofrac": cluster_noise_nofrac[icl],
                        "noise_en_uncal_nofrac": cluster_noise_uncalib_uncalib[icl],
                   
                        
                    })
                data_cl.append(data)
    f.Close()     
    return data_cl      





if "#_#" in args.inputfile: 
    inputfiles = args.inputfile.split("#_#")
else:
    inputfiles = [args.inputfile]


data = [ ]
for file in inputfiles:
    data+=analyze(file)


df = pd.DataFrame(data)
df.to_csv(args.outputfile, sep=";", index=False)

