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
from multiprocessing import Pool
import scipy

parser = argparse.ArgumentParser()
parser.add_argument("-t","--type", type=str, help="ele/gamma", required=True)
parser.add_argument("-o","--out", type=str, help="outputdir", default="out.txt")
parser.add_argument("-s","--simFraction", type=str, help="simFraction")
parser.add_argument("-n","--nfiles", type=int, help="Nfiles", default=1)
parser.add_argument("-sf","--skip-files", type=int, help="Nfiles to skip", default=0)
args = parser.parse_args()


file_ele = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_FULL/hadd/"
file_gamma = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_FULL/hadd/"
nfiles = args.nfiles


if args.type == "ele": file = file_ele
elif args.type == "gamma": file = file_gamma
else: 
    print("Select either ele/gamma type")
    exit(1)

files = [file + f for f in os.listdir(file)[args.skip_files:]][:nfiles]

# chain = R.TChain("recosimdumper/caloTree")
# for f in files[:nfiles]:
#     try:
#         print("Adding to chain: ",f)
#         chain.Add(f)
#     except:
#         print("problems with file")
#         continue


simfraction_thresholds_file = R.TFile(args.simFraction)
simfraction_thresholds = simfraction_thresholds_file.Get("h2_Minimum_simScore_seedBins")



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

def pass_simfraction_threshold(seed_eta, seed_et, cluster_calo_score ):
    '''
    This functions associates a cluster as true matched if it passes a threshold in simfraction
    '''
    iX = min(max(1,simfraction_thresholds.GetXaxis().FindBin(seed_et)      ), simfraction_thresholds.GetNbinsX())
    iY = min(max(1,simfraction_thresholds.GetYaxis().FindBin(abs(seed_eta))), simfraction_thresholds.GetNbinsY())
    thre = simfraction_thresholds.GetBinContent(iX,iY)
    #print(seed_eta, seed_et, cluster_calo_score, thre, cluster_calo_score >= thre )
    return cluster_calo_score >= thre


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
        dphi = 1.075 - 0.25 * aeta
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

def run(fi):
    scipy.random.seed()
    data_cl = [ ]
    print("Working on file: ", fi)
    try:
        f = R.TFile.Open(fi)
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
                                calo_association.get_calo_association(clusters_scores, sort_calo_cl=True, debug=False, min_sim_fraction=1e-4)
        # CaloParticle Pileup information
        cluster_nXtalsPU = ev.pfCluster_simPU_nSharedXtals 
        cluster_PU_simenergy = ev.pfCluster_simEnergy_sharedXtalsPU

        cluster_noise = ev.pfCluster_noise
        cluster_noise_uncalib  = ev.pfCluster_noiseUncalib
        cluster_noise_nofrac = ev.pfCluster_noiseNoFractions
        cluster_noise_uncalib_uncalib = ev.pfCluster_noiseUncalibNoFractions


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
                simen_pu = cluster_PU_simenergy[icl]
                
                #check trheshold with seed eta, et and cluster score
                pass_simfrac = pass_simfraction_threshold(pfCluster_eta[seed],pfCluster_rawEnergy[seed]/math.cosh(pfCluster_eta[seed]), score )
                pusimen_frac = simen_pu / simen_signal
                
                is_in_window, (detaw, dphiw) = in_window(pfCluster_eta[seed], pfCluster_phi[seed], pfCluster_iz[seed],
                                         pfCluster_eta[icl], pfCluster_phi[icl], pfCluster_iz[icl],
                                        deta_up, deta_down, dphi )
                data_cl.append({
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
                    "simen_pu": simen_pu,
                    "simen_sig_frac": simen_signal/pfCluster_rawEnergy[icl],
                    "simen_pu_frac":  simen_pu/pfCluster_rawEnergy[icl],
                    "PUsimen_frac": pusimen_frac ,
                    
                    "noise_en" : cluster_noise[icl],
                    "noise_en_uncal": cluster_noise_uncalib[icl],
                    "noise_en_nofrac": cluster_noise_nofrac[icl],
                    "noise_en_uncal_nofrac": cluster_noise_uncalib_uncalib[icl],
                    
                    "nxtals": pfCluster_nXtals[icl],
                    "is_seed": int(seed == icl),
                    "pass_simfrac_thr": int(pass_simfrac),
                    "in_window": int(is_in_window),
                    "deta_seed": detaw,
                    "dphi_seed": dphiw, 
                    "nxtals_PU": cluster_nXtalsPU[icl],
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
                })
    f.Close()     
    return data_cl      


p = Pool(5)

data = p.map(run, files)

data_join = pd.concat([ pd.DataFrame(data_cl) for data_cl in data ])

print(data_join)
# df_en.to_csv(args.out+"/output_PUfrac_en.txt", sep=';', index=False)
# df_cl.to_csv(args.out+"/output_PUfrac_cls.txt", sep=';', index=False)
store = pd.HDFStore(args.out)
store['df'] = data_join  # save it

store.close()