import ROOT  as R
import calo_association
import sys
import os
from pprint import pprint
import pandas as pd
import math
import argparse 
import random
import string
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-t","--type", type=str, help="ele/gamma", required=True)
parser.add_argument("-o","--out", type=str, help="outputdir", default="out.txt")
parser.add_argument("-s","--simFraction", type=str, help="simFraction")
parser.add_argument("-n","--nfiles", type=int, help="Nfiles", default=1)
parser.add_argument("-sf","--skip-files", type=int, help="Nfiles to skip", default=0)
args = parser.parse_args()


file_ele = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_v2/FourElectronsGunPt1_Dumper_v2_hadd/"
file_gamma = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper/hadd/"
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

def pass_simfraction_threshold(seed_eta, seed_et, cluster_calo_score ):
    '''
    This functions associates a cluster as true matched if it passes a threshold in simfraction
    '''
    iX = min(max(1,simfraction_thresholds.GetXaxis().FindBin(seed_et)      ), simfraction_thresholds.GetNbinsX())
    iY = min(max(1,simfraction_thresholds.GetYaxis().FindBin(abs(seed_eta))), simfraction_thresholds.GetNbinsY())
    thre = simfraction_thresholds.GetBinContent(iX,iY)
    #print(seed_eta, seed_et, cluster_calo_score, thre, cluster_calo_score >= thre )
    return cluster_calo_score >= thre

window_index = 0

def run(fi):
    data_cl = [ ]
    print("Working on file: ", fi)
    f = R.TFile(fi)
    tree = f.Get("recosimdumper/caloTree")
    for i, ev in enumerate(tree):    
        pfCluster_energy = ev.pfCluster_energy
        pfCluster_rawEnergy = ev.pfCluster_rawEnergy
        pfCluster_eta = ev.pfCluster_eta
        pfCluster_phi = ev.pfCluster_phi
        pfCluster_ieta = ev.pfCluster_ieta
        pfCluster_iphi = ev.pfCluster_iphi
        pfCluster_iz = ev.pfCluster_iz
        pfCluster_nXtals = ev.pfCluster_nXtals
        calo_simenergy = ev.caloParticle_simEnergy
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
        # cluster_PU_recoenergy = ev.pfCluster_recoEnergy_sharedXtalsPU
        # total_PU_simenergy = ev.caloParticlePU_totEnergy

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

            for icl, score in clusters:
                simen_signal = pfcluster_calo_score[icl] * calo_simenergy[calo]
                #check trheshold with seed eta, et and cluster score
                pass_simfrac = pass_simfraction_threshold(pfCluster_eta[seed],pfCluster_rawEnergy[seed]/math.cosh(pfCluster_eta[seed]), score )
                pusimen_frac = cluster_PU_simenergy[icl] / simen_signal

                data_cl.append({
                    "wi": "".join([ random.choice(string.ascii_lowercase) for _ in range(7)]),
                    "en": pfCluster_rawEnergy[icl],
                    "et": pfCluster_rawEnergy[icl]/ math.cosh(pfCluster_eta[icl]),
                    "ieta" : pfCluster_ieta[icl],
                    'iphi': pfCluster_iphi[icl],
                    "eta" : pfCluster_eta[icl],
                    'phi': pfCluster_phi[icl],
                    'iz': pfCluster_iz[icl],
                    "simfrac_sig": score, 
                    "simen_sig": simen_signal,
                    "simen_pu": cluster_PU_simenergy[icl],
                    # "recoen_pu": cluster_PU_recoenergy[icl],
                    "simen_sig_frac": simen_signal/pfCluster_rawEnergy[icl],
                    "simen_pu_frac":  cluster_PU_simenergy[icl]/pfCluster_rawEnergy[icl],
                    "PUsimen_frac": pusimen_frac ,
                    "nxtals": pfCluster_nXtals[icl],
                    "is_seed": int(seed == icl),
                    "pass_simfrac_thr": int(pass_simfrac),
                    "nxtals_PU": cluster_nXtalsPU[icl],
                    "nVtx": nVtx, 
                    "obsPU":obsPU,
                    "calo_simen": calo_simenergy[calo],
                    "calo_simet": calo_simenergy[calo]/ math.cosh(calo_simeta[calo]),
                    "calo_geneta": calo_geneta[calo],
                    "calo_genphi": calo_genphi[calo],
                    "calo_simeta": calo_simeta[calo],
                    "calo_simphi": calo_simphi[calo],
                    "calo_genen" : calo_genenergy[calo],
                    "calo_genet" : calo_genenergy[calo] / math.cosh(calo_geneta[calo])
                })
    f.Close()     
    return data_cl      


p = Pool()

data = p.map(run, files)

data_join = pd.concat([ pd.DataFrame(data_cl) for data_cl in data ])

print(data_join)
# df_en.to_csv(args.out+"/output_PUfrac_en.txt", sep=';', index=False)
# df_cl.to_csv(args.out+"/output_PUfrac_cls.txt", sep=';', index=False)
store = pd.HDFStore(args.out)
store['df'] = data_join  # save it

store.close()