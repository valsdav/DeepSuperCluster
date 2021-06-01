import ROOT  as R
import calo_association
import sys
from pprint import pprint
import pandas as pd
import math
from math import cosh
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t","--type", type=str, help="ele/gamma", required=True)
parser.add_argument("-o","--out", type=str, help="outputfile", default="out.txt")
parser.add_argument("-n","--nfiles", type=int, help="Nfiles", default=1)
args = parser.parse_args()


file_ele = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourElectronsGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_Reduced_Dumper_v2/crab_FourElectronsGunPt1_Dumper_v2/210519_163221/0000/output_{}.root"
file_gamma = "/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/FourGammasGunPt1-100_pythia8_StdMixing_Flat55To75_14TeV_112X_mcRun3_2021_realistic_v16_Reduced_Dumper/dumper/210518_081335/0000/output_{}.root"
nfiles = args.nfiles

if args.type == "ele": file = file_ele
elif args.type == "gamma": file = file_gamma
else: 
    print("Select either ele/gamma type")
    exit(1)

chain = R.TChain("recosimdumper/caloTree")
for i in range(nfiles):
    try:
        print("Adding to chain: ",file.format(i+1))
        chain.Add(file.format(i+1))
    except:
        print("problems with file")
        continue

data= [ ] 


def pass_simfraction_threshold(self, seed_eta, seed_et, cluster_calo_score ):
    '''
    This functions associates a cluster as true matched if it passes a threshold in simfraction
    '''
    iX = min(max(1,self.simfraction_thresholds.GetXaxis().FindBin(seed_et)      ), self.simfraction_thresholds.GetNbinsX())
    iY = min(max(1,self.simfraction_thresholds.GetYaxis().FindBin(abs(seed_eta))), self.simfraction_thresholds.GetNbinsY())
    thre = self.simfraction_thresholds.GetBinContent(iX,iY)
    #print(seed_eta, seed_et, cluster_calo_score, thre, cluster_calo_score >= thre )
    return cluster_calo_score >= thre


def dynamic_window(self,eta):
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




for i, ev in enumerate(chain):    
    if i %100==0: print(".", end="")

    pfCluster_energy = ev.pfCluster_energy
    pfCluster_rawEnergy = ev.pfCluster_rawEnergy
    pfCluster_eta = ev.pfCluster_eta
    pfCluster_phi = ev.pfCluster_phi
    pfCluster_ieta = ev.pfCluster_ieta
    pfCluster_iphi = ev.pfCluster_iphi
    pfCluster_iz = ev.pfCluster_iz
    calo_simenergy = ev.caloParticle_simEnergy
    calo_genenergy = ev.caloParticle_genEnergy
    calo_simeta = ev.caloParticle_simEta
    calo_simphi = ev.caloParticle_simPhi
    calo_simiz = ev.caloParticle_simIz
    # calo_isPU = ev.caloParticle_isPU
    # calo_isOOTPU = ev.caloParticle_isOOTPU
    pfcl_f5_r9 = ev.pfCluster_full5x5_r9
    pfcl_f5_sigmaIetaIeta = ev.pfCluster_full5x5_sigmaIetaIeta
    pfcl_f5_sigmaIetaIphi = ev.pfCluster_full5x5_sigmaIetaIphi
    pfcl_f5_sigmaIphiIphi = ev.pfCluster_full5x5_sigmaIphiIphi
    pfcl_f5_swissCross = ev.pfCluster_full5x5_swissCross
    pfcl_r9 = ev.pfCluster_r9
    pfcl_sigmaIetaIeta = ev.pfCluster_sigmaIetaIeta
    pfcl_sigmaIetaIphi = ev.pfCluster_sigmaIetaIphi
    pfcl_sigmaIphiIphi = ev.pfCluster_sigmaIphiIphi
    pfcl_swissCross = ev.pfCluster_swissCross
    pfcl_nxtals = ev.pfCluster_nXtals
    pfcl_etaWidth = ev.pfCluster_etaWidth
    pfcl_phiWidth = ev.pfCluster_phiWidth
    pfclhit_energy = ev.pfClusterHit_rechitEnergy
    pfclhit_fraction = ev.pfClusterHit_fraction
    pfclhit_ieta = ev.pfClusterHit_ieta
    pfclhit_iphi = ev.pfClusterHit_iphi
    pfclhit_iz = ev.pfClusterHit_iz
    nVtx = ev.nVtx
    rho = ev.rho
    obsPU = ev.obsPU
    truePU = ev.truePU

    clusters_scores = ev.pfCluster_sim_fraction

    pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                            calo_association.get_calo_association(clusters_scores, sort_calo_cl=True, debug=False, min_sim_fraction=1e-4)
    # CaloParticle Pileup information
    cluster_nXtalsPU = ev.pfCluster_simPU_nSharedXtals 
    cluster_PU_simenergy = ev.pfCluster_simEnergy_sharedXtalsPU
    cluster_PU_recoenergy = ev.pfCluster_recoEnergy_sharedXtalsPU
    total_PU_simenergy = ev.caloParticlePU_totEnergy

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
        score = clusters[0][1]
        simenergy_signal = pfcluster_calo_score[seed] * calo_simenergy[calo]

        tot_en_clcalo = 0.
        tot_et_clcalo = 0.
        for icl, _ in clusters:
            tot_en_clcalo += pfCluster_rawEnergy[icl]
            tot_et_clcalo += pfCluster_rawEnergy[icl]/ math.cosh(pfCluster_eta[icl])

        data.append({
            "en": pfCluster_rawEnergy[seed],
            "et": pfCluster_rawEnergy[seed]/ math.cosh(pfCluster_eta[seed]),
            "ieta" : pfCluster_ieta[seed],
            'iphi': pfCluster_iphi[seed],
            "eta" : pfCluster_eta[seed],
            'phi': pfCluster_phi[seed],
            'iz': pfCluster_iz[seed],
            "simfrac_sig": score, 
            "simen_sig": simenergy_signal,
            "simen_pu": cluster_PU_simenergy[seed],
            "recoen_pu": cluster_PU_recoenergy[seed],
            "simen_sig_frac": simenergy_signal/pfCluster_rawEnergy[seed],
            "simen_pu_frac":  cluster_PU_simenergy[seed]/pfCluster_rawEnergy[seed],
            "PUsimen_frac":   cluster_PU_simenergy[seed] / simenergy_signal,
            "tot_en_clcalo": tot_en_clcalo,
            "tot_et_clcalo": tot_et_clcalo,
            "calo_en" : calo_simenergy[calo],
            "calo_et": calo_simenergy[calo] / math.cosh(calo_simeta[calo]),
            "calo_simeta" : calo_simeta[calo],
            "calo_simphi": calo_simphi[calo]
        })

    
        
                
        


df = pd.DataFrame(data)
df.to_csv(args.out, sep=';', index=False)