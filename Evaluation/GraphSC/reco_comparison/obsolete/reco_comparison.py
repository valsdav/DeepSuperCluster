from __future__ import print_function
from math import pi, sqrt, cosh
import random
import string
from collections import OrderedDict, defaultdict
from operator import itemgetter, attrgetter
import calo_association
import random
from pprint import pprint
import json
import numpy as np
import ROOT as R

'''
This script extracts the windows and associated clusters from events
coming from RecoSimDumper. 

All windows are created:  seeds inside other windows creates their window
'''


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

# Check if a xtal is in the window
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

    return data


class WindowCreator():

    def __init__(self, simfraction_thresholds,  seed_min_fraction=1e-2, cl_min_fraction=1e-4, simenergy_pu_limit = 1.5):
        self.seed_min_fraction = seed_min_fraction
        self.cluster_min_fraction = cl_min_fraction
        self.simfraction_thresholds = simfraction_thresholds
        self.simenergy_pu_limit = simenergy_pu_limit


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



    def get_windows(self, event, assoc_strategy,  nocalowNmax, min_et_seed=1, debug=False):
        ## output
        output_seeds = []
        output_event = []
        # Branches
        pfCluster_energy = event.pfCluster_energy
        pfCluster_rawEnergy = event.pfCluster_rawEnergy
        pfCluster_eta = event.pfCluster_eta
        pfCluster_phi = event.pfCluster_phi
        pfCluster_ieta = event.pfCluster_ieta
        pfCluster_iphi = event.pfCluster_iphi
        pfCluster_iz = event.pfCluster_iz
        calo_simenergy = event.caloParticle_simEnergy
        calo_simenergy_goodstatus = event.caloParticle_simEnergyGoodStatus
        calo_genenergy = event.caloParticle_genEnergy
        calo_simeta = event.caloParticle_simEta
        calo_simphi = event.caloParticle_simPhi
        calo_geneta = event.caloParticle_genEta
        calo_genphi = event.caloParticle_genPhi
        calo_simiz = event.caloParticle_simIz
        # calo_geniz = event.caloParticle_genIz
        # calo_isPU = event.caloParticle_isPU
        # calo_isOOTPU = event.caloParticle_isOOTPU
        pfcl_nxtals = event.pfCluster_nXtals
        nVtx = event.nVtx
        rho = event.rho
        obsPU = event.obsPU
        truePU = event.truePU

        clusters_scores = getattr(event, "pfCluster_"+assoc_strategy)
        # Get Association between pfcluster and calo
        # Sort the clusters for each calo in order of score. 
        # # This is needed to understand which cluster is the seed of the calo
        # Working only on signal caloparticle
        pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                                calo_association.get_calo_association(clusters_scores, sort_calo_cl=True, debug=False, min_sim_fraction=self.cluster_min_fraction)
        # CaloParticle Pileup information
        cluster_nXtalsPU = event.pfCluster_simPU_nSharedXtals 
        cluster_PU_simenergy = event.pfCluster_simEnergy_sharedXtalsPU
        cluster_signal_simenergy = event.pfCluster_simEnergy_sharedXtals
        cluster_PU_recoenergy = event.pfCluster_recoEnergy_sharedXtalsPU
        total_PU_simenergy = event.caloParticlePU_totEnergy

        # #total PU simenergy in all clusters in the event
        # total_PU_simenergy = sum([simPU for cl, simPU in cluster_PU_simenergy.items()])

        if debug:
            print(">>> Cluster_calo map")
            for cluster, calo in pfcluster_calo_map.items():
                if calo == -1: continue
                print("cl: {} | calo: {} (calo Et: {:.2f}, eta {:.2f}, phi {:.2f})| score: {:.4f}, simEnPU: {:.3f}".format(cluster,calo,
                                            calo_simenergy[calo]/cosh(calo_simeta[calo]) ,calo_simeta[calo],calo_simphi[calo],pfcluster_calo_score[cluster],cluster_PU_simenergy[cluster]))
            print("\n>>> Calo_cluster map")
            for calo, clusters in calo_pfcluster_map.items():
                print("calo: {} | clusters: ".format(calo))
                for cl, sc in clusters:
                    print("\t> cl: {}, Et: {:.2f}, eta: {:.2f}, phi:{:.2f}, score: {:.4f}, simEnPU: {:.3f}".format(cl,pfCluster_rawEnergy[cl]/ cosh(pfCluster_eta[cl]), pfCluster_eta[cl],pfCluster_phi[cl], sc,cluster_PU_simenergy[cl]))
            print()

        #Mustache info
        mustache_seedindex = [s for s in event.superCluster_seedIndex]
        mustache_rawEn = event.superCluster_rawEnergy
        mustache_calibEn = event.superCluster_energy
        mustache_eta = event.superCluster_eta
        mustache_ncls = event.superCluster_nPFClusters
        pfcl_in_mustache = event.superCluster_pfClustersIndex

        #DeepSC info
        deepsc_seedindex = [s for s in event.deepSuperCluster_seedIndex]
        deepsc_rawEn = event.deepSuperCluster_rawEnergy
        deepsc_calibEn = event.deepSuperCluster_energy
        deepsc_eta = event.deepSuperCluster_eta
        deepsc_ncls = event.deepSuperCluster_nPFClusters
        pfcl_in_deepsc = event.deepSuperCluster_pfClustersIndex    
        if debug:
            print("deepSC seeds", deepsc_seedindex)
            print("mustache seeds", mustache_seedindex)
        
        # Look at the caloparticles 
        # Get only the seed
        calomatched_seeds = [ ]
        deepsc_calomatched_seeds = [ ]
        mustache_calomatched_seeds = [ ]
        for calo, clusters in calo_pfcluster_map.items():
            seed = clusters[0][0]
            calomatched_seeds.append(seed)
            seed_score = clusters[0][1]
            seed_eta = pfCluster_eta[seed]
            seed_phi = pfCluster_phi[seed]
            seed_iz = pfCluster_iz[seed]
            seed_en = pfCluster_rawEnergy[seed]
            seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
            
            if seed_score < self.seed_min_fraction:
                if debug: print("SEED SCORE too small")
                continue

            calo_inwindow = in_window(calo_geneta[calo],calo_genphi[calo],
                                      calo_simiz[calo], seed_eta, seed_phi, seed_iz,
                                      *self.dynamic_window(seed_eta))
            if not calo_inwindow:
                print("SEED NOT IN CALO WINDOW")
                continue

            # Now check the basic quantities for DeepSc and Mustache
            deepsc_found = seed in deepsc_seedindex
            mustache_found = seed in mustache_seedindex
            deepsc_index = -1
            mustache_index = -1
            m_cls = []
            d_cls = []
            if deepsc_found:
                deepsc_calomatched_seeds.append(seed)
                deepsc_index = deepsc_seedindex.index(seed)
                d_cls = pfcl_in_deepsc[deepsc_index] 
            if mustache_found:
                mustache_calomatched_seeds.append(seed)
                mustache_index = mustache_seedindex.index(seed)
                m_cls = pfcl_in_mustache[mustache_index]

            if debug:
                print("Calo ({}), Seed ({}), DeepSC clusters {}".format(calo,seed,d_cls))
                print("Calo ({}), Seed ({}), Mustache clusters {}".format(calo,seed,m_cls))                

            true_cls = []
            cls_in_window = [ ]

            #######
            # Loop on all the clusters to find the ones in the windows and count the total and the true ones
            ######
            for icl in range(len(pfCluster_eta)):
                cl_eta = pfCluster_eta[icl]
                cl_phi = pfCluster_phi[icl]
                cl_iz = pfCluster_iz[icl]

                isin, (etaw, phiw) = in_window(seed_eta,seed_phi,seed_iz, cl_eta, cl_phi, cl_iz,
                                                *self.dynamic_window(seed_eta))
                if isin:
                    cls_in_window.append(icl)
                    is_calo_matched =  pfcluster_calo_map[icl] == calo  # we know at this point it is not -1
                    if is_calo_matched:
                        PU_simenfrac = cluster_PU_simenergy[icl] / cluster_signal_simenergy[icl][calo]
                        # First of all check the PU sim energy limit
                        if PU_simenfrac < self.simenergy_pu_limit:
                            #associate the cluster to the caloparticle with simfraction optimized thresholds 
                            pass_simfrac_thres = self.pass_simfraction_threshold(seed_eta, 
                                                                                 seed_et, pfcluster_calo_score[icl] )
                            if pass_simfrac_thres:
                                true_cls.append(icl)

            if debug:
                print("True cls: ", true_cls)
                print("All cls:", cls_in_window)
                
            out = {
                "calomatched" : 1,
                "en_seed": pfCluster_rawEnergy[seed],
                "et_seed": seed_et,
                "en_seed_calib": pfCluster_energy[seed],
                "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                "seed_eta": seed_eta,
                "seed_phi": seed_phi,
                "seed_iz": seed_iz, 

                "in_deepsc" : int(deepsc_found),
                "in_mustache": int(mustache_found),

                "ncls_tot": len(cls_in_window),
                "ncls_true": len(true_cls),
                "ncls_mustache" : mustache_ncls[mustache_index] if mustache_index != -1 else 0,                
                "ncls_deepsc" : deepsc_ncls[deepsc_index] if deepsc_index!=-1 else 0,
                "cls_must" : [c for c in m_cls],
                "cls_deepsc" : [c for c in d_cls],
                                
                "en_mustache_raw": mustache_rawEn[mustache_index] if mustache_index!=-1 else 0, 
                "et_mustache_raw": mustache_rawEn[mustache_index]/cosh(mustache_eta[mustache_index]) if mustache_index!=-1 else 0, 
                "en_mustache_calib": mustache_calibEn[mustache_index]  if mustache_index!=-1 else 0, 
                "et_mustache_calib": mustache_calibEn[mustache_index]/cosh(mustache_eta[mustache_index]) if mustache_index!=-1 else 0,

                "en_deepsc_raw": deepsc_rawEn[deepsc_index] if deepsc_index!=-1 else 0, 
                "et_deepsc_raw": deepsc_rawEn[deepsc_index]/cosh(deepsc_eta[deepsc_index]) if deepsc_index!=-1 else 0, 
                "en_deepsc_calib": deepsc_calibEn[deepsc_index]  if deepsc_index!=-1 else 0, 
                "et_deepsc_calib": deepsc_calibEn[deepsc_index]/cosh(deepsc_eta[deepsc_index]) if deepsc_index!=-1 else 0,
                
                # Sim energy and Gen Enerugy of the caloparticle
                "calo_en_true_gen": calo_genenergy[calo], 
                "calo_et_true_gen": calo_genenergy[calo]/cosh(calo_geneta[calo]),
                "calo_en_true_sim": calo_simenergy_goodstatus[calo], 
                "calo_et_true_sim": calo_simenergy_goodstatus[calo]/cosh(calo_geneta[calo]),
                "calo_geneta": calo_geneta[calo],
                "calo_genphi": calo_genphi[calo],
                "calo_simeta": calo_simeta[calo],
                "calo_simphi": calo_simphi[calo],
                
                # PU information
                "nVtx": nVtx, 
                "rho": rho,
                "obsPU": obsPU, 
                "truePU": truePU,
                "event_tot_simen_PU": total_PU_simenergy,
            }
            output_seeds.append(out)

        # Now analyze not calomatched seeds just looking at all the seeds of the Mustache and DeepSC
        nocalomatched_seeds = [ ]
        for seed in mustache_seedindex + deepsc_seedindex:
            if seed in nocalomatched_seeds: continue
            if seed in calomatched_seeds: continue
            nocalomatched_seeds.append(seed)
            seed_eta = pfCluster_eta[seed]
            seed_phi = pfCluster_phi[seed]
            seed_iz = pfCluster_iz[seed]
            seed_en = pfCluster_rawEnergy[seed]
            seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
            
            deepsc_found = seed in deepsc_seedindex
            mustache_found = seed in mustache_seedindex
            deepsc_index = -1
            mustache_index = -1
            m_cls = []
            d_cls = []
            if deepsc_found:
                deepsc_index = deepsc_seedindex.index(seed)
                d_cls = pfcl_in_deepsc[deepsc_index] 
            if mustache_found:
                mustache_index = mustache_seedindex.index(seed)
                m_cls = pfcl_in_mustache[mustache_index]

                
            out = {
                "calomatched" : 0,
                "en_seed": pfCluster_rawEnergy[seed],
                "et_seed": seed_et,
                "en_seed_calib": pfCluster_energy[seed],
                "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                "seed_eta": seed_eta,
                "seed_phi": seed_phi,
                "seed_iz": seed_iz, 

                "in_deepsc" : int(xdeepsc_found),
                "in_mustache": int(mustache_found),

                "ncls_mustache" : mustache_ncls[mustache_index] if mustache_index != -1 else 0,                
                "ncls_deepsc" : deepsc_ncls[deepsc_index] if deepsc_index!=-1 else 0,
                "cls_must" : [c for c in m_cls],
                "cls_deepsc" : [c for c in d_cls],
                
                "en_mustache_raw": mustache_rawEn[mustache_index] if mustache_index!=-1 else 0, 
                "et_mustache_raw": mustache_rawEn[mustache_index]/cosh(mustache_eta[mustache_index]) if mustache_index!=-1 else 0, 
                "en_mustache_calib": mustache_calibEn[mustache_index]  if mustache_index!=-1 else 0, 
                "et_mustache_calib": mustache_calibEn[mustache_index]/cosh(mustache_eta[mustache_index]) if mustache_index!=-1 else 0,

                "en_deepsc_raw": deepsc_rawEn[deepsc_index] if deepsc_index!=-1 else 0, 
                "et_deepsc_raw": deepsc_rawEn[deepsc_index]/cosh(deepsc_eta[deepsc_index]) if deepsc_index!=-1 else 0, 
                "en_deepsc_calib": deepsc_calibEn[deepsc_index]  if deepsc_index!=-1 else 0, 
                "et_deepsc_calib": deepsc_calibEn[deepsc_index]/cosh(deepsc_eta[deepsc_index]) if deepsc_index!=-1 else 0,

                # Sim energy and Gen Enerugy of the caloparticle
                "calo_en_true_gen": -1, 
                "calo_et_true_gen": -1,
                "calo_en_true_sim": -1,
                "calo_et_true_sim": -1,
                "calo_geneta": -1,
                "calo_genphi": -1,
                "calo_simeta": -1,
                "calo_simphi": -1,
                

                # PU information
                "nVtx": nVtx, 
                "rho": rho,
                "obsPU": obsPU, 
                "truePU": truePU,
                "event_tot_simen_PU": total_PU_simenergy,
            }
            output_seeds.append(out)
            
        ## Now save event-wise information
        out = {
            "nseeds_must" : len(mustache_seedindex),
            "nseeds_deepsc" : len(deepsc_seedindex),
            "nseeds_calomatched" : len(calomatched_seeds),
            "nseeds_nocalomatched" : len(nocalomatched_seeds),
            "nseeds_calomatched_must" : len(mustache_calomatched_seeds),
            "nseeds_calomatched_deepsc" : len(deepsc_calomatched_seeds),
            "seeds_must" : mustache_seedindex,
            "seeds_deepsc" : deepsc_seedindex,
            "seeds_calomatched_must": mustache_calomatched_seeds,
            "seeds_calomatched_deepsc" : deepsc_calomatched_seeds,
            "ncls_tot_must" : sum([len(m) for m in pfcl_in_mustache ]),
            "ncls_tot_deepsc" : sum([len(m) for m in pfcl_in_deepsc ]),
            "ncls_tot" : len(pfCluster_energy),
            
        }
        output_event.append(out)
        
        return output_seeds, output_event
