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


    def get_clusters_inside_window(self,seed_et, seed_eta, seed_phi, seed_iz, cls_eta, cls_phi, cls_iz,
                                   pfcluster_calo_map, pfcluster_calo_score, caloindex):
        true_cls = []
        cls_in_window = [ ]
        #######
        # Loop on all the clusters to find the ones in the windows and count the total and the true ones
        ######
        for icl in range(len(cls_eta)):
            cl_eta = cls_eta[icl]
            cl_phi = cls_phi[icl]
            cl_iz = cls_iz[icl] 
            isin, (etaw, phiw) = in_window(seed_eta,seed_phi,seed_iz, cl_eta, cl_phi, cl_iz,
                                           *self.dynamic_window(seed_eta))
            if isin:
                cls_in_window.append(icl)
                is_calo_matched =  pfcluster_calo_map[icl] == caloindex  # we know at this point it is not -1
                if is_calo_matched:
                    #associate the cluster to the caloparticle with simfraction optimized thresholds 
                    pass_simfrac_thres = self.pass_simfraction_threshold(seed_eta, 
                                                                         seed_et, pfcluster_calo_score[icl] )
                    if pass_simfrac_thres:
                        true_cls.append(icl)
        return cls_in_window, true_cls
                    
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



    def get_windows(self, event, assoc_strategy,  nocalowNmax, min_et_seed=1,
                    sc_collection="superCluster", loop_on_calo=False,  debug=False):
        ## output
        output_object = []
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
        calo_genpt = event.caloParticle_genPt
        calo_simiz = event.caloParticle_simIz
        # calo_geniz = event.caloParticle_genIzt
        # calo_isPU = event.caloParticle_isPU
        # calo_isOOTPU = event.caloParticle_isOOTPU
        pfcl_nxtals = event.pfCluster_nXtals
        nVtx = event.nVtx
        rho = event.rho
        obsPU = event.obsPU
        truePU = event.truePU

        #SuperCluster branches
        sc_rawEn = getattr(event, f"{sc_collection}_rawEnergy")
        #sc_rawESEn = event.superCluster_rawESEnergy
        sc_corrEn = getattr(event,f"{sc_collection}_energy")
        sc_eta = getattr(event,f"{sc_collection}_eta")
        sc_phi = getattr(event,f"{sc_collection}_phi")
        sc_nCls = getattr(event,f"{sc_collection}_nPFClusters")
        sc_seedIndex = [s for s in getattr(event,f"{sc_collection}_seedIndex")]
        pfcl_in_sc = getattr(event,f"{sc_collection}_pfClustersIndex")
        
        # GenParticle info
        genpart_energy = event.genParticle_energy
        genpart_eta = event.genParticle_eta
        genpart_phi = event.genParticle_phi
        genpart_pt = event.genParticle_pt
        

        superCluster_genParticle_dR = getattr(event,f"{sc_collection}_dR_genScore")
        genParticle_superCluster_dR_list = getattr(event,f"genParticle_{sc_collection}_dR_genScore_MatchedIndex")
        genParticle_superCluster_matching = {}
        superCluster_genParticle_matching = {}

        #Gen association
        for igen , sc_vec in enumerate(genParticle_superCluster_dR_list):
            # getting the list (iSC,deltaR  ) to the iGen particle
            dRs = [ (isc,  superCluster_genParticle_dR[isc][igen])  for isc in sc_vec]
            if len(dRs) == 0: continue
            bestSC = list(sorted(dRs, key=itemgetter(1)))[0][0]
            genParticle_superCluster_matching[igen] = bestSC
            superCluster_genParticle_matching[bestSC] = igen

            
        clusters_scores = getattr(event, "pfCluster_"+assoc_strategy)
        # Get Association between pfcluster and calo
        # Sort the clusters for each calo in order of score. 
        # # This is needed to understand which cluster is the seed of the calo
        # Working only on signal caloparticle
        pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
            calo_association.get_calo_association(clusters_scores, sort_calo_cl=True,
                                                  debug=False, min_sim_fraction=self.cluster_min_fraction)

        calo_seeds_init = [(cls[0][0], calo)  for calo, cls in calo_pfcluster_map.items()] # [(seed,caloindex),..]
        calo_seeds =[]
        # clean the calo seeds
        for calo_seed, icalo in calo_seeds_init:
            # Check the minimum sim fraction
            if pfcluster_calo_score[calo_seed] < self.seed_min_fraction:
                print("Seed score too small")
                continue
            # Check if the calo and the seed are in the same window
            calo_inwindow = in_window(calo_geneta[icalo],calo_genphi[icalo],
                                      calo_simiz[icalo], pfCluster_eta[calo_seed],
                                      pfCluster_phi[calo_seed], pfCluster_iz[calo_seed],
                                      *self.dynamic_window(pfCluster_eta[calo_seed]))
            if not calo_inwindow:
                print("Seed not in window")
                continue 
            calo_seeds.append(calo_seed)

        #print("SuperCluster seeds index: ", sc_seedIndex)
        #print("Calo-seeds index: ", calo_seeds)
        calomatched_final_sc = [ ]
        ##################
        ## Object level info
        # Loop on the superCluster and check if they are genMatched or caloMatched
        if not loop_on_calo:
            # Look on all the SuperClusters and check if the seed is a calo-seed
            for iSC in range(len(sc_rawEn)):
                seed = sc_seedIndex[iSC]
                calomatched = seed in calo_seeds
                if calomatched: calomatched_final_sc.append(seed)
                caloindex = pfcluster_calo_map[seed] if calomatched else -999
                genmatched = iSC in  superCluster_genParticle_matching
                genindex = superCluster_genParticle_matching[iSC] if genmatched else -999
                #print(seed, calomatched, genmatched)
                
                seed_eta = pfCluster_eta[seed]
                seed_phi = pfCluster_phi[seed]
                seed_iz = pfCluster_iz[seed]
                seed_en = pfCluster_rawEnergy[seed]
                seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
                
                cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                     pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                          pfcluster_calo_map, pfcluster_calo_score, caloindex)
                missing_cls, correct_cls, spurious_cls = [],[],[]
                for icl in cls_in_window:
                    if icl in true_cls:
                        if icl in pfcl_in_sc[iSC]:
                            correct_cls.append(icl)
                        else:
                            missing_cls.append(icl)
                    else:
                        if icl in pfcl_in_sc[iSC]:
                            spurious_cls.append(icl)

                            
                out = {
                    "calomatched" : int(calomatched),
                    "caloindex": caloindex,
                    "genmatched" : int(genmatched),
                    "genindex": genindex ,
                    "sc_index": iSC,
                    "seed_index": seed,
                    
                    "en_seed": pfCluster_rawEnergy[seed],
                    "et_seed": seed_et,
                    "en_seed_calib": pfCluster_energy[seed],
                    "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                    "seed_eta": seed_eta,
                    "seed_phi": seed_phi,
                    "seed_iz": seed_iz, 
                    
                    "ncls_sel": sc_nCls[iSC],
                    "ncls_sel_true": len(correct_cls),
                    "ncls_sel_false": len(spurious_cls),
                    "ncls_true": len(true_cls),
                    "ncls_tot": len(cls_in_window),
                    "ncls_missing": len(missing_cls),
                    
                    "en_sc_raw": sc_rawEn[iSC], 
                    "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]),
                    #"en_sc_raw_ES" : sc_rawESEn[iSC],
                    #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                    "en_sc_calib": sc_corrEn[iSC], 
                    "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]), 
                    
                    # Sim energy and Gen Enerugy of the caloparticle
                    "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                    "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                    "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                    "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                    "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                    "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                    "calo_genpt": calo_genpt[caloindex] if calomatched else -1,
                    
                    # GenParticle info
                    "genpart_en": genpart_energy[genindex] if genmatched else -1,
                    "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                    "gen_eta": genpart_eta[genindex] if genmatched else -1,
                    "gen_phi": genpart_phi[genindex] if genmatched else -1,
                    "gen_pt": genpart_pt[genindex] if genmatched else -1,
                    
                    
                    # PU information
                    "nVtx": nVtx, 
                    "rho": rho,
                    "obsPU": obsPU, 
                    "truePU": truePU,

                    #Evt number
                    "eventId" : event.eventId,
                    "runId": event.runId
                    
                }
                output_object.append(out)
            ##########################
            ## IF we want to loop on calo-matched seeds instead of SC object
        else:
            for seed in calo_seeds:
                calomatched = True
                # Check if there is a SuperCluster with this calo
                # (The calo seeds have been already filtered by simfraction and inWindow)
                sc_found = seed in sc_seedIndex
                if not sc_found : continue
                iSC = sc_seedIndex.index(seed)
                
                caloindex = pfcluster_calo_map[seed]
                genmatched = iSC in superCluster_genParticle_matching
                genindex = superCluster_genParticle_matching[iSC] if genmatched else -999
                
                seed_eta = pfCluster_eta[seed]
                seed_phi = pfCluster_phi[seed]
                seed_iz = pfCluster_iz[seed]
                seed_en = pfCluster_rawEnergy[seed]
                seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
                
                cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                     pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                          pfcluster_calo_map, pfcluster_calo_score, caloindex)
                missing_cls, correct_cls, spurious_cls = [],[],[]
                for icl in cls_in_window:
                    if icl in true_cls:
                        if icl in pfcl_in_sc[iSC]:
                            correct_cls.append(icl)
                        else:
                            missing_cls.append(icl)
                    else:
                        if icl in pfcl_in_sc[iSC]:
                            spurious_cls.append(icl)
                print(true_cls, correct_cls, spurious_cls, missing_cls)
                            
                out = {
                    "calomatched" : 1,
                    "caloindex": caloindex,
                    "genmatched" : int(genmatched),
                    "genindex": genindex,
                    "sc_index": iSC,
                    "seed_index": seed,
                    
                    "en_seed": pfCluster_rawEnergy[seed],
                    "et_seed": seed_et,
                    "en_seed_calib": pfCluster_energy[seed],
                    "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                    "seed_eta": seed_eta,
                    "seed_phi": seed_phi,
                    "seed_iz": seed_iz, 
                                        
                    "ncls_sel": sc_nCls[iSC],
                    "ncls_sel_true": len(correct_cls),
                    "ncls_sel_false": len(spurious_cls),
                    "ncls_true": len(true_cls),
                    "ncls_tot": len(cls_in_window),
                    "ncls_missing": len(missing_cls),
                    
                    "en_sc_raw": sc_rawEn[iSC], 
                    "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]),
                    #"en_sc_raw_ES" : sc_rawESEn[iSC],
                    #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                    "en_sc_calib": sc_corrEn[iSC], 
                    "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]), 
                    
                    # Sim energy and Gen Enerugy of the caloparticle
                    "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                    "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                    "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                    "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                    "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                    "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                    "calo_genpt": calo_genpt[caloindex] if calomatched else -1,
                    
                    # GenParticle info
                    "genpart_en": genpart_energy[genindex] if genmatched else -1,
                    "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                    "gen_eta": genpart_eta[genindex] if genmatched else -1,
                    "gen_phi": genpart_phi[genindex] if genmatched else -1,
                    "gen_pt": genpart_pt[genindex] if genmatched else -1,
                    
                    
                    # PU information
                    "nVtx": nVtx, 
                    "rho": rho,
                    "obsPU": obsPU, 
                    "truePU": truePU,
                    
                    #Evt number
                    "eventId" : event.eventId,
                    "runId": event.runId
                    
                }
                output_object.append(out)
                    
        return output_object, output_event
