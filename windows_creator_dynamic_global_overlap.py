from __future__ import print_function
from math import pi, sqrt, cosh
import random
import string
from collections import OrderedDict, defaultdict
from operator import itemgetter, attrgetter
import calo_association
import random
import json
import ROOT as R
R.gROOT.ProcessLine(".L Mustache.C+")

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

def is_in_geom_mustache(seed_eta, seed_phi, cl_eta, cl_phi, cl_en ):
    '''
    This functions associates a cluster as true matched only if it is in the mustache
    and if it passes a threshold in simfraction
    '''
    is_in_mustache = False 
    if R.inMustache(seed_eta, seed_phi, cl_en,cl_eta,cl_phi):
        if R.inDynamicDPhiWindow(seed_eta, seed_phi, cl_en, cl_eta, cl_phi):
            is_in_mustache = True
    return is_in_mustache

# Check if a xtal is in the window
def in_window(seed_eta, seed_phi, seed_iz, eta, phi, iz, window_eta, window_phi):
    if seed_iz != iz: return False, (-1,-1)
    etaw = seed_eta - eta
    phiw = DeltaPhi(seed_phi, phi)
    if abs(etaw) <= window_eta and abs(phiw) <= window_phi: 
        return True, (etaw, phiw)
    else:
        return False,(-1,-1)



class WindowCreator():

    def __init__(self, simfraction_thresholds,  seed_min_fraction=1e-2):
        self.seed_min_fraction = seed_min_fraction
        self.simfraction_thresholds = simfraction_thresholds


    def pass_simfraction_threshold(self, seed_eta, seed_et, cluster_calo_score ):
        '''
        This functions associates a cluster as true matched if it passes a threshold in simfraction
        '''
        iX = min(max(1,self.simfraction_thresholds.GetXaxis().FindBin(seed_et)      ), self.simfraction_thresholds.GetNbinsX())
        iY = min(max(1,self.simfraction_thresholds.GetYaxis().FindBin(abs(seed_eta))), self.simfraction_thresholds.GetNbinsY())
        thre = self.simfraction_thresholds.GetBinContent(iX,iY)
        #print(seed_eta, seed_et, cluster_calo_score, thre, cluster_calo_score >= thre )
        return cluster_calo_score >= thre


    def dynamic_window(self, eta,iz):
        # if iz == 0:
        #     return 0.2, 0.6  
        # else:
        #     if abs(eta)< 2.25:
        #         deta = 0.2
        #         x = abs(eta)
        #         dphi =   0.2197*(x**2) - 1.342*x + 2.195
        #         return deta, dphi 
        #     elif abs(eta) >= 2.25:
        #         deta = 0.2
        #         x = 2.25
        #         dphi =  0.2197*(x**2) - 1.342*x + 2.195
        #         return deta, dphi 
        return 0.2,0.6



    def get_windows(self, event, assoc_strategy,  nocalowNmax=True, min_et_seed=1, debug=False):
        # Branches
        pfCluster_energy = event.pfCluster_energy
        pfCluster_rawEnergy = event.pfCluster_rawEnergy
        pfCluster_eta = event.pfCluster_eta
        pfCluster_phi = event.pfCluster_phi
        pfCluster_iz = event.pfCluster_iz
        calo_simenergy = event.caloParticle_simEnergy
        calo_simeta = event.caloParticle_simEta
        calo_simphi = event.caloParticle_simPhi
        calo_simiz = event.caloParticle_simIz
        pfcl_f5_r9 = event.pfCluster_full5x5_r9
        pfcl_f5_sigmaIetaIeta = event.pfCluster_full5x5_sigmaIetaIeta
        pfcl_f5_sigmaIetaIphi = event.pfCluster_full5x5_sigmaIetaIphi
        pfcl_f5_sigmaIphiIphi = event.pfCluster_full5x5_sigmaIphiIphi
        pfcl_f5_swissCross = event.pfCluster_full5x5_swissCross
        pfcl_nxtals = event.pfCluster_nXtals
        pfcl_etaWidth = event.pfCluster_etaWidth
        pfcl_phiWidth = event.pfCluster_phiWidth
        pfclhit_energy = event.pfClusterHit_rechitEnergy
        pfclhit_fraction = event.pfClusterHit_fraction

        clusters_scores = getattr(event, "pfCluster_"+assoc_strategy)
        # Get Association between pfcluster and calo
        # Sort the clusters for each calo in order of score. 
        # This is needed to understand which cluster is the seed of the calo
        pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                                calo_association.get_calo_association(clusters_scores, pfCluster_eta, sort_calo_cl=True, debug=False)

        if debug:
            print(">>> Cluster_calo map")
            for cluster, calo in pfcluster_calo_map.items():
                if calo == -1: continue
                print("cl: {} | calo: {} (calo eta {:.3f}, phi {:.3f})| score: {:.4f}".format(cluster,calo,calo_simeta[calo],calo_simphi[calo],pfcluster_calo_score[cluster]))
            print("\n>>> Calo_cluster map")
            for calo, clusters in calo_pfcluster_map.items():
                print("calo: {} | clusters: {}".format(calo, ["cl: {}, score: {:.4f}".format(cl,sc) for cl, sc in clusters]))
            print()
        #Mustache info
        mustacheseed_pfcls = [s for s in event.superCluster_seedIndex]
        pfcl_in_mustache = event.superCluster_pfClustersIndex
    
        # map of windows, key=window index
        windows_map = OrderedDict()
        windows_nocalomatched = []
        windows_calomatched = [ ]
        seed_clusters = []
        

        # 1) Look for highest energy cluster (corrected energy)
        clenergies_ordered = sorted([ (ic , et) for ic, et in enumerate(
                                map ( lambda k: k[0]/cosh(k[1]), zip( pfCluster_energy, pfCluster_eta) )
                                )], key=itemgetter(1), reverse=True)

        if debug: print(">> Windows formation")
        # Now iterate over clusters in order of energies
        for icl, clenergy_T in clenergies_ordered:
            #print(icl, clenergy_T)

            # No seeds with Et< 1 GeV
            if clenergy_T < min_et_seed: continue

            cl_eta = pfCluster_eta[icl]
            cl_phi = pfCluster_phi[icl]
            cl_iz =  pfCluster_iz[icl]

            # Create new window 
            # - Check if the seed is associated with a calo in the window: calomatched 
            # - Check if the seed is the main cluster of the calo:  caloseed
            # It is required to have seed_min_fraction% of the calo energy and to be "in the window" of the seed
            if pfcluster_calo_map[icl] !=-1 and pfcluster_calo_score[icl]> self.seed_min_fraction:
                caloid = pfcluster_calo_map[icl]
                
                if in_window(calo_simeta[caloid],calo_simphi[caloid],calo_simiz[caloid], cl_eta, cl_phi, cl_iz, 
                                                *self.dynamic_window(cl_eta, cl_phi)):
                    calomatched = caloid
                    # Now check if the seed cluster is the main cluster of the calo
                    if calo_pfcluster_map[caloid][0][0] == icl:
                        caloseed = True
                    else:
                        caloseed =False
                    if debug: 
                        print("Seed-to-calo: cluster: {}, calo: {}, seed_eta: {:.3f}, calo_eta : {:.3f}, seed_score: {:.5f}, is caloseed: {}".format(
                                            icl,caloid,cl_eta,calo_simeta[caloid], pfcluster_calo_score[icl], caloseed))
                else:
                    calomatched = -1
                    caloseed = False
                    if debug: 
                        print("Seed-to-calo [Failed window cut]: cluster: {}, calo: {}, seed_eta: {:.3f}, calo_eta : {:.3f}, seed_score: {:.5f}, is caloseed: {}".format(
                                            icl, caloid,cl_eta,calo_simeta[caloid], pfcluster_calo_score[icl], caloseed))
            else:
                calomatched = -1 
                caloseed = False

            # Save the cluster in the list of associated clusters
            seed_clusters.append(icl)
            # Check if it is a mustache seed
            if icl in mustacheseed_pfcls:
                mustache_seed_index = mustacheseed_pfcls.index(icl)
            else:
                mustache_seed_index = -1

            # Create a unique index
            windex = "".join([ random.choice(string.ascii_lowercase) for _ in range(9)])
            # Let's create  new window:
            new_window = {
                "window_index": windex,
                "seed_index": icl,
                "seed": (cl_eta, cl_phi, cl_iz),

                "clusters": [],

                # The seed of the window is associated with a caloparticle in the window
                "is_seed_calo_matched": calomatched != -1,
                # index of the associated caloparticle
                "calo_index": calomatched,
                # The seed is the cluster associated with the particle with the largest fraction
                "is_seed_calo_seed": caloseed,
                # Mustache info
                "is_seed_mustached_matched": mustache_seed_index != -1,
                "mustache_seed_index": mustache_seed_index,

                "seed_eta": cl_eta,
                "seed_phi": cl_phi, 
                "seed_iz": cl_iz,

                "en_seed": pfCluster_rawEnergy[icl],
                "et_seed": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                "en_seed_calib": pfCluster_energy[icl],
                "et_seed_calib": pfCluster_energy[icl] / cosh(cl_eta),
                "en_true": calo_simenergy[caloseed] if caloseed!=-1 else 0, 
                "et_true": calo_simenergy[caloseed]/cosh(calo_simeta[caloseed]) if caloseed!=-1 else 0, 

                "seed_f5_r9": pfcl_f5_r9[icl],
                "seed_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                "seed_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                "seed_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                "seed_f5_swissCross" : pfcl_f5_swissCross[icl],
                "seed_etaWidth" : pfcl_etaWidth[icl],
                "seed_phiWidth" : pfcl_phiWidth[icl],
                "seed_nxtals" : pfcl_nxtals[icl],
                
            }
            # Save the window
            windows_map[windex] = new_window
            if calomatched == -1:  
                windows_nocalomatched.append(windex)
            else:
                # Save also the window index
                windows_calomatched.append(windex)

        if debug: print(">> Associate clusters...")
        # Now that all the windows have been created let's add all the cluster
        for icl, clenergy_T in clenergies_ordered:
            
            cl_iz = pfCluster_iz[icl]
            cl_eta = pfCluster_eta[icl]
            cl_phi = pfCluster_phi[icl]
            cl_rawen = pfCluster_rawEnergy[icl]
        
            # Fill all the windows
            for window in windows_map.values():
                isin, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz,
                                                *self.dynamic_window(window["seed"][0], window["seed"][2]))
                if isin:
                    # First of all check is the cluster is geometrically in the mustache of the seed
                    in_geom_mustache = is_in_geom_mustache(window["seed_eta"], 
                                                window["seed_phi"], cl_eta, cl_phi, cl_rawen)
                    # If the window is not associated to a calo then in_scluster is always false for the cluster
                    if not window["is_seed_calo_matched"]:
                        is_calo_matched = False   
                        in_scluster = False
                        is_calo_seed = False
                    else: 
                        # We have to check the calo_matching using simfraction threshold
                        # Check if the cluster is associated to the SAME calo as the seed
                        is_calo_matched = pfcluster_calo_map[icl] == window["calo_index"]
                        # If the cluster is associated to the same calo of the seed 
                        # the simfraction threshold by seed eta/et is used
                        if is_calo_matched: 
                            calo_score = pfcluster_calo_score[icl]
                            #filter with simfraction optimized thresholds 
                            in_scluster = self.pass_simfraction_threshold(window["seed_eta"], 
                                                window["et_seed"], pfcluster_calo_score[icl] )
                            # Check if the cluster is the main cluster of the calo associated to the seed
                            if calo_pfcluster_map[window["calo_index"]][0][0] == icl:
                                is_calo_seed = True
                            else:
                                is_calo_seed =False
                        else:
                            calo_score = -1
                            in_scluster = False   
                            is_calo_seed = False            
                            
                        
                    # check if the cluster is inside the same mustache
                    if window["mustache_seed_index"] != -1:
                        in_mustache = icl in pfcl_in_mustache[window["mustache_seed_index"]]
                    else:
                        in_mustache = False
                
                    cevent = {  
                        # "window_index": window["window_index"],
                        "cl_index": icl,
                        # Check if it is the seed
                        "is_seed": window["seed_index"] == icl,
                        # True if the cluster geometrically is in the mustache of the seed
                        "in_geom_mustache" : in_geom_mustache,
                        # True if the seed has a calo and the cluster is associated to the same calo
                        "is_calo_matched": is_calo_matched,
                        # True if the cluster is the main cluster of the calo associated with the seed
                        "is_calo_seed": is_calo_seed,
                        # is_calo_matched & (sim fraction optimized threshold) || is window seed
                        "in_scluster": in_scluster or window["seed_index"] == icl ,
                        # True if the cluster is associated with the same (legacy) mustache as the seed
                        "in_mustache" : in_mustache,
                        # Score of association with the caloparticle of the seed, if present
                        "calo_score": calo_score,

                        "cluster_dphi":phiw ,
                        "cluster_iz" : cl_iz,
                        "en_cluster": pfCluster_rawEnergy[icl],
                        "et_cluster": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                        "en_cluster_calib": pfCluster_energy[icl],
                        "et_cluster_calib": pfCluster_energy[icl] /cosh(cl_eta),
                        
                        # Shower shape variables
                        "cl_f5_r9": pfcl_f5_r9[icl],
                        "cl_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                        "cl_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                        "cl_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                        "cl_f5_swissCross" : pfcl_f5_swissCross[icl],
                        "cl_etaWidth" : pfcl_etaWidth[icl],
                        "cl_phiWidth" : pfcl_phiWidth[icl],
                        "cl_nxtals" : pfcl_nxtals[icl]
                    }
                    if window["seed_eta"] > 0:
                        cevent["cluster_deta"] = cl_eta - window["seed_eta"]
                    else:
                        cevent["cluster_deta"] = window["seed_eta"] - cl_eta
                    
                    # Save the cluster in the window
                    window["clusters"].append(cevent)
                    # In this script save all the clusters in all the windows
                    # Save the cluster only in the first window encountered by Et
                    #break


        if debug:
            print("ALL windows")
            print("N windows:", len(windows_map))
            calo_match = []
            nocalo_match = []
            for w in windows_map.values():
                if w["is_seed_calo_matched"]: calo_match.append(w)
                else: nocalo_match.append(w)
            print("Calo-matched windows: ")
            for w in calo_match:
                print(">> Window: ", w["window_index"], "  Calo Matched: ",  w["is_seed_calo_matched"]," Seed calo-seed: ",w['is_seed_calo_seed'])
                print("\t Seed: {}, Eta:{:.3f}, Phi:{:.3f}, Iz: {:.3f}, En:{:.3f}".format( w['seed_index'], w["seed_eta"], w["seed_phi"],w["seed_iz"], w["en_seed"]))
                print("\t Clusters: ", [cl['cl_index'] for cl in w['clusters']])
                print("\t Calo: {}, Eta:{:.3f}, Phi:{:.3f}, En:{:.3f}".format(w["calo_index"],
                                        calo_simeta[w["calo_index"]],calo_simphi[w["calo_index"]], w["en_true"]))
            print("Not Calo-matched windows: ")
            for w in nocalo_match:
                print(">> Window: ", w["window_index"], "  Calo Matched: ",  w["is_seed_calo_matched"])
                print("\t Seed: {}, Eta:{:.3f}, Phi:{:.3f}, Iz: {:.3f}, En:{:.3f}".format( w['seed_index'], w["seed_eta"], w["seed_phi"],w["seed_iz"], w["en_seed"]))
                print("\t Clusters: ", [cl['cl_index'] for cl in w['clusters']])
                
            print(">>> TOT windows calomatched: ", len(calo_match))
        ###############################
        #### Some metadata
        
        for window in windows_map.values():
            calo_index = window["calo_index"]
            # Check the type of events
            # - Number of pfcluster associated, 
            # - deltaR of the farthest cluster
            # - Energy of the pfclusters
            if calo_index != -1:
                # Get number of associated clusters
                assoc_clusters = list(map(lambda cl: cl["cl_index"], filter(lambda cl: cl["in_scluster"], window["clusters"])))
                max_en_pfcluster = max([pfCluster_energy[i] for i in assoc_clusters])
                window["nclusters_insc"] = len(assoc_clusters)
                window["max_en_cluster_insc"] = max_en_pfcluster
                window["max_deta_cluster_insc"] = max( cl["cluster_deta"] filter(lambda cl: cl["in_scluster"], window["clusters"]))
                window["max_dphi_cluster_insc"] = max( cl["cluster_dphi"] filter(lambda cl: cl["in_scluster"], window["clusters"]))
            else:
                window["nclusters_insc"] = 0
                window["max_en_cluster_insc"] = -1
                window["max_deta_cluster_insc"] = -1
                window["max_dphi_cluster_insc"] = -1
            #General info for all the windows
            window["ncls"] = len(window["clusters"])
            window["max_en_cluster"] = max( cl["en_cluster"] for cl in window["clusters"])
            window["max_deta_cluster"] = max( cl["cluster_deta"] for cl in window["clusters"])
            window["max_dphi_cluster"] = max( cl["cluster_dphi"] for cl in window["clusters"])


        
        # Check if there are more than one windows associated with the same caloparticle
        # windows_calomatched = []
        # for calo, ws in calo_windows.items():
        #     # Take only the first windows, by construction the most energetic one
        #     windows_calomatched.append(ws[0])
        #     if debug:
        #         if len(ws)>1:
        #             print("A lot of windows!")
        #             for w in ws:
        #                 print("Windex: {}, seed et: {}, calo en: {}, calo eta: {}, seed eta: {}".format(w, 
        #                             windows_map[w]["et_seed"], calo_simenergy[calo], calo_simeta[calo], 
        #                             windows_map[w]["seed_eta"]))
        
        # In this version we keep all the windows

        ## Now save only a number of window nocalowindow extracting by random
        if len(windows_nocalomatched)> nocalowNmax:
            windows_to_keep_index = windows_calomatched + random.sample(windows_nocalomatched, nocalowNmax)
        else:
            windows_to_keep_index = windows_calomatched + windows_nocalomatched

        if debug: print("Windows to keep: ", windows_to_keep_index)
        
        windows_to_keep = list(filter(lambda w: w["window_index"] in windows_to_keep_index, windows_map.values()))

        # Now we need to convert list of clusters metadata in list. AOS -> SOA
        output_data = []

        for window in windows_to_keep:
            outw = {k:v for k,v in window.items() if k not in ["seed","clusters"]}
            outw["clusters"] = self.summary_clusters_window(window)
            output_data.append(json.dumps(outw))

        # if debug: print(output_data)
        return output_data



    def summary_clusters_window(self, window):
        clusters_data = {  
            "cl_index": [],
            "is_seed": [],
            # True if the cluster geometrically is in the mustache of the seed
            "in_geom_mustache" : [],
            # True if the seed has a calo and the cluster is associated to the same calo
            "is_calo_matched": [],
            # True if the cluster is the main cluster of the calo associated with the seed
            "is_calo_seed": [],
            # is_calo_matched & (sim fraction optimized threshold)
            "in_scluster": [],
            # True if the cluster is associated with the same (legacy) mustache as the seed
            "in_mustache" : [],
            # Score of association with the caloparticle of the seed, if present
            "calo_score" " : []

            "cluster_deta": [],
            "cluster_dphi":[] ,
            "cluster_iz" : [],
            "en_cluster": [],
            "et_cluster": [],
            "en_cluster_calib": [],
            "et_cluster_calib": [],
            
            # Shower shape variables
            "cl_f5_r9": [],
            "cl_f5_sigmaIetaIeta" : [],
            "cl_f5_sigmaIetaIphi" : [],
            "cl_f5_sigmaIphiIphi" : [],
            "cl_f5_swissCross" : [],
            "cl_etaWidth" : [],
            "cl_phiWidth" : [],
            "cl_nxtals"   : []
        }
        for cl in window["clusters"]:
            for key in clusters_data.keys():
                if type(cl[key]) is bool:
                    clusters_data[key].append(int(cl[key]))
                else:
                    clusters_data[key].append(cl[key])
        return clusters_data