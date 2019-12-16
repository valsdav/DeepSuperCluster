from math import pi, sqrt, cosh

'''
This script extracts the windows and associated clusters from events
coming from RecoSimDumper
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


# Check if a xtal is in the window
def in_window(seed_ieta, seed_iphi, seed_iz, ieta, iphi, iz, window_ieta, window_iphi):
    if seed_iz != iz: return False, (-1,-1)
    ietaw = ieta_distance(seed_ieta,ieta,iz)
    iphiw = iphi_distance(seed_iphi,iphi,iz)
    if abs(ietaw) <= window_ieta and abs(iphiw) <= window_iphi: 
        return True, (ietaw, iphiw)
    else:
        return False,(-1,-1)

# Check if cluster has an hit in the window
def cluster_in_window(window, clhits_ieta, clhits_iphi, clhits_iz):
    for ieta, iphi, iz in zip(clhits_ieta, clhits_iphi, clhits_iz):
        hit_in_wind, (ietaw, iphiw) = in_window(window["seed"][0],window["seed"][1],window["seed"][2],ieta, iphi, iz)
        #print((ieta,iphi,iz), (window["seed"][0],window["seed"][1],window["seed"][2]), ietaw, iphiw)
        if hit_in_wind:
            return True
    return False


def get_windows(event, window_ieta, window_iphi):
    # Branches
    pfCluster_energy = event.pfCluster_energy
    pfCluster_ieta = event.pfCluster_ieta
    pfCluster_iphi = event.pfCluster_iphi
    pfCluster_eta = event.pfCluster_eta
    pfCluster_phi = event.pfCluster_phi
    pfCluster_iz = event.pfCluster_iz

    pfcluster_calo_map = event.pfCluster_sim_fraction_min1_MatchedIndex
    calo_pfcluster_map = event.caloParticle_pfCluster_sim_fraction_min1_MatchedIndex
   
    # map of windows, key=pfCluster seed index
    windows_list = {}
    window_index = -1
    nonseed_clusters = []
    # 1) Look for highest energy cluster
    clenergies_ordered = sorted([ (ic , en) for ic, en in enumerate(pfCluster_energy)], 
                                                    key=itemgetter(1), reverse=True)

    # Now iterate over clusters in order of energies
    for iw, (icl, clenergy) in enumerate(clenergies_ordered):
        cl_ieta = pfCluster_ieta[icl]
        cl_iphi = pfCluster_iphi[icl]
        cl_iz = pfCluster_iz[icl]
        cl_eta = pfCluster_eta[icl]
        cl_phi = pfCluster_phi[icl]

        is_in_window = False
        # Check if it is already in one windows
        for window in windows_list:
            is_in_window, (ietaw, iphiw) = in_window(*window["seed"], cl_ieta, cl_iphi, cl_iz) 
            if is_in_window:
                nonseed_clusters.append(icl)
                break

        # If is not already in some window 
        if not is_in_window: 
            caloseed = pfcluster_calo_map[icl]
            if caloseed == -1:
                nocalowN+=1
                # Not creating too many windows of noise
                if nocalowN> nocalowNmax: continue
            # Let's create  new window:
            new_window = {
                "seed": (cl_ieta, cl_iphi, cl_iz),
                "calo" : caloseed,
                "metadata": {
                    "seed_eta": cl_eta,
                    "seed_phi": cl_phi, 
                    "seed_iz": cl_iz,
                    "en_seed": pfCluster_energy[icl],
                    "en_true": calo_simenergy[caloseed] if caloseed!=-1 else 0, 
                    "is_calo": caloseed != -1
                }, 
                "clusters": []
            }
            
            # Create a unique index
            window_index += 1
            new_window["metadata"]["index"] = windex
            # Save the window
            windows_list.append(new_window)
            # isin, mask = fill_window_cluster(new_window, clxtals_ieta, clxtals_iphi, clxtals_iz, 
            #                     clxtals_energy, clxtals_rechitEnergy, pfcluster_calo_map[icl], fill_mask=True)
            # Save also seed cluster for cluster_masks
            new_window["clusters"].append({
                    "cluster_deta": 0.,
                    "cluster_dphi": 0., 
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_energy[icl],
                    "is_seed": True,
                    "in_scluster":  pfcluster_calo_map[icl] == new_window["calo"],
                })

           
    # Now that all the seeds are inside let's add the non seed
    for icl_noseed in nonseed_clusters:
        cl_ieta = pfCluster_ieta[icl_noseed]
        cl_iphi = pfCluster_iphi[icl_noseed]
        cl_iz = pfCluster_iz[icl_noseed]
        cl_eta = pfCluster_eta[icl_noseed]
        cl_phi = pfCluster_phi[icl_noseed]

        # Fill all the windows
        for window in windows_list:
            isin, (ietaw, iphiw) = in_window(*window["seed"], cl_ieta, cl_iphi, cl_iz)
            if isin:
                cevent = {
                    "cluster_dphi": DeltaPhi(cl_phi, window["metadata"]["seed_phi"]), 
                    "cluster_iz" : cl_iz,
                    "en_cluster": pfCluster_energy[icl_noseed],
                    "is_seed": False,
                    "in_scluster":  pfcluster_calo_map[icl_noseed] == window["calo"]
                }
                if window["metadata"]["seed_eta"] > 0:
                    cevent["cluster_deta"] = cl_eta - window["metadata"]["seed_eta"]
                else:
                    cevent["cluster_deta"] = window["metadata"]["seed_eta"] - cl_eta
                
                window["clusters"].append(cevent)


    return windows_list