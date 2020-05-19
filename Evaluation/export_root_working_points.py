#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np 
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os 
import glob
import root_pandas

parser = argparse.ArgumentParser()

parser.add_argument("-iv", "--input-version", type=str, help="Inputdir", required=True)
parser.add_argument("-n", "--ninputfiles", type=int, help="N. files of input", required=True)
parser.add_argument("-o", "--outputfile", type=str, help="Outputfile", required=True)
parser.add_argument("-m", "--model", type=str, help="model file", required=True)
parser.add_argument("-s", "--scaler", type=str, help="scaler", required=True)
parser.add_argument("-e", "--eta-interval",  type=str, help="eta intervals", required=True)
parser.add_argument("--mustache", action="store_true", help="export for mustache", default=False)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.outputfile),exist_ok=True)

from tensorflow import keras 

if not args.mustache:
    model = keras.models.load_model(args.model)
    scaler = pickle.load(open(args.scaler, "rb"))

inputdir = "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/"
ens = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dnn_thres =  np.linspace(0.3 ,1, 30)[:-1]

cols = ["seed_eta", "seed_phi", "seed_iz","en_seed","et_seed",
        "cluster_deta", "cluster_dphi", "en_cluster", "et_cluster",
       "seed_f5_r9", "seed_f5_sigmaIetaIeta","seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi",
        "seed_f5_swissCross","seed_nxtals", "seed_etaWidth", "seed_phiWidth",
        "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi",
        "cl_f5_swissCross", "cl_nxtals", "cl_etaWidth", "cl_phiWidth"]

datas_ele = []

i = 0

for f in glob.glob(inputdir+ "electrons/numpy_v{}/testing/clusters_data_*.pkl".format(args.input_version)):
    i+=1
    if i >args.ninputfiles: break
    d = pickle.load(open(f, "rb"))
    datas_ele.append( d[(d.is_calo_matched == True)] )

data_ele = pd.concat(datas_ele, ignore_index=True)
data_ele["particle"] = "electron"
print("N events ele: ",len(data_ele))

i = 0
datas_gamma = []
for f in glob.glob(inputdir+ "gammas/numpy_v{}/testing/clusters_data_*.pkl".format(args.input_version)):
    i+=1
    if i >args.ninputfiles: break
    d = pickle.load(open(f, "rb"))
    datas_gamma.append(d[ (d.is_calo_matched == True)])
    
data_gamma = pd.concat(datas_gamma, ignore_index=True)
data_gamma["particle"] = "gamma"
print("N events gamma: ",len(data_gamma))

if data_ele.shape[0]> data_gamma.shape[0]:
    data_val = pd.concat([data_gamma, data_ele.iloc[0:len(data_gamma)]], ignore_index=True)
else:
    data_val = pd.concat([data_gamma.iloc[0:len(data_ele)], data_ele], ignore_index=True)

print(f"N samples ele {data_ele.shape}")
print(f"N samples gamma {data_gamma.shape}")


data = pd.concat([data_ele, data_gamma])

eta_bin = list(map(float, args.eta_interval.split("-")))
print("Working on etabin", eta_bin)

data_val = data[ ( abs(data.seed_eta) >= eta_bin[0]) & ( abs(data.seed_eta) < eta_bin[1] ) ]

if not args.mustache:
    data_val["y"] = model.predict(scaler.transform(data_val[cols].values), batch_size=4096)


results= []

if not args.mustache:
    for thr in dnn_thres:
        #print("DNN threshold: ", thr)
        g = data_val[( (data_val.y >  thr) | (data_val.is_seed==True) ) ].groupby("window_index", sort=False).agg(
                                { "en_cluster_calib": 'sum' ,
                                "en_true": "first", 
                                "et_true": "first",
                                "et_seed_calib": "first",
                                "seed_eta": "first",
                                })
        #print(g)
        g["EoEtrue"] = g["en_cluster_calib"] / g["en_true"]
        g["en_bin"] = pd.cut(g["et_seed_calib"], ens, labels=list(range(len(ens)-1)))
        g["dnn_thre"] = thr
        results.append(g)

    result = pd.concat(results)
    root_pandas.to_root(result[["en_cluster_calib", "en_true", "et_seed_calib","seed_eta", "et_true", "EoEtrue", "dnn_thre"]], 
                            args.outputfile, key="resolution_scan")


elif args.mustache:
    g = data_val[data_val.in_mustache==True].groupby("window_index", sort=False).agg(
                                { "en_cluster_calib": 'sum' ,
                                "en_true": "first", 
                                "et_true": "first",
                                "et_seed_calib": "first",
                                "seed_eta": "first",
                                })
    #print(g)
    g["EoEtrue"] = g["en_cluster_calib"] / g["en_true"]
    g["en_bin"] = pd.cut(g["et_seed_calib"], ens, labels=list(range(len(ens)-1)))
   

    root_pandas.to_root(g[["en_cluster_calib", "en_true", "et_seed_calib","seed_eta", "et_true", "EoEtrue"]], 
                            args.outputfile, key="resolution_scan")

print("DONE")