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

parser = argparse.ArgumentParser()

parser.add_argument("-iv", "--input-version", type=str, help="Inputdir", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-m", "--model", type=str, help="model file", required=True)
parser.add_argument("-s", "--scaler", type=str, help="scaler", required=True)
parser.add_argument("-e", "--eta-intervals", nargs="+", type=str, help="eta intervals", required=True)

args = parser.parse_args()

os.makedirs(args.outputdir,exist_ok=True)

from tensorflow import keras 

model = keras.models.load_model(args.model)
scaler = pickle.load(open(args.scaler, "rb"))

inputdir = "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/"
ens = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dnn_thres =  np.linspace(0.3 ,1, 30)[:-1]


cols = ["seed_eta", "seed_phi", "seed_iz","en_seed","et_seed",
        "cluster_deta", "cluster_dphi", "en_cluster", "et_cluster",
       "seed_f5_r9", "seed_f5_sigmaIetaIeta","seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi","seed_swissCross","seed_nxtals",
        "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi","cl_swissCross", "cl_nxtals"]

datas_ele = []

i = 0

for f in glob.glob(inputdir+ "electrons/numpy_v{}/testing/clusters_data_*.pkl".format(args.input_version)):
    i+=1
    if i >10: break
    d = pickle.load(open(f, "rb"))
    datas_ele.append( d[(d.is_calo_matched == True)] )

data_ele = pd.concat(datas_ele, ignore_index=True)
data_ele["particle"] = "electron"
print("N events ele: ",len(data_ele))

i = 0
datas_gamma = []
for f in glob.glob(inputdir+ "gammas/numpy_v{}/testing/clusters_data_*.pkl".format(args.input_version)):
    i+=1
    if i >10: break
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


def bin_analysis(group):
    ratio_left = group["EoEtrue"].quantile(0.16)
    ratio_right = group["EoEtrue"].quantile(0.84)
    ratio_mean = group[(group.EoEtrue >= ratio_left) & (group.EoEtrue <= ratio_right) ].EoEtrue.mean()
    return pd.Series(  
        { 
             "EoEtrue_68scale": ratio_mean,
             "EoEtrue_68width": (abs(ratio_right-ratio_mean) + abs(ratio_left-ratio_mean))/2, 
             #"EoEtrue_68width": abs(ratio_right- ratio_left), 
             "EoEtrue_scale":  group.EoEtrue.mean(),
             "EoEtrue_rms":  group.EoEtrue.std()
        })


for eta_interval in args.eta_intervals:
    eta_bin = list(map(float, eta_interval.split("-")))
    print("Working on etabin", eta_bin)

    data_val = data[ ( abs(data.seed_eta) >= eta_bin[0]) & ( abs(data.seed_eta) <= eta_bin[1] ) ]
    data_val["y"] = model.predict(scaler.transform(data_val[cols].values), batch_size=4096)


    results= []

    for thr in dnn_thres:
        #print("DNN threshold: ", thr)
        g = data_val[ (data_val.y >  thr) | (data_val.is_seed==True) ].groupby("window_index", sort=False).agg(
                                { "en_cluster": 'sum' ,
                                "en_true": "min", 
                                "et_seed": "first",
                                })
        g["EoEtrue"] = g["en_cluster"] / g["en_true"]
        g["en_bin"] = pd.cut(g["et_seed"], ens, labels=list(range(len(ens)-1)))
        
        scanres = g.groupby("en_bin").apply(bin_analysis)
        scanres["dnn_thre"] = thr
        results.append(scanres)

    result = pd.concat(results)
    # index by en_bin and DNN threshold
    result.reset_index(level=0, inplace=True)

    g = data_val[ (data_val.in_mustache==True) | (data_val.is_seed==True) ].groupby("window_index", sort=False).agg(
                            { "en_cluster": 'sum' ,
                                "en_true": "min", 
                                "et_seed": "first",
                                })
    #print(g)
    g["EoEtrue"] = g["en_cluster"] / g["en_true"]
    g["en_bin"] = pd.cut(g["et_seed"], ens, labels=list(range(len(ens)-1)))
    result_must = g.groupby("en_bin").apply(bin_analysis)
    result_must.reset_index(level=0, inplace=True)


    for enbin in range(len(ens)-1):
        
        fig, [ax1,ax2] = plt.subplots(1,2, figsize=(14,5), dpi=100)
        df = result[(result.en_bin==enbin)]
        x = df.dnn_thre.values
        
        # scale
        y = df["EoEtrue_68scale"].values
        y_must = result_must[result_must.en_bin==enbin]["EoEtrue_68scale"].iloc[0]
        ax1.plot(x, y/y_must, "b",label="median")
        ax1.plot(x, [1.]*len(x), "g--")
        
        
        ax1.legend()
        ax1.set_xlabel("DNN score > x")
        ax1.set_ylabel("DeeSC / Mustache")
        ax1.set_title(f"E SC/Etrue scale:  Et seed [{ens[enbin]}, {ens[enbin+1]}]")
        
        ##### width
        y = df["EoEtrue_68width"].values
        y_must = result_must[result_must.en_bin==enbin]["EoEtrue_68width"].iloc[0]
        
        ax2.plot(x, y/y_must,  "b", label="68% width",)
        ax2.plot(x, [1.]*len(x), "g--")
        
        ax2.legend()
        ax2.set_xlabel("DNN score > x")
        ax2.set_ylabel("DeeSC / Mustache ")
        ax2.set_title(f"E SC/Etrue width:  Et seed [{ens[enbin]}, {ens[enbin+1]}]")
        
        fig.savefig(args.outputdir +f"/scale_width_eta_{eta_bin[0]}-{eta_bin[1]}_et_{ens[enbin]}-{ens[enbin+1]}_seed.png")
        

