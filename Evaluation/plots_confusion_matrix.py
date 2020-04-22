import pickle
import numpy as np 
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
import argparse
from keras.models import load_model

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams["image.origin"] = 'lower'


parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="Inputdir data", required=True)
parser.add_argument("-m","--model", type=str, help="model file", required=True)
parser.add_argument("-s","--scaler", type=str, help="scaler file", required=True)
parser.add_argument("--eta", type=float, nargs="+", help="Eta bins", required=True)
parser.add_argument("--en", type=float, nargs="+", help="ET bins", required = True)
parser.add_argument("--deta", type=float, help="DeltaEta radius", required=True)
parser.add_argument("--dphi", type=float, help="DeltaPhi radius", required = True)
parser.add_argument("-t","--thresholds", type=float, nargs="+", help="Threshold", required = True)
parser.add_argument("-o","--outputdir", type=str, help="outputdir", required=True)
parser.add_argument("-r","--roc", action="store_true",  help="Compute ROC", default=False)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("--vars", type=int, help="Variables (0,1,2,)", default=False)
parser.add_argument("-nf","--nfiles", type=int, help="N input files to read", default=False)

args = parser.parse_args()

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)


scaler = pickle.load(open(args.scaler, "rb"))

model = load_model(args.model)

include_seed = False
datas_val = []

datas_ele = []

for i in range(20, 25):
    f = f"/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/electrons/numpy_v5_mustache/validation/clusters_data_{i}.pkl"
    if not os.path.exists(f):
        print("file not found: ", f)
        continue
    d = pickle.load(open(f, "rb"))
    if include_seed:
        datas_ele.append(d[(d.is_calo_matched) ])
        # Seed not included
    else:
        datas_ele.append(d[(d.is_calo_matched) & (d.is_seed==False)])

    
data_ele = pd.concat(datas_ele, ignore_index=True)
data_ele["particle"] = "electron"
print("N events ele: ",len(data_ele))

datas_gamma = []
for i in range(20, 28):
    f = f"/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/gammas/numpy_v5_mustache/validation/clusters_data_{i}.pkl"
    if not os.path.exists(f):
        print("file not found: ", f)
        continue
    d = pickle.load(open(f, "rb"))
    if include_seed:
        datas_gamma.append(d[(d.is_calo_matched) ])
        # Seed not included
    else:
        datas_gamma.append(d[(d.is_calo_matched) & (d.is_seed==False)])
    
data_gamma = pd.concat(datas_gamma, ignore_index=True)
data_gamma["particle"] = "gamma"
print("N events gamma: ",len(data_gamma))

if data_ele.shape[0]> data_gamma.shape[0]:
    data_val = pd.concat([data_gamma, data_ele.iloc[0:len(data_gamma)]], ignore_index=True)
else:
    data_val = pd.concat([data_gamma.iloc[0:len(data_ele)], data_ele], ignore_index=True)


# for i in range(1, args.nfiles):
#     f = f"{args.inputdir}/electrons/numpy_v2/clusters_data_{i}.pkl"
#     if not os.path.exists(f):
#         print("file not found: ", f)
#         continue
#     d = pickle.load(open(f, "rb"))
#     #Seed included
#     if include_seed:
#         datas_val.append(d[(d.is_calo) ])
#         # Seed not included
#     else:
#         datas_val.append(d[(d.is_calo) & (d.is_seed==False)])
    
# data_ele = pd.concat(datas_val, ignore_index=True)
# data_ele["particle"] = "electron"

# datas_val = []
# for i in range(1, args.nfiles):
#     f = f"{args.inputdir}/gammas/numpy_v2/clusters_data_{i}.pkl"
#     if not os.path.exists(f):
#         print("file not found: ", f)
#         continue
#     d = pickle.load(open(f, "rb"))
#     #Seed included
#     if include_seed:
#         datas_val.append(d[(d.is_calo) ])
#         # Seed not included
#     else:
#         datas_val.append(d[(d.is_calo) & (d.is_seed==False)])
    
# data_gamma = pd.concat(datas_val, ignore_index=True)
# data_gamma["particle"] = "gamma"

# if data_ele.shape[0]> data_gamma.shape[0]:
#     data_val = pd.concat([data_gamma, data_ele.iloc[0:len(data_gamma)]], ignore_index=True)
# else:
    # data_val = pd.concat([data_gamma.iloc[0:len(data_ele)], data_ele], ignore_index=True)

cols_list= [
    ["seed_eta", "seed_phi", "seed_iz","cluster_deta", "cluster_dphi", "en_seed", "en_cluster"],
    ["seed_eta", "seed_phi", "seed_iz","cluster_deta", "cluster_dphi", "en_seed", "en_cluster", 
       "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi","cl_swissCross", "cl_nxtals"],
    ["seed_eta", "seed_phi", "seed_iz","en_seed","et_seed",
        "cluster_deta", "cluster_dphi", "en_cluster", "et_cluster",
       "seed_f5_r9d", "seed_f5_sigmaIetaIeta","seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi","seed_swissCross","seed_nxtals",
        "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi","cl_swissCross", "cl_nxtals"]
]
cols = cols_list[args.vars]

print(">>> Evaluation....")
data_val["y"] = model.predict(scaler.transform(data_val[cols].values), batch_size=2048)

data_out = data_val[data_val.in_scluster== False]
data_in = data_val[data_val.in_scluster == True]

##############
# AUC and ROC

if args.roc:
    print(">>> Computing AUC...")

    from sklearn.metrics import roc_auc_score, roc_curve
    y_test = np.array(data_val.in_scluster, dtype=int)
    pred = data_val["y"].values
    auc = roc_auc_score(y_test,pred)
    print("AUC score: " + str(auc))

    print(">>> Saving ROC curve...")
    fp , tp, th = roc_curve(y_test, pred)
    plt.plot(fp, tp, label="roc")
    plt.plot(fp, th, label="threshold")
    plt.xlabel("false positives")
    plt.ylabel("true positives")
    plt.ylim(-0.05,1.05)
    plt.legend()
    plt.savefig(f"{args.outputdir}/roc_curve.png")

##############
#Scores plots

# plt.hist(data_out["y"], bins=100, label="false", histtype="step")
# plt.hist(data_in["y"], bins=100, label="true", histtype="step")
# plt.yscale("log")
# plt.legend()
# plt.savefig(f"{args.outputdir}/scores.png")

# plt.hist(data_out["y"], bins=100, density=True, label="false", histtype="step")
# plt.hist(data_in["y"], bins=100,density=True, label="true", histtype="step")
# plt.yscale("log")
# plt.legend()
# plt.savefig(f"{args.outputdir}/scores_norm.png")

#######################################################################################


def plot_confusion(threshold, eta_bins, et_bins, y_label, palette, axlim=(0.7, 0.3), ):
    eta_min, eta_max = eta_bins
    et_min, et_max = et_bins
    data_out_0 = data_out[(data_out[y_label] < threshold) & (abs(data_out.seed_eta) > eta_min) & (abs(data_out.seed_eta) < eta_max) &
                        (data_out.en_seed / np.cosh(data_out.seed_eta)  > et_min) & (data_out.en_seed / np.cosh(data_out.seed_eta) < et_max) ]
    data_out_1 = data_out[(data_out[y_label] > threshold) & (abs(data_out.seed_eta) > eta_min) & (abs(data_out.seed_eta) < eta_max) &
                        (data_out.en_seed / np.cosh(data_out.seed_eta)  > et_min) & (data_out.en_seed / np.cosh(data_out.seed_eta) < et_max) ]
    data_in_0 = data_in[(data_in[y_label] < threshold) & (abs(data_in.seed_eta) > eta_min) & (abs(data_in.seed_eta) < eta_max) &
                        (data_in.en_seed / np.cosh(data_in.seed_eta)  > et_min) & (data_in.en_seed / np.cosh(data_in.seed_eta) < et_max) ]
    data_in_1 = data_in[(data_in[y_label] > threshold) & (abs(data_in.seed_eta) > eta_min) & (abs(data_in.seed_eta) < eta_max) &
                        (data_in.en_seed / np.cosh(data_in.seed_eta)  > et_min) & (data_in.en_seed / np.cosh(data_in.seed_eta) < et_max) ]
    nbins = 80
    
    fig = plt.figure(figsize=(7,8), dpi=100)

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2, sharey = ax1)  #Share y-axes with subplot 1
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4, sharey = ax3)  #Share y-axes with subplot 1
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    
    #size = max([ data_out_0.size / 80**2, data_out_1.size / 80**2,data_in_0.size / 80**2, data_in_1.size / 80**2])
    
    h, *_, h11 = ax4.hist2d(data_in_1.cluster_dphi, data_in_1.cluster_deta,   
                    bins=(nbins,nbins), range=((-axlim[0], axlim[0]),(-axlim[1], axlim[1])), cmap=palette, norm=colors.LogNorm())
    
    size = np.max(h)
    *_, h00= ax1.hist2d(data_out_0.cluster_dphi, data_out_0.cluster_deta,
                     bins=(nbins,nbins), range=((-axlim[0], axlim[0]),(-axlim[1], axlim[1])), vmax=size, cmap=palette, norm=colors.LogNorm())
    *_, h01 = ax2.hist2d(data_out_1.cluster_dphi, data_out_1.cluster_deta,  
                     bins=(nbins,nbins), range=((-axlim[0], axlim[0]),(-axlim[1], axlim[1])), vmax=size,cmap=palette, norm=colors.LogNorm())
    *_, h10 = ax3.hist2d(data_in_0.cluster_dphi, data_in_0.cluster_deta,  
                    bins=(nbins,nbins), range=((-axlim[0], axlim[0]),(-axlim[1], axlim[1])), vmax=size,cmap=palette, norm=colors.LogNorm())
    
    #fig.colorbar(h00, ax=ax[0][0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.delaxes(cax1)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h01, cax=cax2, label="N. clusters")
    
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.delaxes(cax3)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h11, cax=cax4, label="N. clusters")
    
    ax1.set_ylabel("$\Delta \eta$")
    ax1.set_xlabel("$\Delta \phi$")
    ax2.set_xlabel("$\Delta \phi$")
    #ax2.set_ylabel("Delta Eta")
    ax3.set_ylabel("$\Delta \eta$")
    ax3.set_xlabel("$\Delta \phi$")
    ax4.set_xlabel("$\Delta \phi$")
    #ax4.set_ylabel("Delta Eta")

    ax1.set_xlim(-axlim[0], axlim[0])
    ax2.set_xlim(-axlim[0], axlim[0])
    ax3.set_xlim(-axlim[0], axlim[0])
    ax4.set_xlim(-axlim[0], axlim[0])
    ax1.set_ylim(-axlim[1], axlim[1])
    ax2.set_ylim(-axlim[1], axlim[1])
    ax3.set_ylim(-axlim[1], axlim[1])
    ax4.set_ylim(-axlim[1], axlim[1])
    
    plt.subplots_adjust(wspace = -.015, hspace=0.25)
    #plt.tight_layout()
    fig.text(0.5, 0.9, "Background", ha="center", va="center", fontsize="large")
    fig.text(0.5, 0.48, "Signal", ha="center", va="center",fontsize="large")
    fig.text(0.13, 0.89, f"Score < {threshold}", va="center")
    fig.text(0.13, 0.47, f"Score < {threshold}",va="center")
    fig.text(0.73, 0.89, f"Score > {threshold}", va="center")
    fig.text(0.73, 0.47, f"Score > {threshold}",va="center")
    
    fig.text(0.02, 0.93, f"${eta_min} < |\eta| < {eta_max}$, ${et_min} < E_{{T}}< {et_max}$", va="center", ha="left")
    fig.savefig(f"{args.outputdir}/confmatrix__thre_{threshold}_eta_{eta_min}_{eta_max}_et_{et_min}_{et_max}.png")
    plt.close(fig)


for tr in args.thresholds:
    print(f">>>> Threshold: {tr}")
    for ieta in range(len(args.eta)-1):
        for ien in range(len(args.en)-1):
            etamin = args.eta[ieta]
            etamax = args.eta[ieta+1]
            enmin = args.en[ien]
            enmax = args.en[ien+1]
            print(f">> Eta: {etamin} - {etamax} | Energy:  {enmin} - {enmax}")
            plot_confusion(tr, (etamin, etamax), (enmin, enmax), y_label="y", axlim=(args.dphi, args.deta), palette="viridis")
