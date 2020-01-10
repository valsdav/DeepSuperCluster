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
parser.add_argument("-m","--model", type=str, help="model file", required=True)
parser.add_argument("--eta", type=float, nargs="+", help="Eta bins", required=True)
parser.add_argument("--en", type=float, nargs="+", help="ET bins", required = True)
parser.add_argument("-t","--thresholds", type=float, nargs="+", help="Threshold", required = True)
parser.add_argument("-o","--outputdir", type=str, help="outputdir", required=True)
parser.add_argument("-r","--roc", action="store_true",  help="Compute ROC", default=False)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)

args = parser.parse_args()

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)


scaler = pickle.load(open("../models/scaler.pkl", "rb"))

model = load_model(args.model)


datas_val = []

for i in range(60, 127):
    f = f"/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/numpy_v3/clusters_data_{i}.pkl"
    if not os.path.exists(f):
        print("file not found: ", f)
        continue
    d = pickle.load(open(f, "rb"))
    datas_val.append(d[(d.is_calo) & (d.is_seed == False)])
        
data_val  = pd.concat(datas_val, ignore_index=True)

cols = ["seed_eta", "seed_phi", "seed_iz","cluster_deta", "cluster_dphi", "en_seed", "en_cluster"]

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

plt.hist(data_out["y"], bins=100, label="false", histtype="step")
plt.hist(data_in["y"], bins=100, label="true", histtype="step")
plt.yscale("log")
plt.legend()
plt.savefig(f"{args.outputdir}/scores.png")

plt.hist(data_out["y"], bins=100, density=True, label="false", histtype="step")
plt.hist(data_in["y"], bins=100,density=True, label="true", histtype="step")
plt.yscale("log")
plt.legend()
plt.savefig(f"{args.outputdir}/scores_norm.png")

def plot_confusion(threshold, eta_bins, et_bins, axlim=(0.6, 0.2)):
    eta_min, eta_max = eta_bins
    et_min, et_max = et_bins
    data_out_0 = data_out[(data_out.y < threshold) & (abs(data_out.seed_eta) > eta_min) & (abs(data_out.seed_eta) < eta_max) &
                        (data_out.en_seed / np.cosh(data_out.seed_eta)  > et_min) & (data_out.en_seed / np.cosh(data_out.seed_eta) < et_max) ]
    data_out_1 = data_out[(data_out.y > threshold) & (abs(data_out.seed_eta) > eta_min) & (abs(data_out.seed_eta) < eta_max) &
                        (data_out.en_seed / np.cosh(data_out.seed_eta)  > et_min) & (data_out.en_seed / np.cosh(data_out.seed_eta) < et_max) ]
    data_in_0 = data_in[(data_in.y < threshold) & (abs(data_in.seed_eta) > eta_min) & (abs(data_in.seed_eta) < eta_max) &
                        (data_in.en_seed / np.cosh(data_in.seed_eta)  > et_min) & (data_in.en_seed / np.cosh(data_in.seed_eta) < et_max) ]
    data_in_1 = data_in[(data_in.y > threshold) & (abs(data_in.seed_eta) > eta_min) & (abs(data_in.seed_eta) < eta_max) &
                        (data_in.en_seed / np.cosh(data_in.seed_eta)  > et_min) & (data_in.en_seed / np.cosh(data_in.seed_eta) < et_max) ]
    nbins = 80
    
    fig = plt.figure(figsize=(7,8), dpi=200)

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
                    bins=(nbins,nbins), range=((-0.6,0.6),(-0.2,0.2)), cmap="plasma", norm=colors.LogNorm())
    
    size = np.max(h)
    *_, h00= ax1.hist2d(data_out_0.cluster_dphi, data_out_0.cluster_deta,
                     bins=(nbins,nbins), range=((-0.6,0.6),(-0.2,0.2)), vmax=size, cmap="plasma", norm=colors.LogNorm())
    *_, h01 = ax2.hist2d(data_out_1.cluster_dphi, data_out_1.cluster_deta,  
                     bins=(nbins,nbins), range=((-0.6,0.6),(-0.2,0.2)), vmax=size,cmap="plasma", norm=colors.LogNorm())
    *_, h10 = ax3.hist2d(data_in_0.cluster_dphi, data_in_0.cluster_deta,  
                    bins=(nbins,nbins), range=((-0.6,0.6),(-0.2,0.2)), vmax=size,cmap="plasma", norm=colors.LogNorm())
    
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
    
    fig.text(0.02, 0.93, f"${eta_min} < |\eta| < {eta_max}$,  ${et_min:.0f} < E_{{T}}< {et_max:.0f}$ GeV", va="center", ha="left")

    fig.savefig(f"{args.outputdir}/confmatrix__thre_{threshold}_eta_{eta_min}_{eta_max}_et_{et_min}_{et_max}.png")



for tr in args.thresholds:
    print(f">>>> Threshold: {tr}")
    for ieta in range(len(args.eta)-1):
        for ien in range(len(args.en)-1):
            etamin = args.eta[ieta]
            etamax = args.eta[ieta+1]
            enmin = args.en[ien]
            enmax = args.en[ien+1]
            print(f">> Eta: {etamin} - {etamax} | Energy:  {enmin} - {enmax}")
            plot_confusion(tr, (etamin, etamax), (enmin, enmax))