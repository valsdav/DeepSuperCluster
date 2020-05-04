import pickle
import numpy as np 
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
import argparse

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams["image.origin"] = 'lower'

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="Inputdir data", required=True)
parser.add_argument("--eta", type=float, nargs="+", help="Eta bins", required=True)
parser.add_argument("--en", type=float, nargs="+", help="ET bins", required = True)
parser.add_argument("--deta", type=float, help="DeltaEta radius", required=True)
parser.add_argument("--dphi", type=float, help="DeltaPhi radius", required = True)
parser.add_argument("-t","--thresholds", type=float, nargs="+", help="Threshold", required = True)
parser.add_argument("-o","--outputdir", type=str, help="outputdir", required=True)
parser.add_argument("-r","--roc", action="store_true",  help="Compute ROC", default=False)
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("-nf","--nfiles", type=int, help="N input files to read", default=False)

args = parser.parse_args()

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

include_seed = False
datas_val = []

for i in range(1, args.nfiles):
    f = f"{args.inputdir}/electrons/numpy_v2_mustache/clusters_data_{i}.pkl"
    if not os.path.exists(f):
        print("file not found: ", f)
        continue
    d = pickle.load(open(f, "rb"))
    #Seed included
    if include_seed:
        datas_val.append(d[(d.is_calo) ])
        # Seed not included
    else:
        datas_val.append(d[(d.is_calo) & (d.is_seed==False)])
    
data_ele = pd.concat(datas_val, ignore_index=True)
data_ele["particle"] = "electron"

datas_val = []
for i in range(1, args.nfiles):
    f = f"{args.inputdir}/gammas/numpy_v2_mustache/clusters_data_{i}.pkl"
    if not os.path.exists(f):
        print("file not found: ", f)
        continue
    d = pickle.load(open(f, "rb"))
    #Seed included
    if include_seed:
        datas_val.append(d[(d.is_calo) ])
        # Seed not included
    else:
        datas_val.append(d[(d.is_calo) & (d.is_seed==False)])
    
data_gamma = pd.concat(datas_val, ignore_index=True)
data_gamma["particle"] = "gamma"

if data_ele.shape[0]> data_gamma.shape[0]:
    data_val = pd.concat([data_gamma, data_ele.iloc[0:len(data_gamma)]], ignore_index=True)
else:
    data_val = pd.concat([data_gamma.iloc[0:len(data_ele)], data_ele], ignore_index=True)


print(">>> Evaluation....")
TP_must = data_val[(data_val.in_scluster==True) & ( data_val.in_mustache==True) ]
TN_must = data_val[(data_val.in_scluster==False) & ( data_val.in_mustache==False) ]
FP_must = data_val[(data_val.in_scluster==False) & ( data_val.in_mustache==True) ]
FN_must = data_val[(data_val.in_scluster==True) & ( data_val.in_mustache==False) ]
T_must = data_val[data_val.in_scluster==True]
F_must = data_val[data_val.in_scluster==False]

data_out = data_val[data_val.in_scluster== False]
data_in = data_val[data_val.in_scluster == True]

#######################################################################################


def plot_confusion_must( eta_bins, et_bins, palette, axlim=(0.7, 0.3), ):
    eta_min, eta_max = eta_bins
    et_min, et_max = et_bins
    data_out_0 = TN_must[(abs(TN_must.seed_eta) > eta_min) & (abs(TN_must.seed_eta) < eta_max) &
                        (TN_must.en_seed / np.cosh(TN_must.seed_eta)  > et_min) & (TN_must.en_seed / np.cosh(TN_must.seed_eta) < et_max) ]
    data_out_1 = FP_must[(abs(FP_must.seed_eta) > eta_min) & (abs(FP_must.seed_eta) < eta_max) &
                        (FP_must.en_seed / np.cosh(FP_must.seed_eta)  > et_min) & (FP_must.en_seed / np.cosh(FP_must.seed_eta) < et_max) ]
    data_in_0 =  FN_must[(abs(FN_must.seed_eta) > eta_min) & (abs(FN_must.seed_eta) < eta_max) &
                        (FN_must.en_seed / np.cosh(FN_must.seed_eta)  > et_min) & (FN_must.en_seed / np.cosh(FN_must.seed_eta) < et_max) ]
    data_in_1 =  TP_must[(abs(TP_must.seed_eta) > eta_min) & (abs(TP_must.seed_eta) < eta_max) &
                        (TP_must.en_seed / np.cosh(TP_must.seed_eta)  > et_min) & (TP_must.en_seed / np.cosh(TP_must.seed_eta) < et_max) ]
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
            plot_confusion((etamin, etamax), (enmin, enmax), axlim=(args.dphi, args.deta))
