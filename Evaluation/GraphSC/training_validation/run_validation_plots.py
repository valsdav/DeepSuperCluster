import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
mpl.rcParams["image.origin"] = 'lower'
mpl.rcParams["figure.dpi"] = '150'
import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc

import mplhep as hep
plt.style.use(hep.style.CMS)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputdir", type=str, help="Input dir")
args = parser.parse_args()


outdir =f"{args.inputdir}/validation_plots/"
os.makedirs(outdir, exist_ok = True)
os.makedirs(outdir + "/dataset_check/", exist_ok = True)
os.makedirs(outdir + "/clusters_check/", exist_ok = True)
os.makedirs(outdir + "/recall_purity/", exist_ok = True)
os.makedirs(outdir + "/window_id/", exist_ok = True)
os.makedirs(outdir + "/debug/", exist_ok = True)
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}")
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}/dataset_check")
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}/clusters_check")
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}/window_id")
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}/recall_purity")
os.system(f"cp /eos/user/d/dvalsecc/www/index.php {outdir}/debug")

print("Loading dataset")
df_ele = pd.read_csv(f"{args.inputdir}/validation_data/validation_dataset_ele.csv", sep=";")
df_gamma = pd.read_csv(f"{args.inputdir}/validation_data/validation_dataset_gamma.csv", sep=";")

df_tot = pd.concat([df_ele, df_gamma], sort=False)

# Add variables
for df in [df_ele, df_gamma, df_tot]:
    df['iz'] = df['seed_iz']
    df['ieta'] = df['seed_ieta']
    df['iphi'] = df['seed_iphi']
    df['eta'] = df['seed_eta'].abs()
    df['phi'] = df['seed_phi']
    df['en'] = df['en_seed']
    df['et'] = df['et_seed']
    df["ncls"] = df["ncls_tot"]

def bin_analysis(col):
    def f(df):
        m = df[col].mean()
        A = (df[col].quantile(0.84) - df[col].quantile(0.16))/2
        B = (df[col].quantile(0.025) - df[col].quantile(0.975))/2
        return pd.Series({
            "m": m,
            "w68": A,
            "w95": B,
            "N": df[col].count()
        })
    return f

def bin_analysis_central(col):
    def f(df):
        m = df[col].mean()
        qM = np.sum(df[col]< m) / df[col].count()
        print(qM)
        if (qM> 0.66):
            A = (df[col].quantile(1) - df[col].quantile(qM-0.34))/2
        elif (qM < 0.34):
            A = (df[col].quantile(qM +0.34) - df[col].quantile(0))/2
        else:
            A = (df[col].quantile(qM +0.34) - df[col].quantile(qM-0.34))/2
        return pd.Series({
            "m": m,
            "qM": qM,
            "w68": A,
            "N": df[col].count()
        })
    return f

def get_CI(histo, CL):
    Y, X = histo
    maxBin = np.argmax(Y)
    Xmax = (X[maxBin] + X[maxBin+1])/2
    # now we have to compute the simmetric interval
    Xr, Xl = 0, 0
    tot = np.sum(Y)
    thr = tot* CL/2
#     print(tot, thr)
    xi = maxBin+1
    Nu = Y[maxBin]/2
    Nd = Nu
    while(xi < len(X)):
        Nu += Y[xi]
        if Nu > thr:
            Xr = (X[xi] + X[xi+1])/2
            break
        else:
            xi+=1
            
    xi = maxBin-1
    while(xi > 0):
        Nd += Y[xi]
        if Nd > thr:
            Xl = (X[xi] + X[xi+1])/2
            break
        else:
            xi-=1
    
    return Xmax, Xr, Xl
    
def get_sigma_eff(df):
    A = (df.quantile(0.84) - df.quantile(0.16)) /2
    B = (df.quantile(0.025) - df.quantile(0.975)) /2
    return A,B

    
def get_quantiles(df):
    return df.quantile(0.025), df.quantile(0.16), df.quantile(0.5), df.quantile(0.84), df.quantile(0.975)



    
def get_roc(df, true_fl, out):
    w_pos = df[df.flavour == true_fl][out].values
    w_neg = df[df.flavour != true_fl][out].values
    tot_pos = w_pos.shape[0]
    tot_neg = w_neg.shape[0]
    fpr, tpr, thresholds = [] , [], [] 
    for i in range(50):
        t = i/30.
        tp = np.sum(w_pos>t) / float(tot_pos)
        fp = np.sum(w_neg>t) / float(tot_neg)
        fpr.append(fp)
        tpr.append(tp)
        thresholds.append(i)
    return fpr, tpr
    

#####################################
####  Seed checks
print("seed checks")

fig, (ax,ay) = plt.subplots(1,2, figsize=(17,8), dpi=200)

A =ax.hist2d(df_ele.eta,df_ele.ncls,  bins=(30,39), range=[(0,3),(1, 40)],   cmap="plasma" , norm=colors.LogNorm())
B = ay.hist2d(df_gamma.eta,df_gamma.ncls,  bins=(30,39), range=[(0,3),(1, 40)], cmap="plasma", norm=colors.LogNorm())
ax.text(0.6, 0.9, "Electron", transform=ax.transAxes)
ay.text(0.6, 0.9, "Photon", transform=ay.transAxes)


ax.set_xlabel("Seed $|\eta|$")
ay.set_xlabel("Seed $|\eta|$")
ax.set_ylabel("N. clusters Tot.")
ay.set_ylabel("N. clusters Tot.")
plt.colorbar(A[3],ax=ax, )
plt.colorbar(B[3],ax=ay, )

fig.savefig(outdir + "/dataset_check/ncluster_seed_eta_check.pdf")
fig.savefig(outdir + "/dataset_check/ncluster_seed_eta_check.png")

fig, (ax,ay) = plt.subplots(1,2, figsize=(17,8), dpi=200)

A =ax.hist2d(df_ele.eta,df_ele.ncls_true,  bins=(30,19), range=[(0,3),(1, 20)],   cmap="plasma" , norm=colors.LogNorm())
B = ay.hist2d(df_gamma.eta,df_gamma.ncls_true,  bins=(30,19), range=[(0,3),(1, 20)], cmap="plasma",  norm=colors.LogNorm() )
ax.text(0.6, 0.9, "Electron", transform=ax.transAxes)
ay.text(0.6, 0.9, "Photon", transform=ay.transAxes)

ax.set_xlabel("Seed $|\eta|$")
ay.set_xlabel("Seed $|\eta|$")
ax.set_ylabel("N. clusters True")
ay.set_ylabel("N. clusters True")
plt.colorbar(A[3],ax=ax, )
plt.colorbar(B[3],ax=ay, )

fig.savefig(outdir + "/dataset_check/ncluster_true_seed_eta_check.pdf")
fig.savefig(outdir + "/dataset_check/ncluster_true_seed_eta_check.png")

fig, (ax,ay) = plt.subplots(1,2, figsize=(17,8), dpi=200)
ncls = [2,3,4,5]
for ncl in ncls:
    ax.hist(df_ele[df_ele.ncls_true ==ncl].en_seed / df_ele[df_ele.ncls_true ==ncl].En_true, bins=40,
            range=(0.4,1), histtype="step", label="N. clusters = {}".format(ncl), linewidth=2, density=True)
    ay.hist(df_gamma[df_gamma.ncls_true ==ncl].en_seed / df_gamma[df_gamma.ncls_true ==ncl].En_true, bins=40, 
            range=(0.4,1), histtype="step", label="N. clusters = {}".format(ncl), linewidth=2, density=True)
ax.legend(loc="upper left")
ay.legend(loc="upper left")
ax.set_ylim(1,20)
ay.set_ylim(1,20)
# ax.set_yscale("log")
ax.text(0.7,0.85,"Electron", transform = ax.transAxes)  
ay.text(0.7,0.85,"Photon", transform = ay.transAxes)  
ax.set_xlabel("Seed E / total true E ")
ay.set_xlabel("Seed E / total true E")
hep.cms.label(rlabel="14 TeV",loc=0, ax=ax)
hep.cms.label(rlabel="14 TeV",loc=0, ax=ay)

fig.savefig(outdir + "/dataset_check/seed_energy_fraction_check.pdf")
fig.savefig(outdir + "/dataset_check/seed_energy_fraction_check.png")


fig, ax = plt.subplots(1,1, figsize=(8,8),dpi=150)
ax.hist(df_ele.et_seed, bins=100, range=(0,100), histtype="step", label="Electron", linewidth=2, density=True)
ax.hist(df_gamma.et_seed, bins=100, range=(0,100), histtype="step", label="Photon", linewidth=2, density=True)
ax.legend()
ax.set_ylim(0, 0.03)
ax.set_xlabel("Seed $E_T$")
hep.cms.text("Preliminary ",loc=0, ax=ax)

fig.savefig(outdir+ "/dataset_check/seed_et_elegamma_comparison.pdf")
fig.savefig(outdir+ "/dataset_check/seed_et_elegamma_comparison.png")


fig, ax = plt.subplots(1,1, figsize=(8,8),dpi=150)
ax.hist(df_ele.eta, bins=50, range=(0,3), histtype="step", label="Electron", linewidth=2, density=True)
ax.hist(df_gamma.eta, bins=50, range=(0,3), histtype="step", label="Photon", linewidth=2, density=True)
ax.legend()
ax.set_xlabel("Seed $|\eta|$")
hep.cms.text("Preliminary ",loc=0, ax=ax)
ax.set_ylim(0, 0.6)

fig.savefig(outdir+ "/dataset_check/seed_eta_elegamma_comparison.pdf")
fig.savefig(outdir+ "/dataset_check/seed_eta_elegamma_comparison.png")



fig, (ax,ay) = plt.subplots(1,2, figsize=(17,8),dpi=100)
ncls = [2,3,4,5]
for ncl in ncls:
    ax.hist(df_ele[df_ele.ncls_true ==ncl].en_seed / df_ele[df_ele.ncls_true ==ncl].En_true_sim_good, bins=50, range=(0.6,1), histtype="step", label="Electron Ncl {}".format(ncl), linewidth=2, density=True)
    ay.hist(df_gamma[df_gamma.ncls_true ==ncl].en_seed / df_gamma[df_gamma.ncls_true ==ncl].En_true_sim_good, bins=50, range=(0.6,1), histtype="step", label="Photon Ncl {}".format(ncl), linewidth=2, density=True)
ax.legend(loc="upper left")
ay.legend(loc="upper left")
ax.set_ylim(1,10)
ay.set_ylim(1,10)
# ax.set_yscale("log")
ax.set_xlabel("Seed En. fraction")
ay.set_xlabel("Seed En. fraction")
hep.cms.text("Preliminary ",loc=0, ax=ax)
hep.cms.text("Preliminary ",loc=0, ax=ay)
# ay.text(0.1,0.1, "Ncl true{}".format(ncl), transform=ax.transAxes)

fig.savefig(outdir + "/dataset_check/seed_en_fraction_elegamma.pdf")
fig.savefig(outdir + "/dataset_check/seed_en_fraction_elegamma.png")


#######################
#number of clusters
print("number of clusters plots")

etas = [0, 0.7, 1.3, 2, 3]

fig, ax = plt.subplots(1,len(etas)-1, figsize=(35,8),dpi=150)
df = df_ele
for i in range(len(etas)-1):
    mask = (df.eta>etas[i]) & (df.eta< etas[i+1])
    A = ax[i].hist2d(df[mask].ncls_sel_must, df[mask].ncls_sel, range=((1, 15), (1,15)), bins=(14,14), density=True, cmin=1e-4)
    hep.cms.label(rlabel="14 TeV", loc=0, ax=ax[i])
    ax[i].text(0.1, 0.8, "$\eta$ {}-{}".format(etas[i], etas[i+1]), transform=ax[i].transAxes)
    
    fig.colorbar(A[3] , label="N. cls", ax=ax[i], shrink=0.9)

    ax[i].set_xlabel("Mustache N. clusters")
    ax[i].set_ylabel("DeepSC N. clusters")
    
fig.savefig(outdir + "/clusters_check/ncluster_distribution_2d.pdf")
fig.savefig(outdir + "/clusters_check/ncluster_distribution_2d.png")



for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):

    fig, ax = plt.subplots(2,4, figsize=(35,18),dpi=150)

    xrange = (0,25)
    bins= 25

    ets = [ [(0,2),(2,4),(4,7),(7,10)],[(10,20),(20,50),(50,70),(70,100)]]

    for j,etss in enumerate(ets): 
        for i, (e1,e2) in enumerate(etss): 
            dfx = df[(df.et>=e1) & (df.et<e2)]

            A = ax[j,i].hist(dfx.ncls_tot,bins=bins, range=xrange, histtype='step', linewidth=2, label='All clusters')
            ax[j,i].hist(dfx.ncls_sel,bins=bins, range=xrange, histtype='step', linewidth=2, label='DeepSC')
            ax[j,i].hist(dfx.ncls_sel_must,bins=bins, range=xrange, histtype='step', linewidth=2, label='Mustache')
            ax[j,i].hist(dfx.ncls_true,bins=bins, range=xrange, histtype='step', linewidth=2, linestyle='dashed', label='True cluster')
            maxY = np.max(A[0])
            ax[j,i].legend(loc="upper left")
            ax[j,i].set_yscale('log')
            ax[j,i].set_ylim(1, 5e4 * maxY)
            ax[j,i].set_xlabel("N. Clusters")
            ax[j,i].text(0.55,0.9,  str(e1) + ' < $E_T^{seed}$< ' + str(e2), transform = ax[j,i].transAxes)  
            ax[j,i].text(0.7, 0.75, flavour, transform=ax[j,i].transAxes)

            hep.cms.text("Preliminary ",loc=0, ax=ax[j,i])

            
    fig.savefig(outdir + f"/clusters_check/ncluster_distribution_log_{flavour}.pdf")
    fig.savefig(outdir + f"/clusters_check/ncluster_distribution_log_{flavour}.png")



for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):

    fig, ax = plt.subplots(2,4, figsize=(35,18),dpi=150)

    xrange = (0,20)
    bins= 20

    ets = [ [(0,2),(2,4),(4,7),(7,10)],[(10,20),(20,50),(50,70),(70,100)]]

    for j,etss in enumerate(ets): 
        for i, (e1,e2) in enumerate(etss): 
            dfx = df[(df.et>=e1) & (df.et<e2)]

            ax[j,i].hist(dfx.ncls_tot,bins=bins, range=xrange, histtype='step', linewidth=2, label='All clusters')
            ax[j,i].hist(dfx.ncls_sel,bins=bins, range=xrange, histtype='step', linewidth=2, label='DeepSC')
            ax[j,i].hist(dfx.ncls_sel_must,bins=bins, range=xrange, histtype='step', linewidth=2, label='Mustache')
            A = ax[j,i].hist(dfx.ncls_true,bins=bins, range=xrange, histtype='step', linewidth=2, linestyle='dashed', label='True cluster')
            maxY = np.max(A[0])
            ax[j,i].legend(loc="upper left")
#             ax[j,i].set_yscale('log')
            ax[j,i].set_ylim(1, 1.7 * maxY)
            ax[j,i].set_xlabel("N. Clusters")
            ax[j,i].text(0.55,0.9,  str(e1) + ' < $E_T^{seed}$< ' + str(e2), transform = ax[j,i].transAxes)  
            ax[j,i].text(0.7, 0.75, flavour, transform=ax[j,i].transAxes)

            hep.cms.text("Preliminary ",loc=0, ax=ax[j,i])

    fig.savefig(outdir + "/clusters_check/nclustur_distribution_"+flavour + ".pdf")
    fig.savefig(outdir + "/clusters_check/nclustur_distribution_"+flavour + ".png")

#################3
# distribution of resolution
print("Resolution plots")

for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):

    fig, ax = plt.subplots(2,4, figsize=(35,20),dpi=150)

    xrange = (0.4, 1.2)
    bins= 1000
    CL = 0.4


    r = [2, 4, 8, 10, 15, 30 ,40, 60,100]
    s = [[], []]
    for i in range(len(r)-1):
        line = i //4 
        s[line].append((r[i],r[i+1]))

    for j,ss in enumerate(s): 
        for i, (e1,e2) in enumerate(ss): 
            dfx = df[(abs(df.et_seed)>=e1) & (abs(df.et_seed)<e2)]

            width, width_2 = get_sigma_eff(dfx.En_ovEtrue_sim_good)
            width_must, width_2_must = get_sigma_eff(dfx.En_ovEtrue_sim_good_mustache)
    #         rms_regr = dfx.En_ovEtrue_gen_corr.std()

            ax[j,i].hist(dfx.En_ovEtrue_sim_good_mustache,bins=bins, range=xrange,histtype='step', linewidth=2, label='Mustache {:.3f} $\sigma$ 68%'.format(width_must))
            A= ax[j,i].hist(dfx.En_ovEtrue_sim_good,bins=bins, range=xrange,histtype='step', linewidth=2, label='DeepSC   {:.3f} $\sigma$ 68%'.format(width))
    #         ax[j,i].hist(dfx.En_ovEtrue_gen_corr,bins=bins, range=xrange,histtype='step', linewidth=2, label='DeepSC +regre    RMS: {:.4f}'.format(rms_regr))
            maxY = np.max(A[0])
            ax[j,i].legend(loc="upper left")
    #         ax[j,i].set_yscale('log')
            ax[j,i].set_ylim(1, 1.5 * maxY)
            ax[j,i].set_xlabel("En/En Sim")
            ax[j,i].text(0.07,0.7,  str(e1) + ' < Seed $E_T$ < ' + str(e2), transform = ax[j,i].transAxes)  
            ax[j,i].text(0.7, 0.7, flavour, transform=ax[j,i].transAxes)

            hep.cms.text("Preliminary ",loc=0, ax=ax[j,i])

    fig.savefig(outdir + "/EovEtrue_"+flavour + ".pdf")
    fig.savefig(outdir + "/EovEtrue_"+flavour + ".png")


fig, (ax, ay) = plt.subplots(1,2, figsize=(18,8),dpi=100, )

df = df_gamma
flavour = "Photon"
ets = [2, 4,6,8, 10,20,40,50,60,70,90,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["eta_bin"] = pd.cut(df.eta, etas, labels=list(range(len(etas)-1)))


res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(etas[:-1]):
    ax.plot(ets[:-1], res[res.eta_bin == ieta].w68/res_must[res_must.eta_bin == ieta].w68, label="Eta:{}-{}".format(etas[ieta], etas[ieta+1]), linewidth=2)
ax.set_ylim(0.6, 1.1)
ax.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ax.legend(ncol=2)
ax.set_xlabel("$E_T$ seed")
ax.plot([0,100],[1,1], linestyle="dashed", color="black")

ax.text(0.75, 0.9, flavour, transform=ax.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ax)


df = df_ele
flavour = "Electron"
df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["eta_bin"] = pd.cut(df.eta, etas, labels=list(range(len(etas)-1)))

res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(etas[:-1]):
    ay.plot(ets[:-1], res[res.eta_bin == ieta].w68/res_must[res_must.eta_bin == ieta].w68, label="Eta:{}-{}".format(etas[ieta], etas[ieta+1]), linewidth=2)
ay.set_ylim(0.6, 1.1)
ay.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ay.legend(ncol=2)
ay.set_xlabel("$E_T$ seed")
ay.plot([0,100],[1,1], linestyle="dashed", color="black")

ay.text(0.75, 0.9, flavour, transform=ay.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ay)

plt.savefig(outdir +"/resol_ratio_by_eta.png")
plt.savefig(outdir +"/resol_ratio_by_eta.pdf")


fig, (ax, ay) = plt.subplots(1,2, figsize=(18,8),dpi=100, )

df = df_gamma
flavour ="Photon"
ets = [2,4,6,10,15,20,30,40,50,60,70,80,90,100]
ncls = [0,1,2,3,4,5,6,7,10,15]
df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))


res = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(ncls[:-1]):
    ax.plot(ets[:-1], res[res.ncls_bin == ieta].w68/res_must[res_must.ncls_bin == ieta].w68, label="Ncls:{}-{}".format(ncls[ieta], ncls[ieta+1]), linewidth=2)
ax.set_ylim(0.6, 1.1)
ax.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ax.legend(ncol=2)
ax.set_xlabel("$E_T$ seed")
ax.plot([0,100],[1,1], linestyle="dashed", color="black")

ax.text(0.75, 0.9, flavour, transform=ax.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ax)

df = df_ele
flavour = "Electron"

ets = [2,4,6,10,15,20,30,40,50,60,70,80,90,100]
ncls = [0,1,2,3,4,5,6,7,10,15]
df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))


res = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(ncls[:-1]):
    ay.plot(ets[:-1], res[res.ncls_bin == ieta].w68/res_must[res_must.ncls_bin == ieta].w68, label="Ncls:{}-{}".format(ncls[ieta], ncls[ieta+1]), linewidth=2)
ay.set_ylim(0.6, 1.1)
ay.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ay.legend(ncol=2)
ay.set_xlabel("$E_T$ seed")
ay.plot([0,100],[1,1], linestyle="dashed", color="black")

ay.text(0.75, 0.9, flavour, transform=ay.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ay)

plt.savefig(outdir +"/resol_ratio_by_ncls_tot.png")
plt.savefig(outdir +"/resol_ratio_by_ncls_tot.pdf")


fig, (ax, ay) = plt.subplots(1,2, figsize=(18,8),dpi=100, )

df = df_gamma
flavour = "Photon"
ets = [2, 4,6, 8, 10,20,40,60,80,100]
etas = [0, 0.3,0.7,1.0, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["eta_bin"] = pd.cut(df.eta, etas, labels=list(range(len(etas)-1)))


res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(ets[:-1]):
    ax.plot(etas[:-1], res[res.et_bin == ieta].w68/res_must[res_must.et_bin == ieta].w68, label="$E_T$:{}-{}".format(ets[ieta], ets[ieta+1]), linewidth=2)
ax.set_ylim(0.6, 1.1)
ax.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ax.legend(ncol=2, fontsize="xx-small")
ax.set_xlabel("$\eta$ seed")
ax.plot([0,3],[1,1], linestyle="dashed", color="black")

ax.text(0.75, 0.9, flavour, transform=ax.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ax)


df = df_ele
flavour = "Electron"

df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
df["eta_bin"] = pd.cut(df.eta, etas, labels=list(range(len(etas)-1)))


res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res.reset_index(level=0, inplace=True)
res.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)
res_must.reset_index(level=0, inplace=True)

for ieta, eta in enumerate(ets[:-1]):
    ay.plot(etas[:-1], res[res.et_bin == ieta].w68/res_must[res_must.et_bin == ieta].w68, label="$E_T$:{}-{}".format(ets[ieta], ets[ieta+1]), linewidth=2)
ay.set_ylim(0.6, 1.1)
ay.set_ylabel("DeepSC / Mustache $\sigma_{eff}$ SIM")
ay.legend(ncol=2, loc="lower right", fontsize="xx-small")
ay.set_xlabel("$\eta$ seed")
ay.plot([0,3],[1,1], linestyle="dashed", color="black")

ay.text(0.75, 0.9, flavour, transform=ay.transAxes)

hep.cms.text("Preliminary", loc=0, ax=ay)

################
print("window ID")

fig, ax = plt.subplots(1,3, figsize=(24,8),dpi=100)

for i, v in enumerate(['w_nomatch','w_ele','w_gamma']):
#     ax[i].hist(df_nomatch[v], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="No matched")
    ax[i].hist(df_ele[v], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Ele")
    ax[i].hist(df_gamma[v], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Gamma")
    ax[i].set_xlabel(v)
    ax[i].legend()
    
plt.savefig(outdir +"/window_id/windows_id_incl.png")
plt.savefig(outdir +"/window_id/windows_id_incl.png")


fig, ax = plt.subplots(1,6, figsize=(48,8),dpi=100)

ets = [1,5,10,20,30,50,100]

for iet in range(len(ets)-1):
#     ax[i].hist(df_nomatch[v], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="No matched")
    ax[iet].hist(df_ele[(df_ele.et > ets[iet] ) & (df_ele.et_seed < ets[iet+1])]["w_gamma"], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Ele")
    ax[iet].hist(df_gamma[(df_gamma.et > ets[iet] ) & (df_gamma.et_seed < ets[iet+1])]["w_gamma"], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Gamma")
    ax[iet].set_xlabel("Ele/gamma discriminator")
    ax[iet].text(0.5,0.65, "Seed $E_T$ {}-{}".format(ets[iet],ets[iet+1]), transform=ax[iet].transAxes)
    ax[iet].legend()
    
plt.savefig(outdir +"/window_id/windows_id_byet.png")
plt.savefig(outdir +"/window_id/windows_id_byet.pdf")


fig, ax = plt.subplots(1,6, figsize=(48,8),dpi=100)

ncls = [1,2,3,5,7,10,15]

for iet in range(len(ncls)-1):
#     ax[i].hist(df_nomatch[v], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="No matched")
    ax[iet].hist(df_ele[(df_ele.ncls >= ncls[iet] ) & (df_ele.ncls < ncls[iet+1])]["w_gamma"], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Ele")
    ax[iet].hist(df_gamma[(df_gamma.ncls >= ncls[iet] ) & (df_gamma.ncls < ncls[iet+1])]["w_gamma"], bins=50, range=(0,1), histtype="step", density=True, linewidth=3, label="Gamma")
    ax[iet].set_xlabel("Ele/gamma discriminator")
    ax[iet].text(0.5,0.65, "N. cls {}-{}".format(ncls[iet],ncls[iet+1]), transform=ax[iet].transAxes)
    ax[iet].legend()
    
plt.savefig(outdir +"/window_id/windows_id_bycls.png")
plt.savefig(outdir +"/window_id/windows_id_bycls.pdf")


fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100, )
ets = [2,4,6,8,10,20,50,100]

for iet in range(len(ets)-1):
    fpr_nomat, tpr_nomat = get_roc(df_tot[(df_tot.et_seed >= ets[iet]) &(df_tot.et_seed < ets[iet+1])], 22, "w_gamma")
    plt.plot(fpr_nomat, tpr_nomat, label="AUC: {:.2f} $E_T$:{}-{}".format(auc(fpr_nomat, tpr_nomat), ets[iet], ets[iet+1]), linewidth=3)
   
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
hep.cms.text("Preliminary ",loc=0)
plt.legend()
plt.tight_layout()
plt.savefig(outdir +"/window_id/window_roc_ET.pdf")
plt.savefig(outdir +"/window_id/window_roc_ET.png")


ets = [2,4,6,8,10,20,50,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
aucs = np.zeros((len(ets)-1, len(etas)-1))

for iet in range(len(ets)-1):
    for ieta in range(len(etas)-1):
        fpr_nomat, tpr_nomat = get_roc(df_tot[(df_tot.et_seed >= ets[iet]) &(df_tot.et_seed < ets[iet+1]) & (df_tot.seed_eta.abs() >= etas[ieta]) &(df_tot.seed_eta.abs() < etas[ieta+1])], 22, "w_gamma")
        aucs[iet,ieta] = auc(fpr_nomat, tpr_nomat)
        
fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100, )
I = ax.imshow(aucs, aspect="auto", vmin=0.5, vmax=0.8)
hep.cms.text("Preliminary ",loc=0, ax=ax)

ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)

fig.colorbar(I, label="Ele vs Gamma AUC")

ax.set_ylabel("Seed $E_T$")
ax.set_xlabel("Seed $\eta$")

fig.savefig(outdir+ "/window_id/roc_seed_et_eta.png")
fig.savefig(outdir+ "/window_id/roc_seed_et_eta.pdf")



ets = [2,4,6,8,10,20,50,100]
ncls = [1,2,3,4,5,6,7,8,9,10,15,20]
aucs = np.zeros((len(ets)-1, len(ncls)-1))

for iet in range(len(ets)-1):
    for ncl in range(len(ncls)-1):
        fpr_nomat, tpr_nomat = get_roc(df_tot[(df_tot.et_seed >= ets[iet]) &(df_tot.et_seed < ets[iet+1]) & (df_tot.ncls >= ncls[ncl]) &(df_tot.ncls < ncls[ncl+1])], 22, "w_gamma")
        aucs[iet,ncl] = auc(fpr_nomat, tpr_nomat)
        
fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100, )
I = ax.imshow(aucs, aspect="auto", vmin=0.5, vmax=0.8)
hep.cms.text("Preliminary ",loc=0, ax=ax)

ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(ncls)

fig.colorbar(I, label="Ele vs Gamma AUC")

ax.set_ylabel("Seed $E_T$")
ax.set_xlabel("N. clusters total")

plt.savefig(outdir +"/window_id/roc_ncls_et.png")
plt.savefig(outdir +"/window_id/roc_ncls_et.pdf")

etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]

fig = plt.figure(dpi=72)
for iet in range(len(etas)-1):
    fpr_nomat, tpr_nomat = get_roc(df_tot[(abs(df_tot.seed_eta) >= etas[iet]) &(abs(df_tot.seed_eta) < etas[iet+1])], 22, "w_gamma")
    plt.plot(fpr_nomat, tpr_nomat, label="AUC: {:.2f} $\eta$:{}-{}".format(auc(fpr_nomat, tpr_nomat), etas[iet], etas[iet+1]), linewidth=3)
   
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend()

plt.savefig(outdir +"/window_id/roc_eta.pdf")
plt.savefig(outdir +"/window_id/roc_eta.png")

ncls = [1,2,3,4,5,6,8,10,15]
fig = plt.figure(dpi=72)
for iet in range(len(ncls)-1):
    fpr_nomat, tpr_nomat = get_roc(df_tot[(df_tot.ncls >= ncls[iet]) &( df_tot.ncls < ncls[iet+1])], 22, "w_gamma")
    plt.plot(fpr_nomat, tpr_nomat, label="AUC: {:.2f}, N. cls:{}".format(auc(fpr_nomat, tpr_nomat),ncls[iet]), linewidth=3)
   
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend()

hep.cms.text("Preliminary ",loc=0)
plt.legend()
plt.tight_layout()
plt.savefig(outdir +"/window_id/roc_ncls.pdf")
plt.savefig(outdir +"/window_id/roc_ncls.png")

################################3

print("Precision and recall")

ets = [0,5,10,15, 20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.479, 1.75,2.,2.3,2.6,3]
ncls = [1,2,3,5,10,12,15,17,20,25,30]
nvtx = [20,30,35,40,45,50,55,60,65,70,75,80,90,100,130]

for df in [df_ele, df_gamma]:
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))
    df["et_bin"] = pd.cut(df.et_seed, ets, labels=list(range(len(ets)-1)))
    df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))
    df["nvtx_bin"] = pd.cut(df.nVtx, nvtx, labels=list(range(len(nvtx)-1)))


for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    res_dsc =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.Et_sel_true/ gr.Et_sel).mean())\
                 .unstack(fill_value=0).stack()

    res_mst =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.Et_sel_must_true / gr.Et_sel_must).mean())\
                 .unstack(fill_value=0).stack()

    vmin=0.9

    fig, (ax, ab) = plt.subplots(1,2, figsize=(18,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    a1 = res_dsc.values.reshape((len(etas)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_mst.values.reshape((len(etas)-1,len(ncls)-1))
    a2[a2==0] = np.nan

    A = ax.imshow(a1.T, vmin=vmin, vmax=1)
    ax.set_xlabel("Seed $\eta$")
    ax.set_ylabel("N. clusters")
    fig.colorbar(A , label="Energy Purity", ax=ax, shrink=0.9)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(etas)

    B = ab.imshow(a2.T, vmin=vmin, vmax=1)
    ab.set_xlabel("Seed $\eta$")
    ab.set_ylabel("N. clusters")
    plt.colorbar(B , label="Energy Purity", ax=ab, shrink=0.9)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(etas)

    ax.text(0.75, 1.02, "DeepSC", transform=ax.transAxes, color='darkorange', fontsize="small", fontweight='roman')
    ab.text(0.7, 1.02, "Mustache", transform=ab.transAxes, color='blue', fontsize="small", fontweight='roman')
    ax.text(0.7, 0.9,  flavour, transform=ax.transAxes, fontsize="small", )
    ab.text(0.7, 0.9, flavour, transform=ab.transAxes, fontsize="small")

    ax.minorticks_off()
    ab.minorticks_off()

    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)


    plt.tight_layout()
    fig.savefig(outdir + "/recall_purity/energy_purity_{}.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/energy_purity_{}.png".format(flavour))


for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    res_dsc =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.Et_sel_true/ gr.Et_true).mean())\
                 .unstack(fill_value=0).stack()

    res_mst =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.Et_sel_must_true / gr.Et_true).mean())\
                 .unstack(fill_value=0).stack()

    vmin = min(min(res_dsc), min(res_mst))
    vmin =0.9

    fig, (ax, ab) = plt.subplots(1,2, figsize=(18,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    a1 = res_dsc.values.reshape((len(etas)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_mst.values.reshape((len(etas)-1,len(ncls)-1))
    a2[a2==0] = np.nan

    A = ax.imshow(a1.T, vmin=vmin, vmax=1)
    ax.set_xlabel("Seed $\eta$")
    ax.set_ylabel("N. clusters")
    fig.colorbar(A , label="Energy Recall", ax=ax, shrink=0.9)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(etas)

    B = ab.imshow(a2.T, vmin=vmin, vmax=1)
    ab.set_xlabel("Seed $\eta$")
    ab.set_ylabel("N. clusters")
    plt.colorbar(B , label="Energy Recall", ax=ab, shrink=0.9)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(etas)

    ax.text(0.75, 1.02, "DeepSC", transform=ax.transAxes, color='orange', fontsize="small", fontweight='roman')
    ab.text(0.7, 1.02, "Mustache", transform=ab.transAxes, color='blue', fontsize="small", fontweight='roman')
    ax.text(0.7, 0.9,  flavour, transform=ax.transAxes, fontsize="small", )
    ab.text(0.7, 0.9, flavour, transform=ab.transAxes, fontsize="small")

    ax.minorticks_off()
    ab.minorticks_off()

    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)

    plt.tight_layout()
    fig.savefig(outdir + "/recall_purity/energy_recall_{}.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/energy_recall_{}.png".format(flavour))



for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    
    ets = [0,5,10,15, 20,40,60,80,100]
    etas = [0, 0.4,0.8, 1.2,1.479, 1.75,2.,2.3,2.6,3]
    ncls = [1,2,3,5,10,12,15,17,20,25,30]
    nvtx = [20,30,35,40,45,50,55,60,65,70,75,80,90,100,130]

    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))
    df["et_bin"] = pd.cut(df.et_seed, ets, labels=list(range(len(ets)-1)))
    df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))
    df["nvtx_bin"] = pd.cut(df.nVtx, nvtx, labels=list(range(len(nvtx)-1)))


    res_dsc =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_true/ gr.ncls_sel).mean())\
                 .unstack(fill_value=0).stack()

    res_mst =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_must_true / gr.ncls_sel_must).mean())\
                 .unstack(fill_value=0).stack()

    vmin=0.5

    fig, (ax, ab) = plt.subplots(1,2, figsize=(18,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    a1 = res_dsc.values.reshape((len(etas)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_mst.values.reshape((len(etas)-1,len(ncls)-1))
    a2[a2==0] = np.nan

    A = ax.imshow(a1.T, vmin=vmin, vmax=1)
    ax.set_xlabel("Seed $\eta$")
    ax.set_ylabel("N. clusters")
    fig.colorbar(A , label="N. clusters Purity", ax=ax, shrink=0.9)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(etas)

    B = ab.imshow(a2.T, vmin=vmin, vmax=1)
    ab.set_xlabel("Seed $\eta$")
    ab.set_ylabel("N. clusters")
    plt.colorbar(B , label="N. clusters Purity", ax=ab, shrink=0.9)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(etas)

    ax.text(0.75, 1.02, "DeepSC", transform=ax.transAxes, color='darkorange', fontsize="small", fontweight='roman')
    ab.text(0.7, 1.02, "Mustache", transform=ab.transAxes, color='blue', fontsize="small", fontweight='roman')
    ax.text(0.7, 0.9,  flavour, transform=ax.transAxes, fontsize="small", )
    ab.text(0.7, 0.9, flavour, transform=ab.transAxes, fontsize="small")

    ax.minorticks_off()
    ab.minorticks_off()

    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)

    plt.tight_layout()
    fig.savefig(outdir + "/recall_purity/cluster_purity_{}.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/cluster_purity_{}.png".format(flavour))

for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    
    ets = [0,5,10,15, 20,40,60,80,100]
    etas = [0, 0.4,0.8, 1.2,1.479, 1.75,2.,2.3,2.6,3]
    ncls = [1,2,3,5,10,12,15,17,20,25,30]
    nvtx = [20,30,35,40,45,50,55,60,65,70,75,80,90,100,130]

    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))
    df["et_bin"] = pd.cut(df.et_seed, ets, labels=list(range(len(ets)-1)))
    df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))
    df["nvtx_bin"] = pd.cut(df.nVtx, nvtx, labels=list(range(len(nvtx)-1)))


    res_dsc =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_true/ gr.ncls_true).mean())\
                 .unstack(fill_value=0).stack()

    res_mst =  df.groupby(["eta_bin", "ncls_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_must_true / gr.ncls_true).mean())\
                 .unstack(fill_value=0).stack()

    vmin=0.8

    fig, (ax, ab) = plt.subplots(1,2, figsize=(18,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    a1 = res_dsc.values.reshape((len(etas)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_mst.values.reshape((len(etas)-1,len(ncls)-1))
    a2[a2==0] = np.nan

    A = ax.imshow(a1.T, vmin=vmin, vmax=1)
    ax.set_xlabel("Seed $\eta$")
    ax.set_ylabel("N. clusters")
    fig.colorbar(A , label="N. clusters Recall", ax=ax, shrink=0.9)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(etas)

    B = ab.imshow(a2.T, vmin=vmin, vmax=1)
    ab.set_xlabel("Seed $\eta$")
    ab.set_ylabel("N. clusters")
    plt.colorbar(B , label="N. clusters Recall", ax=ab, shrink=0.9)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(etas))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(etas)

    ax.text(0.75, 1.02, "DeepSC", transform=ax.transAxes, color='darkorange', fontsize="small", fontweight='roman')
    ab.text(0.7, 1.02, "Mustache", transform=ab.transAxes, color='blue', fontsize="small", fontweight='roman')
    ax.text(0.7, 0.9,  flavour, transform=ax.transAxes, fontsize="small", )
    ab.text(0.7, 0.9, flavour, transform=ab.transAxes, fontsize="small")

    ax.minorticks_off()
    ab.minorticks_off()

    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)
    
    plt.tight_layout()
    fig.savefig(outdir + "/recall_purity/cluster_recall_{}.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/cluster_recall_{}.png".format(flavour))



#######################
print("missed cluster checks")

ncls = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,10.5,15.5, 20.5]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    A  = ax.hist2d(dfb.ncls_sel - dfb.ncls_true, dfb.ncls, bins=(nclsx, ncls), cmap="plasma",cmin=1e-4,vmax=0.08, density=True )#
    fig.colorbar(A[3], ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("N. clusters TOT")
    ax.text(0.75, 0.05, flavour, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(outdir+ "/clusters_check/Ncltot_lost_{}.png".format(flavour))
    fig.savefig(outdir+ "/clusters_check/Ncltot_lost_{}.pdf".format(flavour))


    

ncls = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,10.5,15.5, 20.5]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    H,x,y  = np.histogram2d(dfb.ncls_sel - dfb.ncls_true, dfb.ncls, bins=(nclsx, ncls),)#
    
    Hnorm = H.T / np.sum(H.T, axis=-1).reshape(-1,1)
    Hnorm[Hnorm ==0] = np.nan
    A = plt.imshow(Hnorm, cmap="plasma",)
    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(nclsx))- 0.5, minor=False)
    ax.set_yticklabels(["{:.0f}".format(n+0.5) for n in ncls])
    ax.set_xticklabels(["{:.0f}".format(n+0.5) for n in nclsx])
    
    fig.colorbar(A, ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("N. clusters TOT")
    ax.text(0.75, 0.05, flavour, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(outdir+ "/clusters_check/Ncltot_norm_lost_{}.png".format(flavour))
    fig.savefig(outdir+ "/clusters_check/Ncltot_norm_lost_{}.pdf".format(flavour))


ncls = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    A  = ax.hist2d(dfb.ncls_sel - dfb.ncls_true, dfb.ncls_true, bins=(nclsx, ncls), cmap="plasma",cmin=1e-4,vmax=0.3, density=True )#
    fig.colorbar(A[3], ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("N. clusters True")
    ax.text(0.75, 0.9, flavour, transform=ax.transAxes)
    
    fig.tight_layout()
    fig.savefig(outdir+ "/clusters_check/Ncltrue_lost_{}.png".format(flavour))
    fig.savefig(outdir+ "/clusters_check/Ncltrue_lost_{}.pdf".format(flavour))


    
ncls = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    
    H,x,y  = np.histogram2d(dfb.ncls_sel - dfb.ncls_true, dfb.ncls_true, bins=(nclsx, ncls) )#
    Hnorm = H.T / np.sum(H.T, axis=-1).reshape(-1,1)
    Hnorm[Hnorm ==0] = np.nan
    A = plt.imshow(Hnorm, cmap="plasma",)
    fig.colorbar(A, ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("N. clusters True")
    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(nclsx))- 0.5, minor=False)
    ax.set_yticklabels(["{:.0f}".format(n+0.5) for n in ncls])
    ax.set_xticklabels(["{:.0f}".format(n+0.5) for n in nclsx])
    ax.minorticks_off()
    ab.minorticks_off()
    ax.text(0.75, 0.9, flavour, transform=ax.transAxes)
    
    fig.tight_layout()
    fig.savefig(outdir+ "/clusters_check/Ncltrue_norm_lost_{}.png".format(flavour))
    fig.savefig(outdir+ "/clusters_check/Ncltrue_norm_lost_{}.pdf".format(flavour))

    
etas =  [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    A  = ax.hist2d(dfb.ncls_sel - dfb.ncls_true, abs(dfb.seed_eta), bins=(nclsx, etas),cmap="plasma",cmin=1e-4,vmax=0.4, density=True )#
    fig.colorbar(A[3], ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("Seed $\eta$")
    ax.text(0.75, 0.05, flavour, transform=ax.transAxes)
    fig.savefig(outdir+ "/clusters_check/Ncltrue_norm_lost_{}_byeta.png".format(flavour))
    fig.savefig(outdir+ "/clusters_check/Ncltrue_norm_lost_{}_byeta.pdf".format(flavour))
    

ets = [0,5,10,15, 20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.479, 1.75,2.,2.5,3]
ncls = [1,2,3,5,6,8,10,15]

hs = {}
rs = {}
ps = {}

for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))
    df["et_bin"] = pd.cut(df.et_seed, ets, labels=list(range(len(ets)-1)))
    df["ncls_tot_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))
    df["ncls_true_bin"] = pd.cut(df.ncls_true, ncls, labels=list(range(len(ncls)-1)))
    # df["nvtx_bin"] = pd.cut(df.nVtx, nvtx, labels=list(range(len(nvtx)-1)))

    res_precision =  df.groupby(["ncls_true_bin", "ncls_tot_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_true/ gr.ncls_sel).mean())\
                 .unstack(fill_value=0).stack()

    res_recall =  df.groupby(["ncls_true_bin", "ncls_tot_bin"])\
                 .apply( lambda gr:  (gr.ncls_sel_true/ gr.ncls_true).mean())\
                 .unstack(fill_value=0).stack()

    vmin=0.85
    vmax=1
    # vmin = min([min(res_dsc), min(res_mst)])
    # if vmin==0: vmin=0.8

    fig, (az, ax, ab) = plt.subplots(1,3, figsize=(26,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    H,_,_ = np.histogram2d(df.ncls_true, df.ncls, bins=ncls, density=True)
    H[H==0] = np.nan 
    C = az.imshow(H.T, cmap="plasma" , vmin=0, vmax=0.14)
    hs[flavour] = H.T
    fig.colorbar(C, label="N. windows", ax=az, shrink=0.8)
    az.set_xlabel("N. clusters true")
    az.set_ylabel("N. clusters tot")
    az.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    az.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    az.set_yticklabels(ncls)
    az.set_xticklabels(ncls)
    
    a1 = res_precision.values.reshape((len(ncls)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_recall.values.reshape((len(ncls)-1,len(ncls)-1))
    a2[a2==0] = np.nan

    A = ax.imshow(a1.T, vmin=vmin, vmax=vmax)
    ps[flavour] = a1.T
    ax.set_xlabel("N. clusters true")
    ax.set_ylabel("N. clusters tot")
    fig.colorbar(A , label="N. clusters purity", ax=ax, shrink=0.8)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(ncls)

    B = ab.imshow(a2.T, vmin=vmin, vmax=vmax)
    rs[flavour] = a2.T
    ab.set_xlabel("N. clusters true")
    ab.set_ylabel("N. clusters tot")
    plt.colorbar(B , label="N. clusters efficiency", ax=ab, shrink=0.8)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(ncls)
    ax.minorticks_off()
    ab.minorticks_off()
    
    az.text(0.7, 1.02, flavour, transform=az.transAxes, color='blue', fontsize="small")
    ax.text(0.7, 1.02, flavour, transform=ax.transAxes, color='blue', fontsize="small")
    ab.text(0.7, 1.02, flavour, transform=ab.transAxes, color='blue', fontsize="small")
    
    hep.cms.text("Preliminary ",loc=0, ax=az)
    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)

    fig.savefig(outdir + "/recall_purity/cluster_recall_{}_summary.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/cluster_recall_{}_summary.png".format(flavour))

    


fig, (ax,ay,az) = plt.subplots(1,3, figsize=(24,8),dpi=100)
plt.subplots_adjust( wspace=0.3)


A = ax.imshow(hs["Electron"]/ hs["Photon"], cmap="Spectral", vmin=0, vmax=2)
fig.colorbar(A, ax=ax, label="N cluster Ele/Photon",shrink=0.8)
ax.set_xlabel("N. clusters true")
ax.set_ylabel("N. clusters tot")
ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ax.set_yticklabels(ncls)
ax.set_xticklabels(ncls)

A = ay.imshow(ps["Electron"]/ ps["Photon"], cmap="Spectral", vmin=0.99, vmax=1.01 )
fig.colorbar(A, ax=ay, label="N cluster purity Ele/Photon", shrink=0.8)
ay.set_xlabel("N. clusters true")
ay.set_ylabel("N. clusters tot")
ay.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
ay.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ay.set_yticklabels(ncls)
ay.set_xticklabels(ncls)

A = az.imshow(rs["Electron"]/ rs["Photon"], cmap="Spectral", vmin=0.95, vmax=1.05)
fig.colorbar(A, ax=az, label="N cluster efficiency Ele/Photon",shrink=0.8)
az.set_xlabel("N. clusters true")
az.set_ylabel("N. clusters tot")
az.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
az.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
az.set_yticklabels(ncls)
az.set_xticklabels(ncls)

hep.cms.text("Preliminary ",loc=0, ax=ax)
hep.cms.text("Preliminary ",loc=0, ax=ay)
hep.cms.text("Preliminary ",loc=0, ax=az)

fig.savefig(outdir + "/recall_purity/cluster_recall_ele_photon_ratio_summary.png")
fig.savefig(outdir + "/recall_purity/cluster_recall_ele_photon_ratio_summary.pdf")


ets =  [1,2,4,6,8,10,15,20,30,40,50,60,70,100]
nclsx = [-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5, 1.5 ,2.5 ,3.5, 4.5 ,5.5 ]
for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    fig, ax = plt.subplots(1,1, figsize=(10,8),dpi=100)
    dfb = df[df.ncls_sel != df.ncls_true]
    A  = ax.hist2d(dfb.ncls_sel - dfb.ncls_true, dfb.et_seed, bins=(nclsx, ets),cmap="plasma",cmin=1e-4,vmax=0.015, density=True )#
    fig.colorbar(A[3], ax=ax)   
    ax.set_xlabel("N. cluster selected - N. clusters true")
    ax.set_ylabel("Seed $E_T$")
    ax.text(0.75, 0.05, flavour, transform=ax.transAxes)
    ax.set_yscale("log")
    fig.savefig(outdir+ "/recall_purity/Ncltrue_norm_lost_{}_byet.png".format(flavour))
    fig.savefig(outdir+ "/recall_purity/Ncltrue_norm_lost_{}_byet.png".format(flavour))


ets = [0,5,10,15, 20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.479, 1.75,2.,2.5,3]
ncls = [1,2,3,4,5,6,8,10,15]

hs = {}
rs = {}
ps = {}

for df, flavour in zip([df_ele,df_gamma],["Electron","Photon"]):
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))
    df["et_bin"] = pd.cut(df.et_seed, ets, labels=list(range(len(ets)-1)))
    df["ncls_tot_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))
    df["ncls_true_bin"] = pd.cut(df.ncls_true, ncls, labels=list(range(len(ncls)-1)))
    # df["nvtx_bin"] = pd.cut(df.nVtx, nvtx, labels=list(range(len(nvtx)-1)))

    res_precision =  df.groupby(["ncls_true_bin", "ncls_tot_bin"])\
                 .apply( lambda gr:  (gr.En_sel_true/ gr.En_sel).mean())\
                 .unstack(fill_value=0).stack()

    res_recall =  df.groupby(["ncls_true_bin", "ncls_tot_bin"])\
                 .apply( lambda gr:  (gr.En_sel_true/ gr.En_true).mean())\
                 .unstack(fill_value=0).stack()

    vmin=0.99
    vmax=1
    # vmin = min([min(res_dsc), min(res_mst)])
    # if vmin==0: vmin=0.8

    fig, (az, ax, ab) = plt.subplots(1,3, figsize=(26,8),dpi=100)
    plt.subplots_adjust( wspace=0.3)

    H,_,_ = np.histogram2d(df.ncls_true, df.ncls, bins=ncls, density=True)
    H[H==0] = np.nan 
    C = az.imshow(H.T, cmap="plasma" , vmin=0, vmax=0.14)
    hs[flavour] = H.T
    fig.colorbar(C, label="N. windows", ax=az, shrink=0.8)
    az.set_xlabel("N. clusters true")
    az.set_ylabel("N. clusters tot")
    az.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    az.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    az.set_yticklabels(ncls)
    az.set_xticklabels(ncls)
    
    a1 = res_precision.values.reshape((len(ncls)-1,len(ncls)-1))
    a1[a1==0] = np.nan
    a2 = res_recall.values.reshape((len(ncls)-1,len(ncls)-1))
    a2[a2==0] = np.nan
  
    
    A = ax.imshow(a1.T, vmin=vmin, vmax=vmax)
    ps[flavour] = a1.T
    ax.set_xlabel("N. clusters true")
    ax.set_ylabel("N. clusters tot")
    fig.colorbar(A , label="Energy purity", ax=ax, shrink=0.8)

    ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    ax.set_yticklabels(ncls)
    ax.set_xticklabels(ncls)
    
      
    vmin=0.97
    vmax=1

    B = ab.imshow(a2.T, vmin=vmin, vmax=vmax)
    rs[flavour] = a2.T
    ab.set_xlabel("N. clusters true")
    ab.set_ylabel("N. clusters tot")
    plt.colorbar(B , label="Energy efficiency", ax=ab, shrink=0.8)


    ab.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
    ab.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
    ab.set_yticklabels(ncls)
    ab.set_xticklabels(ncls)
    ax.minorticks_off()
    ab.minorticks_off()
    
    az.text(0.7, 1.02, flavour, transform=az.transAxes, color='blue', fontsize="small")
    ax.text(0.7, 1.02, flavour, transform=ax.transAxes, color='blue', fontsize="small")
    ab.text(0.7, 1.02, flavour, transform=ab.transAxes, color='blue', fontsize="small")
    
    hep.cms.text("Preliminary ",loc=0, ax=az)
    hep.cms.text("Preliminary ",loc=0, ax=ax)
    hep.cms.text("Preliminary ",loc=0, ax=ab)

    fig.savefig(outdir + "/recall_purity/energy_recall_{}_summary.pdf".format(flavour))
    fig.savefig(outdir + "/recall_purity/energy_recall_{}_summary.png".format(flavour))


fig, (ax,ay,az) = plt.subplots(1,3, figsize=(24,8),dpi=100)
plt.subplots_adjust( wspace=0.3)

A = ax.imshow(hs["Electron"]/ hs["Photon"],cmap="Spectral",  vmax=2, vmin=0)
fig.colorbar(A, ax=ax, label="N cluster Ele/Photon",shrink=0.8)
ax.set_xlabel("N. clusters true")
ax.set_ylabel("N. clusters tot")
ax.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ax.set_yticklabels(ncls)
ax.set_xticklabels(ncls)
 
A = ay.imshow(ps["Electron"]/ ps["Photon"], cmap="Spectral",  vmax=1.005, vmin=0.995)
fig.colorbar(A, ax=ay, label="En. purity Ele/Photon", shrink=0.8)
ay.set_xlabel("N. clusters true")
ay.set_ylabel("N. clusters tot")
ay.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
ay.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ay.set_yticklabels(ncls)
ay.set_xticklabels(ncls)

A = az.imshow(rs["Electron"]/ rs["Photon"], cmap="Spectral",  vmax=1.01, vmin=0.99 )
fig.colorbar(A, ax=az, label="En. efficiency Ele/Photon",shrink=0.8)
az.set_xlabel("N. clusters true")
az.set_ylabel("N. clusters tot")
az.set_yticks(np.arange(len(ncls)) - 0.5, minor=False)
az.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
az.set_yticklabels(ncls)
az.set_xticklabels(ncls)

hep.cms.text("Preliminary ",loc=0, ax=ax)
hep.cms.text("Preliminary ",loc=0, ax=ay)
hep.cms.text("Preliminary ",loc=0, ax=az)

fig.savefig(outdir + "/recall_purity/energy_recall_ele_photon_ratio_summary.png")
fig.savefig(outdir + "/recall_purity/energy_recall_ele_photon_ratio_summary.pdf")



##################################3
print('comparison with truth resolution')

fig, (ax, ay,az) = plt.subplots(1,3, figsize=(30, 8),dpi=100, )
plt.subplots_adjust( wspace=0.3)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele.reset_index(level=0, inplace=True)
res_ele.reset_index(level=0, inplace=True)
res_gamma.reset_index(level=0, inplace=True)
res_gamma.reset_index(level=0, inplace=True)


vmax=0.4
vmin =0.01

a1 = res_ele.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
a2 = res_gamma.w68.values.reshape((len(etas)-1,len(ets)-1))
a2[a2==0] = np.nan

A = ax.imshow(a1.T,  cmap="plasma_r", vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC", transform=ax.transAxes,  fontsize="small")

A2 = ay.imshow(a2.T,  cmap="plasma_r")
ay.set_xlabel("Seed $\eta$")
ay.set_ylabel("Seed $E_T$")
fig.colorbar(A2 , label="$\sigma_{eff}$ DeepSC Photon", ax=ay, shrink=0.8)
ay.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ay.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ay.set_yticklabels(ets)
ay.set_xticklabels(etas)
ay.text(0.75, 1.02, "Photon", transform=ay.transAxes,  fontsize="small")
ay.text(0.01, 1.02, "DeepSC", transform=ay.transAxes,  fontsize="small")

A3 = az.imshow((a1/a2).T,  cmap="Spectral_r", vmin=0.7, vmax=1.3)
az.set_xlabel("Seed $\eta$")
az.set_ylabel("Seed $E_T$")
fig.colorbar(A3 , label="$\sigma_{eff}$ DeepSC Ele/Photon", ax=az, shrink=0.8)
az.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
az.set_xticks(np.arange(len(etas))- 0.5, minor=False)
az.set_yticklabels(ets)
az.set_xticklabels(etas)
az.text(0.75, 1.02, "Ele/Photon", transform=az.transAxes,  fontsize="small")
az.text(0.01, 1.02, "DeepSC", transform=az.transAxes,  fontsize="small")

fig.savefig(outdir + "/deepsc_wrt_truth_comparison.png")
fig.savefig(outdir + "/deepsc_wrt_truth_comparison.pdf")



fig, (ax, ay,az) = plt.subplots(1,3, figsize=(30, 8),dpi=100, )
plt.subplots_adjust( wspace=0.3)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele.reset_index(level=0, inplace=True)
res_ele.reset_index(level=0, inplace=True)
res_gamma.reset_index(level=0, inplace=True)
res_gamma.reset_index(level=0, inplace=True)


vmax= 0.4
vmin =0.01

a1 = res_ele.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
a2 = res_gamma.w68.values.reshape((len(etas)-1,len(ets)-1))
a2[a2==0] = np.nan

A = ax.imshow(a1.T,  cmap="plasma_r", vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Sedd $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ Mustache Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache", transform=ax.transAxes,  fontsize="small")

A2 = ay.imshow(a2.T,  cmap="plasma_r", vmin=vmin, vmax=vmax)
ay.set_xlabel("Seed $\eta$")
ay.set_ylabel("Sedd $E_T$")
fig.colorbar(A2 , label="$\sigma_{eff}$ Mustache Photon", ax=ay, shrink=0.8)
ay.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ay.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ay.set_yticklabels(ets)
ay.set_xticklabels(etas)
ay.text(0.75, 1.02, "Photon", transform=ay.transAxes,  fontsize="small")
ay.text(0.01, 1.02, "Mustache", transform=ay.transAxes,  fontsize="small")

A3 = az.imshow((a1/a2).T,  cmap="Spectral_r", vmin=0.7, vmax=1.3)
az.set_xlabel("Seed $\eta$")
az.set_ylabel("Sedd $E_T$")
fig.colorbar(A3 , label="$\sigma_{eff}$ Mustache Ele/Photon", ax=az, shrink=0.8)
az.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
az.set_xticks(np.arange(len(etas))- 0.5, minor=False)
az.set_yticklabels(ets)
az.set_xticklabels(etas)
az.text(0.75, 1.02, "Ele/Photon", transform=az.transAxes,  fontsize="small")
az.text(0.01, 1.02, "Mustache", transform=az.transAxes,  fontsize="small")

fig.savefig(outdir + "/mustache_wrt_truth_comparison.png")
fig.savefig(outdir + "/mustache_wrt_truth_comparison.pdf")




fig, axs = plt.subplots(2,2, figsize=(22, 17),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)


res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)


vmax=1.6
vmin =1
palette = "plasma_r"

a1 = res_ele_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan

a1_dsc = res_ele_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan

a1_mst = res_ele_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan




ax = axs[0,0]
A = ax.imshow((a1_dsc/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")

ax = axs[0,1]
A = ax.imshow((a1_mst/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ Mustace/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")


a1 = res_gamma_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan

a1_dsc = res_gamma_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan

a1_mst = res_gamma_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan



ax = axs[1,0]
A = ax.imshow((a1_dsc/a1).T, cmap=palette  , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")

ax = axs[1,1]
A = ax.imshow((a1_mst/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ Mustace/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")

fig.savefig(outdir + "/performance_comparison_simtruth_algo.pdf")
fig.savefig(outdir + "/performance_comparison_simtruth_algo.png")



fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 0.8
vmin = 1.2
palette = "Spectral_r"

a1 = res_ele_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow(( (a1_dsc/a1) / (b1_dsc/b1)).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/simTruth Ele/Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.55, 1.02, "Electron / Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow(( (a1_mst/a1) / (b1_mst/b1)).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ Mustache/simTruth Ele/Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.55, 1.02, "Electron / Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")

fig.savefig(outdir + "/scale_comparison_simtruth_algo.pdf")
fig.savefig(outdir + "/scale_comparison_simtruth_algo.png")



fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 1
vmin = 0.75
palette = "plasma_r"

a1 = res_ele_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.w68.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.w68.values.reshape((len(etas)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.w68.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow( (a1_dsc/a1_mst).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/Mustache Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / Mustache", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow((b1_dsc/ b1_mst).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/Mustache Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / Mustache", transform=ax.transAxes,  fontsize="small")
fig.savefig(outdir + "/performance_comparison_deepsc_mustache.pdf")
fig.savefig(outdir + "/performance_comparison_deepsc_mustache.png")






fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
ncls = [1,2,3,4,5,6,7,8,10,15]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["ncl_bin"] = pd.cut(df_ele.ncls, ncls, labels=list(range(len(ncls)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["ncl_bin"] = pd.cut(df_gamma.ncls, ncls, labels=list(range(len(ncls)-1)))


res_ele_sim = df_ele.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["ncl_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 0.8
vmin = 1.2
palette = "Spectral_r"

a1 = res_ele_sim.w68.values.reshape((len(ncls)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.w68.values.reshape((len(ncls)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.w68.values.reshape((len(ncls)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.w68.values.reshape((len(ncls)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.w68.values.reshape((len(ncls)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.w68.values.reshape((len(ncls)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow(( (a1_dsc/a1) / (b1_dsc/b1)).T,  cmap=palette , vmin=0.9, vmax=1.10)
ax.set_xlabel("N. cls tot")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ DeepSC/simTruth Ele/Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(ncls)
ax.text(0.55, 1.02, "Electron / Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow(( (a1_mst/a1) / (b1_mst/b1)).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("N. cls tot")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="$\sigma_{eff}$ Mustache/simTruth Ele/Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(ncls))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(ncls)
ax.text(0.55, 1.02, "Electron / Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")
fig.savefig(outdir + "/performance_comparison_simtruth_algo_byseedet_ncls.pdf")
fig.savefig(outdir + "/performance_comparison_simtruth_algo_byseedet_ncls.png")




fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 1.1
vmin = 0.9
palette = "plasma_r"

a1 = res_ele_sim.m.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.m.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.m.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.m.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow( (a1_dsc/a1_mst).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/Mustache Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / Mustache", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow((b1_dsc/ b1_mst).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/Mustache Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / Mustache", transform=ax.transAxes,  fontsize="small")
fig.savefig(outdir + "/scale_comparison_simtruth_deepsc_mustache.pdf")
fig.savefig(outdir + "/scale_comparison_simtruth_deepsc_mustache.png")




fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 1.10
vmin = 0.90
palette = "Spectral"

a1 = res_ele_sim.m.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.m.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.m.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.m.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow( (a1_dsc/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow((b1_dsc/ b1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")
fig.savefig(outdir + "/scale_comparison_simtruth_deepsc.pdf")
fig.savefig(outdir + "/scale_comparison_simtruth_deepsc.png")



fig, axs = plt.subplots(1,2, figsize=(20, 8),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 1.1
vmin = 0.9
palette = "Spectral"

a1 = res_ele_sim.m.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.m.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_mst = res_ele_mst.m.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.m.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan

ax = axs[0]
A = ax.imshow( (a1_mst/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale Mustache/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")


ax = axs[1]
A = ax.imshow((b1_mst/ b1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale Mustache/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")

fig.savefig(outdir + "/scale_comparison_simtruth_mustache.pdf")
fig.savefig(outdir + "/scale_comparison_simtruth_mustache.png")


fig, axs = plt.subplots(2,2, figsize=(22, 17),dpi=100, )
plt.subplots_adjust( wspace=0.2)

ets = [2,4,6,8,10,20,40,60,80,100]
etas = [0, 0.4,0.8, 1.2,1.44, 1.57, 1.75,2.,2.3,2.6,3]
df_ele["et_bin"] = pd.cut(df_ele.et, ets, labels=list(range(len(ets)-1)))
df_ele["eta_bin"] = pd.cut(df_ele.eta, etas, labels=list(range(len(etas)-1)))
df_gamma["et_bin"] = pd.cut(df_gamma.et, ets, labels=list(range(len(ets)-1)))
df_gamma["eta_bin"] = pd.cut(df_gamma.eta, etas, labels=list(range(len(etas)-1)))


res_ele_sim = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_gamma_sim  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
res_ele_sim.reset_index(level=0, inplace=True)
res_ele_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)
res_gamma_sim.reset_index(level=0, inplace=True)


res_ele_dsc = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_gamma_dsc  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
res_ele_dsc.reset_index(level=0, inplace=True)
res_ele_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)
res_gamma_dsc.reset_index(level=0, inplace=True)

res_ele_mst = df_ele.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_gamma_mst  = df_gamma.groupby(["eta_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
res_ele_mst.reset_index(level=0, inplace=True)
res_ele_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)
res_gamma_mst.reset_index(level=0, inplace=True)

vmax= 1.05
vmin = 0.95
palette = "Spectral"

a1 = res_ele_sim.m.values.reshape((len(etas)-1,len(ets)-1))
a1[a1==0] = np.nan
b1 = res_gamma_sim.m.values.reshape((len(etas)-1,len(ets)-1))
b1[b1==0] = np.nan

a1_dsc = res_ele_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
a1_dsc[a1_dsc==0] = np.nan
b1_dsc = res_gamma_dsc.m.values.reshape((len(etas)-1,len(ets)-1))
b1_dsc[b1_dsc==0] = np.nan


a1_mst = res_ele_mst.m.values.reshape((len(etas)-1,len(ets)-1))
a1_mst[a1_mst==0] = np.nan
b1_mst = res_gamma_mst.m.values.reshape((len(etas)-1,len(ets)-1))
b1_mst[b1_mst==0] = np.nan



ax = axs[0,0]
A = ax.imshow((a1_dsc/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")

ax = axs[0,1]
A = ax.imshow((a1_mst/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale Mustace/simTruth Ele", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Electron", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")


ax = axs[1,0]
A = ax.imshow((a1_dsc/a1).T, cmap=palette  , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale DeepSC/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "DeepSC / simTruth", transform=ax.transAxes,  fontsize="small")

ax = axs[1,1]
A = ax.imshow((a1_mst/a1).T,  cmap=palette , vmin=vmin, vmax=vmax)
ax.set_xlabel("Seed $\eta$")
ax.set_ylabel("Seed $E_T$")
fig.colorbar(A , label="Scale Mustace/simTruth Photon", ax=ax, shrink=0.8)
ax.set_yticks(np.arange(len(ets)) - 0.5, minor=False)
ax.set_xticks(np.arange(len(etas))- 0.5, minor=False)
ax.set_yticklabels(ets)
ax.set_xticklabels(etas)
ax.text(0.75, 1.02, "Photon", transform=ax.transAxes,  fontsize="small")
ax.text(0.01, 1.02, "Mustache / simTruth", transform=ax.transAxes,  fontsize="small")

fig.savefig(outdir + "/scale_comparison_simtruth_algo.pdf")
fig.savefig(outdir + "/scale_comparison_simtruth_algo.png")



##########################################
#########################################
######################################
print("RAW resolution comparison plots")

dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):
#     ets = [1.5, 3,4,5, 6,8, 10,15,20,40,60, 80,100]
    ets = [1.00, 4, 8, 12, 16, 20,25 ,30,35,40, 45, 50, 60, 70, 80, 90 ,100]
    etas = [0, 1.479, 3]
    df["et_bin"] = pd.cut(df.Et_true_gen, ets, labels=list(range(len(ets)-1)))
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))


    res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
    res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
#     avgenraw = df.groupby(["et_bin","eta_bin"]).agg({"en_true_sim_good": "mean"})
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    
#     avgenraw.reset_index(level=0, inplace=True)
#     avgenraw.reset_index(level=0, inplace=True)

    fig = plt.figure(figsize=(8,9), dpi=200)
    gs = fig.add_gridspec(2, hspace=0.05, height_ratios=[0.75,0.25])
    axs = gs.subplots(sharex=True)

    errx = []
    x = []
    for i in range(len(ets)-1):
        errx.append((ets[i+1]- ets[i])/2)
        x.append((ets[i+1]+ ets[i])/2)

    mustl = []
    deepl = []


        
    for ieta, eta in enumerate(etas[:-1]):
        l = axs[0].errorbar(x, res_must[res_must.eta_bin == ieta].w68, xerr=errx, label="[{}, {}]".format(etas[ieta], etas[ieta+1]), fmt = ".")
        mustl.append(l)
        
    for ieta, eta in enumerate(etas[:-1]):
        l = axs[0].errorbar(x, res[res.eta_bin == ieta].w68, xerr=errx,label="[{}, {}]".format(etas[ieta], etas[ieta+1]), 
                                marker="s", markerfacecolor='none', c=mustl[ieta].lines[0].get_color(), linestyle='none', elinewidth=0)
        deepl.append(l)
    

    for ieta, eta in enumerate(etas[:-1]):
#         v =res_must[res_must.eta_bin == ieta].w68**2 - res[res.eta_bin == ieta].w68**2
#         var = np.sqrt(np.abs(v))*np.sign(v) / res_must[res_must.eta_bin == ieta].w68
        var = res[res.eta_bin == ieta].w68 / res_must[res_must.eta_bin == ieta].w68
        axs[1].errorbar(x, var, xerr=errx, label="[{}, {}]".format(etas[ieta], etas[ieta+1]), fmt="o", linestyle='none', elinewidth=0 )

    axs[0].set_ylabel("$\sigma (E_{Raw}/E_{Sim})$")

    axs[1].set_xlabel("$E_T^{Gen}$ [GeV]")
    # ax.plot([0,100],[1,1], linestyle="dashed", color="black")
    axs[0].set_ylim(1e-2, 0.4)
    axs[1].set_ylim(0.65, 1.05)
    axs[1].set_ylabel("$\sigma_{DeepSC}/\sigma_{Must}$", fontsize=22)
    axs[0].get_yaxis().set_label_coords(-0.1,1)
    axs[1].get_yaxis().set_label_coords(-0.1,1)
#     axs[1].legend(ncol=2, loc="upper right", fontsize=15)

    axs[0].text(0.55, 0.66, flavour, transform=axs[0].transAxes)
    axs[0].set_yscale("log")

    l1= axs[0].legend(handles=mustl, title="$|\eta_{Gen}|$", title_fontsize=18, loc="upper right", bbox_to_anchor=(0.6, 1), fontsize=18)
    
    ml = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Mustache')
    dl = mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=10, label='DeepSC')
    axs[0].legend(handles=[ml,dl], title="Algorithm", title_fontsize=18, loc="upper right", fontsize=18)
    axs[0].add_artist(l1)
    axs[1].grid(axis="y",which="both")
    axs[0].grid(axis="y", which="both")

    hep.cms.label(rlabel="14 TeV", loc=0, ax=axs[0])
    
    fig.savefig(outdir + "/resolution_byEnergy_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_byEnergy_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_byEnergy_{}_ratio.svg".format(flavour))



dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):

    ets = [1, 5,  15, 30,60]
    iplot = [0,1, 3 ]
    etas = [0, 0.3,0.6,0.9,1.2, 1.485, 1.566, 1.75, 2.,2.25,2.5,2.75,3]
    exclude_bin = 5
    df["et_bin"] = pd.cut(df.Et_true_gen, ets, labels=list(range(len(ets)-1)))
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))


    res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
    res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
#     avgenraw = df.groupby(["et_bin","eta_bin"]).agg({"en_true_sim_good": "mean"})
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
#     avgenraw.reset_index(level=0, inplace=True)
#     avgenraw.reset_index(level=0, inplace=True)
    
    fig = plt.figure(figsize=(8,9), dpi=200)
    gs = fig.add_gridspec(2, hspace=0.05, height_ratios=[0.75,0.25])
    axs = gs.subplots(sharex=True)

    errx = []
    x = []
    for i in range(len(etas)-1):
        errx.append((etas[i+1]- etas[i])/2)
        x.append((etas[i+1]+ etas[i])/2)

    mustl = []
    deepl = []
    
    res.loc[res.eta_bin == exclude_bin, ["w68"]] = 0
    res_must.loc[res_must.eta_bin == exclude_bin, ["w68"]] = 0
    
    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        l = axs[0].errorbar(x, res_must[res_must.et_bin == iet].w68, xerr=errx, label="[{}, {}] GeV".format(ets[iet], ets[iet+1]), fmt = ".")
        mustl.append(l)

    i = 0
    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        l = axs[0].errorbar(x, res[res.et_bin == iet].w68,  xerr=errx ,label="[{}, {}] GeV".format(ets[iet], ets[iet+1]), 
                                c=mustl[i].lines[0].get_color(), marker="s", markerfacecolor='none', linestyle='none',elinewidth=0)
        i+=1
        deepl.append(l)
        
    axs[0].fill_between([1.485, 1.566], [5e-3,5e-3],[0.5,0.5], color="lightgray", alpha=0.5)

    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        #v =res_must[res_must.et_bin == iet].w68**2 - res[res.et_bin == iet].w68**2
        #var = np.sqrt(np.abs(v))*np.sign(v) / res_must[res_must.et_bin == iet].w68
        var = res[res.et_bin==iet].w68 / res_must[res_must.et_bin==iet].w68
        axs[1].errorbar(x, var,xerr=errx, label="$E_T^{Gen} $" + " [{}, {}] GeV".format(ets[iet], ets[iet+1]),  fmt="o", linestyle='none', elinewidth=0)

    axs[0].set_ylabel("$\sigma (E_{Raw}/E_{Sim})$")

    l1= axs[0].legend(handles=mustl, title="$E_T^{Gen}$", title_fontsize=18, loc="upper left", fontsize=18)
    
    ml = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Mustache')
    dl = mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=10, label='DeepSC')
    axs[0].legend(handles=[ml,dl], title="Algorithm", title_fontsize=18, loc="upper right", bbox_to_anchor=(0.7, 1), fontsize=18)
    axs[0].add_artist(l1)

    axs[1].set_xlabel("$|\eta_{Gen}|$")
    axs[0].set_ylim(5e-3,1e1)
    # ax.plot([0,100],[1,1], linestyle="dashed", color="black")

    axs[1].set_ylim(0.65, 1.05)
    axs[1].set_ylabel("$\sigma_{DeepSC}/\sigma_{Must}$", fontsize=22)
    axs[0].get_yaxis().set_label_coords(-0.1,1)
    axs[1].get_yaxis().set_label_coords(-0.1,1)
    
    axs[1].fill_between([1.485, 1.566], [-0.1,-0.1],[1.1,1.1], color="lightgray", alpha=0.5)
      
#     axs[1].legend(ncol=3,prop={'size': 13}, loc="lower left" )

    axs[0].text(0.72, 0.8, flavour, transform=axs[0].transAxes)
    
    axs[0].set_yscale("log")
    axs[0].grid(which="both",axis="y")
    axs[1].grid(which="both",axis="y")

    hep.cms.label(rlabel="14 TeV", loc=0, ax=axs[0])
    
    fig.savefig(outdir + "/resolution_byeta_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_byeta_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_byeta_{}_ratio.svg".format(flavour))


dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):
    nvtx = [20,45,50,55,60,70,80,90,120]
    ets = [1,5,15,30,60]
    iplot =[0,1,3]
    df["nvtx_bin"] = pd.cut(df.obsPU, nvtx, labels=list(range(len(nvtx)-1)))
    df["et_bin"] = pd.cut(df.Et_true_gen, ets, labels=list(range(len(ets)-1)))

    res = df.groupby(["nvtx_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
    res_must = df.groupby(["nvtx_bin","et_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
#     avgenraw = df.groupby(["nvtx_bin","et_bin"]).agg({"en_true_sim_good": "mean"})
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)

    fig = plt.figure(figsize=(8,9), dpi=200)
    gs = fig.add_gridspec(2, hspace=0.05, height_ratios=[0.75,0.25])
    ax, ar = gs.subplots(sharex=True)

    errx = []
    x = []
    for i in range(len(nvtx)-1):
        errx.append((nvtx[i+1]- nvtx[i])/2)
        x.append((nvtx[i+1]+ nvtx[i])/2)
        
    mustl = []
    deepl = []
    
    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        l = ax.errorbar(x, res_must[res_must.et_bin == iet].w68, xerr=errx, label=" [{}, {}] GeV".format(ets[iet], ets[iet+1]), fmt = ".")
        mustl.append(l)
        
    i =0
    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        l = ax.errorbar(x, res[res.et_bin == iet].w68 , xerr=errx,
                    label="[{}, {}] GeV".format(ets[iet], ets[iet+1]), marker="s", markerfacecolor='none', 
                    c=mustl[i].lines[0].get_color(), linestyle='none', elinewidth=0)
        i+=1
        deepl.append(l)
    
    ax.set_ylim(1e-2, 6)
    ax.set_ylabel("$\sigma (E_{Raw}/E_{Sim})$")
    ax.grid(axis="y", which="both")
    ax.get_yaxis().set_label_coords(-0.1,1)
    ax.set_yscale("log")
    
#     ax.legend(ncol=2, fontsize='x-small', loc="upper left", title="Seed $E_T$", title_fontsize="small")
    l1= ax.legend(handles=mustl, title="$E_T^{Gen}$", title_fontsize=18, loc="upper left", fontsize=18)
    
    ml = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Mustache')
    dl = mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=10, label='DeepSC')
    ax.legend(handles=[ml,dl], title="Algo", title_fontsize=18, loc="upper right",bbox_to_anchor=(0.7, 1), fontsize=18)
    ax.add_artist(l1)

    ax.text(0.72, 0.8, flavour, transform=ax.transAxes)
    hep.cms.label( loc=0, ax=ax, rlabel="14 TeV")

    for iet, et in enumerate(ets[:-1]):
        if iet not in iplot: continue
        #v =res_must[res_must.et_bin == iet].w68**2 - res[res.et_bin == iet].w68**2
        #var = np.sqrt(np.abs(v))*np.sign(v) / res_must[res_must.et_bin == iet].w68
        var = res[res.et_bin == iet].w68 / res_must[res_must.et_bin==iet].w68
        ar.errorbar(x, var, xerr=errx, label="$E_T^{Gen}$" + " [{}, {}] GeV".format(ets[iet], ets[iet+1]),fmt="o", linestyle='none', elinewidth=0 )
        
    ar.set_ylim(0.7, 1.05)
    ar.set_ylabel("$\sigma_{DeepSC}/\sigma_{Must}$", fontsize=22)
    
    ar.get_yaxis().set_label_coords(-0.1,1)
    
#     ar.legend(ncol=3,prop={'size': 13}, loc="lower left" )
    
    ar.set_xlabel("$PU_{sim}$")
    ar.grid(axis="y", which="both")

    fig.savefig(outdir + "/resolution_byPU_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_byPU_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_byPU_{}_ratio.svg".format(flavour))


dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):

    fig, ax = plt.subplots(1,1, figsize=(8,9),dpi=200, )

    ets = [2,5,10,20,40,60,80,100]
    ncls = [2,3,4,6,10,15]
    
    legends = ["2", "3", "[4, 6)", "[6,10)","[10, $+\infty$)"]
    df["et_bin"] = pd.cut(df.et, ets, labels=list(range(len(ets)-1)))
    df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))


    res = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
    res_must = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)

    errx = []
    x = []
    for i in range(len(ets)-1):
        errx.append((ets[i+1]- ets[i])/2)
        x.append((ets[i+1]+ ets[i])/2)


    # gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[0.8,0.2])
    # axs = gs.subplots(sharex=True)

    for ieta, eta in enumerate(ncls[:-1]):
        ax.errorbar(x, res[res.ncls_bin == ieta].w68/res_must[res_must.ncls_bin == ieta].w68, xerr=errx, label=legends[ieta], fmt=".")


    ax.set_ylim(0.6, 1.1)
    ax.set_ylabel("DeepSC / Mustache $\sigma (E_{RAW}/E_{SIM})$")
    ax.legend(ncol=2, title="N. clusters")

    ax.set_xlabel("Seed $E_T$ [GeV]")
    ax.plot([min(ets),max(ets)],[1,1], linestyle="dashed", color="black")

    ax.grid(axis="y", which="both")
    ax.text(0.75, 0.9, flavour, transform=ax.transAxes)

    hep.cms.label(rlabel="14 TeV", loc=0, ax=ax)

    fig.savefig(outdir + "/resolution_byncls_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_byncls_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_byncls_{}_ratio.svg".format(flavour))

dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):
#     ets = [1.5, 3,4,5, 6,8, 10,15,20,40,60, 80,100]
    ets = [1.00, 4, 8, 12, 16, 20,25 ,30,35,40, 45, 50, 60, 70, 80, 90 ,100]
    etas = [0, 0.7, 1.479, 3]
    df["et_bin"] = pd.cut(df.Et_true_gen, ets, labels=list(range(len(ets)-1)))
    df["eta_bin"] = pd.cut(abs(df.seed_eta), etas, labels=list(range(len(etas)-1)))


    res = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("EnTrue_ovEtrue_sim_good"))
    res_must = df.groupby(["et_bin","eta_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
#     avgenraw = df.groupby(["et_bin","eta_bin"]).agg({"en_true_sim_good": "mean"})
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    
#     avgenraw.reset_index(level=0, inplace=True)
#     avgenraw.reset_index(level=0, inplace=True)

    fig = plt.figure(figsize=(8,9), dpi=200)
    gs = fig.add_gridspec(2, hspace=0.05, height_ratios=[0.75,0.25])
    axs = gs.subplots(sharex=True)

    errx = []
    x = []
    for i in range(len(ets)-1):
        errx.append((ets[i+1]- ets[i])/2)
        x.append((ets[i+1]+ ets[i])/2)

    mustl = []
    deepl = []


        
    for ieta, eta in enumerate(etas[:-1]):
        l = axs[0].errorbar(x, res_must[res_must.eta_bin == ieta].w68, xerr=errx, label="[{}, {}]".format(etas[ieta], etas[ieta+1]), fmt = ".")
        mustl.append(l)
        
    for ieta, eta in enumerate(etas[:-1]):
        l = axs[0].errorbar(x, res[res.eta_bin == ieta].w68, xerr=errx,label="[{}, {}]".format(etas[ieta], etas[ieta+1]), 
                                marker="s", markerfacecolor='none', c=mustl[ieta].lines[0].get_color(), linestyle='none', elinewidth=0)
        deepl.append(l)
    

    for ieta, eta in enumerate(etas[:-1]):
#         v =res_must[res_must.eta_bin == ieta].w68**2 - res[res.eta_bin == ieta].w68**2
#         var = np.sqrt(np.abs(v))*np.sign(v) / res_must[res_must.eta_bin == ieta].w68
        var = res[res.eta_bin == ieta].w68 / res_must[res_must.eta_bin == ieta].w68
        axs[1].errorbar(x, var, xerr=errx, label="[{}, {}]".format(etas[ieta], etas[ieta+1]), fmt="o", linestyle='none', elinewidth=0 )

    axs[0].set_ylabel("$\sigma (E_{Raw}/E_{Sim})$")

    axs[1].set_xlabel("$E_T^{Gen}$ [GeV]")
    # ax.plot([0,100],[1,1], linestyle="dashed", color="black")
    axs[0].set_ylim(7e-3, 0.4)
    axs[1].set_ylim(0.5, 1.05)
    axs[1].set_ylabel("$\sigma_{Truth}/\sigma_{Must}$", fontsize=22)
    axs[0].get_yaxis().set_label_coords(-0.1,1)
    axs[1].get_yaxis().set_label_coords(-0.1,1)
#     axs[1].legend(ncol=2, loc="upper right", fontsize=15)

    axs[0].text(0.65, 0.66, flavour, transform=axs[0].transAxes)
    axs[0].set_yscale("log")

    l1= axs[0].legend(handles=mustl, title="$|\eta_{Gen}|$", title_fontsize=18, loc="upper right", bbox_to_anchor=(0.6, 1), fontsize=18)
    
    ml = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Mustache')
    dl = mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=10, label='Truth-matched')
    axs[0].legend(handles=[ml,dl], title="Algorithm", title_fontsize=18, loc="upper right", fontsize=18)
    axs[0].add_artist(l1)
    axs[1].grid(axis="y",which="both")
    axs[0].grid(axis="y", which="both")

    hep.cms.label(rlabel="14 TeV", loc=0, ax=axs[0])
    
    fig.savefig(outdir + "/resolution_truth_byEnergy_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_truth_byEnergy_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_truth_byEnergy_{}_ratio.svg".format(flavour))


dfs = [df_ele, df_gamma]
flavours = ["Electron","Photon"]

for df,flavour in zip(dfs, flavours):
#     ets = [1.5, 3,4,5, 6,8, 10,15,20,40,60, 80,100]
    ets = [1.00, 4, 8, 12, 16, 20,25 ,30,35,40, 45, 50, 60, 70, 80, 90 ,100]
    ncls = [1,4,10,30]
    df["et_bin"] = pd.cut(df.Et_true_gen, ets, labels=list(range(len(ets)-1)))
    df["ncls_bin"] = pd.cut(df.ncls, ncls, labels=list(range(len(ncls)-1)))


    res = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good"))
    res_must = df.groupby(["et_bin","ncls_bin"]).apply(bin_analysis("En_ovEtrue_sim_good_mustache"))
#     avgenraw = df.groupby(["et_bin","eta_bin"]).agg({"en_true_sim_good": "mean"})
    res.reset_index(level=0, inplace=True)
    res.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    res_must.reset_index(level=0, inplace=True)
    
#     avgenraw.reset_index(level=0, inplace=True)
#     avgenraw.reset_index(level=0, inplace=True)

    fig = plt.figure(figsize=(8,9), dpi=200)
    gs = fig.add_gridspec(2, hspace=0.05, height_ratios=[0.75,0.25])
    axs = gs.subplots(sharex=True)

    errx = []
    x = []
    for i in range(len(ets)-1):
        errx.append((ets[i+1]- ets[i])/2)
        x.append((ets[i+1]+ ets[i])/2)

    mustl = []
    deepl = []


        
    for icl, cl in enumerate(ncls[:-1]):
        l = axs[0].errorbar(x, res_must[res_must.ncls_bin == icl].w68, xerr=errx, label="[{}, {}]".format(ncls[icl], ncls[icl+1]), fmt = ".")
        mustl.append(l)
        
    for icl, cl in enumerate(ncls[:-1]):
        l = axs[0].errorbar(x, res[res.ncls_bin == icl].w68, xerr=errx,label="[{}, {}]".format(ncls[icl], ncls[icl+1]), 
                                marker="s", markerfacecolor='none', c=mustl[icl].lines[0].get_color(), linestyle='none', elinewidth=0)
        deepl.append(l)
    

    for icl, cl in enumerate(ncls[:-1]):
#         v =res_must[res_must.eta_bin == ieta].w68**2 - res[res.eta_bin == ieta].w68**2
#         var = np.sqrt(np.abs(v))*np.sign(v) / res_must[res_must.eta_bin == ieta].w68
        var = res[res.ncls_bin == icl].w68 / res_must[res_must.ncls_bin == icl].w68
        axs[1].errorbar(x, var, xerr=errx, label="[{}, {}]".format(ncls[icl], ncls[icl+1]), fmt="o", linestyle='none', elinewidth=0 )

    axs[0].set_ylabel("$\sigma (E_{Raw}/E_{Sim})$")

    axs[1].set_xlabel("$E_T^{Gen}$ [GeV]")
    # ax.plot([0,100],[1,1], linestyle="dashed", color="black")
    axs[0].set_ylim(1e-2, 0.4)
    axs[1].set_ylim(0.55, 1.05)
    axs[1].set_ylabel("$\sigma_{DeepSC}/\sigma_{Must}$", fontsize=22)
    axs[0].get_yaxis().set_label_coords(-0.1,1)
    axs[1].get_yaxis().set_label_coords(-0.1,1)
#     axs[1].legend(ncol=2, loc="upper right", fontsize=15)

    axs[0].text(0.7, 0.66, flavour, transform=axs[0].transAxes)
    axs[0].set_yscale("log")

    l1= axs[0].legend(handles=mustl, title="N. clusters", title_fontsize=18, loc="upper right", bbox_to_anchor=(0.6, 1), fontsize=18)
    
    ml = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10, label='Mustache')
    dl = mlines.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=10, label='DeepSC')
    axs[0].legend(handles=[ml,dl], title="Algorithm", title_fontsize=18, loc="upper right", fontsize=18)
    axs[0].add_artist(l1)
    axs[1].grid(axis="y",which="both")
    axs[0].grid(axis="y", which="both")

    hep.cms.label(rlabel="14 TeV", loc=0, ax=axs[0])
    
    fig.savefig(outdir + "/resolution_byEnergy_Ncls_{}_ratio.png".format(flavour))
    fig.savefig(outdir + "/resolution_byEnergy_Ncls_{}_ratio.pdf".format(flavour))
    fig.savefig(outdir + "/resolution_byEnergy_Ncls_{}_ratio.svg".format(flavour))
