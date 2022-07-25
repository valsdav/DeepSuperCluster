from plotting_code import do_plot


input_folder = "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/supercluster_regression/electrons"
output_folder = "/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/RecoPlots/RecoComparison_regression_UL18_v8_SCregression/final"
os.makedirs(output_folder, exist_ok=True)


# Do ele
reco = "DeepSC_AlgoA"
df_1 = pd.HDFStore(f"{input_folder}/ele_UL18_123X_Mustache_v8_object.h5py", "r")["df"]
df_2 = pd.HDFStore(f"{input_folder}/ele_UL18_123X_{reco}_v8_object.h5py", "r")["df"]
df_1.rename(columns={"output_object.csv":"calomatched"}, inplace=True)
df_2.rename(columns={"output_object.csv":"calomatched"}, inplace=True)
df_1.rename(columns={"ncsl_tot":"ncls_tot"}, inplace=True)
df_2.rename(columns={"ncsl_tot":"ncls_tot"}, inplace=True)
df_1 = df_1.iloc[:-1]
df_2 = df_2.iloc[:-1]

df_1["Eraw_ov_Esim"] = df_1.en_sc_raw / df_1.calo_en_sim
df_1["Ecorr_ov_Esim"] = df_1.en_sc_calib / df_1.calo_en_sim
df_2["Eraw_ov_Esim"] = df_2.en_sc_raw / df_2.calo_en_sim
df_2["Ecorr_ov_Esim"] = df_2.en_sc_calib / df_2.calo_en_sim

df_1["Eraw_ov_EGen"] = df_1.en_sc_raw / df_1.genpart_en
df_1["Ecorr_ov_EGen"] = df_1.en_sc_calib / df_1.genpart_en
df_2["Eraw_ov_EGen"] = df_2.en_sc_raw / df_2.genpart_en
df_2["Ecorr_ov_EGen"] = df_2.en_sc_calib / df_2.genpart_en

df_1.rename(columns={"sc_index": "old_index"}, inplace=True)
df_2.rename(columns={"sc_index": "new_index"}, inplace=True)

df_join = df_1.merge(df_2, on=["runId","eventId","caloindex"], suffixes=["_old", "_new"],indicator=True)


print("Energy - eta")
res_d, res_m = do_plot(name="ele_gen_matched_corr_byEt_cruijff_sigmaL",
        df=df_join, 
        res_var="Ecorr_ov_EGen", 
        bins1=[4, 8, 12,16, 20,25 ,30,35,40, 45, 50, 60, 70, 80, 90 ,100],
        bins2=[0, 1, 1.485, 1.566, 3], 
        exclude_bin=2, 
        binlabel1="et", 
        binlabel2="eta", 
        binleg= "$\eta_{Gen}$",
        binvar1="calo_et_gen_new", 
        binvar2="seed_eta_new", 
        nbins_fit=250, 
        prange=0.98, 
        general_label="Electron \n(GEN-matched)", 
        xlabel="$E_T^{Gen}$", 
        ylabel="$\sigma_{L} (E_{Calib}/E_{Gen})$",
        ylabelratio="$\sigma^L_{DeepSC}/\sigma^L_{Must}$", 
        yvar="sigmaL",
        ylims1=(5e-3,1.5),
        ylims2=(0.75, 1.15),
        output_folder=None)



res_d, res_m = do_plot(name="ele_gen_matched_corr_byEt_cruijff_sigmaL",
        df=df_join, 
        res_var="Ecorr_ov_EGen", 
        bins1=[4, 8, 12,16, 20,25 ,30,35,40, 45, 50, 60, 70, 80, 90 ,100],
        bins2=[0, 1, 1.485, 1.566, 3], 
        exclude_bin=2, 
        binlabel1="et", 
        binlabel2="eta", 
        binleg= "$\eta_{Gen}$",
        binvar1="calo_et_gen_new", 
        binvar2="seed_eta_new", 
        nbins_fit=250, 
        prange=0.98, 
        general_label="Electron \n(GEN-matched)", 
        xlabel="$E_T^{Gen}$", 
        ylabel="$\sigma_{L} (E_{Calib}/E_{Gen})$",
        ylabelratio="$\sigma^L_{DeepSC}/\sigma^L_{Must}$", 
        yvar="sigmaL",
        ylims1=(5e-3,1.5),
        ylims2=(0.75, 1.15),
        output_folder=None)










res_d, res_m = do_plot(name="ele_gen_matched_corr_byEtEta",
        df=df_join, 
        res_var="Ecorr_ov_EGen", 
        bins1=[0, 0.5,0.8,1.0,1.2, 1.485, 1.566, 1.75, 2.,2.25,2.5,3],
        bins2=[4,10,20,40,60], 
        binlabel1="eta",
        binlabel2="et",
        binleg="$E_T^{Gen}$", 
        binvar1="seed_eta_new", 
        binvar2="calo_et_gen_new", 
        nbins_fit=250, 
        prange=0.98, 
        exclude_bin=5, 
        general_label="Electron \n(GEN-matched)", 
        xlabel="$|\eta_{Gen}|$", 
        ylabel="$\sigma_{avg}(E_{Calib}/E_{Gen})$",
        ylabelratio="$\sigma_{DeepSC}/\sigma_{Must}$", 
        yvar="sigma_avg",
        ylims1=(5e-3,1e1),
        ylims2=(0.75, 1.15),
        fill_between=[1.485, 1.566],
        output_folder=None)
