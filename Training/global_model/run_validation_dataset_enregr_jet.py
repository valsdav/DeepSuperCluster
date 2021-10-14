import numpy as np
import tf_data
import tensorflow as tf
import loader_jet as loader
import argparse 
from collections import defaultdict
import os
from time import time

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset-version", type=str, help="Dataset version", required=True)
parser.add_argument("--model-dir", type=str, help="Model folder", required=True)
parser.add_argument("--model-weights", type=str, help="Model weights", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-n", "--nevents", type=int, help="Number of events",)
parser.add_argument("-b", "--batch-size", type=int, help="Batch size", default=100)
args = parser.parse_args()


data_path_test = {"ele_match": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/recordio_allinfo_{}/testing/calo_matched/*.proto".format(args.dataset_version),
                  "gamma_match": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/recordio_allinfo_{}/testing/calo_matched/*.proto".format(args.dataset_version),
                    "jet_match": "/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/jets/recordio_allinfo_{}/testing/calo_matched/*.proto".format(args.dataset_version),
                  }

features_dict = {
"cl_features" : [ "en_cluster","et_cluster",
            "cluster_eta", "cluster_phi", 
            "cluster_ieta","cluster_iphi","cluster_iz",
            "cluster_deta", "cluster_dphi",
             "cluster_den_seed","cluster_det_seed",
            
#             "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
#             "cl_sigmaIphiIphi","cl_swissCross","cl_etaWidth","cl_phiWidth"
                 
            "cl_nxtals" ],

 "window_features" : [ "max_en_cluster","max_et_cluster","max_deta_cluster","max_dphi_cluster","max_den_cluster","max_det_cluster",
                    "min_en_cluster","min_et_cluster","min_deta_cluster","min_dphi_cluster","min_den_cluster","min_det_cluster",
                    "mean_en_cluster","mean_et_cluster","mean_deta_cluster","mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ],

# DO NOT CHANGE the list above, it is the one used for the training

# Metadata about the window like true energy, true calo position, useful info
 "window_metadata" :  ["en_true_sim","et_true_sim", "en_true_gen", "et_true_gen",
                    "en_true_sim_good", "et_true_sim_good",
                    "nclusters_insc",
                    "nVtx", "rho", "obsPU", "truePU",
                    "sim_true_eta", "sim_true_phi",  
                    "en_mustache_raw", "et_mustache_raw","en_mustache_calib", "et_mustache_calib",
                    "event_tot_simen_PU","wtot_simen_PU", "wtot_simen_sig" ]   ,
    
    
  "seed_features" : ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                     "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth",
                    ],
}

N_metadata = len(features_dict['window_metadata'])
N_seed_features = len(features_dict['seed_features'])

print("N. metadata: ", N_metadata)
print("N. seed features: ", N_seed_features)

print(">> Load the dataset ")
dataset = loader.load_dataset(data_path_test, features_dict=features_dict,  batch_size=args.batch_size, nevents=args.nevents, training=False, 
                            normalization_files=['normalization_{}_all.npz'.format(args.dataset_version),'normalization_wind_features_{}_all.npz'.format(args.dataset_version)])
X,y,w = tf_data.get(dataset)

print(">> Load the model")
model = loader.get_model(args.model_dir + '/args_load.json', 
                        args.model_dir + '/model.py',
             weights_path=args.model_dir+ "/" + args.model_weights, X=X)

print(">> Model successfully loaded")

print(">> Starting to run on events: ")
data = defaultdict(list)
lastT = time()
for ib, (X, y_true, weight) in enumerate(dataset):
    if ib % 10 == 0: 
        now = time()
        rate = 10* args.batch_size / (now-lastT)
        lastT = now
        nsecond = (args.nevents - args.batch_size*ib) / rate
        print("Events: {} ({:.1f}Hz). Eta: {:.0f}:{:.0f}".format(ib*args.batch_size, rate, nsecond//60, nsecond%60))
        
    y_out = model(X, training=False)
    
    cl_X_initial, wind_X_norm , cl_hits, is_seed,n_cl = X
    (dense_clclass,dense_windclass, en_regr_factor),  mask_cls, _  = y_out
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    y_target = tf.cast(y_clclass, tf.float32)

    pred_prob = tf.nn.sigmoid(dense_clclass) * mask_cls[:,:,tf.newaxis]
    pred_prob_window = tf.nn.softmax(dense_windclass)
    y_pred = tf.cast(pred_prob >= 0.5, tf.float32)
    y_mustache = tf.cast(cl_labels[:,:,-2] == 1 , tf.float32)
    
    En = cl_X[:,:,0:1]    
    Et = cl_X[:,:,1:2]    
    Et_tot_window = tf.squeeze(tf.reduce_sum(Et, axis=1))
    En_tot_window = tf.squeeze(tf.reduce_sum(En, axis=1))
    Et_true = tf.reduce_sum( tf.squeeze(Et * y_target),axis=1)

    Et_sel =  tf.reduce_sum( tf.squeeze(Et * y_pred),axis=1)
    Et_sel_true = tf.reduce_sum( tf.squeeze(Et * y_pred * y_target ),axis=1)
    Et_sel_mustache = tf.reduce_sum( tf.squeeze(Et) * y_mustache, axis=1)  
    Et_sel_mustache_true = tf.reduce_sum( tf.squeeze(Et * y_target) * y_mustache, axis=1)

    En_true_sim = y_metadata[:,0]  
    En_true_sim_good = y_metadata[:,4]
    En_mustache_calib = y_metadata[:,15]

    En_true = tf.reduce_sum( tf.squeeze(En * y_target),axis=1)
    En_true_gen =  y_metadata[:,2]
    En_sel = tf.reduce_sum( tf.squeeze(En * y_pred),axis=1) 
    En_sel_true = tf.reduce_sum( tf.squeeze(En * y_pred * y_target),axis=1)  
    En_sel_corr = En_sel * tf.squeeze(en_regr_factor)  
    En_sel_mustache = tf.reduce_sum( tf.squeeze(En) * y_mustache, axis=1)
    En_sel_mustache_true = tf.reduce_sum( tf.squeeze(En * y_target) * y_mustache, axis=1)

    fn_sum = tf.math.cumsum(y_target*(1 -  y_pred), axis=-2)
    fp_sum = tf.math.cumsum(y_pred*(1-y_target), axis=-2)
    true_sum = tf.math.cumsum(y_target, axis=-2)
    mask_first_false_negative =  (fn_sum == 1) & (y_target == 1)
    mask_first_false_positive =   (fp_sum ==1) & (y_target == 0)
    mask_second_false_negative =   (fn_sum==2) & (y_target == 1)
    mask_second_false_positive =  (fp_sum ==2) & (y_target == 0)
    mask_second = (true_sum == 2) & (y_target ==1)
    mask_third = (true_sum == 3) & (y_target ==1)
    mask_fourth = (true_sum == 4) & (y_target ==1)
    # mask_fifth = (true_sum == 5) & (y_target ==1)

    En_zero = tf.zeros(En.shape)
    En_first_false_negative = tf.squeeze(tf.reduce_sum(tf.where(mask_first_false_negative, En, En_zero), axis=-2))
    En_first_false_positive  = tf.squeeze(tf.reduce_sum(tf.where(mask_first_false_positive, En, En_zero), axis=-2))
    En_second_false_negative = tf.squeeze(tf.reduce_sum(tf.where(mask_second_false_negative, En, En_zero), axis=-2))
    En_second_false_positive  = tf.squeeze(tf.reduce_sum(tf.where(mask_second_false_positive, En, En_zero), axis=-2))

    En_second_true = tf.squeeze(tf.reduce_sum(tf.where(mask_second, En, En_zero), axis=-2))
    En_third_true = tf.squeeze(tf.reduce_sum(tf.where(mask_third, En, En_zero), axis=-2))
    En_fourth_true = tf.squeeze(tf.reduce_sum(tf.where(mask_fourth, En, En_zero), axis=-2))
    # En_fifth_true = tf.squeeze(tf.reduce_sum(tf.where(mask_fifth, En, En_zero), axis=-2))
    
    data['ncls'].append(n_cl.numpy())
    data['ncls_true'].append(tf.reduce_sum(tf.squeeze(y_target), axis=-1).numpy())
    data['ncls_sel'].append(tf.reduce_sum(tf.squeeze(y_pred), axis=-1).numpy())
    data['ncls_sel_true'].append(tf.reduce_sum(tf.squeeze(y_pred*y_target), axis=-1).numpy())
    
    data["En_cl_first_fn"].append(En_first_false_negative.numpy())
    data["En_cl_first_fp"].append(En_first_false_positive.numpy())
    data["En_cl_second_fn"].append(En_second_false_negative.numpy())
    data["En_cl_second_fp"].append(En_second_false_positive.numpy())

    data["En_cl_true_2"].append(En_second_true.numpy())
    data["En_cl_true_3"].append(En_third_true.numpy())
    data["En_cl_true_4"].append(En_fourth_true.numpy())
    # data["En_cl_true_5"].append(En_fifth_true.numpy())
    
    #Mustache selection
    data['ncls_sel_must'].append(tf.reduce_sum(y_mustache, axis=-1).numpy())
    data['ncls_sel_must_true'].append(tf.reduce_sum(tf.squeeze(y_target)*y_mustache, axis=-1).numpy())
    
    data["Et_tot"].append(Et_tot_window.numpy())
    data["En_tot"].append(En_tot_window.numpy())
    
    data['Et_true'].append(Et_true.numpy())
    data['Et_sel'].append(Et_sel.numpy())   
    data['Et_sel_true'].append(Et_sel_true.numpy()) 

    data['En_true'].append(En_true.numpy())
    data['En_true_sim'].append(En_true_sim.numpy())
    data['En_true_sim_good'].append(En_true_sim_good.numpy())
    data['En_true_gen'].append(En_true_gen.numpy())
    data['En_sel'].append(En_sel.numpy()) 
    data['En_sel_true'].append(En_sel_true.numpy()) 
    data['En_sel_corr'].append(En_sel_corr.numpy()) 
    
    data['Et_ovEtrue'].append((Et_sel/Et_true).numpy())   
    data['En_ovEtrue'].append((En_sel/En_true).numpy())   
    data['En_ovEtrue_sim'].append((En_sel/En_true_sim).numpy())  
    data['En_ovEtrue_sim_good'].append((En_sel/En_true_sim_good).numpy()) 

    # Truth selected clusters energy over sim energy
    data['EnTrue_ovEtrue_sim'].append((En_true/En_true_sim).numpy())
    data['EnTrue_ovEtrue_sim_good'].append((En_true/En_true_sim_good).numpy())

    #Mustache energy
    data['Et_sel_must'].append(Et_sel_mustache.numpy())        
    data['En_sel_must'].append(En_sel_mustache.numpy())   
    data['Et_sel_must_true'].append(Et_sel_mustache_true.numpy())        
    data['En_sel_must_true'].append(En_sel_mustache_true.numpy())    
    data['En_sel_must_calib'].append(En_mustache_calib.numpy())    
    
    data['Et_ovEtrue_mustache'].append((Et_sel_mustache/Et_true).numpy())   
    data['En_ovEtrue_mustache'].append((En_sel_mustache/En_true).numpy())   
    data['En_ovEtrue_sim_mustache'].append((En_sel_mustache/En_true_sim).numpy()) 
    data['En_ovEtrue_sim_good_mustache'].append((En_sel_mustache/En_true_sim_good).numpy()) 
    
    data["en_regr_factor"].append(tf.squeeze(en_regr_factor).numpy())
    data["En_ovEtrue_gen"].append((En_sel/En_true_gen).numpy())
    data["En_ovEtrue_gen_calib"].append((En_sel_corr/En_true_gen).numpy())
    
    data["En_ovEtrue_gen_mustache"].append((En_sel_mustache/En_true_gen).numpy())
    data["En_ovEtrue_gen_calib_mustache"].append((En_mustache_calib/En_true_gen).numpy())
   
    data["flavour"].append(y_metadata[:, -1].numpy())

    # seed features
    for iS, s in enumerate(features_dict["seed_features"]):
        data[s].append(y_metadata[:, N_metadata+iS].numpy())
        
    for iW, w in enumerate(features_dict["window_features"]):
        data[w].append(wind_X[:, iW].numpy())
        
    for iM, m in enumerate(features_dict["window_metadata"]):
        data[m].append(y_metadata[:, iM].numpy())
        
    # Now mustache selection
    data["w_nomatch"].append(pred_prob_window[:,0].numpy())
    data["w_ele"].append(pred_prob_window[:,1].numpy())
    data["w_gamma"].append(pred_prob_window[:,2].numpy())
    
    
print("\n\n>> DONE")

print(">> Converting to pandas")
data_final = {}
for k,v in data.items():
    data_final[k] = np.concatenate(v)
    
import pandas as pd
df  = pd.DataFrame(data_final)

df_ele = df[df.flavour == 11]
df_gamma = df[df.flavour == 22]
df_nomatch = df[df.flavour == 0]

print("Saving on disk")

os.makedirs(args.outputdir, exist_ok=True)
df_ele.to_csv(args.outputdir +"/validation_dataset_{}_ele.csv".format(args.dataset_version), sep=";",index=False)
df_gamma.to_csv(args.outputdir +"/validation_dataset_{}_gamma.csv".format(args.dataset_version), sep=";",index=False)
df_nomatch.to_csv(args.outputdir +"/validation_dataset_{}_jetmatch.csv".format(args.dataset_version), sep=";",index=False)

print("DONE!")