import numpy as np
import awk_data
import tensorflow as tf
import loader_awk
import argparse 
from collections import defaultdict
import os
import json
from time import time

parser = argparse.ArgumentParser()

parser.add_argument("--model-config", type=str, help="Model configuration", required=True)
parser.add_argument("--model-weights", type=str, help="Model weights", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-n", "--nevents", type=int, help="Number of events",)
parser.add_argument("-b", "--batch-size", type=int, help="Batch size", default=100)
args = parser.parse_args()


config = json.load(open(args.model_config))

features_dict = config["dataset_conf"]["validation"]["columns"]
# The seed features are taken directly from the cluster
features_dict["seed_features"] = [ ]
for cl_f in features_dict["cl_features"]:
    features_dict["seed_features"] = cl_f.replace("cluster","seed")

def DeltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    return dphi
    
N_metadata = len(features_dict['window_metadata'])
N_seed_features = len(features_dict['seed_features'])

print("N. metadata: ", N_metadata)
print("N. seed features: ", N_seed_features)


print(">> Load the dataset manually to be able to use all the features")



print(">> Load the dataset and model")
model, dataset = loader_awk.get_model_and_dataset(args.model_config, args.model_weights,
                                                  training=False)

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

    cl_X_initial, wind_X, cl_hits, is_seed, mask_cls, mask_rechits = X
    (dense_clclass,dense_windclass, en_regr_factor),  mask_cls_  = y_out
    y_clclass, y_windclass, cl_X, wind_X, y_metadata, cl_labels = y_true
    
    y_target = tf.cast(y_clclass, tf.float32)

    pred_prob = tf.nn.sigmoid(dense_clclass) * mask_cls[:,:,tf.newaxis]
    pred_prob_window = tf.nn.softmax(dense_windclass)
    y_pred = tf.cast(pred_prob >= 0.5, tf.float32)
    y_mustache = tf.cast(cl_labels[:,:,4] == 1 , tf.float32)

    ind_cl_en = features_dict["cl_features"]["en_cluster"]
    ind_cl_et = features_dict["cl_features"]["et_cluster"]
    ind_cl_et = features_dict["cl_features"]["et_cluster"]
    En = cl_X[:,:,ind_cl_en:ind_cl_en+1]    
    En_calib = cl_labels[:,:,-2:-1]
    Et = cl_X[:,:,1:2]    
    Et_tot_window = tf.squeeze(tf.reduce_sum(Et, axis=1))
    En_tot_window = tf.squeeze(tf.reduce_sum(En, axis=1))
    En_tot_window_calib = tf.squeeze(tf.reduce_sum(En_calib, axis=1))
    Et_true = tf.reduce_sum( tf.squeeze(Et * y_target),axis=1)

    Et_sel =  tf.reduce_sum( tf.squeeze(Et * y_pred),axis=1)
    Et_sel_true = tf.reduce_sum( tf.squeeze(Et * y_pred * y_target ),axis=1)
    Et_sel_mustache = tf.reduce_sum( tf.squeeze(Et) * y_mustache, axis=1)  
    Et_sel_mustache_true = tf.reduce_sum( tf.squeeze(Et * y_target) * y_mustache, axis=1)

    ## Todo implement a mapping of variables name to indexes
    En_true_sim = y_metadata[:,0]  
    En_true_sim_good = y_metadata[:,4]
    Et_true_sim_good = y_metadata[:,5]
    En_mustache_regr = y_metadata[:,15]
    En_true_gen =  y_metadata[:,2]
    Et_true_gen =  y_metadata[:,3]

    En_true = tf.reduce_sum( tf.squeeze(En * y_target),axis=1)
    En_true_calib = tf.reduce_sum( tf.squeeze(En_calib * y_target),axis=1)

    En_sel = tf.reduce_sum( tf.squeeze(En * y_pred),axis=1) 
    En_sel_calib = tf.reduce_sum( tf.squeeze(En_calib * y_pred),axis=1) 
    En_sel_true = tf.reduce_sum( tf.squeeze(En * y_pred * y_target),axis=1) 
    En_sel_true_calib = tf.reduce_sum( tf.squeeze(En_calib * y_pred * y_target),axis=1) 
    
    En_sel_corr = En_sel_calib * tf.squeeze(en_regr_factor)
    En_sel_mustache = tf.reduce_sum( tf.squeeze(En) * y_mustache, axis=1)
    En_sel_mustache_calib = tf.reduce_sum( tf.squeeze(En_calib) * y_mustache, axis=1)
    En_sel_mustache_true = tf.reduce_sum( tf.squeeze(En * y_target) * y_mustache, axis=1)
    En_sel_mustache_true_calib = tf.reduce_sum( tf.squeeze(En_calib * y_target) * y_mustache, axis=1)

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
    data["En_tot_calib"].append(En_tot_window_calib.numpy())
    
    data['Et_true'].append(Et_true.numpy())
    data['Et_sel'].append(Et_sel.numpy())   
    data['Et_sel_true'].append(Et_sel_true.numpy()) 

    data['En_true'].append(En_true.numpy())
    data['En_true_sim'].append(En_true_sim.numpy())
    data['En_true_sim_good'].append(En_true_sim_good.numpy())
    data['Et_true_sim_good'].append(Et_true_sim_good.numpy())
    data['En_true_gen'].append(En_true_gen.numpy())
    data['Et_true_gen'].append(Et_true_gen.numpy())

    data['En_sel'].append(En_sel.numpy()) 
    data['En_sel_calib'].append(En_sel_calib.numpy()) 
    data['En_sel_true'].append(En_sel_true.numpy()) 
    data['En_sel_true_calib'].append(En_sel_true_calib.numpy()) 
    data['En_sel_corr'].append(En_sel_corr.numpy()) 
    
    data['Et_ovEtrue'].append((Et_sel/Et_true).numpy())   
    data['En_ovEtrue'].append((En_sel/En_true).numpy())   

    data['En_ovEtrue_sim'].append((En_sel/En_true_sim).numpy())  
    data['En_ovEtrue_sim_good'].append((En_sel/En_true_sim_good).numpy()) 
    data['En_calib_ovEtrue_sim'].append((En_sel_calib/En_true_sim).numpy())  
    data['En_calib_ovEtrue_sim_good'].append((En_sel_calib/En_true_sim_good).numpy()) 

    # Truth selected clusters energy over sim energy
    data['EnTrue_ovEtrue_sim'].append((En_true/En_true_sim).numpy())
    data['EnTrue_ovEtrue_sim_good'].append((En_true/En_true_sim_good).numpy())
    data['EnTrue_calib_ovEtrue_sim'].append((En_true_calib/En_true_sim).numpy())
    data['EnTrue_calib_ovEtrue_sim_good'].append((En_true_calib/En_true_sim_good).numpy())

    #Mustache energy
    data['Et_sel_must'].append(Et_sel_mustache.numpy())        
    data['En_sel_must'].append(En_sel_mustache.numpy())  
    data['En_sel_must_calib'].append(En_sel_mustache_calib.numpy())   
    data['Et_sel_must_true'].append(Et_sel_mustache_true.numpy())        
    data['En_sel_must_true'].append(En_sel_mustache_true.numpy()) 
    # calib == sum of pfClusters calibrated energy
    data['En_sel_must_true_calib'].append(En_sel_mustache_true_calib.numpy())    
    #central regression of the mustache
    data['En_sel_must_regr'].append(En_mustache_regr.numpy())    
    
    data['Et_ovEtrue_mustache'].append((Et_sel_mustache/Et_true).numpy())   
    data['En_ovEtrue_mustache'].append((En_sel_mustache/En_true).numpy()) 
    data['En_calib_ovEtrue_mustache'].append((En_sel_mustache_calib/En_true).numpy())   
    data['En_ovEtrue_sim_mustache'].append((En_sel_mustache/En_true_sim).numpy()) 
    data['En_ovEtrue_sim_good_mustache'].append((En_sel_mustache/En_true_sim_good).numpy()) 
    data['En_calib_ovEtrue_sim_mustache'].append((En_sel_mustache_calib/En_true_sim).numpy()) 
    data['En_calib_ovEtrue_sim_good_mustache'].append((En_sel_mustache_calib/En_true_sim_good).numpy()) 
    
    data["en_regr_factor"].append(tf.squeeze(en_regr_factor).numpy())
    data["En_ovEtrue_gen"].append((En_sel/En_true_gen).numpy())
    data["En_calib_ovEtrue_gen"].append((En_sel_calib/En_true_gen).numpy())
    data["En_ovEtrue_gen_regr"].append((En_sel_corr/En_true_gen).numpy())
    
    data["En_ovEtrue_gen_mustache"].append((En_sel_mustache/En_true_gen).numpy())
    data["En_calib_ovEtrue_gen_mustache"].append((En_sel_mustache_calib/En_true_gen).numpy())
    data["En_ovEtrue_gen_regr_mustache"].append((En_mustache_regr/En_true_gen).numpy())
   
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
df_nomatch = df[df.flavour ==0]

print("Saving on disk")

os.makedirs(args.outputdir, exist_ok=True)
df_ele.to_csv(args.outputdir +"/validation_dataset_ele.csv", sep=";",index=False)
df_gamma.to_csv(args.outputdir +"/validation_dataset_gamma.csv", sep=";",index=False)
# df_nomatch.to_csv(args.outputdir +"/validation_dataset_{}_nomatch.csv".format(args.dataset_version), sep=";",index=False)

print("DONE!")
