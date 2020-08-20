#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import numpy as np 
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 
import glob
import json


mpl.rcParams['figure.figsize'] = (5,5)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams["image.origin"] = 'lower'


# In[2]:


opt_version = 3
numpy_version = 1
base_dir = "/storage/ECAL/deepcluster/models/bayesian_opt_v{}".format(opt_version)
limit_files = 20
cols = ["seed_eta", "seed_iz","en_seed","et_seed",
        "cluster_deta", "cluster_dphi", "en_cluster", "et_cluster",
       "seed_f5_r9", "seed_f5_sigmaIetaIeta","seed_f5_sigmaIetaIphi","seed_f5_sigmaIphiIphi",
        "seed_f5_swissCross","seed_nxtals", "seed_etaWidth", "seed_phiWidth",
        "cl_f5_r9", "cl_f5_sigmaIetaIeta","cl_f5_sigmaIetaIphi","cl_f5_sigmaIphiIphi",
        "cl_f5_swissCross", "cl_nxtals", "cl_etaWidth", "cl_phiWidth"]

os.makedirs(base_dir, exist_ok = True)


# # Data preparation


files_ele = f"/storage/ECAL/training_data/wp_comparison/electrons/numpy_wp_ele_v{numpy_version}/training/"
files_gamma = f"/storage/ECAL/training_data/wp_comparison/gammas/numpy_wp_gamma_v{numpy_version}/training/"

datas_ele = []

i = 0
for f in glob.glob(files_ele+"*.pkl"):
    if i>limit_files :continue
    d = pickle.load(open(f, "rb"))   
    datas_ele.append(d[d.is_seed == False])
    i+=1
    
data_ele = pd.concat(datas_ele, ignore_index=True)
data_ele["particle"] = "electron"
print("N events ele: ",len(data_ele))

datas_gamma = []
i = 0
for f in glob.glob(files_gamma+"*.pkl"):
    if i>limit_files :continue
    d = pickle.load(open(f, "rb"))  
    datas_gamma.append(d[d.is_seed==False])
    i+=1
    
data_gamma = pd.concat(datas_gamma, ignore_index=True)
data_gamma["particle"] = "gamma"
print("N events gamma: ",len(data_gamma))

if data_ele.shape[0]> data_gamma.shape[0]:
    data = pd.concat([data_gamma, data_ele.iloc[0:len(data_gamma)]], ignore_index=True)
else:
    data = pd.concat([data_gamma.iloc[0:len(data_ele)], data_ele], ignore_index=True)
    
del data_gamma
del data_ele


# ## Reweighting
# Only the classes are reweighted, not in Et/eta bins

# In[8]:


w = len(data[(data.is_seed == False) & (data.in_scluster == False)]) / len(data[(data.is_seed == False) & (data.in_scluster==True)])
print("Weight ",w)
data.loc[data.in_scluster,"w"] = w
data.loc[data.in_scluster==False, "w"] = 1.


# # Array preparation
X = data[ cols ].values
truth = data[["in_scluster"]].values
y = np.array(truth[:], dtype=int)
weights = data.w.values


from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pickle.dump(scaler, open(base_dir + "/scaler_v{}.pkl".format(opt_version), "wb"))


print("N. samples:", X.shape[0])


X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_scaled, y, weights, test_size=0.20,
                                                stratify=y)


# Import all the required Libraries
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, Adagrad, RMSprop
from keras.metrics import AUC
from keras import callbacks
from keras import backend as K
from keras import regularizers
from keras.callbacks import *


import sherpa
parameters = [sherpa.Discrete('num_units', [50, 500]),
              sherpa.Discrete('num_layers', [2, 5]),
              sherpa.Continuous('l2_reg', [0, 0.03]),
              sherpa.Continuous('dropout', [0.0,0.4]),
              sherpa.Continuous('lr', [0.1,0.001]),
              sherpa.Ordinal('batch_size', [1024, 2048, 4096]),
              sherpa.Choice('optimizer', ["adam", "adagrad", "rmsprop"])]

activation = "relu"

alg = sherpa.algorithms.GPyOpt(max_num_trials=200,
                                verbosity=True)


study = sherpa.Study(parameters=parameters,
                    algorithm=alg,
                    lower_is_better=True, 
                    output_dir=base_dir)

for trial in study:

    os.makedirs("{}/model_{}".format(base_dir, trial.id), exist_ok=True)
    print(">>>>> Working on trial: ", trial.id)

    json.dump(trial.parameters, open("{}/model_{}/parameters.json".format(base_dir, trial.id), "w"))

    model = Sequential()
    for ilayer in range(trial.parameters['num_layers']):
        if ilayer ==0:
            model.add(Dense(trial.parameters['num_units'], 
                            input_dim=X_val.shape[1],
                            activation=activation,
                            kernel_regularizer=regularizers.l2(trial.parameters['l2_reg'])))
        else:
            model.add(Dense(trial.parameters['num_units'], 
                            activation=activation,
                            kernel_regularizer=regularizers.l2(trial.parameters['l2_reg'])))
        
        model.add(Dropout(trial.parameters["dropout"]))

    model.add(Dense(1, activation="sigmoid"))


    if trial.parameters["optimizer"] == "adam":
        opt = Adam(learning_rate=trial.parameters["lr"])
    elif trial.parameters["optimizer"] == "adagrad":
        opt = Adagrad(learning_rate=trial.parameters["lr"])
    elif trial.parameters["optimizer"] == "rmsprop":
        opt = RMSprop(learning_rate=trial.parameters["lr"])

    model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=[AUC(), "accuracy"])
    model.summary()


    auto_save = ModelCheckpoint("{}/model_{}/model.hd5".format(base_dir,trial.id), monitor='val_loss', 
                        verbose=1, save_best_only=True, save_weights_only=False, 
                        mode='auto', period=1)

    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, 
    #                         patience=3, verbose=1)


    history = model.fit(
                X_train, y_train,
                sample_weight = w_train,
                batch_size = trial.parameters["batch_size"],
                shuffle=True,
                verbose=False,
                epochs=10,
                validation_data = (X_val, y_val, w_val),
                callbacks = [ auto_save, study.keras_callback(trial, objective_name="val_loss")], # early_sto 
                )

    # Save it under the form of a json file
    print(history.history)
    json.dump(str(history.history), open("{}/model_{}/history.json".format(base_dir, trial.id), 'w'))


    study.finalize(trial)

    study.save()

    