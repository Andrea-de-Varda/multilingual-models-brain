#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:19:13 2024

@author: dev
"""

#####################################################################################
# downsampling to check if negative results w/ Pereira are due to small sample size #
#####################################################################################

from sklearn.utils import resample


for n_samples in [50, 100, 250, 500, 1000]:
    print(f"\n{n_samples} SAMPLES")
    for modelname in dict_bestlayer.keys():
        print(f"Processing with {modelname.upper()}...")
        layernum = dict_bestlayer[modelname] # best layer (transfer, study I)
        embeddings = load(modelname)
        
        # refit model on entire data (exp + random), store, use w/ new data
        X = np.vstack([vec[layernum].mean(axis = 0) for vec in embeddings])
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_full = X_scaler.fit_transform(X)
        y_train_full = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_train, y_train = resample(X_train_full, y_train_full, 
                                    replace=False, 
                                    n_samples=n_samples, 
                                    random_state=n_samples)
        
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg.fit(X_train, y_train)
    
        with open(f"../../confirmatory/registered_models/control_{str(n_samples)}/{modelname}", 'wb') as handle:
            pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(f"../../confirmatory/registered_models/control_{str(n_samples)}/normaliz_params/{modelname}", 'wb') as handle:
            pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        # randomized model
        np.random.seed(0)  # seed for reproducibility
        np.random.shuffle(y_train)  # shuffling y
        reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        reg_random.fit(X_train, y_train)
    
        with open(f"../../confirmatory/registered_models/control_{str(n_samples)}/{modelname}_random", 'wb') as handle:
            pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)
            


for modelname in dict_bestlayer.keys():
    print(f"Processing with {modelname.upper()}...")
    layernum = dict_bestlayer[modelname] # best layer (transfer, study I)
    embeddings = load(modelname)
    
    # refit model on entire data (exp + random), store, use w/ new data
    X = np.vstack([vec[layernum].mean(axis = 0) for vec in embeddings])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_full = X_scaler.fit_transform(X)[baseline_filter]
    y_train_full = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()[baseline_filter]
    
    X_train, y_train = resample(X_train_full, y_train_full, 
                                replace=False, 
                                n_samples=500, 
                                random_state=0)
    
    reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/control_baseline_500/{modelname}", 'wb') as handle:
        pickle.dump(reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"../../confirmatory/registered_models/control_baseline_500/normaliz_params/{modelname}", 'wb') as handle:
        pickle.dump([X_scaler, y_scaler], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # randomized model
    np.random.seed(0)  # seed for reproducibility
    np.random.shuffle(y_train)  # shuffling y
    reg_random = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
    reg_random.fit(X_train, y_train)

    with open(f"../../confirmatory/registered_models/control_baseline_500/{modelname}_random", 'wb') as handle:
        pickle.dump(reg_random, handle, protocol=pickle.HIGHEST_PROTOCOL)