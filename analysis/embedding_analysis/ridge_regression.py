import os
import torch
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# I have model embeddings in the form of 
# model name: deepseek 2b
# 10 different runs at temperature of 0.7
# each run contains mean residual stream in the format: n_questions x d_model

# fMRI data
# 10 regions of interest (maybe I can add more surrounding regions, or look at broader networks)
# each region contains z-scored beta values from single trial GLMs in the format: n_questions x n_voxels


# People typically calculate ridge regression between model embeddings and voxel values
# How should I set this up? 

def regression_wrapper(task):
    
    model_path = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/midprocessing/{task}/model_cls"
    models = os.listdir(model_path) # this is now the raw csv files of embeddings
    
    datapath = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/midprocessing/{task}/human"
    participants = os.listdir(datapath)
    
    score_dict = {}
    
    for person in participants:
        
        print(f"Participant: {person}")
        # For every csv file of ROI voxels, which now includes larger regions with combined parcels
        roi_dir = f"{datapath}/{person}"
        ROIs = os.listdir(roi_dir)
        
        roi_dict = {}
        for roi_file in ROIs:
            roi = roi_file[:-4]
            print(f"ROI: {roi}")
            roi_data_path = f"{roi_dir}/{roi_file}"
            roi_df = pd.read_csv(roi_data_path).fillna(0)
            
            X = roi_df.to_numpy()
            
            # for every model - every file
            model_dict = {}
            for m in models:
                print(f"Working on embeddings from {m} for {task}")
                embedding_layer_path = f"{model_path}/{m}"
                layers = os.listdir(embedding_layer_path)
                
                # TODO - need to save the values for each layer
                layer_dict = {}
                for l in layers:
                    layer_path = f"{embedding_layer_path}/{l}"
                    
                    df = pd.read_csv(layer_path)
                
                    y = df.to_numpy()
                    
                    if X.shape[0] != y.shape[0]:
                        print("Shape mismatch, skipping")
                        continue
                    
                    # alpha is the regularization strength
                    rr_model = Ridge(alpha=1.0)  
                    
                    # scores = cross_val_score(rr_model, X, y, cv=3, scoring='r2')
                    scores = cross_val_score(rr_model, y, X, cv=3, scoring='r2') # predicting beta values from tokens seems to improve performance
                    layer_dict[l] = np.mean(scores)
                    
                    # print(f"Mean R² across ROI: {np.mean(scores):.3f}")

                    # save the scores for every embedding layer that I sampled, from every model, for each ROI, for each person
                model_dict[m] = layer_dict
            roi_dict[roi] = model_dict
        score_dict[person] = roi_dict
        # break
    # outputfile  = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/results/token_embeddings_from_beta_values_{task}.pkl"
    outputfile  = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/results/beta_values_from_cls_tokens_{task}.pkl"
    
    with open(outputfile, 'wb') as f:
        pickle.dump(score_dict, f)
'''
person : {
    ROI : {
        model : {
            layer : score
                
        }
    }
}
'''

def main():
    
    # For code and prose
    regression_wrapper('code')
    regression_wrapper('prose')


if __name__ == "__main__":
    main()

                # break
            # break
    
'''
for model in models:
    print(f"Model: {model}")
    model_dir = f"{model_path}/{model}"
    runs = os.listdir(model_dir)
    
    # for every run
    for run in runs:
        run_path = f"{model_dir}/{run}"
        run_df = pd.read_csv(run_path)
        y = run_df.to_numpy()
        
        # print(X.shape, y.shape)
        
        rr_model = Ridge(alpha=1.0)  
        # alpha is the regularization strength
        scores = cross_val_score(rr_model, X, y, cv=3, scoring='r2')
        
        
        print(f"Mean R² across ROI: {np.mean(scores):.3f}")
'''
# break

        # print(f"Voxel {voxel_idx} R² scores: {scores}")
        # print(f"Mean R²: {np.mean(scores):.3f}")
        
        # from sklearn.model_selection import GridSearchCV

        # param_grid = {'alpha': np.logspace(-2, 3, 10)}
        # model = Ridge()
        # grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        # grid.fit(X, Y[:, 0])  # for a single voxel
        # print("Best alpha:", grid.best_params_['alpha'])
        
        


# for every run within the model

# I wonder if I could parallelize this


# print(X)
# break
# break

# either calculate correlation coefficient, or use this is a sample for ridge regression


# Can also try RSA
# need X and Y

# model = Ridge(alpha=1.0)  # alpha is the regularization strength
# scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# print(f"Voxel {voxel_idx} R² scores: {scores}")
# print(f"Mean R²: {np.mean(scores):.3f}")

# from sklearn.model_selection import GridSearchCV

# param_grid = {'alpha': np.logspace(-2, 3, 10)}
# model = Ridge()
# grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
# grid.fit(X, Y[:, 0])  # for a single voxel
# print("Best alpha:", grid.best_params_['alpha'])

# pass


# For every csv file of ROI voxels

# either calculate correlation coefficient, or use this is a sample for ridge regression

