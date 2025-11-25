import os
import re
import pickle
import numpy as np

# TODO find the voxels among all the participants with the smallest range of correlation values
# Goal would be to find the voxels that are most tuned to coding (or prose)

# TODO - might just do this for the best performing model and layer

file = f"/storage1/fmri_model_data/ridge_regression_models/203/deepseek_2b-layer_0-code-correlations.pkl"

with open(file, 'rb') as f:
    data = pickle.load(f)


# separated by task (code, prose)

# iterate through participants

# faster option would be a numpy array where each column is a participant

# keep track of each voxel's values


def main():
    
    datapath = "/storage1/fmri_model_data/ridge_regression_pca_models"
    participants = os.listdir(datapath)
    
    for i,task in enumerate(['code', 'prose']):
        
        values = np.array([i for i in range(1,131066)]) # number of voxels
        for person in participants:
            
            # load data
            person_file = f"{datapath}/{person}/{"TBD"}-layer_{"TBD"}-{task}-correlations.pkl"
            
            with open(person_file, 'rb') as f:
                person_data = pickle.load(f)
                
            values = np.append(values, person_data, axis=0)
            

if __name__ == "__main__":
    main()