import os
import re
import pickle
import numpy as np
import pandas as pd
import nibabel as nib

best_models = ['code-deepseek_6b-ndelays_10-look_ahead_by_0',
               'prose-starcoder2_7b-ndelays_16-look_ahead_by_3']

important_parcels = [9,133,172]

# read in atlases
# atlas_base_path = "/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases"
atlas_base_path = "/home/zachkaras/fmri_model/analysis/pipeline/atlases"

# read in 2d mni mask
mask = nib.load(f"{atlas_base_path}/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx] # contains the schaefer parcel numbers
cortex_vx = np.where(atlas_only_brain != 0)[0]
parcel_nums = atlas_only_brain[cortex_vx]

# Making empty templates to save output
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)

'''
parcels_of_interest = {
    # From previous resting state paper: 133, 172, 9 - just choosing the significant ones

    'theory' : [133,172,192,284,339,395],
    'data_driven' : [9,67,176,183,231,242,341,343,348,380,386,389,391]
}
'''

class Regression_Info(object):
    # [model_name, task, look_ahead, n_delays, layer, stat]
    def __init__(self, model_name=None, task=None, look_ahead=None, n_delays=None, layer=None, stat=None):
        self.model_name = model_name
        self.task       = task
        self.look_ahead = look_ahead
        self.n_delays   = n_delays
        self.layer      = layer
        self.stat       = stat
        
    def __str__(self):
        return f"{self.model_name}, {self.task}, {self.look_ahead}, {self.n_delays}, {self.layer}, {self.stat}"

def parse_regression_info(path):
    
    parts = path.split('-') 
    # example: ['codegemma_7b', 'code', 'look_ahead_by_1', 'ndelays_0', 'layer_28', 'correlations.pkl']
    info = Regression_Info()
    info.model_name = parts[0]
    info.task       = parts[1]
    info.look_ahead = parts[2]
    info.n_delays   = parts[3]
    info.layer      = parts[4]
    info.stat       = (parts[5])[:-4]
    
    return info

def iterate_through_participants(filepath, model_params, stat):
    participants = os.listdir(filepath)
    
    records = []
    # iterating through participants
    
    for p in participants:
        print(p)
        datapath = f"{filepath}/{p}"
        files = os.listdir(datapath)
        stat_files = [f for f in files if re.search(stat, f) and not re.search(r'only_regressor|\+', f)]
        parts = model_params.split('-')
        stat_files = [f for f in stat_files if all(part in f for part in parts)]
        # break
        
        # iterating through the stat files
        # these are vectors of correlation coefficients between predicted and recorded signal
        # Different parameters were adjusted, so the folder contains all the possibilities
        # [model_name, task, look_ahead, n_delays, layer, stat]
        for sf in stat_files:
            
            info = parse_regression_info(sf)
            
            stat_file = f"{datapath}/{sf}"
            with open(stat_file, 'rb') as f:
                try:
                    stat_vec = pickle.load(f) # stat vec is just voxels from the schaefer parcel, about 130k voxels
                except:
                    print(f"issue with {p}: {sf}")
            
            # for the stat vec, I need to align it with indices of important schafer parcels
            parcel_correlations = {}
            for parcel in important_parcels:
                parcel_idx = np.where(parcel_nums == parcel)
                parcel_corrs = stat_vec[parcel_idx]
                parcel_correlations[parcel] = parcel_corrs
            
            new_record = {
                'participant' : p,
                'task' : info.task,
                'model' : info.model_name,
                'ndelays' : info.n_delays,
                'look_ahead' : info.look_ahead,
                'layer' : info.layer,
                **parcel_correlations
            }
            records.append(new_record)
        #     break
        # break
            
    records = pd.DataFrame(records)
    return records


def main():
    # iterate through directories, parse the file names, and accumulate stats
    # parsing file names [model_name, task, look_ahead, n_delays, layer, stat]

    filepath = "/data2/zachkaras/fmri_model_data/ridge_regression_pca_params"
    all_records = []
    for m in best_models:
        records = iterate_through_participants(filepath, m, 'correlations')
        all_records.append(records)
    all_records = pd.concat(all_records, ignore_index=True)

    outpath = "/data/zachkaras/fmri_model_data/intermediate_results"
    
    with open(f"{outpath}/parcels_of_interest-correlations.pkl", 'wb') as f:
        pickle.dump(all_records, f)

if __name__ == "__main__":
    main()