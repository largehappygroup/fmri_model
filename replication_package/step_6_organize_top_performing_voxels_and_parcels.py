import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from collections import defaultdict
import xml.etree.ElementTree as ET

# THIS SCRIPT READS IN ALL THE RESULTS, FORMATTED AS NIFTI FILES WHERE THE VALUE OF EACH VOXEL
# CORRESPONDS TO THE CORRELATION COEFFICIENT BETWEEN THE PREDICTED AND ACTUAL SIGNAL.
# THOSE RESULTS ARE ORGANIZED IN TWO HEAVILY NESTED (ARGUABLY TOO MUCH SO) DICTIONARIES WHERE VALUES
# ARE THE TOP 10,000 CORRELATION COEFFICIENTS, AND SEPARATELY, THE SCHAEFER PARCEL NUMBERS ASSOCIATED WITH 
# THOSE CORRELATION COEFFICIENTS. 

# read in atlases
atlas_base_path = "atlases"

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
hemis = ['Left' if ((parcel <= 200) & (parcel > 0)) else 'Right' for parcel in parcel_nums]

# Loading in atlas and labels for harvard-oxford atlas regions
hox_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
hox = nib.load(hox_data['filename']).get_fdata()
hox_vec = hox.flatten()
hox_only_brain = hox_vec[brain_idx]
hox_nums = hox_only_brain[cortex_vx] # Obtaining voxels in Harvard Oxford atlas that are in the schaefer atlas.
                                     #      The voxels in the Schaefer atlas that correspond to empty space in HarvOx atlas (values of 0) will get filtered out when we match to region names.

hox_label_path = "/home/zachkaras/atlases/HarvardOxford-Cortical.xml"
hox_labels = [label.text for label in ET.parse(hox_label_path).getroot().iter('label')] # indices are the region numbers - need to add one though

# Making empty templates to save output
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)

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


def find_cutoff(vec, threshold = 10**4):
    
    # I can find the threshold point based on sorting, then keep everything in the same place
    copy = vec.copy()
    copy.sort()
    cutoff = copy[-threshold-1]
    return cutoff


def iterate_through_participants(filepath, stat):
    participants = os.listdir(filepath)
    
    records = []
    # iterating through participants
    
    for p in participants:
        print(p)
        datapath = f"{filepath}/{p}"
        files = os.listdir(datapath)
        # stat_files = [f for f in files if re.search(stat, f) and not re.search(r'only_regressor|\+', f)]
        stat_files = [f for f in files if re.search(stat, f) and re.search('\+', f)]
    
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
            
            # filter to top 10k, 10k is default parameter but can be changed with threshold argument
            z_vec = np.arctanh(stat_vec)
            cutoff = find_cutoff(z_vec)

            # print(cutoff, type(stat_vec), stat_vec[0:10])
            try:
                top_voxel_idx = (np.where(z_vec > cutoff))[0]
            except:
                stat_vec = np.array(z_vec)
                top_voxel_idx = (np.where(z_vec > cutoff))[0]
            
            top_voxel_vals = z_vec[top_voxel_idx]
            
            # Using z-scored correlation coefficients for downstream correlation tests
            top_vals = z_vec[np.where(z_vec > cutoff)[0]]
            participant_means = float(np.mean(top_vals))
        
            # parcel nums contains schaefer parcel numbers for each voxel location
            # so this finds the corresponding parcels for the top voxels
            top_parcels = parcel_nums[top_voxel_idx]
            top_parcels = np.array([int(parcel) for parcel in top_parcels])

            # Mapping directly from voxel indices to HarvOx regions
            top_hox_idx = [int(idx) for idx in top_voxel_idx if hox_nums[idx] != 0] # finding locations for top 10k voxels 
            top_regions = hox_nums[top_hox_idx] # mapping to HarvOx region numbers
            top_regions = np.array([int(roi) for roi in top_regions])

            # finding corresponding hemispheres for voxel locations. Need to index by top_hox_idx to properly map to
            # correct locations in hemis list
            top_regions = [f"{hemis[top_hox_idx[i]]} {hox_labels[roi-1]}" for i,roi in enumerate(top_regions)]
            
            new_record = {
                'participant' : p,
                'task' : info.task,
                'model' : info.model_name,
                'ndelays' : info.n_delays,
                'look_ahead' : info.look_ahead,
                'layer' : info.layer,
                'top_voxel_idx' : top_voxel_idx,
                'top_voxel_vals' : top_voxel_vals,
                'participant_means' : participant_means,
                'top_parcels' : top_parcels,
                'top_regions' : top_regions

            }
            records.append(new_record)
        #     break
        # break
            
    records = pd.DataFrame(records)
    return records


def main():
    # iterate through directories, parse the file names, and accumulate stats
    # parsing file names [model_name, task, look_ahead, n_delays, layer, stat]

    # filepath = "/data2/zachkaras/ridge_regression_pca_params_remote"
    filepath = "/data2/zachkaras/fmri_model_data/ridge_regression_pca_params" # behemoth
    # filepath = "/s1/fmri_model_data/ridge_regression_pca_params" # cumberland
    
    # participant_means, stat_collection, parcel_collection 
    records = iterate_through_participants(filepath, 'correlations')
    # records = iterate_through_participants(filepath, 'cosine')

    # outpath = "/s1/fmri_model_data/intermediate_results"
    # outpath = "/data/zachkaras/fmri_model_data/intermediate_results"
    outpath = "/tank/home/zachkaras/fmri_model_data/intermediate_results"
    
    with open(f"{outpath}/all_results_regressor+features.pkl", 'wb') as f:
        pickle.dump(records, f)

if __name__ == "__main__":
    main()
        