import os
import re
import json
import pickle
import numpy as np
import nibabel as nib
from collections import defaultdict

# THIS SCRIPT READS IN ALL THE RESULTS, FORMATTED AS NIFTI FILES WHERE THE VALUE OF EACH VOXEL
# CORRESPONDS TO THE CORRELATION COEFFICIENT BETWEEN THE PREDICTED AND ACTUAL SIGNAL.
# THOSE RESULTS ARE ORGANIZED IN TWO HEAVILY NESTED (ARGUABLY TOO MUCH SO) DICTIONARIES WHERE VALUES
# ARE THE TOP 10,000 CORRELATION COEFFICIENTS, AND SEPARATELY, THE SCHAEFER PARCEL NUMBERS ASSOCIATED WITH 
# THOSE CORRELATION COEFFICIENTS. 


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

def nested_dict():
        return defaultdict(nested_dict)

def convert_back_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: convert_back_to_dict(v) for k, v in d.items()}
    return d

# def convert_to_nifti(values):
#     # working backwards to save correlation values as voxels in MNI space
#     empty_schaefer[cortex_vx] = values
#     empty_mni[brain_idx] = empty_schaefer
#     result_brain = np.reshape(empty_mni, og_shape)

#     # Saving results
#     nifti_result = nib.Nifti1Image(result_brain, affine=atlas.affine, header=atlas.header)
#     nib.save(nifti_result, "test_plotting.nii.gz")
#     return result_brain, nifti_result


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
    
    stat_collection = nested_dict()
    parcel_collection = nested_dict()
    participant_means = nested_dict()
    
    # iterating through participants
    for p in participants:
        print(p)
        datapath = f"{filepath}/{p}"
        files = os.listdir(datapath)
        stat_files = [f for f in files if re.search(stat, f) and not re.search(r'only_regressor|\+', f)] # TODO - will need to update this to look at only regressor and feature+regressor
        
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
            # print("stat vec ", len(stat_vec), stat_vec)
            cutoff = find_cutoff(stat_vec)
            top_voxel_idx = (np.where(stat_vec > cutoff))[0]
            top_voxel_vals = stat_vec[top_voxel_idx]
            
            # Using z-scored correlation coefficients for downstream correlation tests
            z = np.arctanh(stat_vec)
            cutoff = find_cutoff(z)
            top_vals = z[np.where(z > cutoff)[0]]
            participant_means[info.model_name][info.layer][int(p)] = float(np.mean(top_vals))
            

            top_parcels = parcel_nums[top_voxel_idx]
            top_parcels = [int(parcel) for parcel in top_parcels]
            
            # Should I keep one thing consistent, like only look at the best layer for a given model
            # and only the best performing participants?
            # It should depend on the research questions...
            # No I'd like to see across a range of participants
            # I feel like layer 
            parcel_collection[p][info.task][info.model_name][info.n_delays][info.look_ahead][info.layer] = top_parcels
            
            if info.look_ahead not in stat_collection[p][info.task][info.model_name][info.n_delays][info.layer].keys():
                stat_collection[p][info.task][info.model_name][info.n_delays][info.look_ahead][info.layer]  = top_voxel_vals
            else:
                np.append(stat_collection[p][info.task][info.model_name][info.n_delays][info.look_ahead][info.layer], top_voxel_vals)
        #     break
        # break
    
    # print(stat_collection)
    # test = json.loads(json.dumps(parcel_collection))
    # print(test)
    return convert_back_to_dict(participant_means), convert_back_to_dict(stat_collection), convert_back_to_dict(parcel_collection)


def main():
    # iterate through directories, parse the file names, and accumulate stats

    # parsing file names [model_name, task, look_ahead, n_delays, layer, stat]

    filepath = "/data/zachkaras/fmri_model_data/ridge_regression_pca_params"
    participant_means, stat_collection, parcel_collection = iterate_through_participants(filepath, 'correlations')


    # TODO - adjust for the +regressor, no regressor, only regressor results
    with open("/data/zachkaras/fmri_model_data/intermediate_results/no_regressor-top_parcels_per_participant.pkl", 'wb') as f:
        pickle.dump(parcel_collection, f)

    with open("/data/zachkaras/fmri_model_data/intermediate_results/no_regressor-top_correlation_vals_per_participant.pkl", 'wb') as f:
        pickle.dump(stat_collection, f)

    with open("/data/zachkaras/fmri_model_data/intermediate_results/no_regressor-participant_mean_z_correlations.pkl", 'wb') as f:
        pickle.dump(participant_means, f)


if __name__ == "__main__":
    main()
        