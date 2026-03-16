import os
import re
import pickle
import numpy as np
import nibabel as nib
from scipy import stats
from collections import defaultdict

# load in atlases
atlas_base_path = "/home/zachkaras/fmri_model/analysis/pipeline/atlases"

# read in 2d mni mask
mask = nib.load(f"{atlas_base_path}/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx] # contains the schaefer parcel numbers
cortex_vx = np.where(atlas_only_brain != 0)[0] # 131k
parcel_nums = atlas_only_brain[cortex_vx]

right_hemi = np.where(parcel_nums > 200) # 65k
left_hemi = np.where(parcel_nums <= 200) # 65k


# Making empty templates to save output
empty_cortex = np.zeros(cortex_vx.shape)
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)


def find_cutoff(vec, threshold = 10**4):
    
    # I can find the threshold point based on sorting, then keep everything in the same place
    copy = vec.copy()
    copy.sort()
    cutoff = copy[-threshold-1]
    return cutoff

def zscore_corr_coefficients(corrs):
    return np.arctanh(corrs)

def nested_dict():
    return defaultdict(nested_dict)


def calculate_hemi_means(top_voxel_idx, top_voxel_vals):
    temp_cx = empty_cortex.copy()
    temp_cx[top_voxel_idx] = top_voxel_vals

    top_left_vx  = temp_cx[left_hemi]
    top_left_filtered_vx = top_left_vx[np.where(top_left_vx > 0)[0]]
    
    top_right_vx = temp_cx[right_hemi]
    top_right_filtered_vx = top_right_vx[np.where(top_right_vx > 0)[0]]

    left_mean = float(np.mean(top_left_filtered_vx))
    right_mean = float(np.mean(top_right_filtered_vx))
    
    return left_mean, right_mean


# load in best performing models
def iterate_through_models():

    best_models = ['code-deepseek_6b-ndelays_4-look_ahead_by_10', 
                'code-codegemma_7b-ndelays_4-look_ahead_by_10',
                'code-codegemma_7b-ndelays_16-look_ahead_by_0', 
                'prose-codegemma_7b-ndelays_10-look_ahead_by_5', 
                'prose-deepseek_6b-ndelays_20-look_ahead_by_10']

    participant_base_path = "/data/zachkaras/fmri_model_data/ridge_regression_pca_params"
    participants = os.listdir(participant_base_path)

    means = nested_dict()
        
    for m in best_models:
        for p in participants:
            
            files = os.listdir(f"{participant_base_path}/{p}")
            parts = m.split('-')
            task,model,delays,look_ahead = parts[0],parts[1],parts[2],parts[3]
            
            # finding only the relevant filenames
            pattern = f"{model}-{task}-{look_ahead}-{delays}"
            best_files = [f for f in files if re.search(pattern, f) and f.endswith('correlations.pkl')]

            for bf in best_files:

                layer_num = int(re.search(r'layer_(\d+)', bf).group(1)) 
                
                if layer_num not in list(means[m].keys()):
                    means[m][layer_num] = {'left': [], 'right': []}

                datapath = f"{participant_base_path}/{p}/{bf}"
                
                with open(datapath, 'rb') as f:
                    nii = pickle.load(f)

                # z-score - fisher's z transformation since they're correlation coefficients
                z_scored = zscore_corr_coefficients(nii) # 131k


                cutoff = find_cutoff(z_scored) #, (len(z_scored)-1)) # also not significant for all the voxel correlation values
                
                top_voxel_idx = (np.where(z_scored > cutoff))[0] # 10k
                top_voxel_vals = z_scored[top_voxel_idx]
                
                left_mean,right_mean = calculate_hemi_means(top_voxel_idx, top_voxel_vals)

                means[m][layer_num]['left'].append(left_mean)
                means[m][layer_num]['right'].append(right_mean)

    return means

                                                                                                                                                                                                                                                                                 
def calculate_stats(means):
    results = nested_dict()
                            
    for model, layer_data in means.items():                                                                                                                                                                                                                                    
        for layer_num, hemi_data in layer_data.items():
            left  = np.array(hemi_data['left'])                                                                                                                                                                                                                                
            right = np.array(hemi_data['right'])                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                
            t, p = stats.ttest_rel(left, right)                                                                                                                                                                                                                                
            results[model][layer_num] = {                                                                                                                                                                                                                                      
                't': t,                  
                'p': p,                                                                                                                                                                                                                                                        
                'left_mean':  float(np.mean(left)),                                                                                                                                                                                                                            
                'right_mean': float(np.mean(right)),                                                                                                                                                                                                                           
            }                                                                                                                                                                                                                                                                  
            print(f"{model} | layer {layer_num:2d} | "                                                                                                                                                                                                                         
                f"L={np.mean(left):.4f}, R={np.mean(right):.4f} | "                                                                                                                                                                                                          
                f"t={t:.3f}, p={p:.4f}")                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                
    return results  



def main():
    means = iterate_through_models()
    calculate_stats(means)
    # looks like these results are basically not significant - I don't think they will survive correction for multiple comparisons
    # that's just t-tests based on the mean correlation coefficients
    
    # It looks like there are more spikes for the higher performing participants - is there a way to statistically check that?
    # It looks like there are also certain regions that are better modeled by specific layers - the best performing layers show clusters


if __name__ == "__main__":
    main()