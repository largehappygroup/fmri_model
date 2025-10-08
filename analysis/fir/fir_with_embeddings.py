import os
import re
import math
import json
import pickle
import numpy as np
import nibabel as nib
from npp import zscore
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge
from collections import defaultdict
from matplotlib.pyplot import figure, cm

#############################################################################
############## Loading Atlases ##############################################
#############################################################################

atlas_base_path = "/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases"

# read in 2d mni mask
mask = nib.load(f"{atlas_base_path}/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx]
cortex = np.where(atlas_only_brain != 0)[0]

# Making empty templates to save output
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)
    
# def create_histogram(voxcorrs):
#     f = plt.figure(figsize=(8,8))
#     ax = f.add_subplot(1,1,1)
#     ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
#     ax.set_xlabel("Correlation")
#     ax.set_ylabel("Num. voxels");
#     plt.savefig() # TODO

def save_correlation_coefficients_as_nifti(voxcorrs):
    # working backwards to save correlation values as voxels in MNI space
    empty_schaefer[cortex] = voxcorrs
    empty_mni[brain_idx] = empty_schaefer
    result_brain = np.reshape(empty_mni, og_shape)

    # Saving results
    nifti_result = nib.Nifti1Image(result_brain, affine=atlas.affine, header=atlas.header)
    nib.save(nifti_result, "layer_0_clean.nii.gz")
    
def find_top_voxels(corrs):
    max_corrs = corrs.copy()
    max_corrs.sort()
    max_corrs = max_corrs[-5:]
    top_inds = [int(np.where(corrs == c)[0][0]) for c in max_corrs]
    
    return top_inds
    
def make_and_save_plots(outpath, bscorrs, fmri_test, corrs, weights, emb_test, keys_test):
    
    # Plotting training performance
    f = figure()
    ax = f.add_subplot(1,1,1)
    ax.semilogx( np.logspace(1,4,12), bscorrs.mean(2).mean(1), 'o-')
    plt.savefig(f"{outpath}/training_performance.png", dpi=150)
    
    top_voxels = find_top_voxels(corrs)
    pred = np.dot(emb_test, weights)
    
    f = figure(figsize=(15,15))
    
    for i in range(1,6):
        
        ax = f.add_subplot(5,1,i)
        selvox = top_voxels[i-1]

        realresp = ax.plot(fmri_test[:,selvox], 'k')[0]
        predresp = ax.plot(zscore(pred[:,selvox]), 'r')[0]
        ax.set_ylabel(f"Voxel {selvox}")

        ax.set_xlim(0, len(keys_test))
        ax.set_xlabel(f"Time (fMRI time points)")
        ax.legend()
        x_labels = list(keys_test.values())

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)

        ax.legend((realresp, predresp), ("Actual response", "Predicted response (scaled)"));
    plt.savefig(f"{outpath}/top_5_voxels.png", dpi=150)


    
def run_ridge_regression(delRstim, zRresp, delPstim, zPresp):
    alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 1 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20

    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)
    
    return wt, corr, bscorrs
    
def load_and_split_embeddings(embedding_path, split_point):
    
    with open(embedding_path, 'rb') as f:
        embedding = pickle.load(f)
        
    delRstim = embedding[:split_point, :] # delRstim from pickle files
    delPstim = embedding[split_point:, :] # delPstim is prediction
    
    return delRstim, delPstim

def load_keystrokes(keystroke_path, split_point, vol_nums):
    with open(keystroke_path, 'rb') as f:
        keystrokes = pickle.load(f)
    
    filtered_keystrokes = {kv[0]:kv[1] for i,kv in enumerate(keystrokes.items()) if vol_nums[i] == 1}
    
    Rkeys = {kv[0]:kv[1] for i,kv in enumerate(filtered_keystrokes.items()) if i <= split_point}
    Pkeys = {kv[0]:kv[1] for i,kv in enumerate(filtered_keystrokes.items()) if i > split_point}
    
    return Rkeys,Pkeys
    

def fmri_train_test_split(reshaped_scan):
    
    split_point = math.floor((reshaped_scan.shape)[0]*0.9)
    zRresp = reshaped_scan[:split_point,:] # zRresp is fMRI data
    zPresp = reshaped_scan[split_point:, :] # zPresp is fMRI data from prediction
    
    return zRresp, zPresp, split_point
    
def load_and_reshape_fmri_data(fmripath, vols_to_skip):
    fmri_data = nib.load(fmripath)
    scan = fmri_data.get_fdata()

    scan_2d = (np.reshape(scan, [scan.shape[0]*scan.shape[1]*scan.shape[2], scan.shape[3]]))

    means = scan_2d.mean(axis=1, keepdims=True)
    stds = scan_2d.std(axis=1, keepdims=True)

    z_scored_2d_scan = (scan_2d - means) / np.where(stds==0, 1, stds)

    # Only looking at voxels in the MNI brain
    scan_2d_brain = z_scored_2d_scan[brain_idx, :]

    # Only looking at voxels labeled in the Schaefer Atlas
    scan_2d_schaefer = scan_2d_brain[cortex,:]
    scan_2d_schaefer = scan_2d_schaefer.T
    
    vol_nums = np.ones((scan_2d_schaefer.shape)[0])
    vol_nums[vols_to_skip] = 0
    
    scan_2d_schaefer_filtered = scan_2d_schaefer[np.where(vol_nums == 1)]
    
    return scan_2d_schaefer_filtered, vol_nums

def load_task_specific_data(p, task):
    participant_fmri_path = f"/storage1/fmri_model_data/clean_{task}/{p}.nii.gz"
        
    # participant_embedding_base_path = f"/storage1/fmri_model_data/fir_vectors/{p}"
    participant_keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl"
    vols_to_skip_path = f"/storage1/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl"
    
    with open(vols_to_skip_path, 'rb') as f:
        vols_to_skip = pickle.load(f)
        
    atlas_voxels_2d,vol_nums = load_and_reshape_fmri_data(participant_fmri_path, vols_to_skip)
    
    fmri_train,fmri_test,split_point = fmri_train_test_split(atlas_voxels_2d)
    
    keys_train,keys_test = load_keystrokes(participant_keystroke_path, split_point, vol_nums)
        
    return fmri_train, fmri_test, keys_train, keys_test, vol_nums, split_point 
        
    
def main():
    participant_path = f"/storage1/fmri_model_data/fir_vectors"
    participants = os.listdir(participant_path)
    
    def nested_dict():
        return defaultdict(nested_dict)
      
    all_corr_means = nested_dict()
    all_corr_stds = nested_dict()
    
    num_participants = len(participants)
    for i,p in enumerate(participants):     
        # task specific - fMRI, keystrokes, vols_to_skip
        code_fmri_train, code_fmri_test, code_keys_train, code_keys_test, code_split_point = load_task_specific_data(p, 'code')
        prose_fmri_train, prose_fmri_test, prose_keys_train, prose_keys_test, prose_split_point = load_task_specific_data(p, 'prose')
        
        participant_embedding_base_path = f"/storage1/fmri_model_data/fir_vectors/{p}"
        embeddings = os.listdir(participant_embedding_base_path)
        num_embeddings = len(embeddings)
        for ii,emb in enumerate(embeddings):
            meta_data = re.split('-', emb)
            model_name = meta_data[0]
            task = meta_data[1]
            layer = (meta_data[3])[:-4]
            
            split_point = code_split_point if task == 'code' else prose_split_point
            fmri_train = code_fmri_train if task == 'code' else prose_fmri_train
            fmri_test = code_fmri_test if task == 'code' else prose_fmri_test
            keys_test = code_keys_test if task == 'code' else prose_keys_test
            
            embedding_path = f"{participant_embedding_base_path}/{emb}"
            emb_train, emb_test = load_and_split_embeddings(embedding_path, split_point) # TODO task specific split point
            
            print(f"Participant {p} ({i+1}/{num_participants}) Ridge Regression for {model_name}, {layer} ({ii+1}/{num_embeddings})")            
            weights,corrs, bscorrs = run_ridge_regression(fmri_train, emb_train)
            
            
            base_outpath = f"/storage1/fmri_model_data/ridge_regression_models/{p}"
            weights_outfile = f"{base_outpath}/{model_name}-{layer}-{task}-model_weights.pkl" # saving for specific model, layer, and task
            corr_outfile = f"{base_outpath}/{model_name}-{layer}-{task}-correlations.pkl"
            
            make_and_save_plots(base_outpath, bscorrs, fmri_test, corrs, weights, emb_test, keys_test)
            
            # Saving model weights and correlation values between predicted and actual timecourses as output
            if not os.path.exists(base_outpath):
                os.mkdir(base_outpath)
            
            with open(weights_outfile, 'wb') as f:
                pickle.dump(weights, f)
            
            with open(corr_outfile, 'wb') as f:
                pickle.dump(corrs, f)
                
            # Saving summary statistics from training performance
            corrs_mean = np.mean(corrs)
            corrs_std = np.std(corrs)
            
            all_corr_means[p][task][model_name][layer] = corrs_mean
            all_corr_stds[p][task][model_name][layer] = corrs_std
    
    # converting from default dictionaries to regular dictionaries
    all_corr_means = json.loads(json.dumps(all_corr_means))
    all_corr_stds  = json.loads(json.dumps(all_corr_stds))
    
    with open("results/all_corr_means.pkl", 'wb') as f:
        pickle.dump(all_corr_means, f)
    
    with open(f"results/all_corr_standard_deviations.pkl", 'wb') as f:
        pickle.dump(all_corr_stds, f)

if __name__ == "__main__":
    main()