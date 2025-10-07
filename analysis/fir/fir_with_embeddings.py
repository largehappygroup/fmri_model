import os
import re
import math
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge

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

# with open("midprocess/133/code_formatted_keystrokes.pkl", 'rb') as f:
#     keystrokes = pickle.load(f)

    
def create_histogram(voxcorrs):
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(1,1,1)
    ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Num. voxels");
    plt.savefig() # TODO

def save_correlation_coefficients_as_nifti(voxcorrs):
    # working backwards to save correlation values as voxels in MNI space
    empty_schaefer[cortex] = voxcorrs
    empty_mni[brain_idx] = empty_schaefer
    result_brain = np.reshape(empty_mni, og_shape)

    # Saving results
    nifti_result = nib.Nifti1Image(result_brain, affine=atlas.affine, header=atlas.header)
    nib.save(nifti_result, "layer_0_clean.nii.gz")
    
def run_ridge_regression(delRstim, zRresp, delPstim, zPresp):
    alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 1 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20

    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)
    
    # predicted timecourses
    pred = np.dot(delPstim, wt)
    
    # # calculating correlation coefficient between predicted and actual timecourse
    # voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
    # for vi in range(zPresp.shape[1]):
    #     voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
    
    return wt, corr
    
def load_and_split_embeddings(embedding_path, split_point):
    # embedding_path = '/home/zachkaras/fmri/starcoder2_7b_code_fir_embeddings.pkl'
    # embedding_path = '/storage1/fmri_model_data/fir_vectors/test/starcoder2_7b_code_fir_embedding_layer_0.pkl'

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
    # participant_data_path = "/home/zachkaras/fmri/fmri_model_data/midprocess/133/filtered_func_data_clean.nii.gz"
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
    
# def iterate_through_participants(participants, task):

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
    
    # iterate_through_participants(participants, 'code')
    # iterate_through_participants(participants, 'prose')
    
    
    # TODO - keep track of stats here
    # for each model, average correlation coefficient, standard deviation of correlation values
    
    # make plots for predicted voxel activity and the corresponding 
    # can choose voxels with the 5 highest correlation values
    for p in participants:
        # participant_fmri_path = f"/storage1/fmri_model_data/clean_code/{p}.nii.gz"
        
        # task specific - fMRI, keystrokes, vols_to_skip
        code_fmri, code_keystrokes, code_vols_to_skip, code_split_point = load_task_specific_data(p, 'code')
        prose_fmri, prose_keystrokes, prose_vols_to_skip, prose_split_point = load_task_specific_data(p, 'prose')
        
        participant_embedding_base_path = f"/storage1/fmri_model_data/fir_vectors/{p}"
        embeddings = os.listdir(participant_embedding_base_path)
        
        # participant_fmri_path = f"/storage1/fmri_model_data/clean_{task}/{p}.nii.gz"
        
        # participant_keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl"
        # vols_to_skip_path = f"/storage1/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl"
        
        # with open(vols_to_skip_path, 'rb') as f:
        #     vols_to_skip = pickle.load(f)
        
        # atlas_voxels_2d_code,code_vol_nums = load_and_reshape_fmri_data(participant_code_fmri_path, vols_to_skip)
        # fmri_train,fmri_test,split_point = fmri_train_test_split(atlas_voxels_2d)
        
        # keys_train,keys_test = load_keystrokes(participant_keystroke_path, split_point, vol_nums)
        
        
        
        for emb in embeddings:
            embedding_path = f"{participant_embedding_base_path}/{emb}"
            emb_train, emb_test = load_and_split_embeddings(embedding_path)
            
            meta_data = re.split('-', emb)
            model_name = meta_data[0]
            task = meta_data[1]
            layer = meta_data[2]
            
            print(model_name, task, layer)
            
            # weights,corrs = run_ridge_regression(fmri_train, emb_train)
            
            # there are multiple models
            # and multiple layers within each participant's directory
            # what am I trying to do with the embeddings for each?
            # No matter what the embeddings are, I'll be running ridge regression, 
            # then saving the model weights, corrs
            
            # Is there a way to separate out navigation from logic from syntax?
            # can make a regressor for keypresses
            # 
            pass
    # load in files
    # process fMRI
    # load in embeddings
    # iterate through everything
    # save output

if __name__ == "__main__":
    main()