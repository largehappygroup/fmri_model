import os
import math
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# from ridge import bootstrap_ridge

atlas_base_path = "/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases"

# read in 2d mni mask
mask = nib.load(f"{atlas_base_path}/MNI152_T1_2mm_brain_mask.nii.gz")
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"{atlas_base_path}/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx]
cortex = np.where(atlas_only_brain != 0)[0]

participant_data_path = "/home/zachkaras/fmri/fmri_model_data/midprocess/133/filtered_func_data_clean.nii.gz"
fmri_data = nib.load(participant_data_path)
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

scan_2d_schaefer.shape

split_point = math.floor((scan_2d_schaefer.shape)[0]*0.9)
zRresp = scan_2d_schaefer[:split_point,:]
zPresp = scan_2d_schaefer[split_point:, :]

    
embedding_path = '/home/zachkaras/fmri/starcoder2_7b_code_fir_embeddings.pkl'

with open(embedding_path, 'rb') as f:
    embedding = pickle.load(f)
    

with open("midprocess/133/code_formatted_keystrokes.pkl", 'rb') as f:
    keystrokes = pickle.load(f)

# define numpy array where rows are time points and columns are LLM vectors
# delRstim from pickle files


# zRresp is fMRI data


# delPstim is prediction


# zPresp is fMRI data from prediction


alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
nboots = 1 # Number of cross-validation runs.
chunklen = 40 # 
nchunks = 20

# wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
#                                                      alphas, nboots, chunklen, nchunks,
#                                                      singcutoff=1e-10, single_alpha=True)