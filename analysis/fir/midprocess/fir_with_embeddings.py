import os
import math
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge

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

# scan_2d_schaefer.shape

split_point = math.floor((scan_2d_schaefer.shape)[0]*0.9)
zRresp = scan_2d_schaefer[:split_point,:]
zPresp = scan_2d_schaefer[split_point:, :]

    
# embedding_path = '/home/zachkaras/fmri/starcoder2_7b_code_fir_embeddings.pkl'
embedding_path = '/storage1/fmri_model_data/fir_vectors/test/starcoder2_7b_code_fir_embedding_layer_0.pkl'


with open(embedding_path, 'rb') as f:
    embedding = pickle.load(f)
    
delRstim = embedding[:split_point, :]
delPstim = embedding[split_point:, :]
    

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

wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                     alphas, nboots, chunklen, nchunks,
                                                     singcutoff=1e-10, single_alpha=True)


pred = np.dot(delPstim, wt)
voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
for vi in range(zPresp.shape[1]):
    voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
    
    
f = plt.figure(figsize=(8,8))
ax = f.add_subplot(1,1,1)
ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
ax.set_xlabel("Correlation")
ax.set_ylabel("Num. voxels");


# working backwards to save correlation values as voxels in MNI space
empty_schaefer[cortex] = voxcorrs
empty_mni[brain_idx] = empty_schaefer
result_brain = np.reshape(empty_mni, og_shape)

# Saving results
nifti_result = nib.Nifti1Image(result_brain, affine=atlas.affine, header=atlas.header)
nib.save(nifti_result, "layer_0_clean.nii.gz")