import os
import re
import gc
import math
import json
import pickle
import warnings
import numpy as np
import nibabel as nib
from npp import zscore
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge
from collections import defaultdict
from matplotlib.pyplot import figure, cm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

#############################################################################
############## Loading Atlases ##############################################
#############################################################################

# make sure things svae, load, and plot properly

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

#############################################################################
############## Functions ####################################################
#############################################################################
    
# Using cosine similarity as a performance metric between predicted and actual brain signal
def calculate_voxelwise_cosine_similarity(recorded, predicted):
    similarities = []
    for i in range(predicted.shape[1]):
        rec = [recorded[:,i]]
        pre = [predicted[:,i]]
        cos = float((cosine_similarity(rec, pre))[0][0])
        similarities.append(cos)
    return similarities
    
# For plotting purposes, finding the voxels with the highest performance
# TODO - do this for cosine similarity too
def find_top_voxels(corrs):
    max_corrs = corrs.copy()
    max_corrs.sort()
    max_corrs = max_corrs[-5:]
    top_inds = [int(np.where(corrs == c)[0][0]) for c in max_corrs]
    
    return top_inds
    
# Making plots for correlations and predicted vs. actual signal
def make_and_save_plots(outpath, meta_data, bscorrs, fmri_test, stat, predicted_signal, emb_test, keys_test, stat_used):
    
    warnings.filterwarnings("ignore", message="No artists with labels found")
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")
    
    model_name = meta_data[0]
    task = meta_data[1]
    look = meta_data[2]
    delays = meta_data[3]
    layer = (meta_data[5])[:-4]
    
    # Plotting training performance
    f = figure()
    ax = f.add_subplot(1,1,1)
    ax.semilogx( np.logspace(1,4,12), bscorrs.mean(2).mean(1), 'o-')
    plt.savefig(f"{outpath}/{model_name}-{task}-{look}-{delays}-{layer}-training_performance.png", dpi=150)
    
    # Plotting top 5 timecourses along with keystrokes
    top_voxels = find_top_voxels(stat)
    f = figure(figsize=(15,15))
    
    for i in range(1,6):
        
        ax = f.add_subplot(5,1,i)
        selvox = top_voxels[i-1]

        realresp = ax.plot(fmri_test[:,selvox], 'k')[0]
        predresp = ax.plot(zscore(predicted_signal[:,selvox]), 'r')[0]
        ax.set_ylabel(f"Voxel {selvox}")

        ax.set_xlim(0, len(keys_test))
        ax.set_xlabel(f"Time (fMRI time points)")
        ax.legend()
        if i == 5:
            x_labels = list(keys_test.values())
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)

        ax.legend((realresp, predresp), ("Actual response", "Predicted response (scaled)"));
    plt.savefig(f"{outpath}/{model_name}-{task}-{look}-{delays}-{layer}-{stat_used}-top_5_voxels.png", dpi=150)
    plt.close('all')
    gc.collect()


# emb_train, fmri_train, emb_test, fmri_test
def run_ridge_regression(emb_train, fmri_train, emb_test, fmri_test):
    alphas = np.logspace(1, 4, 12) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 5 #1 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(emb_train, fmri_train, emb_test, fmri_test,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)
    
    return wt, corr, bscorrs

# For parallel processing, does all the setup for ridge regression, though the parameter list is grossly long
def ridge_regression_wrapper(emb, participant_embedding_base_path, participant, base_outpath,
                             code_split_point, prose_split_point, 
                             code_fmri_train, code_fmri_test, 
                             prose_fmri_train, prose_fmri_test, 
                             code_keys_test, prose_keys_test):
    
    meta_data = re.split('-', emb)
    model_name = meta_data[0]
    task = meta_data[1]
    look = meta_data[2]
    delays = meta_data[3]
    layer = (meta_data[5])[:-4]
    # task = meta_data[1]
    # layer = (meta_data[3])[:-4]
    # look_ahead = meta_data[2]
    # ndelays = meta_data[3]
    
    # Splitting the data based on the task
    split_point = code_split_point if task == 'code' else prose_split_point
    fmri_train = code_fmri_train if task == 'code' else prose_fmri_train
    fmri_test = code_fmri_test if task == 'code' else prose_fmri_test
    keys_test = code_keys_test if task == 'code' else prose_keys_test
    
    embedding_path = f"{participant_embedding_base_path}/{emb}"
    emb_train, emb_test = load_and_split_embeddings(embedding_path, split_point)
    
    # Running ridge regression here
    print(f"Participant {participant} Ridge Regression for {model_name}, {task}, {layer}, {look}, {delays}")
    weights,corrs,bscorrs = run_ridge_regression(emb_train, fmri_train, emb_test, fmri_test)
    predicted_signal = np.dot(emb_test, weights)
    
    # calculating cosine similarity as a performance metric
    cos_similarities = calculate_voxelwise_cosine_similarity(fmri_test, predicted_signal)
    
    # base_outpath = f"/storage1/fmri_model_data/ridge_regression_pca_models/{participant}"
    # base_outpath = f"/storage1/fmri_model_data/test/{participant}"
    corr_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-correlations.pkl"
    sim_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-cosine_similarities.pkl"
    # std_outfile = f"{base_outpath}/{model_name}-{layer}-{task}-stds.pkl"
    
    
    # Saving model weights and correlation values between predicted and actual timecourses as output
    # UPDATE - not saving model weights for now because the files are huge
    if not os.path.exists(base_outpath):
        os.mkdir(base_outpath)
    
    ### Not saving model weights for now because each file as float64 takes up 18GB
    ###   Each participant has 60 embedding files to test and there are 25 participants
    # with open(weights_outfile, 'wb') as f:
    #     pickle.dump(weights, f)

    with open(corr_outfile, 'wb') as f:
        pickle.dump(corrs, f)
        
    with open(sim_outfile, 'wb') as f:
        pickle.dump(cos_similarities, f)
        
    make_and_save_plots(base_outpath, meta_data, bscorrs, fmri_test, corrs, predicted_signal, emb_test, keys_test, 'correlation')
    make_and_save_plots(base_outpath, meta_data, bscorrs, fmri_test, cos_similarities, predicted_signal, emb_test, keys_test, 'cosine_similarity')
    
    # Trying to free up space
    del weights, corrs, bscorrs, emb_train, emb_test, fmri_train, fmri_test
    gc.collect()
    
    return
    
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
    participant_keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl"
    vols_to_skip_path = f"/storage1/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl"
    
    # try:
    with open(vols_to_skip_path, 'rb') as f:
        vols_to_skip = pickle.load(f)
    # except:
    #     print(f"No data for participant {p}")
    #     return
        
    atlas_voxels_2d,vol_nums = load_and_reshape_fmri_data(participant_fmri_path, vols_to_skip)
    
    fmri_train,fmri_test,split_point = fmri_train_test_split(atlas_voxels_2d)
    
    keys_train,keys_test = load_keystrokes(participant_keystroke_path, split_point, vol_nums)
        
    return fmri_train, fmri_test, keys_train, keys_test, split_point 
        
    
def main():
    # participant_path = f"/storage1/fmri_model_data/fir_vectors_pca"
    participant_path = f"/storage1/fmri_model_data/fir_vectors_pca_params"
    participants = os.listdir(participant_path)
    
    num_participants = len(participants)
    for i,p in enumerate(participants):     
        print(f"Participant {p} ({i+1}/{num_participants}): Loading fMRI data")
        base_outpath = f"/storage1/fmri_model_data/ridge_regression_pca_params/{p}"
        
        if os.path.isdir(base_outpath):
            print(f"Participant output already exists. Skipping")
            continue
        
        try:
            code_fmri_train, code_fmri_test, code_keys_train, code_keys_test, code_split_point = load_task_specific_data(p, 'code')
        except:
            print(f"Could not find data for participant {p} on code")
        
        try:
            prose_fmri_train, prose_fmri_test, prose_keys_train, prose_keys_test, prose_split_point = load_task_specific_data(p, 'prose')
        except:
            print(f"Could not find date for participant {p} on prose")
            
        participant_embedding_base_path = f"/storage1/fmri_model_data/fir_vectors_pca_params/{p}"
        embeddings = os.listdir(participant_embedding_base_path)
        num_embeddings = len(embeddings)

        base_outpath = f"/storage1/fmri_model_data/ridge_regression_pca_params/{p}"

        n_workers = 36
        # os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

        print(f"Participant {p} ({i+1}/{num_participants}): "
            f"processing {num_embeddings} embeddings with {n_workers} workers")

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    ridge_regression_wrapper, 
                    emb, participant_embedding_base_path, p, base_outpath,
                    code_split_point, prose_split_point,
                    code_fmri_train, code_fmri_test,
                    prose_fmri_train, prose_fmri_test,
                    code_keys_test,prose_keys_test
                ): emb
                for emb in embeddings
            }

            for ii, fut in enumerate(as_completed(futures), start=1):
                emb = futures[fut]
                try:
                    fut.result()
                    meta_data = re.split('-', emb)
                    model_name = meta_data[0]
                    task = meta_data[1]
                    layer = (meta_data[3])[:-4]

                except Exception as e:
                    print(f"Error for {emb} (participant {p}): {e}")
                # break
        # break

if __name__ == "__main__":
    main()
