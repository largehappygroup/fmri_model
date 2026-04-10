import os
import re
import gc
import math
import json
import pickle
# import logging
import warnings
import numpy as np
import argparse
import nibabel as nib
from npp import zscore
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge
from collections import defaultdict
from matplotlib.pyplot import figure, cm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

# ALLOWED_CORES = list(range(0,40))
ALLOWED_CORES = list(range(0,32))

parser = argparse.ArgumentParser(description="Toggling GPU usage")
parser.add_argument("--gpu", required=False, default=False, help="Set to True if using the GPU.")
# logger = logging.getLogger(__name__)

args = parser.parse_args()
using_gpu = int(args.gpu)
# print(using_gpu, type(using_gpu))

#############################################################################
############## Loading Atlases ##############################################
#############################################################################

# make sure things svae, load, and plot properly

# atlas_base_path = "/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases"
atlas_base_path = "/home/zachkaras/fmri_model/analysis/pipeline/atlases"

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
    '''
    To select the ridge parameter independently for each voxel, we used 50 iterations
    of cross-validation. Since fMRI data is auto-correlated, for
    each cross-validation run we randomly sampled 40 different
    chunks of the training data, each totaling over 4 minutes.
    The training set comprised 26 stories, totaling 5.4 hours.
    '''
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(emb_train, fmri_train, emb_test, fmri_test,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)

    return wt, corr, bscorrs

def calculate_R2(predicted_signal, actual_signal):
    SS_res = np.sum((actual_signal - predicted_signal) ** 2, axis=0)
    SS_tot = np.sum((actual_signal - actual_signal.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - (SS_res / SS_tot)
    return R2
    

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
    layer = meta_data[5]
    regressor = (meta_data[6])[:-4]

    if re.match("only_regressor", regressor):
        return

    # Splitting the data based on the task
    split_point = code_split_point if task == 'code' else prose_split_point
    fmri_train = code_fmri_train if task == 'code' else prose_fmri_train
    fmri_test = code_fmri_test if task == 'code' else prose_fmri_test
    keys_test = code_keys_test if task == 'code' else prose_keys_test
    
    embedding_path = f"{participant_embedding_base_path}/{emb}"
    emb_train, emb_test = load_and_split_embeddings(embedding_path, split_point)

    # Running ridge regression here
    print(f"Participant {participant} Ridge Regression for {model_name}, {task}, {layer}, {look}, {delays}")

    # I need ridge regression run for full model (regressor+features) and also for the base model (just features)
    # I have the ridge regression models trained for the base model, but I need the full model for all participants
    # because I previously ran just the base model. I'll use the full model to show all the stats
    # and I need to rerun the base model for certain models and parameter configurations to recalculate R^2

    # So right now, I need to run the full model for all participants
    # I'll rerun the base model afterwards for the best model configurations to calculate R^2 values for that

    # base_outpath = f"/s1/fmri_model_data/ridge_regression_pca_models/{participant}"
    # base_outpath = f"/s1/fmri_model_data/test/{participant}"
    corr_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-{regressor}-correlations.pkl"
    sim_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-{regressor}-cosine_similarities.pkl"
    # weights_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-model_weights.pkl"
    keys_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-test_keystrokes.pkl"
    R2_outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-{regressor}-R2.pkl"

    if os.path.isfile(corr_outfile) and os.path.isfile(sim_outfile) and os.path.isfile(keys_outfile) and os.path.isfile(R2_outfile):
        return
    
    weights,corrs,bscorrs = run_ridge_regression(emb_train, fmri_train, emb_test, fmri_test)
    predicted_signal = np.dot(emb_test, weights)
    # calculating cosine similarity as a performance metric
    cos_similarities = calculate_voxelwise_cosine_similarity(fmri_test, predicted_signal)

    R2 = calculate_R2(predicted_signal, fmri_test)
    
    
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
        
    with open(keys_outfile, 'wb') as f:
        pickle.dump(keys_test, f)
        
    with open(R2_outfile, 'wb') as f:
        pickle.dump(R2, f)
        
    # make_and_save_plots(base_outpath, meta_data, bscorrs, fmri_test, corrs, predicted_signal, emb_test, keys_test, 'correlation')
    # make_and_save_plots(base_outpath, meta_data, bscorrs, fmri_test, cos_similarities, predicted_signal, emb_test, keys_test, 'cosine_similarity')
    
    # Trying to free up space
    del weights, corrs, bscorrs, emb_train, emb_test, fmri_train, fmri_test
    gc.collect()
    
    return
    
def load_and_split_embeddings(embedding_path, split_point):
    
    with open(embedding_path, 'rb') as f:
        embedding = pickle.load(f)

        if using_gpu == 1:
            import cupy as cp
            embedding = cp.array(embedding)
            # print("embedding should be converted")

    # print("embedding type: ", type(embedding))
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
    zPresp = reshaped_scan[split_point:, :] # zPresp is fMRI data for prediction
    
    return zRresp, zPresp, split_point
    
def load_and_reshape_fmri_data(fmripath, vols_to_skip):
    fmri_data = nib.load(fmripath)
    scan = fmri_data.get_fdata()

    scan_2d = (np.reshape(scan, [scan.shape[0]*scan.shape[1]*scan.shape[2], scan.shape[3]]))
    
    if using_gpu == 1:
        import cupy as cp
        scan_2d = cp.asarray(scan_2d)
        # print("scan should be converted")
    
    # print("scan type", type(scan_2d))
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
    base = "/data/zachkaras"
    participant_fmri_path = f"{base}/fmri_model_data/clean_{task}/{p}.nii.gz"
    # participant_keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl" # cumberland path
    participant_keystroke_path = f"/home/zachkaras/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl" # behemoth path
    vols_to_skip_path = f"{base}/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl"
    
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

def init_worker():
    os.sched_setaffinity(0,ALLOWED_CORES)
        
    
def main():
    # participant_path = f"/s1/fmri_model_data/fir_vectors_pca"
    # participant_path = f"/s1/fmri_model_data/fir_vectors_pca_params"
    base = "/data2/zachkaras"
    participant_path = f"{base}/fmri_model_data/fir_vectors_pca_params" # behemoth path

    participants = os.listdir(participant_path)
    
    num_participants = len(participants)
    for i,p in enumerate(participants):     
        print(f"Participant {p} ({i+1}/{num_participants}): Loading fMRI data")
        base_outpath = f"/data/zachkaras/fmri_model_data/ridge_regression_pca_params/{p}"
        continue    
        # if os.path.isdir(base_outpath):
        #     print(f"Participant output already exists. Skipping")
        #     continue
        
        try:
            code_fmri_train, code_fmri_test, code_keys_train, code_keys_test, code_split_point = load_task_specific_data(p, 'code')
        except:
            code_fmri_train,code_fmri_test,code_keys_test, code_split_point = None, None, None, None
            print(f"Could not find data for participant {p} on code")
        
        try:
            prose_fmri_train, prose_fmri_test, prose_keys_train, prose_keys_test, prose_split_point = load_task_specific_data(p, 'prose')
        except:
            prose_fmri_train, prose_fmri_test, prose_keys_test, prose_split_point = None, None, None, None
            print(f"Could not find date for participant {p} on prose")
            
        participant_embedding_base_path = f"{base}/fmri_model_data/fir_vectors_pca_params/{p}"
        embeddings = os.listdir(participant_embedding_base_path)
        embeddings = [e for e in embeddings if re.search(r'regressor\+features', e)]

        # filtered to best models by only copying the relevant ones over
        # this was after I moved the data to cumberland since it's so huge

        # embeddings = [e for e in embeddings if re.search('no_regressor', e)] # adding to see the influence of the regressor on performance 2/15/2026
        
        num_embeddings = len(embeddings)

        base_outpath = f"/data2/zachkaras/fmri_model_data/ridge_regression_pca_params/{p}"

        # os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

        print(f"Participant {p} ({i+1}/{num_participants}): "
            f"processing {num_embeddings} embeddings with {len(ALLOWED_CORES)} workers")

        with ProcessPoolExecutor(max_workers=len(ALLOWED_CORES), initializer=init_worker) as ex:
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
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
