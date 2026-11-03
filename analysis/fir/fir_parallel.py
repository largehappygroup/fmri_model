import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
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
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import correlation_score, r2_score, r2_score_split
# from ridge import bootstrap_ridge
from collections import defaultdict
from matplotlib.pyplot import figure, cm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

# TODO - differences between the two tasks
# TODO - for each voxel, mean prediction accuracy over participants
# TODO - correct correlations for multiple comparisons
# TODO - motion as another regressor?
# TODO - no flatmaps
# TODO - trends across layers. variance explained?

# RQs
# How well does the approach work (compared to baselines and confounds)
# Where does it perform well - intersections between participants? Keep as voxels? Check overlap between participants
#   check for confounds (on/off structure of task)
# Layer investigation

# Necessities:
# BERT and GPT-2 as baselines
# I want to answer the question of whether modern LLMs represent different information in different layers
# I suspect that they don't. I can compute the variance explained. 
# I can compare against BERT and GPT-2 and prior research
# the linguistic information encoded within - I'd also need to compute this for code somehow

# Confounding factors

# ALLOWED_CORES = list(range(0,40))
ALLOWED_CORES = list(range(0,25))
NUM_GPU_WORKERS = 4 # processes sharing the GPU; each one batches over all voxels already

parser = argparse.ArgumentParser(description="Toggling GPU usage")
parser.add_argument("--gpu", required=False, default=0, help="Set to 1 if using the GPU.")
# logger = logging.getLogger(__name__)

args = parser.parse_args()
using_gpu = int(args.gpu)
# print(using_gpu, type(using_gpu))

# himalaya moves arrays to the GPU itself, so the data can stay as numpy arrays
backend = set_backend("cupy" if using_gpu == 1 else "numpy")

#############################################################################
############## Loading Atlases ##############################################
#############################################################################

# make sure things save, load, and plot properly

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

def get_band_slices(features, regressors):
    feature_names = ["regressors", "features"]
    features_width = features.shape[1]
    regressors_width = regressors.shape[1]
    return {
        "regressors": slice(0, regressors_width),
        "features": slice(regressors_width, regressors_width+features_width)
    }

# Banded ridge regression: the regressor and the LLM features each get their own regularization,
# so the single regressor column isn't drowned out by (or forced to share an alpha with) thousands of features
def run_banded_ridge_regression(X_train, Y_train, X_test, Y_test, band_slices):
    # n_iter - 100, tutorial uses 20 but says you could/should use more. Another repo uses 100 
    # alphas - np.logspace(1,20,20) - zaid zada paper
    # local_alpha - True, default is True and repos I've found don't overwrite to False
    # score_func - can use default of l2_neg_loss
    # fit_intercept - False because we use centering in make_pipeline
    # n_targets_batch
    # n_alphas_batch
    # n_targets_batch_refit
    solver_params = dict(n_iter=100, alphas=np.logspace(1, 20, 20), n_targets_batch=2000, n_alphas_batch=20, n_targets_batch_refit=2000)

    # fMRI data is autocorrelated, so CV folds are contiguous blocks of time rather than shuffled TRs
    cv = KFold(n_splits=5, shuffle=False)

    column_kernelizer = ColumnKernelizer([(name, Kernelizer(kernel="linear"), idx)
                                          for name, idx in band_slices.items()])
    model = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                                  solver_params=solver_params, cv=cv)

    pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=False), column_kernelizer, model)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    Y_train = Y_train.astype("float32")
    Y_test = Y_test.astype("float32")

    pipeline.fit(X_train, Y_train)

    predicted_signal = pipeline.predict(X_test)
    predicted_split = pipeline.predict(X_test, split=True) # per-band predictions, (n_bands, n_TRs, n_voxels)

    corrs = backend.to_numpy(correlation_score(Y_test, predicted_signal))
    R2 = backend.to_numpy(r2_score(Y_test, predicted_signal))
    # How much of the R^2 is attributable to each band. r2_score_split needs zero-mean data, and the fMRI
    # was z-scored over the whole run, so the test segment is centered first (bands sum to R^2 of centered test data)
    R2_split = backend.to_numpy(r2_score_split(Y_test - Y_test.mean(0), predicted_split))
    R2_split = {name: R2_split[i] for i, name in enumerate(band_slices)}
    cv_scores = backend.to_numpy(model.cv_scores_) # (n_iter, n_voxels)
    deltas = backend.to_numpy(model.deltas_)
    band_alphas = np.exp(-deltas)  #{name: float(np.exp(-deltas[i, :])) for i, name in enumerate(band_slices)}
    corrs = np.nan_to_num(corrs)

    return corrs, R2, R2_split, cv_scores, band_alphas


# For parallel processing, does all the setup for ridge regression, though the parameter list is grossly long
def ridge_regression_wrapper(feat, participant_embedding_base_path, participant, base_outpath,
                             code_split_point, prose_split_point, 
                             code_fmri_train, code_fmri_test, 
                             prose_fmri_train, prose_fmri_test, 
                             code_keys_test, prose_keys_test):
    
    meta_data = re.split('-', feat)
    model_name = meta_data[0]
    task = meta_data[1]
    look = meta_data[2]
    delays = meta_data[3]
    layer = meta_data[5]
    regressor = (meta_data[6])[:-4]
    
    outfile = f"{base_outpath}/{model_name}-{task}-{look}-{delays}-{layer}-{regressor}-output.npz"
    if os.path.isfile(outfile):
        return

    # Splitting the data based on the task
    split_point = code_split_point if task == 'code' else prose_split_point
    fmri_train = code_fmri_train if task == 'code' else prose_fmri_train
    fmri_test = code_fmri_test if task == 'code' else prose_fmri_test
    keys_test = code_keys_test if task == 'code' else prose_keys_test
    
    feature_path = f"{participant_embedding_base_path}/{feat}"
    regressor_path = feature_path[:-len('features.pkl')] + 'regresors.pkl'
    # re.sub('features', 'regressors', feature_path)

    X_train, X_test, band_slices = load_and_split_embeddings(feature_path, regressor_path, split_point)
    # n_delays = int(re.search(r'ndelays_(\d+)', delays).group(1))

    print(f"Participant {participant} Ridge Regression for {model_name}, {task}, {layer}, {look}, {delays}")
    corrs,R2,R2_split,cv_scores,band_alphas = run_banded_ridge_regression(X_train, fmri_train, X_test, fmri_test, band_slices)
    np.savez_compressed(outfile, 
                        correlations=corrs, 
                        r2=R2, 
                        r2_split=np.stack([R2_split[n] for n in band_slices]), 
                        band_labels=np.array(list(band_slices)), 
                        band_alphas=band_alphas)
    del corrs, cv_scores, X_train, X_test, fmri_train, fmri_test
    gc.collect()
    
def load_and_split_embeddings(feature_path, regressor_path, split_point):
    
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    with open(regressor_path, 'rb') as f:
        regressors = pickle.load(f)

    band_slices = get_band_slices(features, regressors)

    # Combining into one matrix
    X = np.hstack((regressors, features))

    # print("embedding type: ", type(embedding))
    X_train = X[:split_point, :] # delRstim from pickle files
    X_test = X[split_point:, :] # delPstim is prediction
    
    return X_train, X_test, band_slices

def load_keystrokes(keystroke_path, split_point, vol_nums):
    with open(keystroke_path, 'rb') as f:
        keystrokes = pickle.load(f)
    
    filtered_keystrokes = {kv[0]:kv[1] for i,kv in enumerate(keystrokes.items()) if vol_nums[i] == 1}
    
    Rkeys = {kv[0]:kv[1] for i,kv in enumerate(filtered_keystrokes.items()) if i < split_point}
    Pkeys = {kv[0]:kv[1] for i,kv in enumerate(filtered_keystrokes.items()) if i >= split_point}
    
    return Rkeys,Pkeys

def fmri_train_test_split(reshaped_scan):
    split_point = math.floor((reshaped_scan.shape)[0]*0.9)
    zRresp = reshaped_scan[:split_point,:] # zRresp is fMRI data
    zPresp = reshaped_scan[split_point:, :] # zPresp is fMRI data for prediction
    return zRresp, zPresp, split_point
    
def load_and_reshape_fmri_data(fmripath, vols_to_skip):
    fmri_data = nib.load(fmripath)
    scan = fmri_data.get_fdata(dtype=np.float32, caching='unchanged') # don't keep a cached copy of the full volume

    scan_2d = (np.reshape(scan, [scan.shape[0]*scan.shape[1]*scan.shape[2], scan.shape[3]]))

    # Only looking at voxels in the MNI brain that are labeled in the Schaefer Atlas.
    # Selecting these before z-scoring avoids making full-volume copies (z-scoring is per voxel, so results are the same)
    scan_2d_schaefer = scan_2d[brain_idx[cortex], :]
    del scan, scan_2d

    means = scan_2d_schaefer.mean(axis=1, keepdims=True)
    stds = scan_2d_schaefer.std(axis=1, keepdims=True)

    scan_2d_schaefer = (scan_2d_schaefer - means) / np.where(stds==0, 1, stds)
    scan_2d_schaefer = scan_2d_schaefer.T
    
    vol_nums = np.ones((scan_2d_schaefer.shape)[0])
    vol_nums[vols_to_skip] = 0
    
    scan_2d_schaefer_filtered = scan_2d_schaefer[np.where(vol_nums == 1)]
    
    return scan_2d_schaefer_filtered, vol_nums

def load_task_specific_data(p, task):
    # base = "/data/zachkaras"
    base = "/tank/home/zachkaras"
    participant_fmri_path = f"{base}/fmri_model_data/clean_{task}/{p}.nii.gz"
    participant_keystroke_path = f"{base}/fmri_model/analysis/fir/midprocess/{p}/{task}_new_keystrokes.pkl" # behemoth path
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
    base = "/tank/home/zachkaras"
    participant_path = f"{base}/fmri_model_data/fir_vectors_pca_params" # behemoth path
    participants = os.listdir(participant_path)
    num_participants = len(participants)

    for i,p in enumerate(participants):     
        print(f"Participant {p} ({i+1}/{num_participants}): Loading fMRI data")
        base_outpath = f"/tank/home/zachkaras/fmri_model_data/ridge_regression_pca_params/{p}"
        # continue    
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
        features = [e for e in embeddings if re.search(r'features', e)]
        num_embeddings = len(features)

        base_outpath = f"/tank/home/zachkaras/fmri_model_data/ridge_regression_pca_params/{p}"
        if not os.path.isfolder(base_outpath):
            os.mkdir(base_outpath)

        num_workers = NUM_GPU_WORKERS if using_gpu == 1 else len(ALLOWED_CORES)
        print(f"Participant {p} ({i+1}/{num_participants}): "
            f"processing {num_embeddings} embeddings with {num_workers} workers")

        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as ex:
            futures = {
                ex.submit(
                    ridge_regression_wrapper, 
                    feat, participant_embedding_base_path, p, base_outpath,
                    code_split_point, prose_split_point,
                    code_fmri_train, code_fmri_test,
                    prose_fmri_train, prose_fmri_test,
                    code_keys_test,prose_keys_test
                ): feat
                for feat in features
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
