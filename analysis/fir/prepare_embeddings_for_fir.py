import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

ALLOWED_CORES = list(range(0, 25))

with open("outlier_volumes.pkl", 'rb') as f:
    outlier_vols = pickle.load(f)

fim_tags = {
    "starcoder2": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "codegemma" : ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
    "deepseek"  : ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
}

def format_keystrokes(keystrokes, model_name):
    # keystrokes is {'prefix', 'suffix', 'middle'} from prepare_prompts_from_keystrokes.py
    prefix, suffix, middle = keystrokes['prefix'], keystrokes['suffix'], keystrokes['middle']
    family = model_name.rsplit('_', 1)[0]  # e.g. "starcoder2_3b" -> "starcoder2"
    if family in fim_tags:
        pre_tag, suf_tag, mid_tag = fim_tags[family]
        # middle goes after the middle tag, so the last token is the most recent keystroke
        return f"{pre_tag}{prefix}{suf_tag}{suffix}{mid_tag}{middle}"
    # gpt2 and encoders weren't trained with FIM, so give them the text in reading order
    return f"{prefix}{middle}{suffix}"

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        # print(f"iteration {d}", dstim.shape)
        
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def reduce_dimensionality(X, task):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # pca = PCA(n_components=256)
    pca = PCA(n_components=0.99) # adjusted to 99% variance on 2/16/2026
    try:
        X_reduced = pca.fit_transform(X_scaled)
    except Exception as e:
        print("Issue with standard PCA, selecting fixed number of components")
        n_components = 256 if task == "code" else 512 # Based on the amount of explained variance when testing PCA components
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        with open("preparing_embedding_error_log.txt", '+a') as f:
            f.write(f"Issue with PCA: {e}\n")
    return X_reduced

def concatenate_task_regressor(participant, task):
    regressor_base_path = f"/tank/home/zachkaras/fmri_model/midprocessing/regressors/questions/{task}"
    questions = [q for q in range(9)]
    
    participant_df = pd.DataFrame(columns=[i for i in range(9)])
    for q in questions:
        regressor_path = f"{regressor_base_path}/{q}/{participant}.csv"
        try:
            regressor_df = pd.read_csv(regressor_path, header=None)
        except:
            continue
        participant_df[q] = regressor_df   
    return participant_df.sum(axis=1).to_numpy()


def prepare_regressors(participant, task):
    num_keys_regressor_path = f"/tank/home/zachkaras/fmri_model/analysis/fir/midprocess/{participant}/{task}_num_keystrokes_regressor.pkl"
    task_regressor = concatenate_task_regressor(participant, task)

    with open(num_keys_regressor_path, 'rb') as f:
        num_keys_regressor = pickle.load(f)

    num_keys_regressor = [(n - np.mean(num_keys_regressor))/np.std(num_keys_regressor) for n in num_keys_regressor]

    task_regressor = [(n - np.mean(task_regressor)) / np.std(task_regressor) for n in task_regressor]
    # Outlier volumes are kept here so delays line up with the full timeline; they're removed after delaying
    regressors = np.vstack((num_keys_regressor, task_regressor)).T
    
    return regressors

def compute_embedding_size(embedding_dict):
    first_key = list(embedding_dict.keys())[0]
    emb = embedding_dict[first_key]['layer_0']
    return emb.shape

# keep track of duplicated layers and condense that into one 
# save volume numbers to get rid of that data in fMRI files
# UPDATE - I tried this but it gets rid of a lot of data and performance seems to drop a lot
def organize_individual_layers(participant, model, task, layers, keystroke_dict, embedding_dict):
    # layer label should be 'layer_8'
    signal = { l : [] for l in layers }
    emb_size = compute_embedding_size(embedding_dict)
    outliers = set() # This catches all volumes that aren't during questions

    # indices of keystroke_dict are 0-indexed
    # check to see if participant has outlier volumes {'task': {'participant' : [vols]}}
    if participant in list(outlier_vols[task].keys()):
        outlier_list = [int(n) for n in outlier_vols[task][participant]]
        outliers.update(outlier_list)

    # Outlier volumes stay in the signal so delays are computed on the full timeline;
    # they're removed after delaying, using the returned vols_to_skip
    for i,(vol,keys) in enumerate(keystroke_dict.items()):
        if keys == '':
            placeholder_embedding = np.zeros(emb_size)
            for l in layers:
                signal[l].append(placeholder_embedding)    
        else:
            keys = format_keystrokes(keys, model)
            embeddings_by_layer = embedding_dict[keys]
            for l, emb in embeddings_by_layer.items():
                signal[l].append(emb)

    signal = { l : np.vstack(v) for l,v in signal.items()} 
    vols_to_skip = list(outliers)
    vols_to_skip.sort()
    
    return signal, vols_to_skip
    

def find_layer_labels(model, keystroke_dict, embedding_dict):
    for k,v in keystroke_dict.items():
        if v != '':
            # the keystroke dictionary structure is vol_number : 'keystrokes'
            # the keystrokes are used as keys for the embedding dict
            # so we find the first non-empty string keystrokes
            # and use that as a key for the embedding dictionary
            # the dicitonary structure of the embedding dictionary is 
            # 'keystrokes' : {'layer_num' : [embeddings] , ...} 
            keystrokes = format_keystrokes(v, model)
            layer_labels = list((embedding_dict[keystrokes]).keys())
            return layer_labels

def process_participant_lookahead(p, task, model_path, model, ndelays, t):
    """Worker: load embeddings once for (participant, look_ahead_by) and process all delay values."""
    emb_datapath = f"{model_path}/{p}/{task}_look_ahead_by_{t}-keystroke_embeddings.pkl"
    try:
        with open(emb_datapath, 'rb') as f:
            embedding_dict = pickle.load(f)
    except Exception as e:
        print(f"can't open embedding for {p}, t={t}: {e}")
        return
    keystroke_path = f"/tank/home/zachkaras/fmri_model/analysis/fir/midprocess/{p}/{task}-look_ahead_by_{t}-formatted_keystrokes.pkl"
    try:
        with open(keystroke_path, 'rb') as f:
            keystroke_dict = pickle.load(f)
    except Exception as e:
        print(f"can't open keystrokes for {p}, t={t}: {e}")
        return

    layers = find_layer_labels(model, keystroke_dict, embedding_dict)
    signal, vols_to_skip = organize_individual_layers(str(p), model, task, layers, keystroke_dict, embedding_dict)

    with open(f"/tank/home/zachkaras/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl", 'wb') as f:
        pickle.dump(vols_to_skip, f)

    regressors = prepare_regressors(p, task)

    outputdir = f"/tank/home/zachkaras/fmri_model_data/fir_vectors_pca_params/{p}"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)

    # Compute PCA once per layer, then apply each delay set
    for l, sig in signal.items():
        all_done = all(
            os.path.exists(f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-regressor+features.pkl")
            and os.path.exists(f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-only_regressor.pkl")
            for d in ndelays
        )
        if all_done:
            continue

        sig_pca = reduce_dimensionality(sig, task)

        for d in ndelays:
            delays = range(1, d + 1)
            feat_outputfile = f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-features.pkl"
            reg_outputfile = f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-regressors.pkl"

            if os.path.exists(feat_outputfile) and os.path.exists(reg_outputfile):
                continue

            assert len(regressors) == len(sig_pca), f"{p} {task} {l}: {len(regressors)} regressor vols vs {len(sig_pca)} embedding vols"

            delayed_sig = make_delayed(sig_pca, delays) if d > 0 else sig_pca
            delayed_regressor = make_delayed(regressors, delays) if d > 0 else regressors

            # Delays are computed on the full timeline above, so outlier volumes are only removed now
            keep = np.ones(delayed_sig.shape[0], dtype=bool)
            keep[vols_to_skip] = False
            delayed_sig = delayed_sig[keep]
            delayed_regressor = delayed_regressor[keep]

            with open(feat_outputfile, 'wb') as f:
                pickle.dump(delayed_sig, f)
            with open(reg_outputfile, 'wb') as f:
                pickle.dump(delayed_regressor, f)


def init_worker():
    os.sched_setaffinity(0, ALLOWED_CORES)

def run_participants(model_path, model, task):
    participants = os.listdir(model_path)
    ndelays = [0, 4, 10, 16, 20]
    look_ahead_by = [0, 1, 3, 5, 10]

    jobs = [(p, t) for p in participants for t in look_ahead_by]
    num_jobs = len(jobs)
    print(f"{model} {task}: {num_jobs} jobs across {len(ALLOWED_CORES)} workers")
    with ProcessPoolExecutor(max_workers=len(ALLOWED_CORES), initializer=init_worker) as ex:
        futures = {
            ex.submit(process_participant_lookahead, p, task, model_path, model, ndelays, t): (p, t)
            for p, t in jobs
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            p, t = futures[fut]
            try:
                fut.result()
                print(f"  [{i}/{num_jobs}] done: {p} t={t}")
            except Exception as e:
                print(f"  [{i}/{num_jobs}] error: {p} t={t}: {e}")


def main():
    # iterate through models
    all_models = f"/tank/home/zachkaras/fmri_model_data/fir_embeddings_params"
    models = os.listdir(all_models)
    
    for m in models:
        print(m)
        model_path = f"{all_models}/{m}"
        run_participants(model_path, m, 'code')
        run_participants(model_path, m, 'prose')

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
