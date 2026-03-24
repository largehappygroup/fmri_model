import os
import pickle
# import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# parser = argparse.ArgumentParser(description="Script to conduct FIR with embeddings from LLMs.")
# parser.add_argument("--ndelays", required=False, default=4, help="This indicates how many delayed copies of the embedding to include (i.e., for how long the keystrokes will influence neural activity)")
# args = parser.parse_args()

# TODO - check 

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


def prepare_regressor(participant, task, vols_to_skip):
    num_keys_regressor_path = f"/home/zachkaras/fmri_model/analysis/fir/midprocess/{participant}/{task}_num_keystrokes_regressor.pkl"
        
    with open(num_keys_regressor_path, 'rb') as f:
        num_keys_regressor = pickle.load(f)
        
    mean = np.mean(num_keys_regressor) 
    std = np.std(num_keys_regressor)

    num_keys_regressor = [(n - mean)/std for n in num_keys_regressor]
    num_keys_regressor = [n for i,n in enumerate(num_keys_regressor) if i not in vols_to_skip]
    num_keys_regressor = np.expand_dims(np.array(num_keys_regressor), axis=1)
    
    return num_keys_regressor


# keep track of duplicated layers and condense that into one 
# save volume numbers to get rid of that data in fMRI files
# UPDATE - I tried this but it gets rid of a lot of data and performance seems to drop a lot
def organize_individual_layers(layers, keystroke_dict, embedding_dict):

    signal = { l : [] for l in layers }
    vols_without_keystrokes = set()
    for vol,keys in keystroke_dict.items():
        
        if keys == '':
            vols_without_keystrokes.add(vol)
            continue
            
        layers = embedding_dict[keys]
        for l, emb in layers.items():
            signal[l].append(emb)

    signal = { l : np.vstack(v) for l,v in signal.items()} 
    vols_to_skip = list(vols_without_keystrokes)
    vols_to_skip.sort()
    
    return signal, vols_to_skip
    

def find_layer_labels(keystroke_dict, embedding_dict):
    for k,v in keystroke_dict.items():
        if v != '':
            # the keystroke dictionary structure is vol_number : 'keystrokes'
            # the keystrokes are used as keys for the embedding dict
            # so we find the first non-empty string keystrokes
            # and use that as a key for the embedding dictionary
            # the dicitonary structure of the embedding dictionary is 
            # 'keystrokes' : {'layer_num' : [embeddings] , ...} 
            layer_labels = list((embedding_dict[v]).keys())
            return layer_labels

def run_participants(model_path, model, task):
    
    participants = os.listdir(model_path)
    
    # need codegemma 7b, code, 4 delays, look ahead by 10
    # codegemma 7b, code, 16 delays, look ahead by 0
    # codegemma 7b, prose, 10 delays, look ahead by 5
    
    for p in participants:
        
        print(p)
        look_ahead_by = [0, 1, 3, 5, 10]
        outputdir = f"/data2/zachkaras/fmri_model_data/fir_vectors_pca_params/{p}"
        
        for t in look_ahead_by:
            emb_datapath = f"{model_path}/{p}/{task}_look_ahead_by_{t}-keystroke_embeddings.pkl"
            try:
                with open(emb_datapath, 'rb') as f:
                    embedding_dict = pickle.load(f)
            except Exception as e:
                print(f"can't open embedding: {e}")
                continue    
                
            keystroke_path = f"/home/zachkaras/fmri_model/analysis/fir/midprocess/{p}/{task}-look_ahead_by_{t}-formatted_keystrokes.pkl"
            try:
                with open(keystroke_path, 'rb') as f:
                    keystroke_dict = pickle.load(f)
            except:
                continue    
            
            # For PCA, can use vector sizes of 985 from semantic tiling, 768 from continuous language, 50 from intracranial EEG
            # I found that the feature vectors need to be smaller than the other dimensions, so 768 and 985 are too big
            # 50 feels too small, so I'm trying 256 for now 10/14/2025
            # print(f"Embedding length for {model}: {len((embedding_dict[list(embedding_dict.keys())[0]])['layer_0'])}")
            layers = find_layer_labels(keystroke_dict, embedding_dict)

            # make a stack for each layer
            # signal has the structure of {'layer_0' : <ndarray of embeddings>, 'layer_5' : <ndarray of embeddings>}
            signal, vols_to_skip = organize_individual_layers(layers, keystroke_dict, embedding_dict)
            
            with open(f"/data/zachkaras/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl", 'wb') as f:
                pickle.dump(vols_to_skip, f)
            
            regressor = prepare_regressor(p, task, vols_to_skip)
            
            for l,sig in signal.items():

                # sshfs to host computed embeddings on behemoth, load in file, then add delays
                remote_dir = f"/data2/zachkaras/fir_vectors_pca_params_remote/{p}"
                remote_file = f"{remote_dir}/{model}-{task}-look_ahead_by_{t}-ndelays_0-fir_embedding-{l}.pkl"
    
                with open(remote_file, 'rb') as f:
                    base_embedding = pickle.load(f)

                sig_with_regressor = np.hstack((regressor, base_embedding))

                ndelays = [0, 4, 10, 16, 20]
                for d in ndelays:
                    delays = range(1,d+1)

                    reg_feat_outputfile = f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-regressor+features.pkl"
                    reg_outputfile = f"{outputdir}/{model}-{task}-look_ahead_by_{t}-ndelays_{d}-fir_embedding-{l}-only_regressor.pkl"
                    
                    if os.path.exists(reg_outputfile) and os.path.exists(reg_feat_outputfile):
                        continue
                    
                    # I've already done PCA on embeddings, so I'm using the base embedding then duplicating it
                    # # 3/23/2026 - looks like I didn't actually save the embeddings with the regressor
                    # # so the current embeddings are just the feature vectors.
                    # # now I need to save the regressor and the combined feature + regressor 
                    # sig_pca = reduce_dimensionality(sig, task)
                    # print("SIG PCA", sig_pca.shape)
                    delayed_sig = make_delayed(sig_with_regressor, delays) if d > 0 else sig_with_regressor
                    delayed_regressor = make_delayed(regressor, delays) if d > 0 else regressor
                    
                    if not os.path.exists(outputdir):
                        os.mkdir(outputdir)
                    
                    with open(reg_feat_outputfile, 'wb') as f:
                        pickle.dump(delayed_sig, f)
                    
                    with open(reg_outputfile, 'wb') as f:
                        pickle.dump(delayed_regressor, f)


def main():
    # iterate through models
    all_models = f"/data/zachkaras/fmri_model_data/fir_embeddings_params"
    models = os.listdir(all_models)
    
    for m in models:
        print(m)
        model_path = f"{all_models}/{m}"
        run_participants(model_path, m, 'code')
        run_participants(model_path, m, 'prose')
    

if __name__=="__main__":
    main()
