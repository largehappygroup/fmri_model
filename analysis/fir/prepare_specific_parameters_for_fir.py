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

def reduce_dimensionality(X):
    # X_t = X.T
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=256)
    # X_pca = pca.fit_transform(X_scaled)
    # return X_pca.T
    return pca.fit_transform(X_scaled)


def prepare_regressor(participant, task, vols_to_skip):
    num_keys_regressor_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{participant}/{task}_num_keystrokes_regressor.pkl"
        
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
    # repeated_keystroke_volumes = set()
    # prev="ENTRYPOINT"
    for vol,keys in keystroke_dict.items():
        
        if keys == '':
            vols_without_keystrokes.add(vol)
            continue
        
        # if keys == prev:
        #     # repeated_keystroke_volumes.add(vol)
        #     # vols_to_skip.add(vol)
        #     vols_without_keystrokes.add(vol)
        #     # print(f"Previous: {prev}\nCurrent: {keys}")
        #     continue
        # else:
        #     prev = keys
            
        layers = embedding_dict[keys]
        for l, emb in layers.items():
            signal[l].append(emb)

    signal = { l : np.vstack(v) for l,v in signal.items()} 
    vols_to_skip = list(vols_without_keystrokes)
    vols_to_skip.sort()
    
    # repeated_keystroke_volumes = list(repeated_keystroke_volumes)
    # repeated_keystroke_volumes.sort()
    # print(len(repeated_keystroke_volumes), repeated_keystroke_volumes)
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
        
def run_parameters(model_path, model, task, params):
    participants = os.listdir(model_path)
    
    for p in participants:
        for combo in params:
            delay = combo['delays']
            delays = range(1,delay+1)
            look_ahead = combo['look_ahead']
            best_layer = combo['layer']
            emb_datapath = f"{model_path}/{p}/{task}_look_ahead_by_{look_ahead}-keystroke_embeddings.pkl"
            try:
                with open(emb_datapath, 'rb') as f:
                    embedding_dict = pickle.load(f)
            except Exception as e:
                print(f"can't open embedding: {e}")
                continue  
            
            keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}-look_ahead_by_{look_ahead}-formatted_keystrokes.pkl"
            try:
                with open(keystroke_path, 'rb') as f:
                    keystroke_dict = pickle.load(f)
            except:
                continue 
            
            layers = find_layer_labels(keystroke_dict, embedding_dict)

            # make a stack for each layer
            # signal has the structure of {'layer_0' : <ndarray of embeddings>, 'layer_5' : <ndarray of embeddings>}
            signal, vols_to_skip = organize_individual_layers(layers, keystroke_dict, embedding_dict)
            # signal = signal[f"layer_{best_layer}"]
            
            regressor = prepare_regressor(p, task, vols_to_skip)
            
            sig = signal[f'layer_{best_layer}']
            
            # for l,sig in signal.items():
            sig_pca = reduce_dimensionality(sig)
            # sig_pca = np.hstack((regressor, sig_pca)) # commenting out to see the effect of the regressor 2/15/2026
            
            delayed_sig = make_delayed(sig_pca, delays) if delay > 0 else sig_pca
            outputdir = f"/s1/fmri_model_data/fir_vectors_pca_params/{p}"
            
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
            
            # with open(f"{outputdir}/{model}-{task}-look_ahead_by_{look_ahead}-ndelays_{delay}-fir_embedding-{best_layer}.pkl", 'wb') as f:
            with open(f"{outputdir}/{model}-{task}-look_ahead_by_{look_ahead}-ndelays_{delay}-fir_embedding-{best_layer}-no_regressor.pkl", 'wb') as f:
                pickle.dump(delayed_sig, f)
            

def main():
    # iterate through models
    # all_models = f"/storage1/fmri_model_data/fir_embeddings"
    # model_basepath = f"/s1/fmri_model_data/fir_embeddings_params"
    model = 'codegemma_7b'
    model_path = f"/s1/fmri_model_data/fir_embeddings_params/{model}"
    # models = os.listdir(all_models)
    
    code_params = [{ 'look_ahead': 10, 'delays' :  4, 'layer' : 20}, 
                   { 'look_ahead':  0, 'delays' : 16, 'layer' : 20}]
    prose_params = [{'look_ahead':  5, 'delays' : 10, 'layer' : 20}]
    
    run_parameters(model_path, model, 'code', code_params)
    run_parameters(model_path, model, 'prose', prose_params)
    

if __name__=="__main__":
    main()
