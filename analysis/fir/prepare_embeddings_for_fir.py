import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Script to conduct FIR with embeddings from LLMs.")
parser.add_argument("--ndelays", required=False, default=4, help="This indicates how many delayed copies of the embedding to include (i.e., for how long the keystrokes will influence neural activity)")
args = parser.parse_args()

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

def run_participants(model_path, model, task):
    
    # TODO - sensitivity analysis to different parameters
    #           - ndelays
    #           - keystroke formatting
    #           - with/without PCA
    # run os listdir on model path to get participants
    participants = os.listdir(model_path)
    # ndelays = 4
    ndelays = args.ndelays # Updated to be a variable parameter on 12/26/2025
    delays = range(1,ndelays+1)

    for p in participants:
        # p = 133
        print(p)
        emb_datapath = f"{model_path}/{p}/{task}_keystroke_embeddings.pkl"
        try:
            with open(emb_datapath, 'rb') as f:
                embedding_dict = pickle.load(f)
        except Exception as e:
            print(f"can't open embedding: {e}")
            continue    
            
        keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_formatted_keystrokes.pkl"
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
        
        # signal_pca = reduce_dimensionality(signal)
        
        with open(f"/storage1/fmri_model_data/vols_to_skip/{p}_{task}_vols_to_skip.pkl", 'wb') as f:
            pickle.dump(vols_to_skip, f)
        
        regressor = prepare_regressor(p, task, vols_to_skip)
        
        for l,sig in signal.items():
            sig = np.hstack((regressor, sig))
            sig_pca = reduce_dimensionality(sig)
            # print("After PCA", sig_pca.shape)
            # with open("test_regressor.pkl", 'wb') as f:
            #     pickle.dump(sig, f)
            
            # delayed_sig = make_delayed(sig, delays)
            delayed_sig = make_delayed(sig_pca, delays)
            # delayed_sig = [np.array(s) for s in sig]
            # print(type(delayed_sig), len(delayed_sig), type(delayed_sig[0]))
            # outputdir = f"/storage1/fmri_model_data/fir_vectors/{p}"
            outputdir = f"/storage1/fmri_model_data/fir_vectors_pca/{p}"
            
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
            
            with open(f"{outputdir}/{model}-{task}-ndelays_{args.ndelays}-fir_embedding-{l}.pkl", 'wb') as f:
                pickle.dump(delayed_sig, f)
            #break
        
        #break


def main():
    # iterate through models
    all_models = f"/storage1/fmri_model_data/fir_embeddings"
    models = os.listdir(all_models)
    
    for m in models:
        print(m)
        model_path = f"{all_models}/{m}"
        run_participants(model_path, m, 'code')
        run_participants(model_path, m, 'prose')
        # break
    # iterate through participants
    

if __name__=="__main__":
    main()
