import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

# def add_time_delays(signal, delays):
    
#     nt,ndim = signal.shape
#     dstims = []
#     for i,d in enumerate(delays):
#         dstim = np.zeros((nt,ndim))
#         if d < 0:
#             dstim[]
#         pass
#     pass


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
    
    # run os listdir on model path to get participants
    participants = os.listdir(model_path)
    ndelays = 4
    delays = range(1,ndelays+1)

    for p in participants:
        p = 133
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
        
        layers = find_layer_labels(keystroke_dict, embedding_dict)

        # make a stack for each layer
        # signal has the structure of {'layer_0' : <ndarray of embeddings>, 'layer_5' : <ndarray of embeddings>}
        signal, vols_to_skip = organize_individual_layers(layers, keystroke_dict, embedding_dict)
        
        with open(f"/storage1/fmri_model_data/{p}_vols_to_skip.pkl", 'wb') as f:
            pickle.dump(vols_to_skip, f)
        
        for l,sig in signal.items():
            delayed_sig = make_delayed(sig, delays)
            
            with open(f"/storage1/fmri_model_data/fir_vectors/test/{model}_code_fir_embedding_{l}.pkl", 'wb') as f:
                pickle.dump(delayed_sig, f)
            # break
        
        
        # maybe TODO - reduce dimensionality of embeddings
        
        break



def main():
    # iterate through models
    all_models = f"/storage1/fmri_model_data/fir_embeddings"
    models = os.listdir(all_models)
    
    for m in models:
        print(m)
        model_path = f"{all_models}/{m}"
        run_participants(model_path, m, 'code')
        #run_participants(model_path, 'prose')
        break
    # iterate through participants
    

if __name__=="__main__":
    main()
