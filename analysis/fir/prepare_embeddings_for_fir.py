import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def organize_individual_layers(layers, keystroke_dict, embedding_dict):

    signal = { l : [] for l in layers }
    vols_without_keystrokes = set()
    for vol,keys in keystroke_dict.items():
        
        if keys == '':
            vols_without_keystrokes.add(vol)
            # TODO record and remove volumes without any keystroke data
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

def run_participants(model_path, task):
    
    # run os listdir on model path to get participants
    participants = os.listdir(model_path)

    for p in participants:
        print(p)
        
        emb_datapath = f"{model_path}/{p}/{task}_keystroke_embeddings.pkl"
        try:
            with open(emb_datapath, 'rb') as f:
                embedding_dict = pickle.load(f)
        except:
            continue    
            
        keystroke_path = f"/home/zachkaras/fmri/fmri_model/analysis/fir/midprocess/{p}/{task}_formatted_keystrokes.pkl"
        try:
            with open(keystroke_path, 'rb') as f:
                keystroke_dict = pickle.load(f)
        except:
            continue    
        
        layers = find_layer_labels(keystroke_dict, embedding_dict)

        # make a stack for each layer
        signal, vols_to_skip = organize_individual_layers(layers, keystroke_dict, embedding_dict)
        
        # TODO downsample signal based on TR times using Lanczos filter
        # try using original code from Huth Tutorial
        
        # Do I need to downsample? The volumes and keystrokes should already line up
        
        # Can I just directly run FIR?
        
        # after I downsample 
        break





def main():
    # iterate through models
    all_models = f"/home/zachkaras/fmri/fmri_model_data/fir_embeddings"
    models = os.listdir(all_models)
    
    for m in models:
        model_path = f"{all_models}/{m}"
        run_participants(model_path, 'code')
        #run_participants(model_path, 'prose')
        break
    # iterate through participants
    

if __name__=="__main__":
    main()
