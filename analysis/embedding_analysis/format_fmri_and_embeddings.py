import os
import re
import math
import torch
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict

parser = argparse.ArgumentParser(description="Method for embedding extraction")

parser.add_argument("--cls", required=False, default=False, help="Opting to extract CLS tokens instead of using mean pooling for embedding extraction.")

#######################################################################
#################### fMRI Variables ###################################
#######################################################################


# load in atlases
# mask file
maskfile = "../pipeline/atlases/MNI152_T1_2mm_brain_mask.nii"
mask_nii = nib.load(maskfile)
mask = mask_nii.get_fdata()

# getting indexes of brain in MNI file
brain_idx = np.argwhere(mask)

# mni brain file
mni_brain_file = "../pipeline/atlases/MNI152_T1_2mm_brain.nii.gz"
mni_brain_nii = nib.load(mni_brain_file)
mni_brain = mni_brain_nii.get_fdata()

# Schaefer atlas
atlas_file = "../pipeline/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
atlas_nii = nib.load(atlas_file)
atlas = atlas_nii.get_fdata()
atlas_1D = atlas[tuple(brain_idx.T)]


#######################################################################
#################### fMRI Processing ##################################
#######################################################################

def process_fmri(brain_path):
    # find ROI voxels from betas

    # format
    try:
        data_nii = nib.load(brain_path)
    except:
        print(f"No data for {brain_path}")
        return 0
    data = data_nii.get_fdata()
    data_1D = data[tuple(brain_idx.T)]
    
    # dictionary mapping each schaefer parcel onto corresponding region
    roi_regions = {
        # Left inferior temporal gyrus
        69 : 'litg', 70 : 'litg', 71 : 'litg', 133 : 'litg', 151 : 'litg', 154 : 'litg',
        
        # Right inferior temporal gyrus
        204 : 'ritg', 209 : 'ritg', 271 : 'ritg', 330 : 'ritg', 338 : 'ritg', 339 : 'ritg',
        
        # Left superior parietal lobule
        72 : 'lspl', 73 : 'lspl', 93 : 'lspl', 95 : 'lspl', 96 : 'lspl', 159 : 'lspl', 160 : 'lspl', 162 : 'lspl', 163 : 'lspl',
        
        # Right superior parietal lobule
        284 : 'rspl', 332 : 'rspl', 333 : 'rspl', 335 : 'rspl', 364 : 'rspl', 365 : 'rspl',
        
        # Lingual Gyrus
        4 : 'lg', 5 : 'lg', 9 : 'lg',
        
        # Left inferior frontal gyrus (Broca's Area)
        104 : 'lifg', 105 : 'lifg', 136 : 'lifg', 166 : 'lifg', 170 : 'lifg', 172 : 'lifg', 175 : 'lifg', 
        
        # Right inferior frontal gyrus 
        309 : 'rifg', 377 : 'rifg', 378 : 'rifg', 
        
        # Right temporo/parietal/occipital
        216 : 'rtpo', 222 : 'rtpo', 273 : 'rtpo', 274 : 'rtpo', 363 : 'rtpo', 364 : 'rtpo'
    }
    
    roi_vals = list(roi_regions.keys())
    
    # Original ROIs: [9, 69, 73, 133, 151, 172, 192, 284, 339, 395]

    # roi_dict = {r : [] for r in roi_vals}
    
    roi_dict = {
        'litg' : [],
        'ritg' : [],
        'lspl' : [],
        'rspl' : [],
        'lg'   : [],
        'lifg' : [],
        'rifg' : [],
        'rtpo' : []
    }
    
    for r in roi_vals:
        roi_idx = np.argwhere(atlas_1D == r)
        roi_data = np.squeeze(data_1D[roi_idx].T)
        
        roi_key = roi_regions[r]
        roi_dict[roi_key].extend(roi_data)
        
    return roi_dict

def beta_processing_wrapper(participants, fmri_path, task):
    
    # iterating through each participant
    for p in participants:
        print(f"Processing data from participant {p}")
        beta_files = [f"{fmri_path}/{question_num}/{p}_beta.nii" for question_num in range(0,9)]

        # making a dictionary where key values are the question numbers
        # Then the values will also be dictionaries
        # whose keys are the roi numbers, and whose values are the voxel values from that roi
        collected_betas = { question_num : {} for question_num in range(9)}
        
        # need to make a for loop through each of the 0-8 questions, then load the beta map for each

        for i,f in enumerate(beta_files):
            print(f"question {i}")
            roi_activity = process_fmri(f)
            if roi_activity == 0:
                continue
            collected_betas[i] = roi_activity
            
        questions_by_roi = defaultdict(dict)
        for question_num,roi_val in collected_betas.items():
            for roi,voxel_vals in roi_val.items():
                questions_by_roi[roi][question_num] = voxel_vals
                
        output_path = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/midprocessing/{task}/human/{p}"
        os.system(f'mkdir {output_path}')

        for roi,questions_activity in questions_by_roi.items():
            df = pd.DataFrame.from_dict(questions_activity)
            df.T.to_csv(f"{output_path}/roi_{roi}.csv")
            
        # break

#######################################################################
############ Embedding Processing #####################################
#######################################################################

'''

def aggregate_runs_by_question(model_path, model_name, task):
    print(f"Processing embeddings for {model_name}")
    
    full_embedding_path = f"{model_path}/{model_name}"
    output_dir = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/midprocessing/{task}/model/{model_name}"
    os.system(f"mkdir {output_dir}")
    
    # runs are basically 'participants'
    runs = range(1,11)
    
    for r in runs:
        print(f"Run {r}")
        questions = range(0,9)
        model_embeddings = {question_num : [] for question_num in questions}
        
        for q in questions:
            embedding_path = f"{full_embedding_path}/question_{q}_run_{r}.pt"
            question_embedding = process_embeddings(embedding_path)
            model_embeddings[q] = question_embedding
        
        
        output_file = f"run_{r}.csv"
        
        df = pd.DataFrame.from_dict(model_embeddings)
        df.T.to_csv(f"{output_dir}/{output_file}")
        # break
'''
def save_embeddings(embeddings, output_dir):
    # this is a dictionary with structure {layer_num : embeddings, }
    for layer,question_embeddings in embeddings.items():
            output_file = f"{layer}.csv"
        
            # saving in the format model_name, layer_num
            df = pd.DataFrame.from_dict(question_embeddings)
            df.T.to_csv(f"{output_dir}/{output_file}")


def pad_model_embeddings(embeddings):
    # embeddings have the structure layer : {question_num : []}
    for layer, question_embedding in embeddings.items():
        max_length = max([emb.shape[0] for emb in question_embedding.values()])
        
        for question, embedding in question_embedding.items():
            pad_width = max_length - embedding.shape[0]
            padded = np.pad(embedding, (0, pad_width), mode='constant')

            embeddings[layer][question] = padded

    return embeddings

def process_layer(layer):
    # this is mean for now, but probably want to use 
    # averaging across tokens to give a feature vector describing each token,
    # rather than the behavior of the features across the tokens
    
    mean_embedding = torch.mean(layer, dim=1)
    return mean_embedding


def process_embeddings(embedding_path, task, model_name, question_num, cls=False, num_samples=4):
    
    print(f"Processing model {model_name} for {task}, question {question_num}")
    embedding = torch.load(embedding_path)
    # print("SHAPE ", embedding['layer_0'].shape)
    # return
    # print(len(embedding.keys()), embedding.keys(), embedding)
    
    # if cls:
        
    #     cls_tokens = [embedding[layer][0] for layer in embedding.keys()]
    #     # print(len(cls_tokens), type(cls_tokens), len(cls_tokens[0]), type(cls_tokens[0]))
    #     return cls_tokens
    # return
    # descriptive variables for accessing some intermediate layers too
    num_layers = len(embedding.keys())
    all_layers = list(embedding.keys())
    num_layers = len(all_layers)
    step = math.floor(num_layers/num_samples)
    
    intermediate_layer_idx = [n for n in range(0, num_layers, step)]
    intermediate_layer_labels = [all_layers[i] for i in intermediate_layer_idx]
    
    
    # I'd like to look at more than just the first layer, but I want to avoid just looking at every layer
    # I could do a sampling across the embeddings, maybe 5?
    # first layer, last layer, 3 intermediate layers
    
    # layered like an onion - aggregating the different layers for a given question
    onion = defaultdict()
    
    for layer_label in intermediate_layer_labels:
        this_layer = embedding[layer_label]
        
        # if cls:
        #     cls_token = this_layer[0]
        #     print(len(cls_token))
        # else:
        processed_layer = this_layer[0] if cls else process_layer(this_layer)
        print(len(processed_layer))
        
        onion[layer_label] = processed_layer
        
    # Saving last layer
    last_layer_label = all_layers[-1]
    last_layer = embedding[last_layer_label][0] if cls else process_layer(embedding[last_layer_label])
    print(len(last_layer))
    
    onion[last_layer_label] = last_layer
    
    return onion

def process_embeddings_wrapper(embedding_path, task, cls):
    embedding_task_path = f"{embedding_path}/{task}"
    embeddings = os.listdir(embedding_task_path)
    
    # Accumulating a list of the model names
    models = set()
    for e in embeddings:
        split_model_name = e.split('_')
        model_name = f"{split_model_name[0]}_{split_model_name[1]}"
        models.add(model_name)
    
    # iterating through those model names to collect data from each question
    for m in models:        
        output_dir = f"/home/zachkaras/fmri/fmri_model/analysis/embedding_analysis/midprocessing/{task}/model_cls/{m}"
        os.system(f'mkdir {output_dir}')
        questions = range(0,9)
        
        # need this structure but for every layer
        # model_embeddings = {question_num : [] for question_num in questions}
        model_embeddings = defaultdict(dict)
        for q in questions:
            
            # this path now corresponds to each layer of a given LLM
            # I need to extract the different layers I care about
            # then save them into their own files
            # How can I return the outputs? Maybe a dictionary?
            embedding_path = f"{embedding_task_path}/{m}_question_{q}.pt"
            
            # this is a dictionary with structure {layer_num : embeddings, }
            # question_embedding_layers = process_embeddings(embedding_path, task, m, q)
            question_embedding_layers = process_embeddings(embedding_path, task, m, q, cls)
            
            # if isinstance(question_embedding_layers, dict):
                # for loop going through all the keys
            for layer,emb in question_embedding_layers.items():
                model_embeddings[layer][q] = emb
            # else:
            #     # otherwise it's a 
            #     # process the cls tokens
            #     pass
            # break
        
        padded_embeddings = pad_model_embeddings(model_embeddings)
        save_embeddings(padded_embeddings, output_dir)
        
        
        # break
            # model_embeddings[q] = question_embedding
    
    
    # models = {(e.split('_'))[0:2] for e in embeddings}
    # print(models)
    
    # embeddings.sort()
    # print(embeddings)
    # return

    # for i,e in enumerate(embeddings):
        
        # can hijack this loop to process all the questions from a specific model
        
            
        # embedding_filepath = f"{embedding_task_path}/{e}"
        # embedding_info = embeddings[i].split('_')
        # model_name = f"{embedding_info[0]}_{embedding_info[1]}"
        # question_num = (embedding_info[-1])[:-3]
        # process_embeddings(embedding_filepath, task, model_name, question_num)
        # break

    #print(embedding_info)

    # embeddings are now in the format of residual stream
    # need to extract the first layer, which is in the format of n_input_tokens x d_model
    # maybe put that into a tensor or numpy array before running ridge regression
    # maybe unwrap?
    # model_path = f"{embedding_path}/{task}"
    # models = os.listdir(model_path)
    # models = [m for m in models if not re.search("csv", m)]
    
    #for m in models:        
    #    aggregate_runs_by_question(model_path, m, task)
        # break


# read in embeddings


# embeddings hierarchy is model/ --> question_{num}_run_{num}.pt
# across the questions, need to concatenate the files based on the runs
# e.g. question_0_run_1.pt
#      question_1_run_1.pt
#      question_2_run_1.pt

# can first save the full tensor as a structure, then take the mean
# so dimensions will go from  n_tokens x d_model x 9 questions
# to 9_questions x d_model

# save in folders like midprocessing/code/embeddings/{model_name}_run_{num}.csv

#######################################################################
############ Main Function ############################################
#######################################################################

def main():
    
    args = parser.parse_args()
    cls = args.cls
    # read in fMRI data
    # read from 
    # code_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/code" #
    # prose_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/prose" #
    code_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/code/questions" # 
    prose_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/prose/questions" # then 0-8/then {participant_id}_beta.nii 

    # --- Processing fMRI Data ---
    # Code
    participants_code = os.listdir(f"{code_fmri_path}/0")
    participants_code = [f[0:3] for f in participants_code]
    participants_code.sort()
    # beta_processing_wrapper(participants_code, code_fmri_path, 'code')

    # Prose
    participants_prose = os.listdir(f"{prose_fmri_path}/0")
    participants_prose = [f[0:3] for f in participants_prose]
    participants_prose.sort()
    # beta_processing_wrapper(participants_prose, prose_fmri_path, 'prose')

    # --- Processing Model Embeddings ---
    embedding_path = "/home/zachkaras/fmri/fmri_model_data/model_embeddings"
    process_embeddings_wrapper(embedding_path, 'code', cls)
    # process_embeddings_wrapper(embedding_path, 'prose', cls)



if __name__=="__main__":
    main()


