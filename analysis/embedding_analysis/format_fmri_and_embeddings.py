import os
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict

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
    
    # find seed values of interest
    # find indices associated with seed values
    # make vectors of just those
    roi_vals = [9, 69, 73, 133, 151, 172, 192, 284, 339, 395]

    roi_dict = {r : [] for r in roi_vals}
    
    for r in roi_vals:
        roi_idx = np.argwhere(atlas_1D == r)
        roi_data = np.squeeze(data_1D[roi_idx].T)
        # print(r, len(roi_data))

        roi_dict[r] = roi_data
    return roi_dict

    # have different AOIs, so data format should be 
    # n_questions x n_roi_voxels

    # beta_map format is 

    # line up embeddings into a format like n_responses x d_embedding

    # line up fmri into a format like n_beta_maps x n_voxels

    # save both

    # save in folders like midprocessing/code/fmri/{participant}_{roi}.csv

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
def process_embeddings(embedding_path, model_name, question_num):
    
    embedding = torch.load(embedding_path)
    
    # pull out first layer
    first_layer = embedding['layer_1']
    print(first_layer.shape)



'''
def process_embeddings(embedding_path):
    embedding = torch.load(embedding_path)
    
    # this is mean for now, but probably want to use 
    mean_embedding = torch.mean(embedding, dim=0)
    return mean_embedding

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

def process_embeddings_wrapper(embedding_path, task):
    embedding_task_path = f"{embedding_path}/{task}"
    embeddings = os.listdir(embedding_task_path)

    for i,e in enumerate(embeddings):
        embedding_filepath = f"{embedding_task_path}/{e}"
        embedding_info = embeddings[i].split('_')
        model_name = f"{embedding_info[0]}_{embedding_info[1]}"
        question_num = (embedding_info[-1])[:-3]
        process_embeddings(embedding_filepath, model_name, question_num)
        break

    #print(embedding_info)

    # embeddings are now in the format of residual stream
    # need to extract the first layer, which is in the format of n_input_tokens x d_model
    # maybe put that into a tensor or numpy array before running ridge regression
    # maybe unwrap?
    model_path = f"{embedding_path}/{task}"
    models = os.listdir(model_path)
    models = [m for m in models if not re.search("csv", m)]
    
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
    process_embeddings_wrapper(embedding_path, 'code')
    #process_embeddings_wrapper(embedding_path, 'prose')



if __name__=="__main__":
    main()


