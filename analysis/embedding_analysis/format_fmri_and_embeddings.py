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
    data_nii = nib.load(brain_path)
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

def beta_processing_wrapper(participants, fmri_path):
    
    # iterating through each participant
    for p in participants:
        print(f"Processing data from participant {p}")
        beta_path = f"{fmri_path}/{p}"
        betas = os.listdir(beta_path)


        # making a dictionary where key values are the question numbers
        # Then the values will also be dictionaries
        # whose keys are the roi numbers, and whose values are the voxel values from that roi
        collected_betas = { question_num : {} for question_num in range(9)}

        for i,f in enumerate(betas):
            print(f"question {i}")
            beta_file = f"{beta_path}/{f}"
            print(beta_file)
            roi_activity = process_fmri(beta_file)
            collected_betas[i] = roi_activity
            # print(roi_activity.keys())
            # print(roi_activity)
            # break
        questions_by_roi = defaultdict(dict)
        for question_num,roi_val in collected_betas.items():
            for roi,voxel_vals in roi_val.items():
                # print(collected_betas.keys(), collected_betas[question_num].keys())
                # collected_betas[question_num][roi]
                # print(voxel_vals)
                # print(voxel_vals)
                questions_by_roi[roi][question_num] = voxel_vals

        for roi,questions_activity in questions_by_roi.items():
            # print(questions_activity.keys())
            
 
            df = pd.DataFrame.from_dict(questions_activity)
            df.T.to_csv(f"/home/zachkaras/fmri/test_roi_{roi}.csv")
            # break

        # print(questions_by_roi.keys(), questions_by_roi[9].keys())
        # # betas_by_seed_values = {roi : voxel_vals for roi in for question_num in collected_betas.keys()}
        # print(collected_betas.keys(), collected_betas[0].keys(), collected_betas[0][9])
        break


#######################################################################
############ Embedding Processing #####################################
#######################################################################

def process_embeddings():
    pass


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
    code_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/code" # then participant number, then betas, which are in patterns of threes 1,4,7,10...
    prose_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/prose" # then participant number, then betas, which are in patterns of threes


    # --- Processing fMRI Data ---
    # Code
    participants_code = os.listdir(code_fmri_path)
    participants_code = [f for f in participants_code if re.match(r'[0-9]{3}', f)]
    beta_processing_wrapper(participants_code, code_fmri_path)

    # Prose
    participants_prose = os.listdir(prose_fmri_path)
    participants_prose = [f for f in participants_prose if re.match(r'[0-9]{3}', f)]
    # beta_processing_wrapper(participants_prose, prose_fmri_path)

    # --- Processing Model Embeddings ---
    code_embeddings_path = "/home/zachkaras/fmri/fmri_model_data/model_embeddings/code"
    prose_embeddings_path = "/home/zachkaras/fmri/fmri_model_data/model_embeddings/prose"

# for loop for participants

if __name__=="__main__":
    main()


