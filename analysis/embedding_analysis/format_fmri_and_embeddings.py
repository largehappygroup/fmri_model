import os
import torch
import numpy as np
import pandas as pd

def process_embeddings():
    pass

def process_fmri():
    pass



# read in embeddings
code_embeddings_path = "/home/zachkaras/fmri/fmri_model_data/model_embeddings/code"
prose_embeddings_path = "/home/zachkaras/fmri/fmri_model_data/model_embeddings/prose"

# embeddings hierarchy is model/ --> question_{num}_run_{num}.pt
# across the questions, need to concatenate the files based on the runs
# e.g. question_0_run_1.pt
#      question_1_run_1.pt
#      question_2_run_1.pt

# can first save the full tensor as a structure, then take the mean
# so dimensions will go from  n_tokens x d_model x 9 questions
# to 9_questions x d_model

# save in folders like midprocessing/code/embeddings/{model_name}_run_{num}.csv



# read in fMRI data
code_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/code/" # then participant number, then betas, which are in patterns of threes 1,4,7,10...
prose_fmri_path = "/home/zachkaras/fmri/fmri_model_data/beta_maps/z_scored/prose/" # then participant number, then betas, which are in patterns of threes


participants_code = os.listdir(code_fmri_path)
participants_prose = os.listdir(prose_fmri_path)
# for loop for participants

# code to pull out every third beta from each folder (1,27,3)

# find ROI voxels from betas

# format


# have different AOIs, so data format should be 
# n_questions x n_roi_voxels

# beta_map format is 

# line up embeddings into a format like n_responses x d_embedding

# line up fmri into a format like n_beta_maps x n_voxels

# save both

# save in folders like midprocessing/code/fmri/{participant}_{roi}.csv
