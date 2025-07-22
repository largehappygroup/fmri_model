
# I have model embeddings in the form of 
# model name: deepseek 2b
# 10 different runs at temperature of 0.7
# each run contains mean residual stream in the format: n_questions x d_model

# fMRI data
# 10 regions of interest (maybe I can add more surrounding regions, or look at broader networks)
# each region contains z-scored beta values from single trial GLMs in the format: n_questions x n_voxels


# People typically calculate ridge regression between model embeddings and voxel values
# How should I set this up? 

