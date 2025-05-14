%% Processing steps for beta maps 

% Only consider voxels within MNI brain
% datapath = "/home/zachkaras/fmri/fmri_model_data/clean";
% files = dir(datapath); for i=3:numel(files); fnames{i-2}=files(i).name; end % find the file names to analyze in the current directory
% nii_template = load_untouch_nii();
maskfile = 'atlases/MNI152_T1_2mm_brain_mask.nii.gz';
mask = niftiread(maskfile); % loads the full 91x109x91 mask
mni_brain_file = 'atlases/MNI152_T1_2mm_brain.nii.gz';
mni_brain = niftiread(mni_brain_file);
brain_idx = find(mask>0); % identifies only the voxels of the brain, not the empty space

% This file can be another wrapper



