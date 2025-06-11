% there are multiple directories with participants' beta maps
% general file structure is
% fmri_model_data/beta_maps/{code/prose}/{all/questions(/loops/nonloops)}/{participant}/
% Need to read through all of them 
% and process the beta maps accordingly.
base_pathname = "/home/zachkaras/fmri/fmri_model_data/beta_maps";
beta_paths = ["code/all",... 
             "code/loops",...
             "code/nonloops",...
             "code/questions/0",...
             "code/questions/1",...
             "code/questions/2",...
             "code/questions/3",...
             "code/questions/4",...
             "code/questions/5",...
             "code/questions/6",...
             "code/questions/7",...
             "code/questions/8",...
             "prose/all",...
             "prose/questions/0",...
             "prose/questions/1",...
             "prose/questions/2",...
             "prose/questions/3",...
             "prose/questions/4",...
             "prose/questions/5",...
             "prose/questions/6",...
             "prose/questions/7",...
             "prose/questions/8"];

% Loading atlases 
% Loading a brain mask file and the atlas files for identifying seed regions
maskfile = '/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/MNI152_T1_2mm_brain_mask.nii.gz';
mask = niftiread(maskfile); % loads the full 91x109x91 mask
mni_brain_file = '/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/MNI152_T1_2mm_brain.nii.gz';
mni_brain = niftiread(mni_brain_file);
brain_idx = find(mask>0); % identifies only the voxels of the brain, not the empty space

% % Loading and create an empty template image for writing NIfTI files:
nii_template = load_untouch_nii(maskfile); % load the brain mask for an example in the right (91x109x91) space
nii_template.img = zeros(size(nii_template.img)); % replace the mask with all 0s
empty_brain = nii_template.img; % create an empty_brain image in the template's shape

atlas = niftiread('/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'); % let's pretend this is a result or seed region file
atlas_2d_brain = atlas(brain_idx); % reshapes to 2d within the brain


%% Look at each directory with beta maps
for i=1:length(beta_paths)
    curr_path = sprintf("%s/%s",base_pathname, beta_paths(i));
    
    % getting list of participants for current beta maps folder
    participants = dir(curr_path);
    for ii=3:numel(participants)
        if participants(ii).isdir
            fnames{ii-2} = participants(ii).name;
        end
    end
    
    % Processing beta maps for each participant
    for ii=1:numel(fnames)
        beta_map_path = sprintf("%s/%s", curr_path, fnames{ii});
        disp(beta_map_path)
        z_beta_map = process_beta_map(beta_map_path, brain_idx, atlas_2d_brain);
        if z_beta_map == 0 % If there wasn't a beta map to be found
            continue
        end
        z_brain = empty_brain;
        z_brain(brain_idx) = z_beta_map;
        outpath = sprintf("%s/z_scored/%s/%s_beta.nii", base_pathname, beta_paths(i), fnames{ii});
        disp(outpath)
        write_nii_cc(nii_template, z_brain, outpath);
    end
end


%% For processing, filter by only voxels that are within the brain

% z-score

% filter to top 10% of active voxels (maybe, this could be done down the road)
% Based on the 'shared representations' paper, the researchers filtered the
% voxels in the language and multiple demand system to the top 10% of
% voxels that responded during the localization task


function z_betas = process_beta_map(beta_map_path, brain_idx, empty_brain, atlas_2d_brain)
    beta_filename = sprintf("%s/beta_0001.nii", beta_map_path);
    try
        beta_map = niftiread(beta_filename);
    catch
        z_betas = 0;
        return
    end
    % make beta map 2d and filter by voxels in the brain
    % There are NaNs in the beta maps because it seems like the
    % participants' masks didn't exactly align with the MNI coordinates at
    % some point during preprocessing (ICA or more likely pipeline?)
    beta_map = beta_map(brain_idx);
    z_betas = (beta_map - nanmean(beta_map)) / nanstd(beta_map);
    
end
