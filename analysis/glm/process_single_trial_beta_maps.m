maskfile = '/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/MNI152_T1_2mm_brain_mask.nii.gz';
mask = niftiread(maskfile); % loads the full 91x109x91 mask
mni_brain_file = '/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/MNI152_T1_2mm_brain.nii.gz';
mni_brain = niftiread(mni_brain_file);
brain_idx = find(mask>0); % identifies only the voxels of the brain, not the empty space

% % Loading and create an empty template image for writing NIfTI files:
nii_template = load_untouch_nii(maskfile); % load the brain mask for an example in the right (91x109x91) space
nii_template.img = zeros(size(nii_template.img)); % replace the mask with all 0s
empty_brain = nii_template.img; % create an empty_brain image in the template's shape


base_pathname = "/home/zachkaras/fmri/fmri_model_data/beta_maps";
generate_paths(base_pathname, brain_idx, empty_brain, base_pathname, nii_template)

% atlas = niftiread('/home/zachkaras/fmri/fmri_model/analysis/pipeline/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'); % let's pretend this is a result or seed region file
% atlas_2d_brain = atlas(brain_idx); % reshapes to 2d within the brain

function generate_paths(path, brain_idx, empty_brain, base_pathname, nii_template)

    beta_paths_code = [];
    beta_paths_prose = [];
    
    code_path = sprintf("%s/code/questions", path);
    prose_path = sprintf("%s/prose/questions", path);

    code_folders = dir(code_path);
    prose_folders = dir(prose_path);

    pluck_beta_maps(code_folders, 'code', brain_idx, empty_brain, base_pathname, nii_template);
    pluck_beta_maps(prose_folders, 'prose', brain_idx, empty_brain, base_pathname, nii_template);

end

function pluck_beta_maps(folders, task, brain_idx, empty_brain, base_pathname, nii_template)

    question_betas = 'beta_00(01|04|07|10|13|16|19|22|25)';

    for i=3:length(folders)
        person = folders(i).name;

        outputdir = sprintf("%s/z_scored/%s/%s", base_pathname, task, person);
        mkdir(outputdir)

        fprintf("\nProcessing beta maps for participant %s\n", person)
        currdir = sprintf("%s/%s", folders(i).folder, person);

        all_betas = dir(currdir);
        files = {all_betas.name};

        matches = ~cellfun('isempty', regexp(files, question_betas));
        specific_betas = all_betas(matches);

        process_beta_map(specific_betas, task, person, brain_idx, empty_brain, base_pathname, nii_template)
        
    end
end


function process_beta_map(beta_maps, task, person, brain_idx, empty_brain, base_pathname, nii_template)

    % global brain_idx empty_brain base_pathname  

    % iterating through beta maps of given participant's directory
    fprintf("processing question ")
    for i=1:length(beta_maps)
        fprintf("%d (%s)... ", i, beta_maps(i).name)

        % need file name
        curr_beta = sprintf("%s/%s", beta_maps(i).folder, beta_maps(i).name);
        % disp(curr_beta)
        try
            beta_map = niftiread(curr_beta);
        catch
            fprintf("issue reading beta map for %s", curr_beta)
            continue
        end

        beta_map = beta_map(brain_idx);
        z_betas = (beta_map - nanmean(beta_map)) / nanstd(beta_map);
        z_brain = empty_brain;
        z_brain(brain_idx) = z_betas;

        outpath = sprintf("%s/z_scored/%s/%s/question_%d_beta.nii", base_pathname, task, person, i);
        % disp(outpath)
        write_nii_cc(nii_template, z_brain, outpath);

    end
end