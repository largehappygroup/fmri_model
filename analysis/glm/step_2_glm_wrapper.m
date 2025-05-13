%% Performing GLM
datapath = "/home/zachkaras/fmri/fmri_model_data/clean";
files = dir(datapath); for i=3:numel(files); fnames{i-2}=files(i).name; end % find the file names to analyze in the current directory
% nii_template = load_untouch_nii();
maskfile = 'atlases/MNI152_T1_2mm_brain_mask.nii.gz';
% mask = niftiread(maskfile); % loads the full 91x109x91 mask
% mni_brain_file = 'atlases/MNI152_T1_2mm_brain.nii.gz';
% mni_brain = niftiread(mni_brain_file);
% brain_idx = find(mask>0); % identifies only the voxels of the brain, not the empty space

% Loading and create an empty template image for writing NIfTI files:
nii_template = load_untouch_nii(maskfile); % load the brain mask for an example in the right (91x109x91) space

question_nums = dir("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions");

% Iterate through participants
for i=1:numel(fnames)

    person = char(fnames(i));
    person = person(1:3);
    fprintf("Running GLMs for participant %s\n", person)

    % Load brain data
    brain_path = sprintf("%s/%s", datapath, string(fnames(i)));
    brain_data = niftiread(brain_path);
    nframes = size(brain_data,4);
    TR = 0.8;

    % hdr = niftiinfo(fmri_filepath);
    % nframes = hdr.ImageSize(4);
    % TR = hdr.PixelDimensions(4);

    % Load in regressor(s) for each loops and nonloops
    loop_regressor_path = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/loops_nonloops/loops/%s.csv", person);
    loop_regressor = readmatrix(loop_regressor_path);

    nonloop_regressor_path = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/loops_nonloops/nonloops/%s.csv", person);
    nonloop_regressor = readmatrix(nonloop_regressor_path);

    % Perform GLM on loops and nonloops
    loop_betas = perform_glm(brain_data, TR, nframes, loop_regressor);
    nonloop_betas = perform_glm(brain_data, TR, nframes, nonloop_regressor);

    % Saving results
    disp("Saving results for loops and nonloops")
    loop_outfile = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/beta_maps/loops/%s_loops.mat", person);
    nonloop_outfile = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/beta_maps/nonloops/%s_nonloops.mat", person);

    save(loop_outfile, "loop_betas");
    save(nonloop_outfile, "nonloop_betas");
    
    % Perform GLM for each question number
    fprintf("Working on GLM for question...")
    for ii=3:numel(question_nums)
        q = question_nums(ii).name;
        fprintf("%s...", q);
        question_path = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions/%s/%s.csv", q, person);
        try
            question_regressor = readmatrix(question_path);
        catch
            fprintf("Participant %s may not have a file associated with question %s\n", person, q)
            continue
        end

        question_betas = perform_glm(brain_data, TR, nframes, question_regressor);

        % Saving output for each question
        question_outfile = sprintf("/home/zachkaras/fmri/fmri_model/midprocessing/beta_maps/questions/%s/%s_q%s.mat", q, person, q);
        save(question_outfile, "question_betas");
    end

    % break

end

% filename = sprintf("/home/zachkaras/fmri/fmri_model_data/results/%s", person);
% write_nii_cc(nii_template, loop_betas, filename);
% compress_file = sprintf("gzip /home/zachkaras/fmri/fmri_model_data/results/%s.nii", person);
% system(compress_file);






