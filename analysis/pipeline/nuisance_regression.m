%% Nuisance Regression
% Reads in motion parameters from each participant
% then regresses them out of the signal at each voxel
% saves a clean functional file in data/clean

% for loop iterating through participants in preprocessed folder
datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess_prose";
files = dir(datapath); for i=3:numel(files); fnames{i-2}=files(i).name; end

for i=1:numel(fnames)
    tic
    % if isempty(regexp(fnames{i}, 'out'))
    %     continue
    % end
    disp(i)
    disp(fnames{i})
    name = fnames{i};
    name = regexp(fnames{i}, '.nii.gz', 'split');
    name = name{1};
    
    fprintf("regressing out nuisances for %s\n", name)
    brain_path = sprintf("%s/%s/filtered_func_data_clean.nii.gz", datapath, name);
    motion_filepath = sprintf("%s/%s/mc/prefiltered_func_data_mcf.par", datapath, name);
    
    % disp(brain_path)
    disp("reading nifti file")
    try
    
        brain_data = niftiread(brain_path);
    catch
        fprintf("Participant %s does not have a nifti file", name)
        continue
    end
    len = size(brain_data,4);

    % design matrix and data
    disp("creating design matrix")
    X = make_design_matrix(motion_filepath, len);
    Y = reshape(brain_data, [(91*109*91),len])';

    % fitting parameters to brain signal
    disp("removing noise")
    b = X\Y;
    Yhat = X*b;
    YC = Y-Yhat;
    YC = reshape(YC', [91,109,91,len]);     

    disp("saving")
    disp(name)
    outdir = "/home/zachkaras/fmri/fmri_model_data/clean_prose";
    outfile = sprintf("%s/%s", outdir, name);
    disp(outfile)
    niftiwrite(YC, outfile);
    compress_file = sprintf("gzip %s/%s.nii", outdir, name);
    system(compress_file);
    toc
    % break

end

function X = make_design_matrix(path, len)
    % mean offset, linear, and quadratic trends
    % n = 600; % num volumes
    mean_offset = ones(len,1);
    linear_trend = (1:len)';
    % quad_trend = (1:len)'.^2;

    % load parameters (clip to 600 volumes)
    Motion = importdata(path);
    Motion = Motion(1:len, :);
    dMP = diff(Motion);
    dMP = [dMP(1,:); dMP];

    % removed quadratic trend because all the timecourses turned into parabolas
    X = [mean_offset, linear_trend, Motion, dMP]; 
end

