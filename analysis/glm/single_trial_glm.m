% if  ~exist('./data/nsdflocexampledataset.mat','file')
%     % download data with curl
%     system('curl -L --output ./data/nsdflocexampledataset.mat https://osf.io/g42tm/download')
% end

% load('/home/zachkaras/fmri/fmri_model/analysis/glm/data/nsdflocexampledataset.mat')
% calculating single trial betas using GLMsingle

% output directories
outputdir_root = "/home/zachkaras/fmri/fmri_model_data/beta_maps";
code_outputdir = sprintf("%s/code/questions", outputdir_root);

tr = 0.8;
stimdur = 60;
% prose_outputdir = sprintf("%s/prose/questions", outputdir_root);


% iterate_through_participants(prose_datapath, "prose")
code_datapath = "/home/zachkaras/fmri/fmri_model_data/midprocess/";
iterate_through_participants(code_datapath, "code", code_outputdir, stimdur, tr)

% need to format excel files as design matrix


% need to load in corresponding brain data
function iterate_through_participants(datapath, task, outputdir, stimdur, tr)
    
    files = dir(datapath); 
    for i=3:numel(files) 
        if ~contains(files(i).name, 'pkl')
            fnames{i-2}=files(i).name; 
        end
    end

    
    % disp(X)
    % if task == "prose"
    %     blocknum = 1;
    % elseif task == "code"
    %     blocknum = 3;
    % end

    for i=3:length(fnames)
        brain_data_path = sprintf("%s/%s/filtered_func_data_clean.nii.gz", datapath, fnames{i});
        % disp("l")
        brain_data = niftiread(brain_data_path);
        design = create_design_matrix(fnames{i});

        opt = struct('wantmemoryoutputs',[1 1 1 1]);

        % This example saves output .mat files to the folder
        % "example2outputs/GLMsingle". If these outputs don't already exist, we
        % will perform the time-consuming call to GLMestimatesingletrial.m;
        % otherwise, we will just load from disk.
        full_outputdir = sprintf('%s/GLMsingle', outputdir);
        if ~exist(full_outputdir,'dir')
        
            [results] = GLMestimatesingletrial(design,brain_data,stimdur,tr,full_outputdir,opt);
        
            % We assign outputs of GLMestimatesingletrial to "models" structure.
            % Note that results{1} contains GLM estimates from an ONOFF model,
            % where all images are treated as the same condition. These estimates
            % could be potentially used to find cortical areas that respond to
            % visual stimuli. We want to compare beta weights between conditions
            % therefore we are not going to store the ONOFF GLM results.
        
            clear models;
            models.FIT_HRF = results{2};
            models.FIT_HRF_GLMdenoise = results{3};
            models.FIT_HRF_GLMdenoise_RR = results{4};
        
        else
            % Load existing file outputs if they exist
            results = load([outputdir '/GLMsingle/TYPEB_FITHRF.mat']);
            models.FIT_HRF = results;
            results = load([outputdir '/GLMsingle/TYPEC_FITHRF_GLMDENOISE.mat']);
            models.FIT_HRF_GLMdenoise = results;
            results = load([outputdir '/GLMsingle/TYPED_FITHRF_GLMDENOISE_RR.mat']);
            models.FIT_HRF_GLMdenoise_RR = results;
        end
        break

    end
end


function X = create_design_matrix(person)
    base_datapath = "/home/zachkaras/fmri/fmri_model/midprocessing/regressors/questions";

    X = [];
    for i=0:8
        regressor_path = sprintf("%s/%d/%s.csv", base_datapath, i, person);
        try
            regressor = readtable(regressor_path);
        catch
            fprintf("Couldn't find file for %s on question %d\n", person, i)
            continue
        end
        
        X = [X, table2array(regressor)];
    end
end




